"""
Script to analyze how token geometry evolves during training.
Trains a PostLN LLaMA-style transformer for a specific number of iterations
and compares token geometry before and after training.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.model.utils import create_random_model
from src.analyzer.geometry import GeometryAnalyzer
from src.utils.wikitext_dataset import WikiTextDataset
from src.analyzer.visualization import plot_similarity_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_tiktoken_tokenizer():
    """
    Create a tiktoken tokenizer directly to avoid import issues.
    
    Returns:
        Tiktoken tokenizer
    """
    try:
        import tiktoken
        from src.tokenizer.base import BaseTokenizer
        
        class TiktokenTokenizer(BaseTokenizer):
            def __init__(self, model_name='gpt2'):
                self.encoding = tiktoken.get_encoding(model_name)
                self.vocab_size = self.encoding.n_vocab
                logger.info(f"Created Tiktoken tokenizer with model {model_name}, vocab size: {self.vocab_size}")
            
            def encode(self, text):
                return self.encoding.encode(text)
            
            def decode(self, tokens):
                return self.encoding.decode(tokens)
        
        return TiktokenTokenizer()
    except Exception as e:
        logger.error(f"Error creating Tiktoken tokenizer: {e}")
        
        # Fall back to BPE
        try:
            from src.tokenizer.bpe import BPETokenizer
            if os.path.exists('data/embedding/encoder.json'):
                logger.info("Falling back to BPE tokenizer")
                return BPETokenizer('data/embedding/encoder.json', 'data/embedding/vocab.bpe')
        except Exception:
            pass
            
        # Fall back to character tokenizer as last resort
        from src.tokenizer.character import CharTokenizer
        logger.info("Falling back to character tokenizer")
        return CharTokenizer(256)

def create_model(vocab_size, n_layer=12, use_swiglu=True, ln_type="postln", use_initial_ln=True, mixln_split=0.25):
    """
    Create a GPT model with the specified parameters.
    
    Args:
        vocab_size: Size of the vocabulary
        n_layer: Number of transformer layers
        use_swiglu: Whether to use SwiGLU activation
        ln_type: Layer normalization type (preln, postln, periln, mixln)
        use_initial_ln: Whether to use initial layer normalization
        mixln_split: Fraction of layers to use postln in mixln architecture
        
    Returns:
        GPT model
    """
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=1024,
        n_layer=n_layer,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=False,
        ln=ln_type,
        use_initial_ln=use_initial_ln,
        mixln_split=mixln_split,
        use_swiglu=use_swiglu,
        initializer_range=0.02  # Add standard deviation for weight initialization
    )
    
    model_type = "LLaMA" if use_swiglu else "Standard"
    model_name = f"{model_type}-{ln_type.capitalize()}"
    
    return GPT(config), model_name

def train_for_iterations(model, dataset, iterations=100, batch_size=8, 
                        learning_rate=1e-5, device='cuda'):
    """
    Train model for a specific number of iterations.
    
    Args:
        model: Model to train
        dataset: Dataset to train on
        iterations: Number of iterations to train for
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Losses during training
    """
    model = model.to(device)
    model.train()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    losses = []
    iteration = 0
    
    progress_bar = tqdm(total=iterations, desc="Training")
    
    while iteration < iterations:
        # Loop through batches
        for x, y in dataloader:
            if iteration >= iterations:
                break
                
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            losses.append(loss.item())
            
            # Update iteration counter
            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log every 10 iterations
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}/{iterations} | Loss: {loss.item():.4f}")
    
    progress_bar.close()
    return losses

def analyze_token_geometry(model, tokenizer, prompt, device='cuda'):
    """
    Analyze token geometry for a model.
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer to use
        prompt: Prompt to analyze
        device: Device to run on
        
    Returns:
        Dictionary with complete analysis results including:
        - similarity_matrices: Cosine similarity matrices per layer
        - metrics: Token geometry metrics (cosine_sim, token_norm, update_norm)
        - tokens: The tokenized input
    """
    model = model.to(device)
    model.eval()
    
    # Create analyzer
    analyzer = GeometryAnalyzer(model, device)
    
    # Analyze prompt
    analysis_result = analyzer.analyze_single_prompt(prompt, tokenizer)
    
    # Clean up
    analyzer.cleanup()
    
    return analysis_result

def plot_geometry_comparison(initial_results, final_results, output_dir):
    """
    Plot comparison of token geometry before and after training,
    including similarity matrices and additional metrics.
    
    Args:
        initial_results: Analysis results for initial model
        final_results: Analysis results for final model
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract similarity matrices from results
    initial_matrices = initial_results.get('similarity_matrices', {})
    final_matrices = final_results.get('similarity_matrices', {})
    
    # Extract metrics
    initial_metrics = initial_results.get('metrics', {})
    final_metrics = final_results.get('metrics', {})
    
    # Get common layers for similarity matrices
    common_layers = sorted(set(initial_matrices.keys()) & set(final_matrices.keys()))
    
    if not common_layers:
        logger.error("No common layers between initial and final models")
        return
    
    # Select a subset of layers to visualize
    sample_layers = [
        common_layers[0],  # First layer
        common_layers[len(common_layers)//3],  # ~33% through
        common_layers[2*len(common_layers)//3],  # ~66% through
        common_layers[-1]  # Last layer
    ]
    
    # Create a grid of plots - columns: before/after, rows: layers
    fig, axs = plt.subplots(len(sample_layers), 3, figsize=(15, 5*len(sample_layers)))
    fig.suptitle("Token Geometry Evolution During Training", fontsize=16)
    
    # Column titles
    col_titles = ["Initial Model", "Trained Model", "Difference"]
    for j, title in enumerate(col_titles):
        axs[0, j].set_title(title, fontsize=14)
    
    # Plot each layer
    for i, layer_idx in enumerate(sample_layers):
        # Row labels
        axs[i, 0].set_ylabel(f"Layer {layer_idx}", fontsize=14)
        
        # Get matrices
        init_matrix = initial_matrices[layer_idx]
        final_matrix = final_matrices[layer_idx]
        diff_matrix = final_matrix - init_matrix
        
        # Determine common color scale
        vmin = min(init_matrix.min(), final_matrix.min())
        vmax = max(init_matrix.max(), final_matrix.max())
        
        # Plot initial matrix
        im0 = axs[i, 0].imshow(init_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 0].axis('off')
        
        # Plot final matrix
        im1 = axs[i, 1].imshow(final_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[i, 1].axis('off')
        
        # Plot difference matrix with different color scale
        diff_scale = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
        im2 = axs[i, 2].imshow(diff_matrix, cmap='coolwarm', vmin=-diff_scale, vmax=diff_scale)
        axs[i, 2].axis('off')
    
    # Add colorbars
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    fig.colorbar(im1, cax=cbar_ax1, label="Cosine Similarity")
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(im2, cax=cbar_ax2, label="Difference")
    
    # Save the figure
    output_path = os.path.join(output_dir, "token_geometry_evolution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved token geometry evolution plot to {output_path}")
    
    # Create plot of average similarity changes
    plt.figure(figsize=(10, 6))
    
    # Calculate average similarity for each layer (excluding diagonal)
    init_avgs = []
    final_avgs = []
    layers = []
    
    for layer_idx in common_layers:
        # Get matrices
        init_matrix = initial_matrices[layer_idx]
        final_matrix = final_matrices[layer_idx]
        
        # Create mask to ignore diagonal
        mask = np.ones_like(init_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Calculate average similarities
        init_avg = np.mean(init_matrix[mask])
        final_avg = np.mean(final_matrix[mask])
        
        init_avgs.append(init_avg)
        final_avgs.append(final_avg)
        layers.append(layer_idx)
    
    # Plot
    plt.plot(layers, init_avgs, 'b-', marker='o', label="Initial Model")
    plt.plot(layers, final_avgs, 'r-', marker='s', label="Trained Model")
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Average Token Similarity", fontsize=12)
    plt.title("Average Token Similarity Across Layers", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(output_dir, "average_similarity_evolution.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved average similarity evolution plot to {output_path}")
    
    # Also plot the differences in average similarity
    plt.figure(figsize=(10, 6))
    
    # Calculate differences
    diff_avgs = [final - init for init, final in zip(init_avgs, final_avgs)]
    
    # Plot
    plt.bar(layers, diff_avgs, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Change in Avg. Token Similarity", fontsize=12)
    plt.title("Impact of Training on Token Similarity", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Color positive/negative bars differently
    for i, v in enumerate(diff_avgs):
        color = 'green' if v > 0 else 'red'
        plt.bar(layers[i], v, color=color, alpha=0.7)
    
    # Save the figure
    output_path = os.path.join(output_dir, "similarity_change_by_layer.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved similarity change plot to {output_path}")
    
    # Now plot the additional metrics (token norms and update norms)
    plot_metrics_comparison(initial_metrics, final_metrics, common_layers, output_dir)

def plot_metrics_comparison(initial_metrics, final_metrics, common_layers, output_dir):
    """
    Plot comparison of token metrics before and after training.
    
    Args:
        initial_metrics: Metrics from initial model
        final_metrics: Metrics from final model
        common_layers: List of common layers between models
        output_dir: Output directory
    """
    # Make sure we have metrics to plot
    if not initial_metrics or not final_metrics:
        logger.warning("No metrics available for comparison")
        return
    
    # Define metrics to plot
    metric_names = ['cosine_sim', 'token_norm', 'update_norm']
    metric_labels = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Change Between Layers)'
    }
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Token Metrics Evolution During Training", fontsize=16)
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Get metric values for each model
        # Check if we have this metric in both models
        if metric_name not in initial_metrics or metric_name not in final_metrics:
            ax.text(0.5, 0.5, f"Metric '{metric_name}' not available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            continue
            
        init_values = []
        final_values = []
        layers = []
        
        # Get common layers for this metric
        init_layers = sorted(initial_metrics[metric_name].keys())
        final_layers = sorted(final_metrics[metric_name].keys())
        common_metric_layers = sorted(set(init_layers) & set(final_layers))
        
        if not common_metric_layers:
            ax.text(0.5, 0.5, f"No common layers for '{metric_name}'", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            continue
        
        # Collect values
        for layer in common_metric_layers:
            init_values.append(initial_metrics[metric_name][layer])
            final_values.append(final_metrics[metric_name][layer])
            layers.append(layer)
        
        # Plot values
        ax.plot(layers, init_values, 'b-', marker='o', label="Initial Model")
        ax.plot(layers, final_values, 'r-', marker='s', label="Trained Model")
        
        # Add labels and grid
        ax.set_ylabel(metric_labels.get(metric_name, metric_name), fontsize=12)
        ax.set_title(f"Evolution of {metric_labels.get(metric_name, metric_name)}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Also plot the difference as a bar chart in a twin axis
        ax2 = ax.twinx()
        diff_values = [final - init for init, final in zip(init_values, final_values)]
        bars = ax2.bar(layers, diff_values, alpha=0.3, color='gray', width=0.4)
        ax2.set_ylabel('Change', color='gray')
        
        # Color bars based on sign
        for j, bar in enumerate(bars):
            if diff_values[j] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    # Set common x-axis label
    axes[-1].set_xlabel("Layer", fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "metrics_evolution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics evolution plot to {output_path}")

def main():
    """Main function to run training and analyze token geometry evolution."""
    parser = argparse.ArgumentParser(description='Analyze token geometry evolution during training')
    parser.add_argument('--data-dir', type=str, default='data/wikitext-2',
                       help='Directory containing wikitext-2 dataset')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations to train for')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for training')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--ln', type=str, choices=['preln', 'postln', 'periln', 'mixln'], default='postln',
                       help='Layer normalization architecture (default: postln)')
    parser.add_argument('--no-initial-ln', action='store_true',
                       help='Disable initial layer normalization after embeddings')
    parser.add_argument('--mixln-split', type=float, default=0.25,
                       help='Fraction of layers to use postln in mixln architecture (default: 0.25)')
    parser.add_argument('--output-dir', type=str, default='outputs/geometry_evolution',
                       help='Output directory')
    parser.add_argument('--prompt', type=str, 
                       default="The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms. Unlike recurrent neural networks, transformers process all tokens in parallel, making them efficient for training on large datasets. This fundamental shift in architecture enabled the development of models with unprecedented performance on a wide range of language tasks.",
                       help='Prompt to analyze for token geometry')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get tokenizer
    tokenizer = create_tiktoken_tokenizer()
    
    # Check if wikitext dataset exists
    train_file = os.path.join(args.data_dir, 'train.txt')
    if not os.path.exists(train_file):
        logger.error(f"Wikitext training file not found at {train_file}")
        logger.info("Please run scripts/download_wikitext.py first")
        return
    
    # Create dataset
    logger.info(f"Loading dataset from {train_file}")
    dataset = WikiTextDataset(train_file, tokenizer, block_size=128)
    
    # Create model with selected LN style
    model, model_name = create_model(
        vocab_size=tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257,
        n_layer=args.layers,
        use_swiglu=True,    # SwiGLU (LLaMA-style)
        ln_type=args.ln,
        use_initial_ln=not args.no_initial_ln,
        mixln_split=args.mixln_split
    )
    
    # Save initial model
    initial_model_path = os.path.join(args.output_dir, "initial_model")
    os.makedirs(initial_model_path, exist_ok=True)
    model.save_pretrained(initial_model_path)
    logger.info(f"Saved initial model to {initial_model_path}")
    
    # Analyze initial token geometry
    logger.info("Analyzing initial token geometry...")
    initial_analysis = analyze_token_geometry(model, tokenizer, args.prompt, args.device)
    
    # Train for specified iterations
    logger.info(f"Training for {args.iterations} iterations with lr={args.learning_rate}...")
    losses = train_for_iterations(
        model=model,
        dataset=dataset,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Analyze final token geometry
    logger.info("Analyzing final token geometry...")
    final_analysis = analyze_token_geometry(model, tokenizer, args.prompt, args.device)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.iterations + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save loss plot
    loss_plot_path = os.path.join(args.output_dir, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved training loss plot to {loss_plot_path}")
    
    # Compare token geometry before and after training
    logger.info("Comparing token geometry before and after training...")
    plot_geometry_comparison(initial_analysis, final_analysis, args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()