"""
Experiment to analyze the effect of warming up a model with random data.

This script:
1. Creates a random dataset
2. Warms up a PreLN model without initial normalization on this random data
   (architecture is configurable via --ln and --use-initial-ln parameters)
3. Analyzes the token geometry before and after warmup
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.model.utils import create_random_model, get_device
from src.analyzer.geometry import GeometryAnalyzer
from src.utils.data import get_tokenizer
from src.analyzer.visualization import plot_token_geometry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class RandomDataset(Dataset):
    """
    Dataset that generates random token sequences and labels.
    """
    
    def __init__(self, vocab_size, seq_length, size=1000, seed=42):
        """
        Initialize the random dataset.
        
        Args:
            vocab_size: Size of the vocabulary
            seq_length: Length of each sequence
            size: Number of sequences to generate
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.size = size
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate random data
        self.data = torch.randint(0, vocab_size, (size, seq_length))
        
        # Generate random target labels
        # For more realistic targets, shift the input by 1 position to predict the next token
        # This is similar to language modeling
        self.targets = torch.roll(self.data, -1, dims=1)
        # Replace the last column with random tokens (since we rolled, the last column has no real "next token")
        self.targets[:, -1] = torch.randint(0, vocab_size, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class WikitextRandomLabelDataset(Dataset):
    """
    Dataset that uses real text sequences from wikitext-2 but with random target labels.
    This allows for training with realistic input distribution but random outputs.
    """
    
    def __init__(self, tokenizer, seq_length, size=1000, seed=42, split='train'):
        """
        Initialize the Wikitext dataset with random labels.
        
        Args:
            tokenizer: Tokenizer to use for encoding text
            seq_length: Length of each sequence
            size: Number of sequences to generate (maximum)
            seed: Random seed for reproducibility
            split: Which split of the dataset to use ('train', 'validation', or 'test')
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
        self.seq_length = seq_length
        self.size = size
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load text from wikitext-2 file
        dataset_path = os.path.join('data', 'wikitext-2', f'{split}.txt')
        logger.info(f"Loading wikitext-2 data from {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error loading wikitext-2 data: {e}")
            raise
        
        # Tokenize the entire text
        tokens = self.tokenize_text(text)
        
        # Create sequences of the specified length
        sequences = []
        for i in range(0, len(tokens) - seq_length, seq_length // 2):  # 50% overlap for more data
            if len(sequences) >= size:
                break
            sequences.append(tokens[i:i + seq_length])
        
        self.size = min(size, len(sequences))
        logger.info(f"Created {self.size} sequences from wikitext-2")
        
        # Convert sequences to tensor
        self.data = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in sequences[:self.size]])
        
        # Generate random target labels
        self.targets = torch.randint(0, self.vocab_size, (self.size, seq_length))
    
    def tokenize_text(self, text):
        """Tokenize text using the provided tokenizer."""
        if hasattr(self.tokenizer, 'encode'):
            # TikToken and some other tokenizers
            return self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, 'tokenize'):
            # Character tokenizers and some others
            return [self.tokenizer.token_to_id(t) for t in self.tokenizer.tokenize(text)]
        else:
            # Fallback for other tokenizers
            raise ValueError("Tokenizer does not have encode or tokenize method")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_with_random_data(model, tokenizer, vocab_size, seq_length, num_iterations=100, 
                         batch_size=32, learning_rate=1e-4, device='cuda', use_real_input=True):
    """
    Train a model with either completely random data or real input with random labels.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer to use for encoding text (needed for WikitextRandomLabelDataset)
        vocab_size: Vocabulary size
        seq_length: Sequence length
        num_iterations: Number of iterations to train
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        use_real_input: Whether to use real text input with random labels (True) or completely random data (False)
        
    Returns:
        List of losses during training
    """
    model = model.to(device)
    model.train()
    
    # Create dataset
    if use_real_input:
        logger.info("Using real wikitext-2 inputs with random labels")
        dataset = WikitextRandomLabelDataset(
            tokenizer=tokenizer,
            seq_length=seq_length,
            size=batch_size * (num_iterations + 10),
            split='train'
        )
    else:
        logger.info("Using completely random data")
        dataset = RandomDataset(
            vocab_size=vocab_size,
            seq_length=seq_length,
            size=batch_size * (num_iterations + 10)
        )
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    losses = []
    iteration = 0
    
    progress_bar = tqdm(total=num_iterations, desc="Random Warmup")
    
    for x, y in dataloader:
        if iteration >= num_iterations:
            break
            
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Update progress
        iteration += 1
        progress_bar.update(1)
        
        # Log every 10 iterations
        if iteration % 10 == 0:
            progress_bar.set_description(f"Random Warmup [Loss: {loss.item():.4f}]")
    
    progress_bar.close()
    return losses


def analyze_model(model, prompt, tokenizer, device='cuda'):
    """
    Analyze token geometry for a model.
    
    Args:
        model: Model to analyze
        prompt: Prompt to analyze
        tokenizer: Tokenizer to use
        device: Device to run on
        
    Returns:
        Complete analysis results
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


def compare_metrics(initial_metrics, warmed_metrics, random_metrics, output_dir):
    """
    Compare metrics between initial, random-warmed, and real-data models.
    
    Args:
        initial_metrics: Metrics from initial random model
        warmed_metrics: Metrics from model warmed up on random data
        random_metrics: Metrics from model warmed up on real data (if provided)
        output_dir: Output directory
    """
    # Define metrics to plot
    metric_names = ['cosine_sim', 'token_norm', 'update_norm']
    metric_labels = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Change Between Layers)'
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Effect of Random Data Warmup on Token Metrics", fontsize=16)
    
    # Color and marker styles for each model
    styles = {
        'Initial': {'color': 'blue', 'marker': 'o', 'label': 'Initial Random Model'},
        'Random': {'color': 'red', 'marker': 's', 'label': 'After Random Warmup'}
    }
    
    if random_metrics:
        styles['Real'] = {'color': 'green', 'marker': '^', 'label': 'After Real Data Training'}
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Check if all models have this metric
        if (metric_name not in initial_metrics or 
            metric_name not in warmed_metrics or 
            (random_metrics and metric_name not in random_metrics)):
            ax.text(0.5, 0.5, f"Metric '{metric_name}' not available for all models", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            continue
        
        # Get layers from initial model
        layers = sorted(initial_metrics[metric_name].keys())
        
        # Plot each model's metrics
        ax.plot(
            layers, 
            [initial_metrics[metric_name][layer] for layer in layers],
            marker=styles['Initial']['marker'],
            color=styles['Initial']['color'],
            label=styles['Initial']['label']
        )
        
        ax.plot(
            layers, 
            [warmed_metrics[metric_name][layer] for layer in layers],
            marker=styles['Random']['marker'],
            color=styles['Random']['color'],
            label=styles['Random']['label']
        )
        
        if random_metrics:
            ax.plot(
                layers, 
                [random_metrics[metric_name][layer] for layer in layers],
                marker=styles['Real']['marker'],
                color=styles['Real']['color'],
                label=styles['Real']['label']
            )
        
        # Set labels and grid
        ax.set_ylabel(metric_labels.get(metric_name, metric_name), fontsize=12)
        ax.set_title(f"Effect of Warmup on {metric_labels.get(metric_name, metric_name)}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Set common x-axis label
    axes[-1].set_xlabel("Layer", fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "warmup_effect_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved warmup effect metrics plot to {output_path}")
    
    # Now plot the differences between random warmup and initial model
    plot_metric_differences(initial_metrics, warmed_metrics, random_metrics, 
                          output_dir, prefix="warmup_diff")


def plot_metric_differences(initial_metrics, warmed_metrics, random_metrics, output_dir, prefix="diff"):
    """
    Plot differences in metrics between models.
    
    Args:
        initial_metrics: Metrics from initial random model
        warmed_metrics: Metrics from model warmed up on random data
        random_metrics: Metrics from model warmed up on real data (if provided)
        output_dir: Output directory
        prefix: Prefix for output files
    """
    # Define metrics to plot
    metric_names = ['cosine_sim', 'token_norm', 'update_norm']
    metric_labels = {
        'cosine_sim': 'Change in Average Cosine Similarity',
        'token_norm': 'Change in Token Representation Norm',
        'update_norm': 'Change in Update Norm'
    }
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Changes in Token Metrics After Warmup", fontsize=16)
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Check if all models have this metric
        if (metric_name not in initial_metrics or 
            metric_name not in warmed_metrics):
            ax.text(0.5, 0.5, f"Metric '{metric_name}' not available for all models", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            continue
        
        # Get layers from initial model
        layers = sorted(initial_metrics[metric_name].keys())
        
        # Calculate differences
        random_diff = [warmed_metrics[metric_name][layer] - initial_metrics[metric_name][layer] 
                      for layer in layers]
        
        # Plot the differences as bar charts
        bars1 = ax.bar(layers, random_diff, alpha=0.7, color='red', label='Random Warmup Effect')
        
        if random_metrics and metric_name in random_metrics:
            real_diff = [random_metrics[metric_name][layer] - initial_metrics[metric_name][layer] 
                        for layer in layers]
            # Plot with slight offset for visibility
            bars2 = ax.bar([layer + 0.3 for layer in layers], real_diff, 
                          alpha=0.7, color='green', label='Real Data Effect', width=0.3)
        
        # Set labels and grid
        ax.set_ylabel(metric_labels.get(metric_name, metric_name), fontsize=12)
        ax.set_title(f"Effect of Warmup on {metric_labels.get(metric_name, metric_name)}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add horizontal zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Set common x-axis label
    axes[-1].set_xlabel("Layer", fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{prefix}_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metric differences plot to {output_path}")


def main():
    """Main function to run random warmup experiment."""
    parser = argparse.ArgumentParser(description='Analyze effect of random data warmup')
    
    # Data and model arguments
    parser.add_argument('--output-dir', type=str, default='outputs/random_warmup',
                      help='Directory to save results')
    parser.add_argument('--tokenizer-dir', type=str, default='tokenizer/',
                      help='Path to local tokenizer directory')
    parser.add_argument('--prompt', type=str, default=None,
                      help='Prompt to analyze (defaults to a standard test prompt)')
    parser.add_argument('--seq-length', type=int, default=128,
                      help='Sequence length for random data')
    parser.add_argument('--use-real-input', action='store_true',
                      help='Use real text input from wikitext-2 with random labels instead of completely random data')
    
    # Model architecture
    parser.add_argument('--n-layer', type=int, default=8,
                      help='Number of transformer layers')
    parser.add_argument('--n-head', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--n-embd', type=int, default=384,
                      help='Embedding dimension')
    parser.add_argument('--ln', type=str, choices=['preln', 'postln', 'periln', 'mixln'], default='preln',
                      help='Layer normalization architecture (default: preln)')
    parser.add_argument('--use-initial-ln', action='store_true',
                      help='Enable initial layer normalization after embeddings (disabled by default)')
    parser.add_argument('--mixln-split', type=float, default=0.25,
                      help='Fraction of layers to use postln in mixln architecture (default: 0.25)')
    parser.add_argument('--no-swiglu', action='store_true',
                      help='Use GELU activation instead of SwiGLU')
    
    # Training arguments
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of training iterations for random warmup')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cuda, mps, or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = get_device()
    logger.info(f"Using device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_dir=args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    
    # Set default prompt if not provided
    if args.prompt is None:
        args.prompt = "This is a test prompt for analyzing the effect of random data warmup on token geometry in transformer models."
    
    # Create initial model with specified architecture
    logger.info(f"Creating initial model with {args.ln} architecture...")
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.seq_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
        bias=False,
        ln=args.ln,
        use_initial_ln=args.use_initial_ln,
        mixln_split=args.mixln_split,
        use_swiglu=not args.no_swiglu,
        initializer_range=0.02  # Add standard deviation for weight initialization
    )
    model = GPT(config)
    logger.info(f"Created model with {model.get_num_params()/1e6:.2f}M parameters")
    logger.info(f"Architecture: {args.ln.capitalize()}, RMSNorm, {'SwiGLU' if not args.no_swiglu else 'GELU'}, "
               f"{'with' if args.use_initial_ln else 'without'} initial normalization")
    
    # Save initial model
    initial_model_path = os.path.join(args.output_dir, "initial_model")
    os.makedirs(initial_model_path, exist_ok=True)
    model.save_pretrained(initial_model_path)
    logger.info(f"Saved initial model to {initial_model_path}")
    
    # Analyze initial token geometry
    logger.info("Analyzing initial token geometry...")
    initial_analysis = analyze_model(model, args.prompt, tokenizer, args.device)
    
    # Plot initial token geometry
    logger.info("Plotting initial token geometry...")
    plot_token_geometry(
        initial_analysis['metrics'], 
        title=f"Token Geometry for Initial Random Model",
        output_path=os.path.join(args.output_dir, "initial_token_geometry.png")
    )
    
    # Train with random data
    logger.info(f"Training with {'real input and' if args.use_real_input else ''} random {'labels' if args.use_real_input else 'data'} for {args.iterations} iterations...")
    losses = train_with_random_data(
        model=model,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        seq_length=args.seq_length,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        use_real_input=args.use_real_input
    )
    
    # Save random-warmed model
    warmed_model_path = os.path.join(args.output_dir, "random_warmed_model")
    os.makedirs(warmed_model_path, exist_ok=True)
    model.save_pretrained(warmed_model_path)
    logger.info(f"Saved random-warmed model to {warmed_model_path}")
    
    # Analyze random-warmed token geometry
    logger.info("Analyzing token geometry after random warmup...")
    warmed_analysis = analyze_model(model, args.prompt, tokenizer, args.device)
    
    # Plot warmed token geometry
    logger.info("Plotting warmed token geometry...")
    plot_token_geometry(
        warmed_analysis['metrics'], 
        title=f"Token Geometry After Random Data Warmup",
        output_path=os.path.join(args.output_dir, "warmed_token_geometry.png")
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.iterations + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Random Data Training Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save loss plot
    loss_plot_path = os.path.join(args.output_dir, "random_training_loss.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved training loss plot to {loss_plot_path}")
    
    # Compare metrics
    logger.info("Comparing token metrics before and after random warmup...")
    compare_metrics(
        initial_analysis['metrics'], 
        warmed_analysis['metrics'], 
        None,  # No real data training for now
        args.output_dir
    )
    
    logger.info(f"Experiment complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()