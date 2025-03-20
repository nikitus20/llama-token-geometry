"""
Experiment to compare token geometry across different layer normalization architectures.
This script analyzes token geometry in randomly initialized models with different LN architectures.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.model.utils import create_random_model, get_device
from src.analyzer.geometry import GeometryAnalyzer
from src.utils.data import get_tokenizer, load_prompts
from src.analyzer.visualization import plot_similarity_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def analyze_token_geometry(model, tokenizer, prompt, device):
    """
    Analyze token geometry for a model.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer to use
        prompt: Text prompt to analyze
        device: Device to run on
        
    Returns:
        Dictionary of geometry metrics
    """
    analyzer = GeometryAnalyzer(model, device)
    # Use analyze_single_prompt instead to get similarity matrices
    result = analyzer.analyze_single_prompt(prompt, tokenizer)
    
    # Restructure the result for easier access in plotting functions
    metrics = {
        'similarity_matrix': result['similarity_matrices'],
        'cosine_sim': {},
        'token_norm': {},
        'update_norm': {}
    }
    
    # Copy metrics from the result
    if 'metrics' in result and result['metrics']:
        for metric_name, layer_values in result['metrics'].items():
            metrics[metric_name] = layer_values
    
    analyzer.cleanup()
    return metrics

def plot_comparison_grid(all_metrics, ln_types, output_dir):
    """
    Plot a grid comparing metrics across different LN architectures.
    
    Args:
        all_metrics: Dictionary of metrics for each LN type
        ln_types: List of LN types
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get common layers
    common_layers = sorted(set.intersection(*[set(all_metrics[ln]['similarity_matrix'].keys()) for ln in ln_types]))
    
    # Select representative layers to visualize (evenly spaced)
    if len(common_layers) > 4:
        indices = np.linspace(0, len(common_layers) - 1, 4).astype(int)
        sample_layers = [common_layers[i] for i in indices]
    else:
        sample_layers = common_layers
    
    # Plot similarity matrices
    fig = plt.figure(figsize=(len(ln_types) * 5, len(sample_layers) * 4))
    gs = GridSpec(len(sample_layers), len(ln_types), figure=fig)
    
    # Column headers (LN types)
    for j, ln in enumerate(ln_types):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(f"{ln.capitalize()} LN", fontsize=16)
        ax.axis('off')
    
    max_vals = []
    min_vals = []
    
    # Find global min/max for consistent colormap
    for ln in ln_types:
        for layer_idx in sample_layers:
            sim_matrix = all_metrics[ln]['similarity_matrix'][layer_idx]
            max_vals.append(sim_matrix.max())
            min_vals.append(sim_matrix.min())
    
    vmin = min(min_vals)
    vmax = max(max_vals)
    
    # Plot each layer and LN type
    for i, layer_idx in enumerate(sample_layers):
        # Row labels
        ax = fig.add_subplot(gs[i, 0])
        ax.text(-0.1, 0.5, f"Layer {layer_idx}", fontsize=14, 
                transform=ax.transAxes, va='center', ha='right')
        ax.axis('off')
        
        for j, ln in enumerate(ln_types):
            ax = fig.add_subplot(gs[i, j])
            sim_matrix = all_metrics[ln]['similarity_matrix'][layer_idx]
            im = ax.imshow(sim_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(os.path.join(output_dir, f"ln_comparison_similarity.png"), dpi=150)
    plt.close()
    
    # Plot metrics across layers
    metrics_to_plot = ['cosine_sim', 'token_norm', 'update_norm']
    metric_names = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Layer-to-Layer Change)'
    }
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 12), sharex=True)
    
    for i, metric_name in enumerate(metrics_to_plot):
        ax = axes[i]
        
        for ln in ln_types:
            if metric_name in all_metrics[ln]:
                layers = sorted(all_metrics[ln][metric_name].keys())
                values = [all_metrics[ln][metric_name][layer] for layer in layers]
                ax.plot(layers, values, marker='o', linewidth=2, label=ln.capitalize())
        
        ax.set_ylabel(metric_names.get(metric_name, metric_name), fontsize=12)
        ax.set_title(f"{metric_names.get(metric_name, metric_name)} Across Layers", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc='best')
    
    axes[-1].set_xlabel("Layer", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ln_comparison_metrics.png"), dpi=150)
    plt.close()

def main():
    """Main function to run the LN comparison experiment."""
    parser = argparse.ArgumentParser(description='Compare token geometry across layer normalization architectures')
    parser.add_argument('--output-dir', type=str, default='outputs/ln_comparison',
                      help='Directory to save results')
    parser.add_argument('--n-layer', type=int, default=12,
                      help='Number of transformer layers')
    parser.add_argument('--prompt', type=str, default=None,
                      help='Text prompt to analyze (defaults to a standard prompt)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cuda, mps, or cpu)')
    parser.add_argument('--compare-initial-ln', action='store_true',
                      help='Compare models with and without initial LN')
    parser.add_argument('--tokenizer', type=str, choices=['huggingface', 'tiktoken', 'bpe', 'char'], default='huggingface',
                      help='Tokenizer to use (huggingface, tiktoken, bpe, or char)')
    parser.add_argument('--local-tokenizer', type=str, default=None,
                      help='Path to local HuggingFace tokenizer directory (for offline use)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device is None:
        args.device = get_device()
    logger.info(f"Using device: {args.device}")
    
    # Get tokenizer
    tokenizer = get_tokenizer(tokenizer_type=args.tokenizer, local_tokenizer_path=args.local_tokenizer)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    
    # Set default prompt if not provided
    if args.prompt is None:
        args.prompt = "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms. Unlike recurrent neural networks, transformers process all tokens in parallel, making them efficient for training on large datasets."
    
    # Layer normalization types to compare
    ln_types = ['preln', 'postln', 'periln', 'mixln']
    
    # Results dictionary
    all_metrics = {}
    
    # Create and analyze models with each LN type
    for ln_type in ln_types:
        logger.info(f"Creating and analyzing {ln_type} model...")
        
        # Create model
        model = create_random_model(
            ln_type=ln_type,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            use_initial_ln=True,
            device=args.device
        )
        
        # Analyze token geometry
        metrics = analyze_token_geometry(model, tokenizer, args.prompt, args.device)
        all_metrics[ln_type] = metrics
        
        # Release memory
        del model
        torch.cuda.empty_cache() if args.device == 'cuda' else None
    
    # Plot comparison
    logger.info("Plotting comparison results...")
    plot_comparison_grid(all_metrics, ln_types, args.output_dir)
    
    # If requested, compare models with and without initial LN
    if args.compare_initial_ln:
        logger.info("Comparing models with and without initial layer normalization...")
        initial_ln_metrics = {}
        
        for use_initial in [True, False]:
            ln_suffix = "with_initial" if use_initial else "no_initial"
            initial_ln_metrics[ln_suffix] = {}
            
            for ln_type in ln_types:
                logger.info(f"Creating and analyzing {ln_type} model with initial_ln={use_initial}...")
                
                # Create model
                model = create_random_model(
                    ln_type=ln_type,
                    vocab_size=vocab_size,
                    n_layer=args.n_layer,
                    use_initial_ln=use_initial,
                    device=args.device
                )
                
                # Analyze token geometry
                metrics = analyze_token_geometry(model, tokenizer, args.prompt, args.device)
                initial_ln_metrics[ln_suffix][ln_type] = metrics
                
                # Release memory
                del model
                torch.cuda.empty_cache() if args.device == 'cuda' else None
        
        # For each LN type, plot comparison between with and without initial LN
        for ln_type in ln_types:
            with_metrics = initial_ln_metrics["with_initial"][ln_type]
            without_metrics = initial_ln_metrics["no_initial"][ln_type]
            
            # Get common layers
            common_layers = sorted(set(with_metrics['similarity_matrix'].keys()) & 
                                 set(without_metrics['similarity_matrix'].keys()))
            
            # Select representative layers (evenly spaced)
            if len(common_layers) > 4:
                indices = np.linspace(0, len(common_layers) - 1, 4).astype(int)
                sample_layers = [common_layers[i] for i in indices]
            else:
                sample_layers = common_layers
            
            # Plot comparison
            fig, axes = plt.subplots(len(sample_layers), 2, figsize=(10, 3 * len(sample_layers)))
            
            # Find global min/max
            all_matrices = []
            for layer in sample_layers:
                all_matrices.append(with_metrics['similarity_matrix'][layer])
                all_matrices.append(without_metrics['similarity_matrix'][layer])
                
            vmin = min(matrix.min() for matrix in all_matrices)
            vmax = max(matrix.max() for matrix in all_matrices)
            
            # Column headers
            axes[0, 0].set_title(f"With Initial LN", fontsize=14)
            axes[0, 1].set_title(f"Without Initial LN", fontsize=14)
            
            for i, layer in enumerate(sample_layers):
                # Add row label
                axes[i, 0].set_ylabel(f"Layer {layer}", fontsize=12)
                
                # With initial LN
                im = axes[i, 0].imshow(with_metrics['similarity_matrix'][layer], 
                                     cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
                
                # Without initial LN
                axes[i, 1].imshow(without_metrics['similarity_matrix'][layer], 
                                cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 1].set_xticks([])
                axes[i, 1].set_yticks([])
            
            # Add colorbar
            plt.tight_layout()
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
            
            # Save plot
            plt.savefig(os.path.join(args.output_dir, f"{ln_type}_initial_ln_comparison.png"), dpi=150)
            plt.close()
            
            # Plot metrics comparison
            metrics_to_plot = ['cosine_sim', 'token_norm', 'update_norm']
            metric_names = {
                'cosine_sim': 'Average Cosine Similarity',
                'token_norm': 'Token Representation Norm',
                'update_norm': 'Update Norm (Layer-to-Layer Change)'
            }
            
            fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 12), sharex=True)
            
            for j, metric_name in enumerate(metrics_to_plot):
                ax = axes[j]
                
                # Get data for both configurations
                if metric_name in with_metrics and metric_name in without_metrics:
                    with_layers = sorted(with_metrics[metric_name].keys())
                    with_values = [with_metrics[metric_name][layer] for layer in with_layers]
                    
                    without_layers = sorted(without_metrics[metric_name].keys())
                    without_values = [without_metrics[metric_name][layer] for layer in without_layers]
                    
                    ax.plot(with_layers, with_values, 'b-', marker='o', linewidth=2, 
                          label=f"With Initial LN")
                    ax.plot(without_layers, without_values, 'r-', marker='s', linewidth=2, 
                          label=f"Without Initial LN")
                    
                    ax.set_ylabel(metric_names.get(metric_name, metric_name), fontsize=12)
                    ax.set_title(f"{ln_type.capitalize()}: {metric_names.get(metric_name, metric_name)}", 
                               fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Only add legend to the first plot
                    if j == 0:
                        ax.legend(loc='best')
            
            axes[-1].set_xlabel("Layer", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{ln_type}_initial_ln_metrics.png"), dpi=150)
            plt.close()
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()