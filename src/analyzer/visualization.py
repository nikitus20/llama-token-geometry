"""
Visualization utilities for token geometry analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, List, Any, Union

def plot_token_geometry(metrics: Dict[str, Dict[str, Dict[int, float]]], 
                       title: Optional[str] = None, 
                       output_path: Optional[str] = None) -> None:
    """
    Plot token geometry metrics across layers (supporting the new format).
    
    Args:
        metrics: Dictionary of metrics from GeometryAnalyzer.analyze_prompts
                Expected structure:
                {
                    'mean': {
                        'cosine_sim': {layer_idx: value, ...},
                        'token_norm': {layer_idx: value, ...},
                        'update_norm': {layer_idx: value, ...}
                    },
                    'variance': {...}
                }
        title: Optional title for the plot
        output_path: Optional path to save the plot
    """
    # For backwards compatibility, check if the input is the old format
    if not isinstance(metrics, dict) or ('mean' not in metrics and 'cosine_sim' not in metrics):
        # Handle old format (just a dict of cosine similarities)
        layers = sorted(metrics.keys())
        cosine_sims = [metrics[layer] for layer in layers]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, cosine_sims, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Layer')
        plt.ylabel('Average Cosine Similarity')
        plt.title(title or 'Token Geometry: Average Cosine Similarity Across Layers')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for min and max values if there are values to analyze
        if cosine_sims:  # Check if the list is not empty
            min_idx = np.argmin(cosine_sims)
            max_idx = np.argmax(cosine_sims)
            plt.annotate(f'Min: {cosine_sims[min_idx]:.4f}', 
                        xy=(layers[min_idx], cosine_sims[min_idx]),
                        xytext=(10, -20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            plt.annotate(f'Max: {cosine_sims[max_idx]:.4f}', 
                        xy=(layers[max_idx], cosine_sims[max_idx]),
                        xytext=(10, 20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()
        return
    
    # New format - comprehensive visualization
    mean_metrics = metrics.get('mean', {})
    var_metrics = metrics.get('variance', {})
    
    if 'cosine_sim' in metrics and 'token_norm' not in metrics:
        # Handle transitional format: just a dict with metric types
        mean_metrics = metrics
        var_metrics = {}
    
    # Create figure with subplots for each metric
    num_metrics = len(mean_metrics)
    if num_metrics == 0:
        return
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]  # Make axes always a list for consistent indexing
    
    # Set overall title
    fig.suptitle(title or 'Token Geometry Analysis', fontsize=16)
    
    # Define nice labels for each metric
    metric_labels = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Change Between Layers)'
    }
    
    # Plot each metric
    for i, (metric_name, layer_values) in enumerate(mean_metrics.items()):
        ax = axes[i]
        
        # Get layers and values
        layers = sorted(layer_values.keys())
        values = [layer_values[layer] for layer in layers]
        
        # Get variance if available
        if var_metrics and metric_name in var_metrics:
            variances = [var_metrics[metric_name].get(layer, 0) for layer in layers]
            std_devs = np.sqrt(variances)
            
            # Plot with error bars
            ax.errorbar(layers, values, yerr=std_devs, marker='o', linestyle='-', 
                        capsize=4, label=metric_labels.get(metric_name, metric_name))
        else:
            # Plot without error bars
            ax.plot(layers, values, marker='o', linestyle='-', linewidth=2,
                   label=metric_labels.get(metric_name, metric_name))
        
        # Set labels and grid
        ax.set_ylabel(metric_labels.get(metric_name, metric_name))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add min/max annotations only if there are values to analyze
        if values:  # Check if the list is not empty
            min_idx = np.argmin(values)
            max_idx = np.argmax(values)
            ax.annotate(f'Min: {values[min_idx]:.4f}', 
                      xy=(layers[min_idx], values[min_idx]),
                      xytext=(10, -20),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            ax.annotate(f'Max: {values[max_idx]:.4f}', 
                      xy=(layers[max_idx], values[max_idx]),
                      xytext=(10, 20),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel('Layer')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_similarity_matrix(sim_matrix: np.ndarray, title: Optional[str] = None, 
                          output_path: Optional[str] = None) -> None:
    """
    Plot a token similarity matrix as a heatmap.
    
    Args:
        sim_matrix: Token similarity matrix
        title: Optional title for the plot
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis', vmin=-1, vmax=1, center=0, 
                square=True, xticklabels=False, yticklabels=False)
    plt.title(title or 'Token Similarity Matrix')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_architecture_comparison(results: Dict[str, Union[Dict[int, float], Dict[str, Dict[str, Dict[int, float]]]]], 
                               output_dir: str, n_layer: int = 24, 
                               metric_name: str = 'cosine_sim') -> None:
    """
    Plot comparison between all tested architectures.
    
    Args:
        results: Dictionary of metrics for each architecture. Supports both old and new format.
        output_dir: Directory to save results
        n_layer: Number of transformer layers visualized
        metric_name: Which metric to use for comparison ('cosine_sim', 'token_norm', or 'update_norm')
    """
    # Extract the right metric from each model's results
    processed_results = {}
    for model_name, model_results in results.items():
        # Check if this is the new format
        if isinstance(model_results, dict) and 'mean' in model_results:
            # New format - extract the mean values for the specified metric
            processed_results[model_name] = model_results['mean'].get(metric_name, {})
        elif isinstance(model_results, dict) and metric_name in model_results:
            # Transitional format - with multiple metrics but no mean/variance structure
            processed_results[model_name] = model_results[metric_name]
        else:
            # Old format or already extracted metric dict
            processed_results[model_name] = model_results
    
    # Get all layers across all models
    all_layers = set()
    for model_metrics in processed_results.values():
        all_layers.update(model_metrics.keys())
    layers = sorted(all_layers)
    
    # Set metric label
    metric_labels = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Change Between Layers)'
    }
    metric_label = metric_labels.get(metric_name, metric_name)
    
    # Create output directory for each metric
    metric_output_dir = os.path.join(output_dir, metric_name.replace('_', '-'))
    os.makedirs(metric_output_dir, exist_ok=True)
    
    # LINE CHART
    plt.figure(figsize=(max(12, n_layer//2), 8))  # Scale figure width with number of layers
    
    colors = ['royalblue', 'firebrick', 'forestgreen', 'darkorange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', '*']
    
    for i, (model_name, metric_values) in enumerate(processed_results.items()):
        # Ensure all layers have values
        values = [metric_values.get(layer, 0) for layer in layers]
        plt.plot(
            layers, values, 
            marker=markers[i % len(markers)], 
            linestyle='-', 
            linewidth=2, 
            label=model_name, 
            color=colors[i % len(colors)]
        )
    
    plt.xlabel('Layer')
    plt.ylabel(metric_label)
    plt.title(f'Token Geometry Comparison: {metric_label} Across Architectures ({n_layer} layers)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # For many layers, rotate x labels for better readability
    if n_layer > 12:
        plt.xticks(rotation=45)
    
    # Save line chart
    plt.savefig(os.path.join(metric_output_dir, f"architecture_comparison_{metric_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save numerical results to CSV
    results_df = pd.DataFrame({'layer': layers})
    
    for model_name, metric_values in processed_results.items():
        # Ensure all layers have values
        values = [metric_values.get(layer, 0) for layer in layers]
        results_df[f'{model_name}_{metric_name}'] = values
    
    # Add difference columns for this metric
    if "LLaMA-PreLN" in processed_results and "LLaMA-PostLN" in processed_results:
        llama_preln = [processed_results["LLaMA-PreLN"].get(layer, 0) for layer in layers]
        llama_postln = [processed_results["LLaMA-PostLN"].get(layer, 0) for layer in layers]
        results_df[f'LLaMA_PreLN_vs_PostLN_diff_{metric_name}'] = np.array(llama_preln) - np.array(llama_postln)
    
    if "Standard-PreLN" in processed_results and "Standard-PostLN" in processed_results:
        std_preln = [processed_results["Standard-PreLN"].get(layer, 0) for layer in layers]
        std_postln = [processed_results["Standard-PostLN"].get(layer, 0) for layer in layers]
        results_df[f'Standard_PreLN_vs_PostLN_diff_{metric_name}'] = np.array(std_preln) - np.array(std_postln)
    
    if "LLaMA-PreLN" in processed_results and "Standard-PreLN" in processed_results:
        llama_preln = [processed_results["LLaMA-PreLN"].get(layer, 0) for layer in layers]
        std_preln = [processed_results["Standard-PreLN"].get(layer, 0) for layer in layers]
        results_df[f'LLaMA_vs_Standard_PreLN_diff_{metric_name}'] = np.array(llama_preln) - np.array(std_preln)
    
    # Save to CSV
    results_df.to_csv(os.path.join(metric_output_dir, f"architecture_comparison_{metric_name}.csv"), index=False)
    
    # Also save to main output directory for backwards compatibility
    if metric_name == 'cosine_sim':
        results_df.to_csv(os.path.join(output_dir, "architecture_comparison.csv"), index=False)