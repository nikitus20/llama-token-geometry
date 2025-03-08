"""
Visualization utilities for token geometry analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, List

def plot_token_geometry(avg_cosine_sims: Dict[int, float], 
                       title: Optional[str] = None, 
                       output_path: Optional[str] = None) -> None:
    """
    Plot average cosine similarity across layers.
    
    Args:
        avg_cosine_sims: Dictionary of average cosine similarities per layer
        title: Optional title for the plot
        output_path: Optional path to save the plot
    """
    layers = sorted(avg_cosine_sims.keys())
    cosine_sims = [avg_cosine_sims[layer] for layer in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, cosine_sims, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Layer')
    plt.ylabel('Average Cosine Similarity')
    plt.title(title or 'Token Geometry: Average Cosine Similarity Across Layers')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for min and max values
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


def plot_architecture_comparison(results: Dict[str, Dict[int, float]],
                               output_dir: str, n_layer: int = 24) -> None:
    """
    Plot comparison between all tested architectures.
    
    Args:
        results: Dictionary of cosine similarities for each architecture
        output_dir: Directory to save results
        n_layer: Number of transformer layers visualized
    """
    # Get all layers across all models
    all_layers = set()
    for model_results in results.values():
        all_layers.update(model_results.keys())
    layers = sorted(all_layers)
    
    # LINE CHART
    plt.figure(figsize=(max(12, n_layer//2), 8))  # Scale figure width with number of layers
    
    colors = ['royalblue', 'firebrick', 'forestgreen', 'darkorange']
    markers = ['o', 's', '^', 'D']
    
    for i, (model_name, cosine_sims) in enumerate(results.items()):
        # Ensure all layers have values
        sims = [cosine_sims.get(layer, 0) for layer in layers]
        plt.plot(
            layers, sims, 
            marker=markers[i % len(markers)], 
            linestyle='-', 
            linewidth=2, 
            label=model_name, 
            color=colors[i % len(colors)]
        )
    
    plt.xlabel('Layer')
    plt.ylabel('Average Cosine Similarity')
    plt.title(f'Token Geometry Comparison: Architecture Variants ({n_layer} layers)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # For many layers, rotate x labels for better readability
    if n_layer > 12:
        plt.xticks(rotation=45)
    
    # Save line chart
    plt.savefig(os.path.join(output_dir, "architecture_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save numerical results to CSV
    results_df = pd.DataFrame({'layer': layers})
    
    for model_name, cosine_sims in results.items():
        # Ensure all layers have values
        sims = [cosine_sims.get(layer, 0) for layer in layers]
        results_df[f'{model_name}_cosine_similarity'] = sims
    
    # Add difference columns
    if "LLaMA-PreLN" in results and "LLaMA-PostLN" in results:
        llama_preln = [results["LLaMA-PreLN"].get(layer, 0) for layer in layers]
        llama_postln = [results["LLaMA-PostLN"].get(layer, 0) for layer in layers]
        results_df['LLaMA_PreLN_vs_PostLN_diff'] = np.array(llama_preln) - np.array(llama_postln)
    
    if "Standard-PreLN" in results and "Standard-PostLN" in results:
        std_preln = [results["Standard-PreLN"].get(layer, 0) for layer in layers]
        std_postln = [results["Standard-PostLN"].get(layer, 0) for layer in layers]
        results_df['Standard_PreLN_vs_PostLN_diff'] = np.array(std_preln) - np.array(std_postln)
    
    if "LLaMA-PreLN" in results and "Standard-PreLN" in results:
        llama_preln = [results["LLaMA-PreLN"].get(layer, 0) for layer in layers]
        std_preln = [results["Standard-PreLN"].get(layer, 0) for layer in layers]
        results_df['LLaMA_vs_Standard_PreLN_diff'] = np.array(llama_preln) - np.array(std_preln)
    
    results_df.to_csv(os.path.join(output_dir, "architecture_comparison.csv"), index=False)