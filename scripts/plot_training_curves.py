#!/usr/bin/env python
"""
Script to plot training curves from multiple models.
"""

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_data(model_dir):
    """Load training and validation metrics for a model."""
    # Load model configuration
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config file found for {model_dir}")
        model_name = os.path.basename(model_dir)
        config = {"model_name": model_name}
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load training metrics
    train_path = os.path.join(model_dir, "metrics", "train_metrics.csv")
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    
    # Load validation metrics
    val_path = os.path.join(model_dir, "metrics", "val_metrics.csv")
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    
    return config, train_df, val_df

def plot_training_curves(model_dirs, output_dir=None, plot_type="loss", smooth=0):
    """Plot training curves from multiple models."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Define colors for different models
    colors = sns.color_palette("husl", len(model_dirs))
    
    # Load and plot data for each model
    legend_entries = []
    
    for i, model_dir in enumerate(model_dirs):
        config, train_df, val_df = load_model_data(model_dir)
        model_name = config.get("model_name", os.path.basename(model_dir))
        color = colors[i]
        
        if plot_type == "loss" and train_df is not None:
            # Plot training loss
            if smooth > 0:
                train_df['smoothed_loss'] = train_df['loss'].rolling(window=smooth, min_periods=1).mean()
                plt.plot(train_df['step'], train_df['smoothed_loss'], color=color, alpha=0.8)
            else:
                plt.plot(train_df['step'], train_df['loss'], color=color, alpha=0.8)
            legend_entries.append(f"{model_name} (train)")
            
        elif plot_type == "perplexity" and val_df is not None:
            # Plot validation perplexity
            plt.plot(val_df['step'], val_df['perplexity'], color=color, marker='o', alpha=0.8)
            legend_entries.append(f"{model_name} (val)")
            
        elif plot_type == "both":
            # Plot both training loss and validation perplexity
            if train_df is not None:
                if smooth > 0:
                    train_df['smoothed_loss'] = train_df['loss'].rolling(window=smooth, min_periods=1).mean()
                    plt.plot(train_df['step'], train_df['smoothed_loss'], color=color, linestyle='-', alpha=0.6)
                else:
                    plt.plot(train_df['step'], train_df['loss'], color=color, linestyle='-', alpha=0.6)
                legend_entries.append(f"{model_name} (train)")
            
            if val_df is not None:
                plt.plot(val_df['step'], val_df['perplexity'], color=color, linestyle='--', marker='o', alpha=0.8)
                legend_entries.append(f"{model_name} (val)")
    
    # Set plot labels and legend
    plt.xlabel("Training Steps")
    if plot_type == "loss":
        plt.ylabel("Training Loss")
        plt.title("Training Loss Curves")
    elif plot_type == "perplexity":
        plt.ylabel("Validation Perplexity")
        plt.title("Validation Perplexity Curves")
    else:
        plt.ylabel("Loss / Perplexity")
        plt.title("Training and Validation Curves")
    
    plt.legend(legend_entries)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show plot
    if output_dir is not None:
        output_path = os.path.join(output_dir, f"{plot_type}_curves.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from multiple models')
    parser.add_argument('--model-dirs', type=str, nargs='+', required=True,
                       help='List of model directories to plot')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (if not provided, plots will be shown)')
    parser.add_argument('--plot-type', type=str, choices=['loss', 'perplexity', 'both'], default='both',
                       help='Type of plot to generate')
    parser.add_argument('--smooth', type=int, default=10,
                       help='Window size for loss smoothing (0 to disable)')
    
    args = parser.parse_args()
    plot_training_curves(args.model_dirs, args.output_dir, args.plot_type, args.smooth)

if __name__ == "__main__":
    main()