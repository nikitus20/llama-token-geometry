#!/usr/bin/env python
"""
Script to analyze token geometry across models with different normalization techniques.
This compares both randomly initialized models and their warmed-up versions.
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import glob
from collections import defaultdict

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import GPT
from src.utils.data import get_tokenizer
from src.analyzer.geometry import GeometryAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")
    return device

def load_prompt_completion_pairs(file_path):
    """
    Load prompt-completion pairs from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of prompt-completion dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        if not isinstance(data, list):
            logger.error(f"Expected a list in {file_path}, got {type(data)}")
            return []
            
        # Validate each item
        valid_pairs = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(f"Item {i} is not a dictionary, skipping")
                continue
                
            if 'prompt' not in item or 'completion' not in item:
                logger.warning(f"Item {i} missing prompt or completion, skipping")
                continue
                
            valid_pairs.append(item)
            
        return valid_pairs
    except Exception as e:
        logger.error(f"Error loading prompt-completion pairs: {e}")
        return []

def create_sample_prompt_completion_pairs(output_file):
    """
    Create a sample prompt-completion pairs file.
    
    Args:
        output_file: Path to save the sample file
    """
    sample_pairs = [
        {
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        },
        {
            "prompt": "Explain neural networks.",
            "completion": "Neural networks are computing systems inspired by biological brains. They process information through interconnected nodes organized in layers."
        },
        {
            "prompt": "Define gradient descent.",
            "completion": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent."
        },
        {
            "prompt": "What is backpropagation?",
            "completion": "Backpropagation is an algorithm for training neural networks by calculating gradients of the loss function with respect to the weights."
        },
        {
            "prompt": "Explain transformers.",
            "completion": "Transformers are neural network architectures that use self-attention mechanisms to process sequential data in parallel."
        }
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(sample_pairs, f, indent=2)
        
    logger.info(f"Created sample prompt-completion pairs file at {output_file}")
    return sample_pairs

def load_models_summary(summary_file):
    """
    Load model summary information.
    
    Args:
        summary_file: Path to the summary JSON file
        
    Returns:
        Dictionary with model summary information
    """
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        logger.error(f"Error loading models summary: {e}")
        return None

def find_model_paths(models_dir):
    """
    Find all model paths in a directory.
    
    Args:
        models_dir: Directory containing model subdirectories
        
    Returns:
        Dictionary mapping model names to paths for initial and final models
    """
    model_paths = {}
    
    # Look for normalization-specific model directories
    norm_dirs = glob.glob(os.path.join(models_dir, "*_model"))
    
    for norm_dir in norm_dirs:
        norm_type = os.path.basename(norm_dir).replace("_model", "")
        initial_path = os.path.join(norm_dir, "initial_model")
        final_path = os.path.join(norm_dir, "final_model")
        
        if os.path.exists(initial_path) and os.path.exists(final_path):
            model_paths[norm_type] = {
                "initial": initial_path,
                "final": final_path
            }
    
    return model_paths

def analyze_model(model_path, pairs, tokenizer, device, output_dir, model_name, max_pairs=None):
    """
    Analyze a model's token geometry across multiple prompt-completion pairs.
    
    Args:
        model_path: Path to the saved model
        pairs: List of prompt-completion pairs
        tokenizer: Tokenizer
        device: Device to run on
        output_dir: Directory to save results
        model_name: Name to identify the model (e.g., "preln_initial")
        max_pairs: Maximum number of pairs to analyze
        
    Returns:
        Aggregated results dictionary
    """
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    logger.info(f"Loading model from {model_path}")
    try:
        model = GPT.from_pretrained(model_path, device=device)
        model.to(device)
        model.eval()
        
        # Detect normalization type from model config
        norm_type = "unknown"
        if hasattr(model, 'config'):
            # The normalization type is stored in the 'ln' field in GPTConfig
            if hasattr(model.config, 'ln'):
                norm_type = model.config.ln
            
            # For older models or compatibility
            elif hasattr(model.config, 'norm_type'):
                norm_type = model.config.norm_type
                
        logger.info(f"Detected normalization type: {norm_type}")
        
        # Create analyzer
        analyzer = GeometryAnalyzer(model, device=device, track_gradients=True)
        
        # Initialize storage for aggregated metrics
        all_metrics = {
            'cosine_sim': {},
            'token_norm': {},
            'update_norm': {},
            'gradient_norm': {},
            'gradient_update_correlation': {}
        }
        
        # Limit number of pairs if specified
        if max_pairs and max_pairs < len(pairs):
            import random
            selected_pairs = random.sample(pairs, max_pairs)
        else:
            selected_pairs = pairs
            
        logger.info(f"Analyzing {len(selected_pairs)} prompt-completion pairs for {model_name}")
        
        # Process each pair
        for i, pair in enumerate(tqdm(selected_pairs, desc=f"Analyzing {model_name}")):
            prompt = pair['prompt']
            completion = pair['completion']
            
            try:
                # Run analysis
                results = analyzer.analyze_with_completion(prompt, completion, tokenizer)
                
                # Aggregate metrics
                if 'metrics' in results and results['metrics']:
                    metrics = results['metrics']
                    for metric_name, layer_values in metrics.items():
                        if not layer_values:  # Skip empty metrics
                            continue
                            
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = {}
                            
                        for layer_idx, value in layer_values.items():
                            if layer_idx not in all_metrics[metric_name]:
                                all_metrics[metric_name][layer_idx] = []
                            all_metrics[metric_name][layer_idx].append(value)
            except Exception as e:
                logger.error(f"Error processing pair {i+1} for model {model_name}: {e}")
                continue
        
        # Clean up analyzer
        analyzer.cleanup()
        
        # Compute mean and variance
        aggregated_results = {
            'model_name': model_name,
            'model_path': model_path,
            'norm_type': norm_type,
            'mean': {},
            'variance': {}
        }
        
        for metric_name, layer_data in all_metrics.items():
            aggregated_results['mean'][metric_name] = {}
            aggregated_results['variance'][metric_name] = {}
            
            for layer_idx, values in layer_data.items():
                if values:  # Check if we have data
                    values_array = np.array(values)
                    aggregated_results['mean'][metric_name][layer_idx] = float(np.mean(values_array))
                    aggregated_results['variance'][metric_name][layer_idx] = float(np.var(values_array))
        
        # Save aggregated results
        results_file = os.path.join(model_output_dir, "aggregated_results.json")
        with open(results_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        return aggregated_results
    
    except Exception as e:
        logger.error(f"Error analyzing model {model_path}: {e}")
        return None

def plot_comparison(results_dict, metric_name, output_path, title=None, include_stage=True):
    """
    Create comparative plots for a specific metric across different models.
    
    Args:
        results_dict: Dictionary mapping model names to their aggregated results
        metric_name: The metric to compare (e.g., 'cosine_sim', 'token_norm', etc.)
        output_path: Path to save the plot
        title: Custom title for the plot
        include_stage: Whether to distinguish between initial and final models
    """
    plt.figure(figsize=(14, 8))
    
    # Define color map for different normalization types
    norm_types = set()
    for result in results_dict.values():
        if 'norm_type' in result:
            norm_types.add(result['norm_type'])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(norm_types)))
    norm_colors = {norm: color for norm, color in zip(norm_types, colors)}
    
    # Define line styles for initial vs final models
    line_styles = {
        'initial': ':',    # Dotted
        'final': '-'       # Solid
    }
    
    # Plot each model's metric
    for model_name, result in results_dict.items():
        if ('mean' not in result or 
            metric_name not in result['mean'] or 
            not result['mean'][metric_name]):
            logger.warning(f"Missing {metric_name} data for {model_name}")
            continue
            
        norm_type = result.get('norm_type', 'unknown')
        
        # Determine if initial or final model
        stage = 'initial' if 'initial' in model_name else 'final'
        
        # Get layer data
        data = result['mean'][metric_name]
        
        # Convert string layer indices to int and sort
        layers = []
        for layer in data.keys():
            try:
                if layer == '-1':  # Skip embedding layer for plotting
                    continue
                layers.append(int(layer) if isinstance(layer, str) else layer)
            except:
                continue
                
        layers = sorted(layers)
        
        # Get values for each layer, supporting both string and int keys
        values = []
        for layer in layers:
            if str(layer) in data:
                values.append(data[str(layer)])
            elif layer in data:
                values.append(data[layer])
                
        if not values:
            continue
        
        # Set color by norm type and line style by stage
        color = norm_colors.get(norm_type, 'gray')
        linestyle = line_styles[stage] if include_stage else '-'
        
        # Create label
        if include_stage:
            label = f"{norm_type} ({stage})"
        else:
            label = norm_type
            
        # Plot the data
        plt.plot(layers, values, marker='o', linestyle=linestyle, 
                color=color, label=label, linewidth=2 if stage == 'final' else 1)
    
    # Add labels and legend
    metric_labels = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Change Between Layers)',
        'gradient_norm': 'Gradient Norm Across Layers',
        'gradient_update_correlation': 'Correlation (Gradient ↔ Update)'
    }
    
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f"Comparison of {metric_labels.get(metric_name, metric_name)} Across Models", fontsize=14)
        
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric_labels.get(metric_name, metric_name), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create a legend with smaller font and more columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, fontsize='small', frameon=True)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_initial_vs_final(results_dict, metric_name, norm_type, output_path):
    """
    Create a plot comparing initial vs final models for a specific normalization type.
    
    Args:
        results_dict: Dictionary mapping model names to their aggregated results
        metric_name: The metric to compare
        norm_type: The normalization type to focus on
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Find initial and final models for this norm type
    initial_data = None
    final_data = None
    
    for model_name, result in results_dict.items():
        if result.get('norm_type') == norm_type:
            if 'initial' in model_name:
                initial_data = result
            elif 'final' in model_name:
                final_data = result
    
    if not initial_data or not final_data:
        logger.warning(f"Missing data for {norm_type} comparison")
        return
    
    # Function to extract and plot data
    def extract_and_plot(result, label, color, linestyle):
        if ('mean' not in result or
            metric_name not in result['mean'] or
            not result['mean'][metric_name]):
            return
        
        data = result['mean'][metric_name]
        
        # Convert string layer indices to int and sort
        layers = []
        for layer in data.keys():
            try:
                if layer == '-1':  # Skip embedding layer for plotting
                    continue
                layers.append(int(layer) if isinstance(layer, str) else layer)
            except:
                continue
                
        layers = sorted(layers)
        
        # Get values for each layer, supporting both string and int keys
        values = []
        for layer in layers:
            if str(layer) in data:
                values.append(data[str(layer)])
            elif layer in data:
                values.append(data[layer])
                
        if not values:
            return
        
        # Plot the data
        plt.plot(layers, values, marker='o', linestyle=linestyle, 
                color=color, label=label, linewidth=2)
    
    # Plot both models
    extract_and_plot(initial_data, "Initial Model", 'blue', '--')
    extract_and_plot(final_data, "After Warmup", 'red', '-')
    
    # Add labels and formatting
    metric_labels = {
        'cosine_sim': 'Average Cosine Similarity',
        'token_norm': 'Token Representation Norm',
        'update_norm': 'Update Norm (Change Between Layers)',
        'gradient_norm': 'Gradient Norm Across Layers',
        'gradient_update_correlation': 'Correlation (Gradient ↔ Update)'
    }
    
    plt.title(f"{metric_labels.get(metric_name, metric_name)} for {norm_type.upper()}: Initial vs. After Warmup", fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric_labels.get(metric_name, metric_name), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_summary_table(results_dict, output_path):
    """
    Generate a summary table comparing metrics across normalization types.
    
    Args:
        results_dict: Dictionary mapping model names to their aggregated results
        output_path: Path to save the summary
    """
    # Group results by normalization type and stage (initial/final)
    grouped_results = defaultdict(dict)
    
    for model_name, result in results_dict.items():
        if 'norm_type' not in result:
            continue
            
        norm_type = result['norm_type']
        stage = 'initial' if 'initial' in model_name else 'final'
        
        grouped_results[norm_type][stage] = result
    
    # Metrics to analyze
    metrics = ['cosine_sim', 'token_norm', 'update_norm', 'gradient_norm', 'gradient_update_correlation']
    
    # Create markdown summary
    summary = "# Normalization Techniques Comparison\n\n"
    summary += "This document compares different normalization techniques in transformer models,\n"
    summary += "focusing on token geometry metrics both before and after warmup training.\n\n"
    
    # Add descriptions of each normalization technique
    summary += "## Normalization Techniques\n\n"
    summary += "- **PreLN**: Normalization before attention & MLP blocks\n"
    summary += "- **PostLN**: Normalization after attention & MLP blocks\n"
    summary += "- **PeriLN**: Normalization both before and after attention & MLP blocks\n"
    summary += "- **MixLN**: Mixed approach with some layers using PreLN and others using PostLN\n"
    summary += "- **PreDyT**: Dynamic normalization before blocks\n"
    summary += "- **PostDyT**: Dynamic normalization after blocks\n"
    summary += "- **DeepNorm**: Special normalization with scaled residual connections\n\n"
    
    for metric in metrics:
        metric_labels = {
            'cosine_sim': 'Average Cosine Similarity',
            'token_norm': 'Token Representation Norm',
            'update_norm': 'Update Norm (Change Between Layers)',
            'gradient_norm': 'Gradient Norm',
            'gradient_update_correlation': 'Gradient-Update Correlation'
        }
        
        summary += f"## {metric_labels.get(metric, metric)}\n\n"
        summary += "| Normalization Type | Initial | After Warmup | Change |\n"
        summary += "|-------------------|---------|--------------|--------|\n"
        
        # Compute layer-averaged value for each norm type
        for norm_type in sorted(grouped_results.keys()):
            initial_value = "N/A"
            final_value = "N/A"
            change = "N/A"
            
            # Get initial value
            if 'initial' in grouped_results[norm_type]:
                initial_result = grouped_results[norm_type]['initial']
                if ('mean' in initial_result and metric in initial_result['mean'] and 
                    initial_result['mean'][metric]):
                    values = [v for v in initial_result['mean'][metric].values() if v is not None]
                    if values:
                        initial_value = f"{np.mean(values):.4f}"
            
            # Get final value
            if 'final' in grouped_results[norm_type]:
                final_result = grouped_results[norm_type]['final']
                if ('mean' in final_result and metric in final_result['mean'] and 
                    final_result['mean'][metric]):
                    values = [v for v in final_result['mean'][metric].values() if v is not None]
                    if values:
                        final_value = f"{np.mean(values):.4f}"
            
            # Compute change if both values are available
            if initial_value != "N/A" and final_value != "N/A":
                init_val = float(initial_value)
                final_val = float(final_value)
                change_pct = ((final_val - init_val) / init_val) * 100 if init_val != 0 else 0
                change = f"{change_pct:+.2f}%"
            
            summary += f"| {norm_type} | {initial_value} | {final_value} | {change} |\n"
        
        summary += "\n"
    
    # Add analysis section
    summary += "## Analysis and Observations\n\n"
    summary += "### Key Findings\n\n"
    summary += "- **Token Similarity**: How token representations relate to each other\n"
    summary += "- **Token Norms**: How the magnitude of token representations changes\n"
    summary += "- **Gradient Behavior**: How gradients flow through the network\n\n"
    
    summary += "### Normalization Comparison\n\n"
    summary += "Different normalization techniques show varying behaviors in:\n\n"
    summary += "1. **Stability**: How consistent metrics are across layers\n"
    summary += "2. **Change after training**: How much metrics change after warming up the model\n"
    summary += "3. **Gradient propagation**: How effectively gradients flow through the network\n\n"
    
    # Save summary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Summary table saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze token geometry across models with different normalization techniques')
    
    # Model options
    parser.add_argument('--models-dir', default='saved_models/norm_comparison',
                      help='Directory containing generated models with different normalizations')
    parser.add_argument('--models-summary', default=None,
                      help='Path to models summary JSON file (generated by generate_norm_models.py)')
    
    # Data options
    parser.add_argument('--prompt-completion-file', default=None,
                      help='Path to JSON file with prompt-completion pairs')
    parser.add_argument('--create-sample-data', action='store_true',
                      help='Create sample prompt-completion pairs if none provided')
    parser.add_argument('--max-pairs', type=int, default=3,
                      help='Maximum number of prompt-completion pairs to analyze per model')
    
    # Other options
    parser.add_argument('--tokenizer-dir', default='tokenizer/',
                      help='Path to tokenizer directory')
    parser.add_argument('--output-dir', default='outputs/norm_geometry_analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--device', default=None,
                      help='Device to run on (cuda, mps, or cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device or get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(model_dir=args.tokenizer_dir)
    
    # Load or create prompt-completion pairs
    if args.prompt_completion_file:
        pairs = load_prompt_completion_pairs(args.prompt_completion_file)
        if not pairs:
            logger.warning("Failed to load prompt-completion pairs, creating sample data")
            sample_file = os.path.join(args.output_dir, "sample_prompts.json")
            pairs = create_sample_prompt_completion_pairs(sample_file)
    elif args.create_sample_data:
        sample_file = os.path.join(args.output_dir, "sample_prompts.json")
        pairs = create_sample_prompt_completion_pairs(sample_file)
    else:
        logger.error("No prompt-completion data provided. Use --prompt-completion-file or --create-sample-data")
        return
    
    logger.info(f"Using {len(pairs)} prompt-completion pairs for analysis")
    
    # Get model paths
    model_paths = {}
    
    if args.models_summary:
        # Load from summary file
        summary = load_models_summary(args.models_summary)
        if summary and 'model_paths' in summary:
            model_paths = summary['model_paths']
    else:
        # Find models in the directory
        model_paths = find_model_paths(args.models_dir)
    
    if not model_paths:
        logger.error("No models found for analysis")
        return
    
    logger.info(f"Found {len(model_paths)} model types with initial and final versions")
    
    # Analyze all models
    results_dict = {}
    
    for norm_type, paths in model_paths.items():
        # Analyze initial model
        initial_result = analyze_model(
            model_path=paths['initial'],
            pairs=pairs,
            tokenizer=tokenizer,
            device=device,
            output_dir=args.output_dir,
            model_name=f"{norm_type}_initial",
            max_pairs=args.max_pairs
        )
        
        if initial_result:
            results_dict[f"{norm_type}_initial"] = initial_result
        
        # Analyze final (warmed-up) model
        final_result = analyze_model(
            model_path=paths['final'],
            pairs=pairs,
            tokenizer=tokenizer,
            device=device,
            output_dir=args.output_dir,
            model_name=f"{norm_type}_final",
            max_pairs=args.max_pairs
        )
        
        if final_result:
            results_dict[f"{norm_type}_final"] = final_result
    
    if not results_dict:
        logger.error("No successful model analyses. Exiting.")
        return
    
    # Save all results to a single file
    with open(os.path.join(args.output_dir, "all_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Generate comparative visualizations
    logger.info("Generating comparative visualizations")
    comparison_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Compare all models for each metric
    metrics = ['cosine_sim', 'token_norm', 'update_norm', 'gradient_norm', 'gradient_update_correlation']
    
    for metric in metrics:
        # All models in one plot
        plot_comparison(
            results_dict=results_dict,
            metric_name=metric,
            output_path=os.path.join(comparison_dir, f"all_models_{metric}.png"),
            title=f"Comparison of All Models: {metric}"
        )
        
        # Just final models
        final_results = {k: v for k, v in results_dict.items() if 'final' in k}
        plot_comparison(
            results_dict=final_results,
            metric_name=metric,
            output_path=os.path.join(comparison_dir, f"final_models_{metric}.png"),
            title=f"Comparison of Warmed-Up Models: {metric}",
            include_stage=False
        )
        
        # Just initial models
        initial_results = {k: v for k, v in results_dict.items() if 'initial' in k}
        plot_comparison(
            results_dict=initial_results,
            metric_name=metric,
            output_path=os.path.join(comparison_dir, f"initial_models_{metric}.png"),
            title=f"Comparison of Random Models: {metric}",
            include_stage=False
        )
    
    # 2. Compare initial vs final for each normalization type
    for norm_type in set(result['norm_type'] for result in results_dict.values()):
        norm_dir = os.path.join(comparison_dir, norm_type)
        os.makedirs(norm_dir, exist_ok=True)
        
        for metric in metrics:
            plot_initial_vs_final(
                results_dict=results_dict,
                metric_name=metric,
                norm_type=norm_type,
                output_path=os.path.join(norm_dir, f"{norm_type}_{metric}_comparison.png")
            )
    
    # 3. Generate summary table and analysis
    generate_summary_table(
        results_dict=results_dict,
        output_path=os.path.join(args.output_dir, "normalization_comparison.md")
    )
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 