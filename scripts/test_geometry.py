#!/usr/bin/env python
"""
Script to analyze token geometry of a saved model.
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

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import GPT
from src.utils.data import get_tokenizer
from src.analyzer.geometry import GeometryAnalyzer
from src.analyzer.visualization import plot_token_geometry, plot_similarity_matrix

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
            "prompt": "Explain how neural networks work.",
            "completion": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) that process and transmit information."
        },
        {
            "prompt": "What is gradient descent?",
            "completion": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving toward the steepest descent as defined by the negative of the gradient."
        },
        {
            "prompt": "How does backpropagation work?",
            "completion": "Backpropagation is an algorithm used to train neural networks by calculating gradients of the loss function with respect to the weights, propagating these gradients backwards through the network."
        }
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(sample_pairs, f, indent=2)
        
    logger.info(f"Created sample prompt-completion pairs file at {output_file}")
    return sample_pairs

def analyze_single_example(model, prompt, completion, tokenizer, device, output_dir):
    """
    Analyze a single prompt-completion pair.
    
    Args:
        model: GPT model
        prompt: Text prompt
        completion: Text completion
        tokenizer: Tokenizer
        device: Device to run on
        output_dir: Directory to save results
        
    Returns:
        Analysis results
    """
    # Create analyzer
    analyzer = GeometryAnalyzer(model, device=device, track_gradients=True)
    
    try:
        # Analyze with completion
        logger.info(f"Analyzing prompt: \"{prompt[:50]}...\"")
        results = analyzer.analyze_with_completion(prompt, completion, tokenizer)
        
        # Clean up analyzer
        analyzer.cleanup()
        
        # Save results to file
        results_file = os.path.join(output_dir, "analysis_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if key == 'similarity_matrices':
                    # Skip large matrices in JSON output
                    continue
                elif isinstance(value, dict):
                    serializable_results[key] = {k: v for k, v in value.items()}
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
                    
            json.dump(serializable_results, f, indent=2)
            
        # Save tokens for reference
        with open(os.path.join(output_dir, "tokens.txt"), 'w') as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Completion: {completion}\n\n")
            f.write(f"Prompt tokens ({len(results['prompt_tokens'])} tokens):\n")
            f.write(str(results['prompt_tokens']) + "\n\n")
            f.write(f"Completion tokens ({len(results['completion_tokens'])} tokens):\n")
            f.write(str(results['completion_tokens']))
            
        # Plot visualizations for the metrics
        if 'metrics' in results:
            plot_token_geometry(
                results['metrics'],
                title="Token Geometry Analysis",
                output_path=os.path.join(output_dir, "token_geometry.png")
            )
            
        # Plot similarity matrices
        if 'similarity_matrices' in results:
            matrices_dir = os.path.join(output_dir, "similarity_matrices")
            os.makedirs(matrices_dir, exist_ok=True)
            
            for layer_idx, matrix in results['similarity_matrices'].items():
                try:
                    plot_similarity_matrix(
                        matrix,
                        title=f"Layer {layer_idx} Token Similarity",
                        output_path=os.path.join(matrices_dir, f"layer_{layer_idx}_similarity.png")
                    )
                except Exception as e:
                    logger.error(f"Error plotting similarity matrix for layer {layer_idx}: {e}")
                    
        return results
    except Exception as e:
        logger.error(f"Error analyzing example: {e}")
        return None

def batch_analysis(model, pairs, tokenizer, device, output_dir, max_pairs=None):
    """
    Analyze a batch of prompt-completion pairs.
    
    Args:
        model: GPT model
        pairs: List of prompt-completion pairs
        tokenizer: Tokenizer
        device: Device to run on
        output_dir: Directory to save results
        max_pairs: Maximum number of pairs to analyze
        
    Returns:
        Aggregated results
    """
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
        logger.info(f"Limiting analysis to {max_pairs} pairs (out of {len(pairs)})")
        # Use a random subset
        import random
        selected_pairs = random.sample(pairs, max_pairs)
    else:
        selected_pairs = pairs
        
    logger.info(f"Analyzing {len(selected_pairs)} prompt-completion pairs")
    
    # Process each pair
    for i, pair in enumerate(tqdm(selected_pairs, desc="Analyzing pairs")):
        prompt = pair['prompt']
        completion = pair['completion']
        
        try:
            # Analyze the pair
            pair_dir = os.path.join(output_dir, f"pair_{i+1}")
            os.makedirs(pair_dir, exist_ok=True)
            
            # Run analysis
            results = analyzer.analyze_with_completion(prompt, completion, tokenizer)
            
            # Save pair info
            with open(os.path.join(pair_dir, "pair_info.txt"), 'w') as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Completion: {completion}\n\n")
                if 'loss' in results:
                    f.write(f"Loss: {results['loss']}\n")
                    
            # Aggregate metrics
            if 'metrics' in results:
                metrics = results['metrics']
                for metric_name, layer_values in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = {}
                        
                    for layer_idx, value in layer_values.items():
                        if layer_idx not in all_metrics[metric_name]:
                            all_metrics[metric_name][layer_idx] = []
                        all_metrics[metric_name][layer_idx].append(value)
        except Exception as e:
            logger.error(f"Error processing pair {i+1}: {e}")
            continue
    
    # Clean up analyzer
    analyzer.cleanup()
    
    # Compute mean and variance
    aggregated_results = {
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
    results_file = os.path.join(output_dir, "aggregated_results.json")
    with open(results_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
        
    # Plot aggregated metrics
    plot_token_geometry(
        aggregated_results,
        title=f"Aggregated Token Geometry Analysis ({len(selected_pairs)} samples)",
        output_path=os.path.join(output_dir, "aggregated_metrics.png")
    )
    
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Analyze token geometry of a saved model')
    parser.add_argument('--model-path', required=True, help='Path to the saved model')
    parser.add_argument('--tokenizer-dir', default='tokenizer/', help='Path to tokenizer directory')
    parser.add_argument('--output-dir', default='outputs/token_geometry', help='Directory to save outputs')
    parser.add_argument('--device', default=None, help='Device to run on (cuda, mps, or cpu)')
    parser.add_argument('--no-gradient-tracking', action='store_true', help='Disable gradient tracking')
    parser.add_argument('--batch-size', type=int, default=None, help='Number of prompt-completion pairs to process')
    
    # Input data options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt', help='Text prompt to analyze')
    group.add_argument('--prompt-completion-file', help='Path to JSON file with prompt-completion pairs')
    group.add_argument('--create-sample-data', action='store_true', help='Create sample prompt-completion pairs')
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device or get_device()
    
    # Create the output directory
    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analysis results will be saved to {output_dir}")
    
    # Handle the create-sample-data option
    if args.create_sample_data:
        sample_file = os.path.join('data', 'sample_prompt_completion_pairs.json')
        create_sample_prompt_completion_pairs(sample_file)
        logger.info(f"Created sample data at {sample_file}")
        logger.info(f"To use this file, run: python -m scripts.test_geometry --model-path {args.model_path} --prompt-completion-file {sample_file}")
        return
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(model_dir=args.tokenizer_dir)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = GPT.from_pretrained(args.model_path, device=device)
    model.to(device)
    model.eval()
    
    # Process based on input type
    if args.prompt:
        # Single prompt analysis
        prompt = args.prompt
        completion = None  # No completion for single prompt analysis
        
        # Create directory for this analysis
        single_output_dir = os.path.join(output_dir, "single_prompt_analysis")
        os.makedirs(single_output_dir, exist_ok=True)
        
        # Run the analysis
        analyze_single_example(model, prompt, completion, tokenizer, device, single_output_dir)
        
    elif args.prompt_completion_file:
        # Batch analysis with prompt-completion pairs
        pairs = load_prompt_completion_pairs(args.prompt_completion_file)
        
        if not pairs:
            logger.error("No valid prompt-completion pairs found. Exiting.")
            return
            
        logger.info(f"Loaded {len(pairs)} prompt-completion pairs")
        
        # Create directory for batch analysis
        batch_output_dir = os.path.join(output_dir, "batch_analysis")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Run batch analysis
        batch_analysis(model, pairs, tokenizer, device, batch_output_dir, args.batch_size)
        
        # Also analyze first pair individually
        first_pair = pairs[0]
        logger.info("Analyzing first pair individually for detailed visualization")
        
        # Create directory for detailed analysis
        detailed_dir = os.path.join(output_dir, "detailed_first_pair")
        os.makedirs(detailed_dir, exist_ok=True)
        
        # Run detailed analysis
        analyze_single_example(
            model, 
            first_pair['prompt'], 
            first_pair['completion'],
            tokenizer,
            device,
            detailed_dir
        )
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()