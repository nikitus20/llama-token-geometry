"""
Analysis script for token geometry in transformer models.
"""

import os
import sys
import argparse
import logging
import torch

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import GPT
from src.model.utils import create_random_model, get_model_info, get_available_models, get_device
from src.tokenizer.base import BaseTokenizer
from src.utils.data import load_prompts, get_tokenizer
from src.analyzer.geometry import GeometryAnalyzer
from src.analyzer.visualization import (
    plot_token_geometry, 
    plot_similarity_matrix, 
    plot_architecture_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def analyze_similarity_heatmaps(prompts, output_dir, models, tokenizer, device):
    """
    Create similarity matrix heatmaps for each layer of different model architectures.
    
    Args:
        prompts: List of text prompts
        output_dir: Directory to save results
        models: Dictionary of model names -> model instances
        tokenizer: Tokenizer to use
        device: Device to run on
    """
    # Create output directory
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Choose sample prompt (take first 50 chars of first prompt for clearer visualization)
    if prompts and len(prompts) > 0:
        sample_prompt = prompts[0][:50]
    else:
        sample_prompt = "Sample prompt for token geometry analysis."
        
    logger.info(f"Analyzing prompt: '{sample_prompt}'")
    
    # Analyze with each model
    sim_matrices = {}
    for model_name, model in models.items():
        analyzer = GeometryAnalyzer(model, device)
        
        try:
            matrices, tokens = analyzer.analyze_single_prompt(sample_prompt, tokenizer)
            sim_matrices[model_name] = matrices
            logger.info(f"Successfully analyzed {model_name} model with prompt.")
        except Exception as e:
            logger.error(f"Error analyzing {model_name} model: {e}")
            sim_matrices[model_name] = {}
            
        analyzer.cleanup()
    
    # Plot heatmaps for each layer for each model
    all_layers = set()
    for model_results in sim_matrices.values():
        all_layers.update(model_results.keys())
    all_layers = sorted(all_layers)
    
    for layer_idx in all_layers:
        for model_name, matrices in sim_matrices.items():
            if layer_idx in matrices:
                plot_similarity_matrix(
                    matrices[layer_idx],
                    title=f"{model_name} Layer {layer_idx} Token Similarity",
                    output_path=os.path.join(heatmap_dir, f"{model_name.lower()}_layer{layer_idx}_similarity.png")
                )
    
    # Generate difference heatmaps at key layers
    key_layers = [0, len(all_layers)//4, len(all_layers)//2, len(all_layers)-1]  # Beginning, 25%, 50%, and end layers
    
    # Compare PreLN vs PostLN
    for pretrained_name in [name for name in models if not name.startswith(("LLaMA", "Standard"))]:
        for layer_idx in key_layers:
            if (layer_idx in sim_matrices.get("LLaMA-PreLN", {}) and 
                layer_idx in sim_matrices.get(pretrained_name, {})):
                diff_matrix = sim_matrices["LLaMA-PreLN"][layer_idx] - sim_matrices[pretrained_name][layer_idx]
                plot_similarity_matrix(
                    diff_matrix,
                    title=f"Layer {layer_idx} LLaMA-PreLN vs {pretrained_name} Difference",
                    output_path=os.path.join(heatmap_dir, f"llama_preln_vs_{pretrained_name.lower()}_layer{layer_idx}_diff.png")
                )


def compare_architectures(prompts, output_dir, tokenizer, n_layer=24, device='cuda', 
                        include_trained_model=True, trained_model_path=None):
    """
    Compare token geometry between different model architectures.
    
    Args:
        prompts: List of prompts to analyze
        output_dir: Directory to save results
        tokenizer: Tokenizer to use
        n_layer: Number of layers to use for random models
        device: Device to run on
        include_trained_model: Whether to include a trained model in the comparison
        trained_model_path: Path to trained model directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Vocabulary size
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 256
    
    # Create dict to store results
    results = {}
    
    # Create models
    models = {}
    
    # Random models: LLaMA and standard variants
    configs = [
        {"name": "LLaMA-PreLN", "pre_ln": True, "use_rms_norm": True, "use_swiglu": True},
        {"name": "LLaMA-PostLN", "pre_ln": False, "use_rms_norm": True, "use_swiglu": True},
        {"name": "Standard-PreLN", "pre_ln": True, "use_rms_norm": False, "use_swiglu": False},
        {"name": "Standard-PostLN", "pre_ln": False, "use_rms_norm": False, "use_swiglu": False}
    ]
    
    for config in configs:
        logger.info(f"Creating {config['name']} model with {n_layer} layers...")
        models[config["name"]] = create_random_model(
            pre_ln=config["pre_ln"], 
            vocab_size=vocab_size, 
            device=device, 
            n_layer=n_layer,
            use_rms_norm=config["use_rms_norm"],
            use_swiglu=config["use_swiglu"]
        )
    
    # Add trained model if available
    if include_trained_model and trained_model_path and os.path.exists(trained_model_path):
        try:
            logger.info(f"Loading trained model from {trained_model_path}...")
            trained_model = GPT.from_pretrained(trained_model_path, device=device)
            trained_name = os.path.basename(trained_model_path)
            models[trained_name] = trained_model
            logger.info(f"Added trained model '{trained_name}' to comparison")
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
    
    # Analyze each model
    for model_name, model in models.items():
        logger.info(f"Analyzing {model_name} model...")
        analyzer = GeometryAnalyzer(model, device)
        cosine_sims = analyzer.analyze_prompts(prompts, tokenizer)
        
        # Save individual results
        plot_token_geometry(
            cosine_sims, 
            title=f"Token Geometry for {model_name} Model",
            output_path=os.path.join(output_dir, f"{model_name.lower()}_token_geometry.png")
        )
        analyzer.cleanup()
        
        # Store results for comparison
        results[model_name] = cosine_sims
    
    # Plot comparison of all architectures
    plot_architecture_comparison(results, output_dir, n_layer)
    
    # Generate heatmaps
    analyze_similarity_heatmaps(prompts, output_dir, models, tokenizer, device)
    
    logger.info(f"Results saved to {output_dir}")
    
    # Clean up models
    models = {}
    torch.cuda.empty_cache()


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description='Analyze token geometry in transformer models')
    parser.add_argument('--prompts', type=str, default='data/prompts.txt',
                       help='File containing prompts to analyze')
    parser.add_argument('--num-prompts', type=int, default=None,
                       help='Number of prompts to use (None for all)')
    parser.add_argument('--output-dir', type=str, default='outputs/token_geometry',
                       help='Directory to save results')
    parser.add_argument('--use-bpe', action='store_true', default=True,
                       help='Use BPE tokenizer instead of character tokenizer')
    parser.add_argument('--no-bpe', dest='use_bpe', action='store_false',
                       help='Do not use BPE tokenizer (use character tokenizer instead)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--layers', type=int, default=12,
                       help='Number of transformer layers to use (default: 12)')
    parser.add_argument('--trained-model', type=str, default='postln_model_warmup',
                       help='Name of trained model to include in analysis')
    parser.add_argument('--no-trained-model', action='store_false', dest='include_trained_model',
                       help='Do not include trained model in analysis')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda, mps, or cpu)')
    parser.add_argument('--tokenizer', type=str, choices=['tiktoken', 'bpe', 'char'], default='tiktoken',
                       help='Tokenizer to use (tiktoken, bpe, or char)')
    
    args = parser.parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set device
    if args.device is None:
        args.device = get_device()
    logger.info(f"Using device: {args.device}")
    
    try:
        # Load prompts
        prompts = load_prompts(args.prompts, args.num_prompts)
        logger.info(f"Loaded {len(prompts)} prompts")
        
        if len(prompts) == 0:
            logger.warning("No prompts loaded. Using default sample prompts.")
            prompts = load_prompts()
        
        # Get tokenizer
        tokenizer = get_tokenizer(tokenizer_type=args.tokenizer)
        
        # Check for trained model
        trained_model_path = None
        if args.include_trained_model:
            model_dir = os.path.join('saved_models', args.trained_model)
            if os.path.exists(model_dir):
                trained_model_path = model_dir
                logger.info(f"Found trained model at {trained_model_path}")
            else:
                logger.warning(f"Trained model not found at {model_dir}")
                available_models = get_available_models('saved_models')
                if available_models:
                    logger.info(f"Available trained models: {', '.join(available_models)}")
        
        # Run comparison
        compare_architectures(
            prompts=prompts,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            n_layer=args.layers,
            device=args.device,
            include_trained_model=args.include_trained_model,
            trained_model_path=trained_model_path
        )
        
        logger.info(f"Token geometry analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during token geometry analysis: {e}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    main()