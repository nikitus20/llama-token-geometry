"""
Test script for verifying model and analyzer functionality.
"""

import os
import sys
import torch
import logging
import argparse
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import GPT
from src.model.utils import create_random_model, get_device
from src.utils.data import get_tokenizer
from src.analyzer.geometry import GeometryAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test creating models with different configurations."""
    logger.info("Testing model creation...")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    configurations = [
        {"name": "LLaMA-PreLN", "ln_type": "preln", "use_rms_norm": True, "use_swiglu": True},
        {"name": "LLaMA-PostLN", "ln_type": "postln", "use_rms_norm": True, "use_swiglu": True},
        {"name": "LLaMA-PeriLN", "ln_type": "periln", "use_rms_norm": True, "use_swiglu": True},
        {"name": "LLaMA-MixLN", "ln_type": "mixln", "use_rms_norm": True, "use_swiglu": True},
        {"name": "Standard-PreLN", "ln_type": "preln", "use_rms_norm": False, "use_swiglu": False},
        {"name": "Standard-PostLN", "ln_type": "postln", "use_rms_norm": False, "use_swiglu": False}
    ]
    
    for config in configurations:
        logger.info(f"Creating {config['name']} model...")
        model = create_random_model(
            ln_type=config["ln_type"],
            use_rms_norm=config["use_rms_norm"],
            use_swiglu=config["use_swiglu"],
            n_layer=2,  # Use small model for testing
            device=device
        )
        logger.info(f"{config['name']} model created successfully with {model.get_num_params()/1e6:.2f}M parameters")
        
        # Test forward pass
        dummy_input = torch.randint(0, 100, (1, 10), device=device)
        logits, _ = model(dummy_input)
        logger.info(f"Forward pass successful, output shape: {logits.shape}")
    
    logger.info("All model configurations created successfully!")


def test_tokenizer():
    """Test tokenizer functionality."""
    logger.info("Testing tokenizers...")
    
    # Test character tokenizer
    tokenizer = get_tokenizer(tokenizer_type="char")
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    logger.info(f"Character tokenizer: '{test_text}' -> {tokens} -> '{decoded}'")
    
    # Try to test BPE tokenizer if available
    try:
        bpe_tokenizer = get_tokenizer(tokenizer_type="bpe")
        bpe_tokens = bpe_tokenizer.encode(test_text)
        bpe_decoded = bpe_tokenizer.decode(bpe_tokens)
        logger.info(f"BPE tokenizer: '{test_text}' -> {bpe_tokens} -> '{bpe_decoded}'")
    except Exception as e:
        logger.warning(f"BPE tokenizer test skipped: {e}")
        
    # Try to test Tiktoken tokenizer if available
    try:
        tiktoken_tokenizer = get_tokenizer(tokenizer_type="tiktoken")
        tiktoken_tokens = tiktoken_tokenizer.encode(test_text)
        tiktoken_decoded = tiktoken_tokenizer.decode(tiktoken_tokens)
        logger.info(f"Tiktoken tokenizer: '{test_text}' -> {tiktoken_tokens} -> '{tiktoken_decoded}'")
    except Exception as e:
        logger.warning(f"Tiktoken tokenizer test skipped: {e}")
    
    logger.info("Tokenizer tests completed!")


def test_geometry_analyzer():
    """Test geometry analyzer functionality."""
    logger.info("Testing geometry analyzer...")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create a small model for testing
    model = create_random_model(n_layer=2, device=device)
    
    # Create analyzer
    analyzer = GeometryAnalyzer(model, device=device)
    
    # Test with a simple input
    tokenizer = get_tokenizer(tokenizer_type="char")
    test_text = "This is a test sentence for the analyzer."
    tokens = tokenizer.encode(test_text)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Test computing token metrics
    metrics = analyzer.compute_token_metrics(input_ids)
    logger.info(f"Computed token metrics: {metrics}")
    
    # Test analyzing single prompt
    analysis_result = analyzer.analyze_single_prompt(test_text, tokenizer)
    logger.info(f"Analyzed prompt, got {len(analysis_result['similarity_matrices'])} similarity matrices and metrics: {analysis_result['metrics']}")
    
    # Cleanup
    analyzer.cleanup()
    
    logger.info("Geometry analyzer tests completed!")


def test_pretrained_model_loading():
    """Test loading a pretrained model if available."""
    logger.info("Testing pretrained model loading...")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Check for saved models
    saved_models_dir = 'saved_models'
    if not os.path.exists(saved_models_dir):
        logger.warning(f"No saved models directory found at {saved_models_dir}")
        return
    
    # Look for model directories
    for item in os.listdir(saved_models_dir):
        model_dir = os.path.join(saved_models_dir, item)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "model.pt")):
            try:
                logger.info(f"Found model at {model_dir}, attempting to load...")
                model = GPT.from_pretrained(model_dir, device=device)
                logger.info(f"Successfully loaded model with {model.get_num_params()/1e6:.2f}M parameters")
                
                # Test forward pass
                dummy_input = torch.randint(0, 100, (1, 10), device=device)
                logits, _ = model(dummy_input)
                logger.info(f"Forward pass successful, output shape: {logits.shape}")
                return
            except Exception as e:
                logger.error(f"Error loading model from {model_dir}: {e}")
    
    logger.warning("No pretrained models found or loaded")


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description='Run tests for Token Geometry Analyzer')
    parser.add_argument('--skip-model-creation', action='store_true',
                       help='Skip model creation tests')
    parser.add_argument('--skip-tokenizer', action='store_true',
                       help='Skip tokenizer tests')
    parser.add_argument('--skip-analyzer', action='store_true',
                       help='Skip analyzer tests')
    parser.add_argument('--skip-pretrained', action='store_true',
                       help='Skip pretrained model loading tests')
    args = parser.parse_args()
    
    # Run tests
    if not args.skip_model_creation:
        test_model_creation()
    
    if not args.skip_tokenizer:
        test_tokenizer()
    
    if not args.skip_analyzer:
        test_geometry_analyzer()
    
    if not args.skip_pretrained:
        test_pretrained_model_loading()
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    main()