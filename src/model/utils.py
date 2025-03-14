"""
Utility functions for creating and managing models.
"""

import os
import torch
import logging
from typing import Optional

from src.model.config import GPTConfig
from src.model.transformer import GPT

logger = logging.getLogger(__name__)

def get_device() -> str:
    """
    Get the best available device for running models.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def create_random_model(
    vocab_size=256, 
    n_layer=12, 
    n_head=12, 
    n_embd=768, 
    ln_type="postln", 
    use_initial_ln=True, 
    mixln_split=0.25, 
    use_swiglu=True, 
    device=None
):
    """
    Create a randomly initialized model with the specified architecture.
    
    Args:
        vocab_size: Size of the vocabulary
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        ln_type: Layer normalization architecture (preln, postln, periln, mixln)
        use_initial_ln: Whether to apply normalization after embeddings
        mixln_split: Fraction of layers to use postln in mixln architecture
        use_swiglu: Whether to use SwiGLU activation
        device: Device to move the model to
        
    Returns:
        Randomly initialized model
    """
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=1024,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        ln=ln_type,
        use_initial_ln=use_initial_ln,
        mixln_split=mixln_split,
        use_swiglu=use_swiglu,
        norm_eps=1e-6,
        initializer_range=0.02
    )
    
    logger.info(f"Creating model with {n_layer} layers, " +
               f"RMSNorm, " +
               f"{'SwiGLU' if use_swiglu else 'GELU'}, " +
               f"{ln_type.capitalize()} architecture, " +
               f"{'with' if use_initial_ln else 'without'} initial normalization")
    model = GPT(config)
    
    # Move to specified device if provided
    if device:
        model = model.to(device)
        
    return model


def get_model_info(model: GPT) -> str:
    """
    Get a string description of a model's architecture.
    
    Args:
        model: The GPT model
        
    Returns:
        String description of the model
    """
    config = model.config
    norm_type = config.ln.capitalize()
    # We now always use RMSNorm, so only check for SwiGLU to determine if it's LLaMA-style
    model_type = 'LLaMA' if config.use_swiglu else 'Standard'
    initial = '+Initial' if config.use_initial_ln else ''
    return f"{model_type}-{norm_type}{initial}"


def get_available_models(models_dir: str = "saved_models") -> list:
    """
    Get a list of available pretrained models.
    
    Args:
        models_dir: Directory where models are saved
        
    Returns:
        List of model names
    """
    if not os.path.exists(models_dir):
        return []
    
    # Look for directories that contain both model.pt and config.pt
    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "model.pt")) and \
               os.path.exists(os.path.join(item_path, "config.pt")):
                models.append(item)
    
    return models