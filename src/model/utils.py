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

def create_random_model(pre_ln: bool = True, 
                       vocab_size: Optional[int] = None, 
                       device: Optional[str] = None, 
                       n_layer: int = 24,
                       use_rms_norm: bool = True, 
                       use_swiglu: bool = True) -> GPT:
    """
    Create a randomly initialized model.
    
    Args:
        pre_ln: Whether to use PreLN architecture
        vocab_size: Vocabulary size
        device: Device to place the model on
        n_layer: Number of transformer layers to use
        use_rms_norm: Whether to use RMSNorm (LLaMA-style)
        use_swiglu: Whether to use SwiGLU activation (LLaMA-style)
        
    Returns:
        A randomly initialized GPT model
    """
    if vocab_size is None:
        # Default to BPE vocabulary size if available
        try:
            from src.tokenizer.bpe import BPETokenizer
            if os.path.exists('data/embedding/encoder.json'):
                tokenizer = BPETokenizer('data/embedding/encoder.json', 'data/embedding/vocab.bpe')
                vocab_size = tokenizer.vocab_size
            else:
                vocab_size = 256  # Default for character-level
        except Exception as e:
            logger.error(f"Error creating tokenizer: {e}")
            vocab_size = 256  # Fallback to character-level
    
    # Configure model with specified number of layers
    config = GPTConfig(
        block_size=256,
        vocab_size=vocab_size,
        n_layer=n_layer,   # Configurable number of layers
        n_head=min(12, n_layer),  # Scale attention heads with layers but cap at 12
        n_embd=384,   # Embedding dimension
        dropout=0.1,
        bias=False,
        pre_ln=pre_ln,
        use_rms_norm=use_rms_norm,
        use_swiglu=use_swiglu,
        norm_eps=1e-6
    )
    
    logger.info(f"Creating model with {n_layer} layers, " +
               f"{'RMSNorm' if use_rms_norm else 'LayerNorm'}, " +
               f"{'SwiGLU' if use_swiglu else 'GELU'}, " +
               f"{'PreLN' if pre_ln else 'PostLN'} architecture")
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
    return (f"{'LLaMA' if config.use_rms_norm and config.use_swiglu else 'Standard'}-" +
            f"{'PreLN' if config.pre_ln else 'PostLN'}")


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