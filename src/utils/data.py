"""
Data utilities for loading and processing prompts.
"""

import os
import logging
from typing import List, Optional
from src.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Local tokenizer path
DEFAULT_TOKENIZER_PATH = 'tokenizer/'

def load_prompts(prompt_file: str = 'data/prompts.txt', num_samples: Optional[int] = None) -> List[str]:
    """
    Load prompts from a file, or return a small set for testing.
    
    Args:
        prompt_file: Path to file containing prompts
        num_samples: Optional number of prompts to use
        
    Returns:
        List of prompts
    """
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            # Use a subset for quick testing if specified
            if num_samples and num_samples < len(prompts):
                prompts = prompts[:num_samples]
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts from {prompt_file}: {e}")
            return _get_sample_prompts()
    else:
        logger.warning(f"Prompt file {prompt_file} not found. Creating sample prompts.")
        return _get_sample_prompts()


def _get_sample_prompts() -> List[str]:
    """Return a set of sample prompts for testing."""
    return [
        "Once upon a time in a galaxy far, far away",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "It was the best of times, it was the worst of times",
        "In the beginning, God created the heavens and the earth"
    ]


def get_tokenizer(tokenizer_type: str = None, model_name: str = None, 
                 local_tokenizer_path: str = None, model_dir: str = DEFAULT_TOKENIZER_PATH):
    """
    Get a HuggingFace tokenizer from local files.
    
    Args:
        tokenizer_type: Kept for backward compatibility, ignored
        model_name: Kept for backward compatibility, ignored
        local_tokenizer_path: Path to local tokenizer directory (takes precedence over model_dir)
        model_dir: Path to local tokenizer directory (default)
            
    Returns:
        A Tokenizer instance
    """
    # Support legacy parameters
    path = local_tokenizer_path if local_tokenizer_path is not None else model_dir
    
    if tokenizer_type is not None:
        logger.warning(f"tokenizer_type parameter '{tokenizer_type}' is deprecated and ignored. " 
                      "Using HuggingFace tokenizer only.")
    
    if model_name is not None:
        logger.warning(f"model_name parameter '{model_name}' is deprecated and ignored. " 
                      "Using local tokenizer from {path}")
    
    return Tokenizer(model_dir=path)


def create_prompts_file(prompts: List[str], output_path: str = 'data/prompts.txt') -> None:
    """
    Create a prompts file with the given prompts.
    
    Args:
        prompts: List of prompts to write to file
        output_path: Path to write the prompts file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    logger.info(f"Created prompts file at {output_path} with {len(prompts)} prompts")