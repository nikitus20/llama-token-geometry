"""
Data utilities for loading and processing prompts.
"""

import os
import logging
from typing import List, Optional

from src.tokenizer.base import BaseTokenizer
from src.tokenizer.character import CharTokenizer
from src.tokenizer.bpe import BPETokenizer

logger = logging.getLogger(__name__)

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


def get_tokenizer(tokenizer_type: str = "tiktoken") -> BaseTokenizer:
    """
    Get a tokenizer based on the specified type.
    
    Args:
        tokenizer_type: Type of tokenizer to use ("tiktoken", "bpe", or "char")
        
    Returns:
        A tokenizer instance
    """
    # Try tiktoken first (best option)
    if tokenizer_type.lower() == "tiktoken":
        try:
            logger.info("Using Tiktoken tokenizer (GPT-2 encoding)")
            return TiktokenTokenizer("gpt2")
        except Exception as e:
            logger.error(f"Error loading Tiktoken tokenizer: {e}")
            logger.info("Falling back to BPE tokenizer")
            tokenizer_type = "bpe"
    
    # Try BPE next
    if tokenizer_type.lower() == "bpe" and os.path.exists('data/embedding/encoder.json') and os.path.exists('data/embedding/vocab.bpe'):
        try:
            logger.info("Using BPE tokenizer from embedding files")
            return BPETokenizer('data/embedding/encoder.json', 'data/embedding/vocab.bpe')
        except Exception as e:
            logger.error(f"Error loading BPE tokenizer: {e}")
            logger.info("Falling back to character tokenizer")
    
    # Character tokenizer as last resort
    logger.info("Using character-level tokenizer")
    return CharTokenizer(256)


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