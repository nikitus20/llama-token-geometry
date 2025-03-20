"""
Data utilities for loading and processing prompts.
"""

import os
import json
import logging
import random
import torch
from typing import List, Optional, Dict, Tuple
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

def load_prompt_completion_pairs(file_path: str, num_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load prompt-completion pairs from a file.
    
    Args:
        file_path: Path to JSON file with prompt-completion pairs
        num_samples: Optional number of pairs to sample
        
    Returns:
        List of dictionaries with 'prompt' and 'completion' keys
    """
    if not os.path.exists(file_path):
        logger.warning(f"Prompt-completion file {file_path} not found. Creating sample pairs.")
        return _get_sample_prompt_completion_pairs()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                # JSONL format (one JSON object per line)
                pairs = [json.loads(line) for line in f if line.strip()]
            else:
                # Regular JSON format (array of objects)
                pairs = json.load(f)
                
        # Ensure they have the expected format
        valid_pairs = [p for p in pairs if isinstance(p, dict) and 'prompt' in p and 'completion' in p]
        
        if len(valid_pairs) < len(pairs):
            logger.warning(f"Found {len(valid_pairs)} valid prompt-completion pairs out of {len(pairs)} total.")
        
        # Sample if needed
        if num_samples and num_samples < len(valid_pairs):
            valid_pairs = random.sample(valid_pairs, num_samples)
            
        return valid_pairs
    except Exception as e:
        logger.error(f"Error loading prompt-completion pairs: {e}")
        return _get_sample_prompt_completion_pairs()
    
def _get_sample_prompt_completion_pairs() -> List[Dict[str, str]]:
    """Return a set of sample prompt-completion pairs for testing."""
    return [
        {
            "prompt": "Explain the concept of deep learning to a 5-year-old child.",
            "completion": "Deep learning is like teaching a computer to learn from examples, just like how you learn. When you see lots of dogs, you start to recognize what a dog looks like. Computers using deep learning do the same thing!"
        },
        {
            "prompt": "What are the main differences between Python and JavaScript?",
            "completion": "Python and JavaScript differ in several ways: Python is commonly used for data analysis and backend development, while JavaScript was built for web browsers. Python syntax is known for readability with indentation, whereas JavaScript uses curly braces."
        },
        {
            "prompt": "Summarize the plot of Romeo and Juliet.",
            "completion": "Romeo and Juliet is about two teenagers from feuding families in Verona who fall in love. After meeting at a party, they secretly marry, but a series of tragic events leads to their deaths, which ultimately reconciles their families."
        }
    ]

def encode_prompt_completion_pairs(pairs: List[Dict[str, str]], 
                                  tokenizer, 
                                  max_length: int = 1024) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Encode a list of prompt-completion pairs for gradient analysis.
    
    Args:
        pairs: List of dictionaries with 'prompt' and 'completion' keys
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        List of tuples containing (input_ids, target_ids)
    """
    import torch
    
    encoded_pairs = []
    
    for pair in pairs:
        prompt = pair['prompt']
        completion = pair['completion']
        
        # Encode the full text (prompt + completion)
        full_text = prompt.strip() + " " + completion.strip()
        encoded_full = tokenizer.encode(full_text)
        
        # Encode just the prompt
        encoded_prompt = tokenizer.encode(prompt)
        prompt_length = len(encoded_prompt)
        
        # Ensure we're within max_length
        if len(encoded_full) > max_length:
            encoded_full = encoded_full[:max_length]
            
        # If prompt is already too long, we'll need to truncate
        if prompt_length >= max_length:
            prompt_length = max_length - 1  # Leave at least one token for completion
            
        # Input includes the prompt (we'll predict the completion)
        input_ids = torch.tensor(encoded_full[:prompt_length], dtype=torch.long)
        
        # Targets are the completion tokens
        target_ids = torch.tensor(encoded_full[prompt_length:], dtype=torch.long)
        
        # Check if we have a valid pair
        if len(input_ids) > 0 and len(target_ids) > 0:
            encoded_pairs.append((input_ids, target_ids))
    
    return encoded_pairs