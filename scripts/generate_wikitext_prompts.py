#!/usr/bin/env python
"""
Script to generate a large number of prompt-completion pairs from WikiText-2 data
for more comprehensive analysis of model token geometry.
"""

import os
import sys
import argparse
import logging
import json
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.utils.data import get_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_wikitext_data(file_path):
    """
    Load text data from WikiText-2 file.
    
    Args:
        file_path: Path to the WikiText-2 text file
        
    Returns:
        List of paragraphs from the WikiText-2 file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into paragraphs (non-empty lines)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Filter out section headers and very short paragraphs
        filtered_paragraphs = [p for p in paragraphs if not p.startswith('=') and len(p) > 50]
        
        logger.info(f"Loaded {len(filtered_paragraphs)} usable paragraphs from {file_path}")
        return filtered_paragraphs
    
    except Exception as e:
        logger.error(f"Error loading WikiText-2 data: {e}")
        return []

def generate_prompt_completion_pairs(paragraphs, tokenizer, num_pairs=1000, prompt_length=100, completion_length=20):
    """
    Generate prompt-completion pairs from WikiText-2 paragraphs.
    
    Args:
        paragraphs: List of paragraphs from WikiText-2
        tokenizer: Tokenizer for tokenizing text
        num_pairs: Number of pairs to generate
        prompt_length: Target length of prompts in tokens
        completion_length: Target length of completions in tokens
        
    Returns:
        List of dictionaries with 'prompt' and 'completion' keys
    """
    pairs = []
    
    # Encode all paragraphs
    encoded_paragraphs = []
    for paragraph in tqdm(paragraphs, desc="Encoding paragraphs"):
        try:
            tokens = tokenizer.encode(paragraph)
            if len(tokens) > prompt_length + completion_length + 10:  # Add some buffer
                encoded_paragraphs.append(tokens)
        except Exception as e:
            logger.warning(f"Error encoding paragraph: {e}")
    
    if not encoded_paragraphs:
        logger.error("No usable encoded paragraphs found")
        return []
    
    logger.info(f"Found {len(encoded_paragraphs)} paragraphs of sufficient length")
    
    # Generate pairs
    attempts = 0
    max_attempts = num_pairs * 5  # Set a limit to avoid infinite loops
    
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        
        # Pick a random paragraph
        paragraph_tokens = random.choice(encoded_paragraphs)
        
        if len(paragraph_tokens) < prompt_length + completion_length:
            continue
        
        # Choose a random starting point that allows for both prompt and completion
        max_start = len(paragraph_tokens) - (prompt_length + completion_length)
        if max_start <= 0:
            continue
            
        start_idx = random.randint(0, max_start)
        
        # Extract prompt and completion tokens
        prompt_tokens = paragraph_tokens[start_idx:start_idx + prompt_length]
        completion_tokens = paragraph_tokens[start_idx + prompt_length:start_idx + prompt_length + completion_length]
        
        # Decode back to text
        try:
            prompt_text = tokenizer.decode(prompt_tokens)
            completion_text = tokenizer.decode(completion_tokens)
            
            # Skip if either is empty or too short
            if len(prompt_text.strip()) < 10 or len(completion_text.strip()) < 5:
                continue
                
            pairs.append({
                "prompt": prompt_text,
                "completion": completion_text
            })
            
            if len(pairs) % 100 == 0:
                logger.info(f"Generated {len(pairs)} prompt-completion pairs")
                
        except Exception as e:
            logger.warning(f"Error decoding tokens: {e}")
    
    logger.info(f"Generated {len(pairs)} prompt-completion pairs in {attempts} attempts")
    return pairs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate prompt-completion pairs from WikiText-2 data')
    parser.add_argument('--output-file', default='outputs/wikitext_prompts.json', help='Output JSON file')
    parser.add_argument('--data-file', default='data/wikitext-2/train.txt', help='WikiText-2 data file')
    parser.add_argument('--num-pairs', type=int, default=1000, help='Number of prompt-completion pairs to generate')
    parser.add_argument('--prompt-length', type=int, default=100, help='Target length of prompts in tokens')
    parser.add_argument('--completion-length', type=int, default=20, help='Target length of completions in tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--tokenizer', default='huggyllama/llama-7b', help='Tokenizer to use')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load WikiText-2 data
    paragraphs = load_wikitext_data(args.data_file)
    if not paragraphs:
        logger.error("No paragraphs found in WikiText-2 data")
        return
    
    # Generate prompt-completion pairs
    pairs = generate_prompt_completion_pairs(
        paragraphs=paragraphs,
        tokenizer=tokenizer,
        num_pairs=args.num_pairs,
        prompt_length=args.prompt_length,
        completion_length=args.completion_length
    )
    
    if not pairs:
        logger.error("Failed to generate any prompt-completion pairs")
        return
    
    # Save to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(pairs)} prompt-completion pairs to {args.output_file}")

if __name__ == "__main__":
    main() 