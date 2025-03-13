import os
import sys
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project's tokenizer
from src.utils.data import get_tokenizer, create_prompts_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_random_prompts(num_prompts=100, min_tokens=10, max_tokens=50, output_file='data/prompts.txt', 
                      tokenizer_type='tiktoken', wikitext_file='data/wikitext-2/validation.txt'):
    """
    Generate random prompts by sampling from the WikiText-2 dataset.
    
    Args:
        num_prompts: Number of prompts to generate
        min_tokens: Minimum number of tokens per prompt
        max_tokens: Maximum number of tokens per prompt
        output_file: Path to save the prompts
        tokenizer_type: Type of tokenizer to use
        wikitext_file: Path to the WikiText-2 file to sample from
    """
    logger.info(f"Generating {num_prompts} random prompts from WikiText-2 using {tokenizer_type} tokenizer")
    
    # Check if WikiText-2 file exists
    if not os.path.exists(wikitext_file):
        logger.error(f"WikiText-2 file not found at {wikitext_file}")
        logger.info("Please run 'python scripts/download_wikitext.py' to download the dataset")
        return []
    
    # Get tokenizer
    tokenizer = get_tokenizer(tokenizer_type=tokenizer_type)
    
    # Load the WikiText-2 file
    try:
        with open(wikitext_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Loaded {len(text)} characters from {wikitext_file}")
        
        # Split into paragraphs (non-empty lines)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        logger.info(f"Found {len(paragraphs)} paragraphs")
        
        # Filter out section headers and very short paragraphs
        filtered_paragraphs = [p for p in paragraphs if not p.startswith('=') and len(p) > 50]
        logger.info(f"After filtering, {len(filtered_paragraphs)} paragraphs remain")
        
        if not filtered_paragraphs:
            logger.error("No suitable paragraphs found in the WikiText-2 file")
            return []
        
        # Generate prompts
        prompts = []
        
        # Try to generate the requested number of prompts
        attempts = 0
        max_attempts = num_prompts * 2  # Allow for some failures
        
        with tqdm(total=num_prompts) as pbar:
            while len(prompts) < num_prompts and attempts < max_attempts:
                attempts += 1
                
                # Select a random paragraph
                paragraph = random.choice(filtered_paragraphs)
                
                # Tokenize the paragraph
                tokens = tokenizer.encode(paragraph)
                
                # If the paragraph is too short, continue
                if len(tokens) < min_tokens:
                    continue
                
                # If the paragraph is longer than max_tokens, select a random slice
                if len(tokens) > max_tokens:
                    # Find a random starting point
                    start_idx = random.randint(0, len(tokens) - max_tokens)
                    # Take a slice of random length between min and max tokens
                    slice_length = random.randint(min_tokens, max_tokens)
                    tokens_slice = tokens[start_idx:start_idx + slice_length]
                    
                    # Decode back to text
                    prompt = tokenizer.decode(tokens_slice)
                else:
                    # Use the whole paragraph
                    prompt = paragraph
                
                # Add to prompts if not already included
                if prompt not in prompts:
                    prompts.append(prompt)
                    pbar.update(1)
        
        logger.info(f"Successfully generated {len(prompts)} prompts")
        
        # Save prompts
        create_prompts_file(prompts, output_file)
        logger.info(f"Saved prompts to {output_file}")
        
        return prompts
    
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        return []

def main():
    """Main function to parse arguments and generate prompts."""
    parser = argparse.ArgumentParser(description='Generate random prompts from WikiText-2 for testing')
    parser.add_argument('--num-prompts', type=int, default=100,
                       help='Number of prompts to generate')
    parser.add_argument('--min-tokens', type=int, default=10,
                       help='Minimum number of tokens per prompt')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum number of tokens per prompt')
    parser.add_argument('--output-file', type=str, default='data/prompts.txt',
                       help='Path to save the prompts')
    parser.add_argument('--tokenizer', type=str, choices=['tiktoken', 'bpe', 'char'], default='tiktoken',
                       help='Tokenizer to use (tiktoken, bpe, or char)')
    parser.add_argument('--wikitext-file', type=str, default='data/wikitext-2/validation.txt',
                       help='Path to the WikiText-2 file to sample from')
    
    args = parser.parse_args()
    
    # Generate prompts
    generate_random_prompts(
        num_prompts=args.num_prompts,
        min_tokens=args.min_tokens, 
        max_tokens=args.max_tokens,
        output_file=args.output_file,
        tokenizer_type=args.tokenizer,
        wikitext_file=args.wikitext_file
    )

if __name__ == "__main__":
    main()