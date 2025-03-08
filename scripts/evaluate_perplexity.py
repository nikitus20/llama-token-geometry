"""
Script to evaluate perplexity on a trained model.
"""

import os
import sys
import argparse
import logging
import torch

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import GPT
from src.utils.data import get_tokenizer
from src.utils.wikitext_dataset import WikiTextDataset, WikiTextPerplexityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to evaluate perplexity on a test set."""
    parser = argparse.ArgumentParser(description='Evaluate perplexity on a test set')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model directory')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['wikitext', 'text'], default='wikitext',
                       help='Dataset type (wikitext or custom text file)')
    parser.add_argument('--data-dir', type=str, default='data/wikitext-2',
                       help='Directory containing wikitext-2 dataset')
    parser.add_argument('--split', type=str, choices=['test', 'validation', 'train'], default='test',
                       help='Dataset split to evaluate on')
    parser.add_argument('--text-file', type=str,
                       help='Path to custom text file for evaluation (used when dataset=text)')
    parser.add_argument('--use-bpe', action='store_true',
                       help='Use BPE tokenizer instead of character tokenizer')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model directory not found: {args.model}")
        return
    
    # Load model
    try:
        logger.info(f"Loading model from {args.model}...")
        model = GPT.from_pretrained(args.model, device=args.device)
        logger.info(f"Model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Get tokenizer
    tokenizer = get_tokenizer(use_bpe=args.use_bpe)
    
    # Create evaluation dataset
    if args.dataset == 'wikitext':
        # Get the appropriate file path
        data_file = os.path.join(args.data_dir, f"{args.split}.txt")
        
        if not os.path.exists(data_file):
            logger.error(f"Dataset file not found: {data_file}")
            logger.info("Please run 'python scripts/download_wikitext.py' to download the dataset")
            return
        
        # Create dataset
        logger.info(f"Creating evaluation dataset from {data_file}...")
        eval_dataset = WikiTextDataset(data_file, tokenizer, model.config.block_size, is_eval=True)
    else:
        # Use custom text file
        if not args.text_file or not os.path.exists(args.text_file):
            logger.error("Text file not provided or not found")
            return
        
        # Create dataset
        from src.utils.data import TextDataset
        logger.info(f"Creating evaluation dataset from {args.text_file}...")
        eval_dataset = TextDataset(args.text_file, tokenizer, model.config.block_size)
    
    # Create evaluator
    evaluator = WikiTextPerplexityEvaluator(model, eval_dataset, args.device)
    
    # Evaluate perplexity
    logger.info("Evaluating perplexity...")
    perplexity = evaluator.evaluate(args.batch_size)
    
    logger.info(f"Perplexity on {args.split} set: {perplexity:.2f}")
    
    # Save results to file
    results_file = os.path.join(args.model, f"{args.split}_perplexity.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Perplexity: {perplexity:.4f}\n")
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()