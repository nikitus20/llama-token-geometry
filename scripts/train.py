"""
Training script for GPT models.
Supports custom architectures and warmup schedules.
"""

import os
import sys
import argparse
import logging
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src.utils.wikitext_dataset import WikiTextDataset, WikiTextPerplexityEvaluator

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.tokenizer.base import BaseTokenizer
from src.utils.data import get_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Simple dataset for training with a text corpus."""
    
    def __init__(self, text_path: str, tokenizer: BaseTokenizer, block_size: int):
        """
        Initialize dataset from a text file.
        
        Args:
            text_path: Path to text file
            tokenizer: Tokenizer for tokenizing text
            block_size: Maximum sequence length
        """
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Load text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
        logger.info(f"Loaded {len(self.tokens)} tokens from {text_path}")
        
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx):
        # Get chunk of tokens
        chunk = self.tokens[idx:idx + self.block_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def train(
    model: GPT,
    train_dataset,
    val_dataset=None,
    output_dir: str = 'saved_models',
    model_name: str = 'model',
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    save_interval: int = 200,
    eval_interval: int = 100,
    device: str = 'cuda',
    plot_results: bool = True
):
    """
    Train a GPT model with perplexity evaluation.
    
    Args:
        model: GPT model to train
        train_dataset: Dataset for training
        val_dataset: Optional dataset for validation and perplexity calculation
        output_dir: Directory to save model
        model_name: Name for saved model
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_steps: Maximum number of training steps
        warmup_steps: Number of warmup steps
        save_interval: Save model every N steps
        eval_interval: Evaluate perplexity every N steps
        device: Device to train on
        plot_results: Whether to plot training and validation metrics
    """
    model = model.to(device)
    model.train()
    
    # Create dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create scheduler
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, max_steps)
    
    # Create evaluator if validation set is provided
    evaluator = None
    if val_dataset is not None:
        evaluator = WikiTextPerplexityEvaluator(model, val_dataset, device)
    
    # Training loop
    step = 0
    total_loss = 0.0
    start_time = time.time()
    
    # Track metrics
    train_losses = []
    val_perplexities = []
    steps = []
    best_perplexity = float('inf')
    
    logger.info(f"Starting training for {max_steps} steps with {warmup_steps} warmup steps")
    
    while step < max_steps:
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"Training")):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                avg_loss = total_loss / 10
                elapsed = time.time() - start_time
                logger.info(f"Step {step+1}/{max_steps} | Loss: {avg_loss:.4f} | "
                           f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.2f}s")
                train_losses.append(avg_loss)
                steps.append(step + 1)
                total_loss = 0.0
                start_time = time.time()
            
            # Evaluate perplexity
            if evaluator is not None and (step + 1) % eval_interval == 0:
                model.eval()
                start_eval_time = time.time()
                perplexity = evaluator.evaluate(batch_size=batch_size)
                eval_time = time.time() - start_eval_time
                
                logger.info(f"Step {step+1}/{max_steps} | "
                           f"Validation Perplexity: {perplexity:.2f} | {eval_time:.2f}s")
                
                val_perplexities.append(perplexity)
                
                # Save best model based on perplexity
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_model_path = os.path.join(output_dir, f"{model_name}_best")
                    model.save_pretrained(best_model_path)
                    logger.info(f"New best perplexity: {perplexity:.2f} | "
                               f"Saved best model to {best_model_path}")
                
                model.train()
            
            # Save checkpoint
            if (step + 1) % save_interval == 0:
                save_path = os.path.join(output_dir, model_name)
                model.save_pretrained(save_path)
                logger.info(f"Saved checkpoint at step {step+1} to {save_path}")
                
                # Also save training plots
                if plot_results and len(steps) > 0:
                    _plot_training_metrics(
                        steps, train_losses, val_perplexities,
                        os.path.join(output_dir, f"{model_name}_training_plot.png")
                    )
            
            step += 1
            if step >= max_steps:
                break
    
    # Save final model
    save_path = os.path.join(output_dir, model_name)
    model.save_pretrained(save_path)
    logger.info(f"Training completed. Final model saved to {save_path}")
    
    # Final evaluation
    if evaluator is not None:
        model.eval()
        final_perplexity = evaluator.evaluate(batch_size=batch_size)
        logger.info(f"Final validation perplexity: {final_perplexity:.2f}")
        
        # Record final metrics in a text file
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Final validation perplexity: {final_perplexity:.2f}\n")
            f.write(f"Best validation perplexity: {best_perplexity:.2f}\n")
    
    # Save training plots
    if plot_results and len(steps) > 0:
        _plot_training_metrics(
            steps, train_losses, val_perplexities,
            os.path.join(output_dir, f"{model_name}_training_plot.png")
        )
    
    return model


def _plot_training_metrics(steps, train_losses, val_perplexities, output_path):
    """
    Plot training metrics and save to file.
    
    Args:
        steps: List of step numbers
        train_losses: List of training losses
        val_perplexities: List of validation perplexities
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, 'b-', label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Plot validation perplexity if available
    if val_perplexities:
        # Create step numbers for perplexity (evaluated less frequently)
        perplexity_steps = [steps[0] + i * (steps[-1] - steps[0]) / (len(val_perplexities) - 1) 
                           for i in range(len(val_perplexities))] if len(val_perplexities) > 1 else [steps[-1]]
        
        plt.subplot(1, 2, 2)
        plt.plot(perplexity_steps, val_perplexities, 'r-', label='Validation Perplexity')
        plt.xlabel('Steps')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Main function to parse arguments and train model."""
    parser = argparse.ArgumentParser(description='Train a GPT model')
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--dataset', type=str, choices=['wikitext', 'text'], default='wikitext',
                         help='Dataset type (wikitext or custom text file)')
    data_group.add_argument('--data-dir', type=str, default='data/wikitext-2',
                         help='Directory containing wikitext-2 dataset (train.txt, validation.txt, test.txt)')
    data_group.add_argument('--text-file', type=str,
                         help='Path to custom text file for training (used when dataset=text)')
    data_group.add_argument('--block-size', type=int, default=128,
                         help='Maximum sequence length')
    data_group.add_argument('--use-bpe', action='store_true',
                         help='Use BPE tokenizer instead of character tokenizer')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--pre-ln', action='store_true',
                          help='Use Pre-LN architecture (default: Post-LN)')
    model_group.add_argument('--use-rms-norm', action='store_true',
                          help='Use RMSNorm instead of LayerNorm')
    model_group.add_argument('--use-swiglu', action='store_true',
                          help='Use SwiGLU activation instead of GELU')
    model_group.add_argument('--n-layer', type=int, default=6,
                          help='Number of transformer layers')
    model_group.add_argument('--n-head', type=int, default=6,
                          help='Number of attention heads')
    model_group.add_argument('--n-embd', type=int, default=384,
                          help='Embedding dimension')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--output-dir', type=str, default='saved_models',
                          help='Directory to save model')
    train_group.add_argument('--model-name', type=str, default='postln_model_warmup',
                          help='Name for the saved model')
    train_group.add_argument('--batch-size', type=int, default=4,
                          help='Batch size for training')
    train_group.add_argument('--learning-rate', type=float, default=3e-4,
                          help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                          help='Weight decay')
    train_group.add_argument('--max-steps', type=int, default=1000,
                          help='Maximum number of training steps')
    train_group.add_argument('--warmup-steps', type=int, default=100,
                          help='Number of warmup steps')
    train_group.add_argument('--save-interval', type=int, default=200,
                          help='Save model every N steps')
    train_group.add_argument('--eval-interval', type=int, default=100,
                          help='Evaluate perplexity every N steps')
    train_group.add_argument('--no-validation', action='store_true',
                          help='Disable validation (no perplexity calculation)')
    train_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                          help='Device to train on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get tokenizer
    tokenizer = get_tokenizer(use_bpe=args.use_bpe)
    
    # Create datasets
    if args.dataset == 'wikitext':
        if not os.path.exists(args.data_dir) or not os.path.exists(os.path.join(args.data_dir, 'train.txt')):
            logger.error(f"Wikitext dataset not found at {args.data_dir}")
            logger.info("Please run 'python scripts/download_wikitext.py' to download the dataset")
            return
        
        # Create training dataset
        train_path = os.path.join(args.data_dir, 'train.txt')
        train_dataset = WikiTextDataset(train_path, tokenizer, args.block_size, is_eval=False)
        
        # Create validation dataset if requested
        val_dataset = None
        if not args.no_validation:
            val_path = os.path.join(args.data_dir, 'validation.txt')
            if os.path.exists(val_path):
                val_dataset = WikiTextDataset(val_path, tokenizer, args.block_size, is_eval=True)
            else:
                logger.warning(f"Validation file not found at {val_path}, skipping validation")
    else:
        # Use custom text file
        if not args.text_file or not os.path.exists(args.text_file):
            logger.error("Text file not provided or not found")
            return
        
        # For custom text files, we'll use the TextDataset class
        from torch.utils.data import random_split
        
        # Create dataset
        dataset = TextDataset(args.text_file, tokenizer, args.block_size)
        
        # Split into train and validation
        if args.no_validation:
            train_dataset = dataset
            val_dataset = None
        else:
            # Use 90% for training, 10% for validation
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 256,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
        bias=False,
        pre_ln=args.pre_ln,
        use_rms_norm=args.use_rms_norm,
        use_swiglu=args.use_swiglu
    )
    
    model = GPT(config)
    
    logger.info(f"Created model with {model.get_num_params()/1e6:.2f}M parameters")
    logger.info(f"Architecture: {'PreLN' if args.pre_ln else 'PostLN'}, "
               f"{'RMSNorm' if args.use_rms_norm else 'LayerNorm'}, "
               f"{'SwiGLU' if args.use_swiglu else 'GELU'}")
    
    # Train model
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        device=args.device
    )


if __name__ == "__main__":
    main()