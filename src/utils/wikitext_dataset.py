"""
Dataset classes for Wikitext-2 training and evaluation.
"""

import os
import logging
import torch
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)

class WikiTextDataset(Dataset):
    """Dataset for training with Wikitext-2."""
    
    def __init__(self, file_path, tokenizer, block_size, is_eval=False):
        """
        Initialize WikiText dataset.
        
        Args:
            file_path: Path to Wikitext file
            tokenizer: Tokenizer for tokenizing text
            block_size: Maximum sequence length
            is_eval: Whether this dataset is for evaluation (affects handling of incomplete blocks)
        """
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.is_eval = is_eval
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Loaded {len(text)} characters from {file_path}")
        
        # Tokenize text
        self.tokens = tokenizer.encode(text)
        logger.info(f"Encoded into {len(self.tokens)} tokens")
        
        # Create examples
        self._create_examples()
    
    def _create_examples(self):
        """Create training/evaluation examples from tokens."""
        # For training, we want consecutive blocks with overlap
        # For evaluation, we want non-overlapping blocks
        
        if self.is_eval:
            # For evaluation, create non-overlapping blocks
            self.examples = []
            
            # Process whole text into blocks to calculate accurate perplexity
            for i in range(0, len(self.tokens) - self.block_size, self.block_size):
                # Get block
                block = self.tokens[i:i + self.block_size]
                
                # Create input and target
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:], dtype=torch.long)
                
                self.examples.append((x, y))
                
            # Handle remaining tokens
            if len(self.tokens) % self.block_size != 0:
                remaining = self.tokens[-(self.block_size + 1):]
                if len(remaining) > 1:  # Ensure we have at least one pair of tokens
                    x = torch.tensor(remaining[:-1], dtype=torch.long)
                    y = torch.tensor(remaining[1:], dtype=torch.long)
                    self.examples.append((x, y))
        else:
            # For training, create overlapping blocks
            self.examples = []
            for i in range(0, len(self.tokens) - self.block_size):
                # Get block
                block = self.tokens[i:i + self.block_size + 1]  # +1 to account for target
                
                # Create input and target
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:], dtype=torch.long)
                
                self.examples.append((x, y))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class WikiTextPerplexityEvaluator:
    """Class for evaluating perplexity on Wikitext."""
    
    def __init__(self, model, dataset, device='cuda'):
        """
        Initialize perplexity evaluator.
        
        Args:
            model: GPT model to evaluate
            dataset: WikiTextDataset for evaluation
            device: Device to run evaluation on
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        
    def evaluate(self, batch_size=4):
        """
        Evaluate perplexity on the dataset.
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(self.dataset), batch_size):
                batch = self.dataset[i:min(i + batch_size, len(self.dataset))]
                
                # Prepare batch
                x_batch = torch.stack([item[0] for item in batch]).to(self.device)
                y_batch = torch.stack([item[1] for item in batch]).to(self.device)
                
                # Forward pass
                logits, loss = self.model(x_batch, y_batch)
                
                # Accumulate loss
                total_loss += loss.item() * y_batch.numel()
                total_tokens += y_batch.numel()
        
        # Calculate average negative log-likelihood
        avg_nll = total_loss / total_tokens
        
        # Perplexity is exp(avg_nll)
        perplexity = np.exp(avg_nll)
        
        return perplexity