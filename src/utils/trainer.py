"""
Simple trainer class for model warmup and training.
"""

import os
import torch
import torch.nn.functional as F
import time
import logging
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TrainerConfig:
    """Configuration for the Trainer."""
    
    def __init__(
        self,
        max_epochs=1,
        batch_size=8,
        learning_rate=3e-4,
        lr_decay=True,
        num_warmup_steps=100,
        max_steps=None,
        num_workers=0,
        save_interval=1000,
        checkpoint_dir=None,
        device=None,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.num_warmup_steps = num_warmup_steps
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                              'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                              'cpu')

class Trainer:
    """Simple trainer for model warmup and training."""
    
    def __init__(self, model, train_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.device = config.device
        self.step = 0
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Set up learning rate scheduler if needed
        if config.lr_decay:
            num_training_steps = len(train_dataset) // config.batch_size * config.max_epochs
            if config.max_steps is not None:
                num_training_steps = min(num_training_steps, config.max_steps)
            
            self.lr_scheduler = self.get_lr_scheduler(
                self.optimizer, 
                num_warmup_steps=config.num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.lr_scheduler = None
    
    def get_lr_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
        """Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return LambdaLR(optimizer, lr_lambda)
    
    def save_checkpoint(self):
        if self.config.checkpoint_dir is None:
            return
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Save model checkpoint
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config.__dict__
        }
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_{self.step}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """Run training."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        # Set model to training mode
        self.model.train()
        
        # Total steps counter
        total_steps = 0
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, batch in pbar:
                # Move batch to device and handle different batch formats
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                    # For simple TensorDataset with single tensor
                    x, y = batch[:, :-1], batch[:, 1:]
                elif isinstance(batch, tuple) and len(batch) == 2:
                    # For datasets that return (input, target) pairs
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                elif isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], torch.Tensor):
                    # For DataLoader unpacking a TensorDataset with a single tensor
                    batch_tensor = batch[0].to(self.device)
                    x, y = batch_tensor[:, :-1], batch_tensor[:, 1:]
                elif isinstance(batch, list) and len(batch) == 2 and all(isinstance(t, torch.Tensor) for t in batch):
                    # For list containing [input_tensor, target_tensor]
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    # For custom datasets with different formats
                    raise ValueError(f"Unsupported batch format: {type(batch)}. Contents: {batch}")
                
                # Forward pass
                outputs = self.model(x)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    # If model returns a tuple (logits, ...), take the first element as logits
                    logits = outputs[0]
                else:
                    # If model returns just logits
                    logits = outputs
                
                # Reshape if needed (for standard transformer outputs)
                if len(logits.shape) == 3:
                    # For sequence models like transformers:
                    # Logits shape is typically [batch_size, sequence_length, vocab_size]
                    # We need to reshape to [batch_size*sequence_length, vocab_size]
                    batch_size, seq_len, vocab_size = logits.shape
                    logits = logits.reshape(-1, vocab_size)
                    
                    # Similarly, reshape targets to [batch_size*sequence_length]
                    y = y.reshape(-1)
                
                # Calculate loss (making sure dimensions match)
                # logits should be [N, vocab_size] and y should be [N] where N is batch_size*seq_len
                if logits.size(0) != y.size(0):
                    raise ValueError(f"Mismatched dimensions: logits {logits.shape}, targets {y.shape}")
                
                loss = F.cross_entropy(logits, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update LR scheduler if exists
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                
                # Update step counter
                self.step += 1
                total_steps += 1
                
                # Save checkpoint if needed
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                # Check if we've reached the maximum steps
                if self.config.max_steps is not None and total_steps >= self.config.max_steps:
                    logger.info(f"Reached maximum steps ({self.config.max_steps}), stopping training")
                    # Save final checkpoint
                    self.save_checkpoint()
                    return
        
        # Save final checkpoint
        self.save_checkpoint() 