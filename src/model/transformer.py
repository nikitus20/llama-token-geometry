"""
Full transformer model implementation.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, Dict, Any

from src.model.config import GPTConfig
from src.model.layers import RMSNorm, Block

logger = logging.getLogger(__name__)

class GPT(nn.Module):
    """
    GPT-style transformer model with configurable architecture.
    Supports LLaMA-style configurations with RMSNorm and SwiGLU.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))
        
        # Always use RMSNorm for final normalization layer
        self.transformer.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        
        # Add an initial normalization layer after embeddings for all architectures
        if config.use_initial_ln:
            self.transformer.ln_emb = RMSNorm(config.n_embd, eps=config.norm_eps)
            logger.info(f"Added initial normalization layer after embeddings for {config.ln} architecture")
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights
        self.transformer.wte.weight = self.lm_head.weight 

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        logger.info(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    
    def _init_weights(self, module):
        config = self.config
        if isinstance(module, nn.Linear):
            std = config.initializer_range
            if hasattr(module, 'is_deeppost_layer') and module.is_deeppost_layer:
                # Special initialization for deeppost layers
                torch.nn.init.xavier_normal_(module.weight, gain=(config.n_layer * 8) ** 0.25)
            elif hasattr(module, 'is_scaled_layer') and module.is_scaled_layer:
                # Scaled initialization for certain layers
                scaled_std = std / (2 * config.n_layer) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
            else:
                # Standard initialization
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            std = config.initializer_range
            # Use the same standard deviation as other layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Optional target tokens for computing loss
            
        Returns:
            tuple of (logits, loss) where loss is None if targets is None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Apply initial normalization layer after embeddings if configured
        if self.config.use_initial_ln and hasattr(self.transformer, 'ln_emb'):
            x = self.transformer.ln_emb(x)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def save_pretrained(self, path: str) -> None:
        """
        Save model weights and config to disk.
        
        Args:
            path: Directory path where to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(path, "model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Save config as dict
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        config_path = os.path.join(path, "config.pt")
        torch.save(config_dict, config_path)
        
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'GPT':
        """
        Load a pretrained model from disk.
        
        Args:
            path: Directory path containing the saved model
            device: Device to load the model on
            
        Returns:
            Loaded GPT model
        """
        # Load config
        config_path = os.path.join(path, "config.pt")
        config_dict = torch.load(config_path, map_location=device)
        config = GPTConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(path, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model