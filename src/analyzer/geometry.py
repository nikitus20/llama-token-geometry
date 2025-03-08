"""
Analyzer for token geometry in transformer models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from src.model.transformer import GPT

logger = logging.getLogger(__name__)

class GeometryAnalyzer:
    """
    Analyzes token representations in a GPT model to understand
    how token geometry changes across layers.
    """
    
    def __init__(self, model: GPT, device: str = 'cuda'):
        """
        Initialize geometry analyzer for a GPT model.
        
        Args:
            model: GPT model
            device: device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Register hooks to capture intermediate token representations
        self.hooks = []
        self.layer_outputs = {}
        
        # Register hooks for each transformer block
        # Using a function to properly capture layer_idx in closure
        def get_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                self.layer_outputs[layer_idx] = output.detach()
            return hook_fn
            
        for i, block in enumerate(model.transformer.h):
            self.hooks.append(block.register_forward_hook(get_hook_fn(i)))
            
        # Also register hook for the final layer norm
        self.hooks.append(
            model.transformer.ln_f.register_forward_hook(
                get_hook_fn(len(model.transformer.h))
            )
        )
        
    def compute_token_cosine_similarities(self, input_ids: torch.Tensor) -> Dict[int, float]:
        """
        Compute average cosine similarity between tokens at each layer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Dictionary of average cosine similarities per layer
        """
        batch_size, seq_len = input_ids.size()
        self.layer_outputs = {}
        
        # Forward pass to collect layer outputs
        with torch.no_grad():
            self.model(input_ids)
            
        # Compute average cosine similarity at each layer
        avg_cosine_sims = {}
        
        for layer_idx, layer_output in self.layer_outputs.items():
            # Layer output shape: [batch_size, seq_len, hidden_dim]
            
            # Vectorized computation of cosine similarities for all batches
            batch_sims = []
            
            for b in range(batch_size):
                # Normalize token vectors for cosine similarity
                tokens = layer_output[b]  # [seq_len, hidden_dim]
                
                # More numerically stable normalization
                tokens_norm = F.normalize(tokens, p=2, dim=1)
                
                # Compute all pairwise cosine similarities
                sim_matrix = torch.mm(tokens_norm, tokens_norm.t())  # [seq_len, seq_len]
                
                # Create mask to exclude self-similarity on diagonal
                mask = torch.ones_like(sim_matrix) - torch.eye(seq_len, device=self.device)
                masked_sim = sim_matrix * mask
                
                # Compute average similarity (excluding self-similarity)
                # Using masked_select for better efficiency
                avg_sim = masked_sim.masked_select(mask.bool()).mean().item()
                batch_sims.append(avg_sim)
            
            avg_cosine_sims[layer_idx] = np.mean(batch_sims)
            
        return avg_cosine_sims
    
    def analyze_prompts(self, prompts: List[str], tokenizer) -> Dict[int, float]:
        """
        Analyze a set of prompts and compute average token geometry metrics.
        
        Args:
            prompts: List of text prompts
            tokenizer: Tokenizer to convert prompts to token IDs
            
        Returns:
            Dictionary of average cosine similarities per layer
        """
        all_cosine_sims = {}
        
        # Process prompts in batches
        batch_size = 4  # Small batch size to avoid memory issues
        for i in tqdm(range(0, len(prompts), batch_size), desc="Analyzing prompts"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize prompts
            try:
                # Tokenize and pad batch
                batch_tokens = [tokenizer.encode(p) for p in batch_prompts]
                max_len = min(self.model.config.block_size, max(len(t) for t in batch_tokens))
                
                # Pad tokens to max length
                padded_tokens = [
                    t[:max_len] + [0] * (max_len - len(t)) if len(t) > 0 else [0] * max_len
                    for t in batch_tokens
                ]
                
                batch_input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=self.device)
            except Exception as e:
                logger.error(f"Error tokenizing batch: {e}")
                continue
            
            # Compute cosine similarities
            try:
                cosine_sims = self.compute_token_cosine_similarities(batch_input_ids)
                
                # Accumulate results
                for layer_idx, avg_sim in cosine_sims.items():
                    if layer_idx not in all_cosine_sims:
                        all_cosine_sims[layer_idx] = []
                    all_cosine_sims[layer_idx].append(avg_sim)
            except Exception as e:
                logger.error(f"Error computing cosine similarities: {e}")
                continue
        
        # Compute average across all prompts
        avg_all_cosine_sims = {
            layer_idx: np.mean(sims) for layer_idx, sims in all_cosine_sims.items()
            if len(sims) > 0  # Only include layers with data
        }
        
        return avg_all_cosine_sims
    
    def analyze_single_prompt(self, prompt: str, tokenizer) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """
        Analyze token geometry for a single prompt.
        
        Args:
            prompt: Text prompt to analyze
            tokenizer: Tokenizer
            
        Returns:
            Tuple of (similarity matrices per layer, tokens)
        """
        # Tokenize prompt
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {e}")
            return {}, []
        
        # Forward pass
        self.layer_outputs = {}
        try:
            with torch.no_grad():
                self.model(input_ids)
        except Exception as e:
            logger.error(f"Error during forward pass: {e}", exc_info=True)
            return {}, tokens
        
        # Calculate similarity matrices for each layer
        sim_matrices = {}
        for layer_idx, output in self.layer_outputs.items():
            try:
                # Get embeddings for the batch (just one example here)
                embeddings = output[0]  # [seq_len, hidden_dim]
                
                # Normalize embeddings
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                
                # Compute similarity matrix
                sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
                
                # Store similarity matrix
                sim_matrices[layer_idx] = sim_matrix.cpu().numpy()
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx}: {e}")
                continue
        
        return sim_matrices, tokens
    
    def cleanup(self) -> None:
        """Remove all hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []