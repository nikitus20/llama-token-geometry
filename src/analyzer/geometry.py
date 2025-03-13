"""
Enhanced analyzer for token geometry in transformer models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm
from src.model.transformer import GPT

logger = logging.getLogger(__name__)

class GeometryAnalyzer:
    """
    Analyzes token representations in a GPT model to understand
    how token geometry changes across layers.
    
    Features:
    - Cosine similarity between tokens
    - Token representation norms
    - Update norms between consecutive layers
    - Variance of metrics across different prompts
    """
    
    def __init__(self, model: GPT, device: str = None):
        """
        Initialize geometry analyzer for a GPT model.
        
        Args:
            model: GPT model
            device: device to run on (cuda, mps, or cpu)
        """
        from src.model.utils import get_device
        if device is None:
            device = get_device()
        self.device = device
        self.model = model
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
        
    def compute_token_metrics(self, input_ids: torch.Tensor) -> Dict[str, Dict[int, float]]:
        """
        Compute various token metrics at each layer:
        - Average cosine similarity between tokens
        - Average token norm
        - Average update norm (difference between consecutive layers)
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Dictionary of metrics, each containing a dictionary of values per layer
        """
        batch_size, seq_len = input_ids.size()
        self.layer_outputs = {}
        
        # Forward pass to collect layer outputs
        with torch.no_grad():
            self.model(input_ids)
            
        # Initialize metric dictionaries
        avg_cosine_sims = {}
        avg_token_norms = {}
        avg_update_norms = {}
        
        # Sort layer indices to ensure proper calculation of updates
        layer_indices = sorted(self.layer_outputs.keys())
        
        for i, layer_idx in enumerate(layer_indices):
            layer_output = self.layer_outputs[layer_idx]
            # Layer output shape: [batch_size, seq_len, hidden_dim]
            
            # Initialize lists to collect batch metrics
            batch_sims = []
            batch_norms = []
            batch_updates = []
            
            for b in range(batch_size):
                # Get token vectors for this batch
                tokens = layer_output[b]  # [seq_len, hidden_dim]
                
                # 1. Compute token norms
                token_norms = torch.norm(tokens, p=2, dim=1)  # [seq_len]
                avg_norm = token_norms.mean().item()
                batch_norms.append(avg_norm)
                
                # 2. Compute cosine similarities
                # Normalize token vectors for cosine similarity
                tokens_norm = F.normalize(tokens, p=2, dim=1)
                
                # Compute all pairwise cosine similarities
                sim_matrix = torch.mm(tokens_norm, tokens_norm.t())  # [seq_len, seq_len]
                
                # Create mask to exclude self-similarity on diagonal
                mask = torch.ones_like(sim_matrix) - torch.eye(seq_len, device=self.device)
                masked_sim = sim_matrix * mask
                
                # Compute average similarity (excluding self-similarity)
                avg_sim = masked_sim.masked_select(mask.bool()).mean().item()
                batch_sims.append(avg_sim)
                
                # 3. Compute update norms (if not the first layer)
                if i > 0:
                    prev_layer_idx = layer_indices[i-1]
                    prev_tokens = self.layer_outputs[prev_layer_idx][b]
                    
                    # Compute update vectors (difference between consecutive layers)
                    updates = tokens - prev_tokens  # [seq_len, hidden_dim]
                    
                    # Compute norm of updates
                    update_norms = torch.norm(updates, p=2, dim=1)  # [seq_len]
                    avg_update = update_norms.mean().item()
                    batch_updates.append(avg_update)
            
            # Average across batches
            avg_cosine_sims[layer_idx] = np.mean(batch_sims)
            avg_token_norms[layer_idx] = np.mean(batch_norms)
            
            # Only store update norms for layers after the first one
            if i > 0:
                avg_update_norms[layer_idx] = np.mean(batch_updates)
        
        # Package all metrics in a dictionary
        metrics = {
            'cosine_sim': avg_cosine_sims,
            'token_norm': avg_token_norms,
            'update_norm': avg_update_norms
        }
        
        return metrics
    
    def analyze_prompts(self, prompts: List[str], tokenizer) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Analyze a set of prompts and compute average token geometry metrics with variances.
        
        Args:
            prompts: List of text prompts
            tokenizer: Tokenizer to convert prompts to token IDs
            
        Returns:
            Dictionary with structure:
            {
                'mean': {
                    'cosine_sim': {layer_idx: mean_value, ...},
                    'token_norm': {layer_idx: mean_value, ...},
                    'update_norm': {layer_idx: mean_value, ...}
                },
                'variance': {
                    'cosine_sim': {layer_idx: variance_value, ...},
                    'token_norm': {layer_idx: variance_value, ...},
                    'update_norm': {layer_idx: variance_value, ...}
                }
            }
        """
        # Initialize dictionaries to collect metrics across all prompts
        all_metrics = {
            'cosine_sim': {},
            'token_norm': {},
            'update_norm': {}
        }
        
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
            
            # Compute metrics
            try:
                batch_metrics = self.compute_token_metrics(batch_input_ids)
                
                # Accumulate results for each metric type
                for metric_name, layer_values in batch_metrics.items():
                    for layer_idx, value in layer_values.items():
                        if layer_idx not in all_metrics[metric_name]:
                            all_metrics[metric_name][layer_idx] = []
                        all_metrics[metric_name][layer_idx].append(value)
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
                continue
        
        # Compute mean and variance across all prompts
        result = {
            'mean': {},
            'variance': {}
        }
        
        for metric_name, layer_values in all_metrics.items():
            result['mean'][metric_name] = {}
            result['variance'][metric_name] = {}
            
            for layer_idx, values in layer_values.items():
                if len(values) > 0:
                    result['mean'][metric_name][layer_idx] = np.mean(values)
                    result['variance'][metric_name][layer_idx] = np.var(values)
        
        return result
    
    def analyze_single_prompt(self, prompt: str, tokenizer) -> Dict[str, Any]:
        """
        Analyze token geometry for a single prompt.
        
        Args:
            prompt: Text prompt to analyze
            tokenizer: Tokenizer
            
        Returns:
            Dictionary containing metrics and token information
        """
        # Tokenize prompt
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {e}")
            return {'tokens': [], 'metrics': {}, 'similarity_matrices': {}}
        
        # Forward pass
        self.layer_outputs = {}
        try:
            with torch.no_grad():
                self.model(input_ids)
        except Exception as e:
            logger.error(f"Error during forward pass: {e}", exc_info=True)
            return {'tokens': tokens, 'metrics': {}, 'similarity_matrices': {}}
        
        # Calculate metrics and similarity matrices for each layer
        metrics = self.compute_token_metrics(input_ids)
        sim_matrices = {}
        
        # Sort layer indices for consistent processing
        layer_indices = sorted(self.layer_outputs.keys())
        
        for layer_idx in layer_indices:
            try:
                # Get embeddings for the batch (just one example here)
                embeddings = self.layer_outputs[layer_idx][0]  # [seq_len, hidden_dim]
                
                # Normalize embeddings
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                
                # Compute similarity matrix
                sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
                
                # Store similarity matrix
                sim_matrices[layer_idx] = sim_matrix.cpu().numpy()
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx}: {e}")
                continue
        
        return {
            'tokens': tokens,
            'metrics': metrics, 
            'similarity_matrices': sim_matrices
        }
    
    def cleanup(self) -> None:
        """Remove all hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def visualize_metrics(self, metrics_dict, title="Token Geometry Analysis", figsize=(14, 10)):
        """
        Visualize metrics across layers.
        
        Args:
            metrics_dict: Dictionary with structure from analyze_prompts
            title: Plot title
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
            fig.suptitle(title, fontsize=16)
            
            # Get the mean metrics and variances
            mean_metrics = metrics_dict['mean']
            var_metrics = metrics_dict['variance']
            
            metric_names = ['cosine_sim', 'token_norm', 'update_norm']
            metric_titles = ['Average Cosine Similarity', 'Average Token Norm', 'Average Update Norm']
            
            for i, (metric_name, metric_title) in enumerate(zip(metric_names, metric_titles)):
                ax = axes[i]
                
                # Get the mean values by layer
                layers = sorted(mean_metrics[metric_name].keys())
                mean_values = [mean_metrics[metric_name][layer] for layer in layers]
                
                # Get corresponding variances
                var_values = [var_metrics[metric_name].get(layer, 0) for layer in layers]
                std_values = np.sqrt(var_values)
                
                # Plot mean with error bars (standard deviation)
                ax.errorbar(layers, mean_values, yerr=std_values, marker='o', linestyle='-', 
                           capsize=4, label=f'{metric_title} across layers')
                
                ax.set_ylabel(metric_title)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add a horizontal line at 0 for context if values might be negative
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Set y limits with some padding
                min_val = np.min(np.array(mean_values) - np.array(std_values))
                max_val = np.max(np.array(mean_values) + np.array(std_values))
                padding = (max_val - min_val) * 0.1
                ax.set_ylim(min_val - padding, max_val + padding)
            
            # Configure the x-axis (shared across subplots)
            axes[-1].set_xlabel('Layer Index')
            axes[-1].set_xticks(layers)
            
            # Adjust layout and save/show
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization.")
            return None