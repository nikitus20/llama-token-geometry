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
    - Gradient norms across layers (how gradients propagate backwards)
    - Variance of metrics across different prompts
    """
    
    def __init__(self, model: GPT, device: str = None, track_gradients: bool = True):
        """
        Initialize geometry analyzer for a GPT model.
        
        Args:
            model: GPT model
            device: device to run on (cuda, mps, or cpu)
            track_gradients: whether to track gradient norms during analysis (default: True)
        """
        from src.model.utils import get_device
        if device is None:
            device = get_device()
        self.device = device
        self.model = model
        self.model.to(device)
        self.track_gradients = track_gradients
        
        # Store original training state to restore later
        self.was_training = self.model.training
        
        # Register hooks to capture intermediate token representations
        self.hooks = []
        self.layer_outputs = {}
        self.gradient_norms = {}
        
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
        
        # Register gradient hooks if requested
        if self.track_gradients:
            self.grad_hooks = []
            
            def get_grad_hook_fn(layer_idx):
                def grad_hook_fn(module, grad_input, grad_output):
                    # Store the gradient norm of the output
                    if isinstance(grad_output, tuple):
                        grad_output = grad_output[0]
                    if grad_output is not None:
                        self.gradient_norms[layer_idx] = grad_output.detach().norm().item()
                return grad_hook_fn
            
            for i, block in enumerate(model.transformer.h):
                self.grad_hooks.append(block.register_full_backward_hook(get_grad_hook_fn(i)))
    
    def enhance_gradient_tracking(self):
        """
        Enhanced gradient tracking that captures more detailed gradient statistics.
        Call this after initializing the GeometryAnalyzer but before analysis.
        """
        if not hasattr(self, 'gradient_cosine_sims'):
            self.track_gradients = True
            self.gradient_norms = {}
            self.gradient_cosine_sims = {}  # Track cosine similarity between gradients
            self.gradient_update_corr = {}  # Track correlation between gradients and updates
            
            # Register gradient hooks
            # If hooks already exist, remove them first
            if hasattr(self, 'grad_hooks') and self.grad_hooks:
                for hook in self.grad_hooks:
                    hook.remove()
            
            self.grad_hooks = []
            
            def get_grad_hook_fn(layer_idx):
                def grad_hook_fn(module, grad_input, grad_output):
                    if isinstance(grad_output, tuple):
                        grad_output = grad_output[0]
                    if grad_output is not None:
                        # Store gradient norm
                        self.gradient_norms[layer_idx] = grad_output.detach().norm().item()
                        
                        # Store the raw gradients for further analysis
                        if not hasattr(self, 'raw_gradients'):
                            self.raw_gradients = {}
                        self.raw_gradients[layer_idx] = grad_output.detach().clone()
                        
                        # If we have the previous layer's output stored, compute correlation
                        # between gradients and the updates (changes from previous layer)
                        if hasattr(self, 'layer_outputs'):
                            layer_indices = sorted(self.layer_outputs.keys())
                            # Find the current layer's index in our tracked outputs
                            if layer_idx in layer_indices:
                                current_idx = layer_indices.index(layer_idx)
                                # Make sure we have a previous layer to compare with
                                if current_idx > 0:
                                    prev_layer_idx = layer_indices[current_idx-1]
                                    
                                    # Safety check to ensure both layers exist in outputs
                                    if layer_idx in self.layer_outputs and prev_layer_idx in self.layer_outputs:
                                        curr_output = self.layer_outputs[layer_idx]
                                        prev_output = self.layer_outputs[prev_layer_idx]
                                        
                                        # Compute update vector (difference between consecutive layers)
                                        update = curr_output - prev_output
                                        
                                        # Reshape to match for correlation calculation
                                        grad_flat = grad_output.view(-1)
                                        update_flat = update.view(-1)
                                        
                                        # Compute cosine similarity between gradient and update
                                        norm_grad = torch.norm(grad_flat)
                                        norm_update = torch.norm(update_flat)
                                        
                                        if norm_grad > 0 and norm_update > 0:
                                            cosine_sim = torch.dot(grad_flat, update_flat) / (norm_grad * norm_update)
                                            self.gradient_update_corr[layer_idx] = cosine_sim.item()
                return grad_hook_fn
            
            # Register hooks for each transformer block
            for i, block in enumerate(self.model.transformer.h):
                self.grad_hooks.append(block.register_full_backward_hook(get_grad_hook_fn(i)))
            
            # Also register hook for the final layer norm
            self.grad_hooks.append(
                self.model.transformer.ln_f.register_full_backward_hook(
                    get_grad_hook_fn(len(self.model.transformer.h))
                )
            )
            
            # Optional: Register hook for embedding layer to track gradients all the way to embeddings
            # Note: embeddings need special handling because they're not part of the forward pass hooks
            self.grad_hooks.append(
                self.model.transformer.wte.register_full_backward_hook(
                    get_grad_hook_fn(-1)  # Use same special index for embedding layer
                )
            )
        
    def compute_token_metrics(self, input_ids: torch.Tensor) -> Dict[str, Dict[int, float]]:
        """
        Compute various token metrics at each layer:
        - Average cosine similarity between tokens
        - Average token norm
        - Average update norm (difference between consecutive layers)
        - Average gradient norm (if track_gradients is True)
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Dictionary of metrics, each containing a dictionary of values per layer
        """
        batch_size, seq_len = input_ids.size()
        self.layer_outputs = {}
        
        # Set model to eval mode for forward pass
        original_mode = self.model.training
        self.model.eval()
        
        # Store the original input embeddings for first layer update calculation
        with torch.no_grad():
            # Get input embeddings directly from the embedding layer
            input_embeds = self.model.transformer.wte(input_ids)  # [batch_size, seq_len, hidden_dim]
            # Store these as a special key for embeddings
            self.layer_outputs[-1] = input_embeds
        
        # If tracking gradients, we need to ensure we have gradients enabled
        if self.track_gradients:
            self.gradient_norms = {}
            
            # Enable gradients for the forward pass
            with torch.set_grad_enabled(True):
                # Forward pass to collect layer outputs
                logits, _ = self.model(input_ids)
                
                # We need a loss to backpropagate
                # Simple loss: predict the next token
                if input_ids.size(1) > 1:  # Check if sequence length is at least 2 tokens
                    shifted_logits = logits[:, :-1, :].contiguous()
                    shifted_targets = input_ids[:, 1:].contiguous()
                    loss = F.cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), 
                                        shifted_targets.view(-1))
                else:
                    # For very short sequences, just use a dummy loss on the full logits
                    # to ensure we can get gradients flowing
                    dummy_targets = torch.zeros(input_ids.size(0), dtype=torch.long, device=self.device)
                    loss = F.cross_entropy(logits[:, 0, :], dummy_targets)
                
                # Backward pass to collect gradients
                loss.backward()
        else:
            # Standard forward pass without gradients
            with torch.no_grad():
                self.model(input_ids)
                
        # Initialize metric dictionaries
        avg_cosine_sims = {}
        avg_token_norms = {}
        avg_update_norms = {}
        avg_gradient_norms = {}
        
        # Sort layer indices to ensure proper calculation of updates
        layer_indices = sorted(self.layer_outputs.keys())
        
        for i, layer_idx in enumerate(layer_indices):
            # Skip the embedding layer for cosine sim and token norm calculations
            if layer_idx < 0:
                continue
                
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
                
                # Only compute mean if we have more than one token (otherwise it's meaningless)
                if seq_len > 1:
                    # Compute average similarity (excluding self-similarity)
                    avg_sim = masked_sim.masked_select(mask.bool()).mean().item()
                    batch_sims.append(avg_sim)
                
                # 3. Compute update norms (for ALL layers, including first transformer layer)
                if i > 0:  # We can always compute this since we stored the embedding outputs at key -1
                    prev_layer_idx = layer_indices[i-1]
                    prev_tokens = self.layer_outputs[prev_layer_idx][b]
                    
                    # Compute update vectors (difference between consecutive layers)
                    updates = tokens - prev_tokens  # [seq_len, hidden_dim]
                    
                    # Compute norm of updates
                    update_norms = torch.norm(updates, p=2, dim=1)  # [seq_len]
                    avg_update = update_norms.mean().item()
                    batch_updates.append(avg_update)
            
            # Average across batches
            avg_cosine_sims[layer_idx] = np.mean(batch_sims) if batch_sims else 0.0
            avg_token_norms[layer_idx] = np.mean(batch_norms) if batch_norms else 0.0
            
            # Only store update norms for layers after the first one
            if i > 0:
                avg_update_norms[layer_idx] = np.mean(batch_updates) if batch_updates else 0.0
        
        # Add gradient norms if tracking gradients
        if self.track_gradients:
            avg_gradient_norms = self.gradient_norms
        
        # Reset model to original training mode
        self.model.train(original_mode)
        
        # Zero gradients to prevent interference with any ongoing training
        if self.track_gradients:
            self.model.zero_grad()
        
        # Package all metrics in a dictionary
        metrics = {
            'cosine_sim': avg_cosine_sims,
            'token_norm': avg_token_norms,
            'update_norm': avg_update_norms
        }
        
        # Add gradient norms if available
        if self.track_gradients:
            metrics['gradient_norm'] = avg_gradient_norms
        
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
                    'update_norm': {layer_idx: mean_value, ...},
                    'gradient_norm': {layer_idx: mean_value, ...},
                    'gradient_update_correlation': {layer_idx: mean_value, ...}
                },
                'variance': {
                    'cosine_sim': {layer_idx: variance_value, ...},
                    'token_norm': {layer_idx: variance_value, ...},
                    'update_norm': {layer_idx: variance_value, ...},
                    'gradient_norm': {layer_idx: variance_value, ...},
                    'gradient_update_correlation': {layer_idx: variance_value, ...}
                }
            }
        """
        # Initialize dictionaries to collect metrics across all prompts
        all_metrics = {
            'cosine_sim': {},
            'token_norm': {},
            'update_norm': {},
            'gradient_norm': {},
            'gradient_update_correlation': {}
        }
        
        # Make sure gradient tracking is enhanced
        self.enhance_gradient_tracking()
        
        # Analyze each prompt
        for prompt in tqdm(prompts, desc="Analyzing prompts"):
            # Tokenize prompt
            encoded = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Run basic token metrics
            metrics = self.compute_token_metrics(encoded)
            
            # Run enhanced gradient analysis
            grad_metrics = self.analyze_gradients(encoded)
            
            # Merge metrics
            metrics.update(grad_metrics)
            
            # Collect metrics for each prompt
            for metric_name, layer_values in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = {}
                    
                for layer_idx, value in layer_values.items():
                    if layer_idx not in all_metrics[metric_name]:
                        all_metrics[metric_name][layer_idx] = []
                    all_metrics[metric_name][layer_idx].append(value)
        
        # Compute mean and variance for each metric
        results = {
            'mean': {},
            'variance': {}
        }
        
        for metric_name, layer_data in all_metrics.items():
            results['mean'][metric_name] = {}
            results['variance'][metric_name] = {}
            
            for layer_idx, values in layer_data.items():
                values_array = np.array(values)
                results['mean'][metric_name][layer_idx] = np.mean(values_array)
                results['variance'][metric_name][layer_idx] = np.var(values_array)
        
        return results
    
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
        
        # Clear any existing gradients if tracking gradients
        if self.track_gradients:
            self.model.zero_grad()
        
        # Compute metrics (including gradient norms if tracking)
        metrics = self.compute_token_metrics(input_ids)
        
        # Calculate similarity matrices
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
    
    def cleanup(self):
        """
        Remove all hooks to prevent memory leaks.
        Call this method when you're done using the analyzer.
        """
        # Remove forward hooks
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            
        # Remove gradient hooks
        if hasattr(self, 'grad_hooks'):
            for hook in self.grad_hooks:
                hook.remove()
            self.grad_hooks = []
            
        # Restore model training state
        if hasattr(self, 'was_training'):
            self.model.train(self.was_training)
        
        # Clear stored data
        self.layer_outputs = {}
        self.gradient_norms = {}
        if hasattr(self, 'raw_gradients'):
            del self.raw_gradients
        if hasattr(self, 'gradient_update_corr'):
            self.gradient_update_corr = {}
        if hasattr(self, 'gradient_cosine_sims'):
            self.gradient_cosine_sims = {}
        
        print("GeometryAnalyzer hooks and stored data cleaned up")

    def visualize_metrics(self, metrics_dict, title="Token Geometry Analysis", figsize=(14, 10)):
        """
        Visualize metrics across layers.
        
        Args:
            metrics_dict: Dictionary with structure from analyze_prompts
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if visualization failed
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get the mean metrics and variances
            mean_metrics = metrics_dict.get('mean', {})
            var_metrics = metrics_dict.get('variance', {})
            
            # If old format (no mean/variance structure), use as is
            if 'mean' not in metrics_dict and len(metrics_dict) > 0:
                mean_metrics = metrics_dict
                var_metrics = {}
            
            # Define metrics to display with labels
            available_metrics = list(mean_metrics.keys())
            metric_labels = {
                'cosine_sim': 'Average Cosine Similarity',
                'token_norm': 'Token Representation Norm',
                'update_norm': 'Update Norm (Change Between Layers)',
                'gradient_norm': 'Gradient Norm Across Layers',
                'gradient_update_correlation': 'Correlation (Gradient ↔ Update)'
            }
            
            # Filter to only include metrics that exist in the data
            metric_names = [m for m in [
                'cosine_sim', 'token_norm', 'update_norm', 
                'gradient_norm', 'gradient_update_correlation'
            ] if m in available_metrics]
            
            if not metric_names:
                logger.warning("No metrics found to visualize")
                return None
                
            metric_titles = [metric_labels.get(name, name) for name in metric_names]
            
            # Create figure with dynamic number of subplots based on metrics
            num_plots = len(metric_names)
            fig, axes = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots/3), sharex=True)
            
            # Ensure axes is always a list for consistent indexing
            if num_plots == 1:
                axes = [axes]
                
            fig.suptitle(title, fontsize=16)
            
            # Create color scheme for different metrics
            colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
            
            for i, (metric_name, metric_title) in enumerate(zip(metric_names, metric_titles)):
                ax = axes[i]
                color = colors[i]
                
                # Get the metric values by layer
                metric_data = mean_metrics[metric_name]
                if not metric_data:
                    # Skip empty metrics
                    ax.text(0.5, 0.5, f"No data for {metric_title}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue
                    
                # Get layers and values, handling potential embedding layer
                layers = sorted(metric_data.keys())
                
                # For visualization, we may want to convert layer indices to more intuitive labels
                # e.g., -1 -> "Embeddings", 0-N -> "Layer 1-N+1", N+1 -> "Final"
                layer_labels = []
                for layer in layers:
                    if layer == -1:
                        layer_labels.append("Emb")
                    elif layer == len(self.model.transformer.h):
                        layer_labels.append("Final")
                    else:
                        layer_labels.append(str(layer + 1))  # 1-indexed for display
                
                values = [metric_data[layer] for layer in layers]
                
                # Get variance if available
                if var_metrics and metric_name in var_metrics:
                    var_data = var_metrics[metric_name]
                    std_values = [np.sqrt(var_data.get(layer, 0)) for layer in layers]
                    
                    # Plot with error bars (standard deviation)
                    ax.errorbar(layers, values, yerr=std_values, marker='o', linestyle='-', 
                               capsize=4, color=color, label=metric_title)
                else:
                    # Plot without error bars
                    ax.plot(layers, values, marker='o', linestyle='-', linewidth=2, 
                           color=color, label=metric_title)
                
                # Customize plot based on metric type
                if metric_name == 'gradient_update_correlation':
                    # Add horizontal line at 0 for correlation
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    # Set a reasonable y-range for correlations
                    ax.set_ylim(-1.1, 1.1)
                    # Add context for correlation interpretation
                    ax.text(0.98, 0.05, "Positive: aligned directions", 
                           horizontalalignment='right', transform=ax.transAxes, 
                           fontsize=9, style='italic', alpha=0.7)
                    ax.text(0.98, 0.95, "Negative: opposing directions", 
                           horizontalalignment='right', transform=ax.transAxes, 
                           fontsize=9, style='italic', alpha=0.7)
                
                # Set labels and grid
                ax.set_ylabel(metric_title)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Add a horizontal line at 0 for context if values might be negative
                if min(values) < 0 and metric_name != 'gradient_update_correlation':
                    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Set y limits with some padding
                if 'std_values' in locals() and std_values:
                    min_val = np.min(np.array(values) - np.array(std_values))
                    max_val = np.max(np.array(values) + np.array(std_values))
                else:
                    min_val = np.min(values)
                    max_val = np.max(values)
                
                # Special y-limit handling for different metrics
                if metric_name != 'gradient_update_correlation':  # Already handled above
                    padding = max(0.1, (max_val - min_val) * 0.1)
                    ax.set_ylim(min_val - padding, max_val + padding)
                
                # Set x-axis ticks with custom labels
                ax.set_xticks(layers)
                ax.set_xticklabels(layer_labels)
            
            # Configure the x-axis (shared across subplots)
            if 'layers' in locals() and len(layers) > 0:
                axes[-1].set_xlabel('Layer')
            
            # Adjust layout and save/show
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            return fig
            
        except Exception as e:
            logger.warning(f"Error in visualization: {e}")
            return None

    def analyze_gradients(self, input_ids, target_ids=None):
        """
        Perform a forward and backward pass to analyze gradients.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            target_ids: Optional target token IDs (if None, will use input_ids shifted right)
            
        Returns:
            Dictionary of gradient metrics
        """
        self.model.zero_grad()
        
        # If track_gradients is not enabled, enable it
        if not hasattr(self, 'grad_hooks') or not self.grad_hooks:
            self.enhance_gradient_tracking()
        
        # Ensure we're using enhanced gradient tracking
        if not hasattr(self, 'gradient_update_corr'):
            self.enhance_gradient_tracking()
            
        # Save original layer outputs before analysis
        original_layer_outputs = self.layer_outputs.copy() if hasattr(self, 'layer_outputs') else {}
        self.layer_outputs = {}
            
        # Set model to eval mode for consistent results
        original_mode = self.model.training
        self.model.eval()
        
        # Store the original input embeddings for first layer update calculation
        with torch.no_grad():
            # Get input embeddings directly from the embedding layer
            input_embeds = self.model.transformer.wte(input_ids)  # [batch_size, seq_len, hidden_dim]
            # Store these as a special key for embeddings
            self.layer_outputs[-1] = input_embeds
        
        # Enable gradient computation
        with torch.set_grad_enabled(True):
            # Forward pass
            logits, _ = self.model(input_ids)
            
            # If target_ids is not provided, create them by shifting input_ids right
            if target_ids is None:
                target_ids = input_ids.clone()
                if target_ids.size(1) > 1:
                    target_ids = target_ids[:, 1:].contiguous()
                    # Add a random token at the end
                    pad = torch.randint(0, self.model.config.vocab_size, 
                                       (target_ids.size(0), 1), 
                                       device=input_ids.device)
                    target_ids = torch.cat([target_ids, pad], dim=1)
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
        
        # Collect gradient metrics
        gradient_metrics = {
            'gradient_norm': self.gradient_norms.copy(),
        }
        
        # Add gradient-update correlation if available
        if hasattr(self, 'gradient_update_corr') and self.gradient_update_corr:
            gradient_metrics['gradient_update_correlation'] = self.gradient_update_corr.copy()
        
        # Reset model state
        self.model.train(original_mode)
        self.model.zero_grad()
        
        # Restore original layer outputs
        self.layer_outputs = original_layer_outputs
        
        return gradient_metrics
    
    def analyze_with_completion(self, prompt, completion, tokenizer):
        """
        Analyze model gradients using a prompt-completion pair.
        This provides a more realistic gradient analysis using actual completions.
        
        Args:
            prompt: The input prompt text
            completion: The expected completion text
            tokenizer: Tokenizer to encode the texts
            
        Returns:
            Dictionary containing analysis results and gradient metrics
        """
        import torch
        import torch.nn.functional as F
        
        # Tokenize the prompt and completion
        try:
            # Encode prompt tokens
            prompt_tokens_list = tokenizer.encode(prompt)
            prompt_tokens = torch.tensor([prompt_tokens_list], dtype=torch.long).to(self.device)
            
            # Store the model's original training state
            original_mode = self.model.training
            self.model.eval()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Ensure enhanced gradient tracking is enabled
            if not hasattr(self, 'gradient_update_corr'):
                self.enhance_gradient_tracking()
                
            # Store original layer outputs
            original_layer_outputs = self.layer_outputs.copy() if hasattr(self, 'layer_outputs') else {}
            self.layer_outputs = {}
            
            # Store input embeddings for first layer update calculations
            with torch.no_grad():
                input_embeds = self.model.transformer.wte(prompt_tokens)
                self.layer_outputs[-1] = input_embeds
                
            # Simplified approach for gradient computation using next-token prediction
            with torch.enable_grad():
                # Forward pass on prompt
                logits, _ = self.model(prompt_tokens)
                
                # For gradient computation, we'll use a simplified approach:
                # predict the next token in the sequence (shifted targets)
                batch_size, seq_len, vocab_size = logits.size()
                
                if seq_len > 1:
                    # Shift logits and targets for next-token prediction
                    logits = logits[:, :-1, :].contiguous()  # Remove last prediction
                    targets = prompt_tokens[:, 1:].contiguous()  # Remove first token (predict next tokens)
                    
                    # Ensure shapes are compatible for loss computation
                    # logits shape: [batch_size, seq_len-1, vocab_size]
                    # targets shape: [batch_size, seq_len-1]
                    
                    # Reshape for cross entropy: [batch_size * (seq_len-1), vocab_size]
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    
                    # Compute loss
                    loss = F.cross_entropy(logits_flat, targets_flat)
                else:
                    # For very short sequences, predict a dummy target
                    # This is just to ensure we can compute gradients
                    dummy_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    loss = F.cross_entropy(logits[:, 0, :], dummy_targets)
                
                # Backward pass to compute gradients
                loss.backward()
            
            # Compute token geometry metrics
            token_metrics = {}
            token_metrics['cosine_sim'] = {}
            token_metrics['token_norm'] = {}
            token_metrics['update_norm'] = {}
            
            # Sort layer indices for consistent processing
            layer_indices = sorted(self.layer_outputs.keys())
            
            # Calculate metrics for each layer
            for i, layer_idx in enumerate(layer_indices):
                # Skip embedding layer for certain metrics
                if layer_idx < 0:
                    continue
                    
                layer_output = self.layer_outputs[layer_idx]
                tokens = layer_output[0]  # Get first batch item
                
                # 1. Compute token norms
                token_norms = torch.norm(tokens, p=2, dim=1)
                token_metrics['token_norm'][layer_idx] = token_norms.mean().item()
                
                # 2. Compute cosine similarities
                tokens_norm = F.normalize(tokens, p=2, dim=1)
                sim_matrix = torch.mm(tokens_norm, tokens_norm.t())
                
                # Create mask to exclude self-similarity
                seq_len = tokens.size(0)
                mask = torch.ones_like(sim_matrix) - torch.eye(seq_len, device=self.device)
                masked_sim = sim_matrix * mask
                
                # Only compute mean if we have more than one token
                if seq_len > 1:
                    token_metrics['cosine_sim'][layer_idx] = masked_sim.masked_select(mask.bool()).mean().item()
                else:
                    token_metrics['cosine_sim'][layer_idx] = 0.0
                
                # 3. Compute update norms for all layers after first
                if i > 0:
                    prev_layer_idx = layer_indices[i-1]
                    prev_tokens = self.layer_outputs[prev_layer_idx][0]
                    updates = tokens - prev_tokens
                    update_norms = torch.norm(updates, p=2, dim=1)
                    token_metrics['update_norm'][layer_idx] = update_norms.mean().item()
            
            # Collect gradient metrics
            gradient_metrics = {
                'gradient_norm': self.gradient_norms.copy() if hasattr(self, 'gradient_norms') else {},
            }
            
            # Add gradient-update correlation if available
            if hasattr(self, 'gradient_update_corr') and self.gradient_update_corr:
                gradient_metrics['gradient_update_correlation'] = self.gradient_update_corr.copy()
                
            # Calculate similarity matrices
            similarity_matrices = {}
            for layer_idx in layer_indices:
                if layer_idx < 0:
                    continue
                try:
                    # Get embeddings
                    embeddings = self.layer_outputs[layer_idx][0]
                    
                    # Normalize embeddings
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                    
                    # Compute similarity matrix
                    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
                    
                    # Store similarity matrix
                    similarity_matrices[layer_idx] = sim_matrix.cpu().numpy()
                except Exception as e:
                    logger.error(f"Error processing layer {layer_idx}: {e}")
                    continue
                    
            # Restore model state
            self.model.train(original_mode)
            self.model.zero_grad()
            
            # Restore original layer outputs
            self.layer_outputs = original_layer_outputs
            
            # For reference, also encode the completion
            try:
                completion_tokens_list = tokenizer.encode(completion)
                completion_tokens = [completion_tokens_list]
            except:
                completion_tokens = [[]]
            
            # Package results
            results = {
                'prompt_tokens': prompt_tokens.cpu().numpy()[0].tolist(),
                'completion_tokens': completion_tokens[0],
                'metrics': {**token_metrics, **gradient_metrics},
                'similarity_matrices': similarity_matrices,
                'loss': loss.item()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analyze_with_completion: {e}")
            # Return empty results on error
            return {
                'prompt_tokens': [],
                'completion_tokens': [],
                'metrics': {},
                'similarity_matrices': {},
                'error': str(e)
            }