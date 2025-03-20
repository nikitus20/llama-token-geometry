#!/usr/bin/env python
"""
Script to generate multiple models with different normalization techniques 
and run a brief warmup training for each.
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import GPT, GPTConfig
from src.utils.data import get_tokenizer
from src.utils.trainer import Trainer, TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")
    return device

def load_warmup_data(file_path, tokenizer):
    """
    Load data from a text file for model warmup.
    
    Args:
        file_path: Path to text data file
        tokenizer: Tokenizer
        
    Returns:
        List of text samples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n\n')  # Split by paragraph
        
        # Filter out empty strings
        data = [sample for sample in data if sample.strip()]
        
        # If we have long texts, split them further
        if len(data) < 50 and any(len(sample) > 1000 for sample in data):
            new_data = []
            for sample in data:
                if len(sample) > 1000:
                    # Split into smaller chunks
                    sentences = sample.split('.')
                    chunks = []
                    current_chunk = []
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        current_chunk.append(sentence)
                        if len('.'.join(current_chunk)) > 500:
                            chunks.append('.'.join(current_chunk) + '.')
                            current_chunk = []
                    
                    if current_chunk:
                        chunks.append('.'.join(current_chunk) + '.')
                    
                    new_data.extend(chunks)
                else:
                    new_data.append(sample)
            
            data = new_data
        
        logger.info(f"Loaded {len(data)} samples for warmup training")
        return data
    
    except Exception as e:
        logger.error(f"Error loading warmup data: {e}")
        # Return a few dummy samples if loading fails
        return [
            "This is a sample text for model training.",
            "Machine learning models need data to train on.",
            "Language models learn to predict the next token in a sequence."
        ]

def create_model_with_norm(norm_type, config_overrides=None, seed=None, init_scale=0.01, initial_ln_scale=0.1):
    """
    Create a model with a specific normalization technique.
    
    Args:
        norm_type: Type of normalization ('preln', 'postln', 'periln', 'mixln', 'predyt', 'postdyt', 'deepnorm')
        config_overrides: Dictionary of configuration overrides
        seed: Random seed for model initialization
        init_scale: Scaling factor for weight initialization
        initial_ln_scale: Scaling factor for the initial normalization layer
        
    Returns:
        Initialized GPT model
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Default small config
    config = {
        'vocab_size': 32000,
        'block_size': 128,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 192,
        'dropout': 0.0,
        'bias': True,
        'use_initial_ln': True,  # Apply normalization after embeddings
        'initial_ln_scale': initial_ln_scale,  # Scaling factor for initial normalization
        'use_swiglu': True,      # Use SwiGLU activation (LLaMA-style)
        'norm_eps': 1e-6,        # Epsilon value for normalization layers
        'initializer_range': init_scale,  # Use the provided init scale parameter
        'max_position_embeddings': 512,  # Maximum sequence length for rotary embeddings
        'rope_base': 10000,  # Base for rotary embeddings
        'scale_attn_weights': False,
        'scale_mlp_output': False,
        'deeppost': False
    }
    
    # Apply overrides
    if config_overrides:
        config.update(config_overrides)
    
    # Add normalization config
    config['ln'] = norm_type
    
    # Special handling for different normalization types
    if norm_type == 'deepnorm':
        # DeepNorm parameters will be computed automatically based on model depth
        pass
    elif norm_type == 'mixln':
        # Set mixln_split to determine the fraction of layers to use postln
        config['mixln_split'] = 0.25  # Default: first 25% of layers use postln
    elif norm_type in ['predyt', 'postdyt']:
        # We don't add dyt_init_alpha to the config object since it's not a field in GPTConfig
        # The model code uses getattr to access this, so we'll keep the default value
        pass
    
    # Create config object
    model_config = GPTConfig(**config)
    
    # Create model
    model = GPT(model_config)
    
    return model

def warmup_model(model, data, tokenizer, device, num_iterations=100, batch_size=8, output_dir=None):
    """
    Run a brief training warmup for the model.
    
    Args:
        model: The model to warm up
        data: List of text samples
        tokenizer: Tokenizer
        device: Device to run on
        num_iterations: Number of warmup iterations
        batch_size: Batch size for training
        output_dir: Directory to save checkpoints
        
    Returns:
        Trained model
    """
    model.to(device)
    
    # Tokenize data
    encoded_data = []
    for sample in data:
        try:
            tokens = tokenizer.encode(sample)
            if len(tokens) > 3:  # Lower threshold to include more samples
                encoded_data.append(tokens)
        except Exception as e:
            logger.warning(f"Error encoding sample: {e}")
    
    if not encoded_data:
        logger.error("No valid encoded samples for training")
        return model
    
    # Combine shorter sequences to create inputs of adequate length
    combined_data = []
    current_sequence = []
    
    # Sort by length to combine similar length samples
    encoded_data.sort(key=len)
    
    for tokens in encoded_data:
        current_sequence.extend(tokens)
        # When we have enough tokens, add to the combined data
        while len(current_sequence) >= model.config.block_size + 1:  # +1 for target shifting
            combined_data.append(current_sequence[:model.config.block_size + 1])
            current_sequence = current_sequence[model.config.block_size:]
    
    # Add any remaining sequence that's long enough
    if len(current_sequence) >= 10:  # Allow for shorter sequences at the end
        # Pad if needed to reach minimum viable length
        if len(current_sequence) < model.config.block_size + 1:
            padding_needed = model.config.block_size + 1 - len(current_sequence)
            current_sequence.extend([tokenizer.pad_token_id] * padding_needed)
        combined_data.append(current_sequence)
    
    # Ensure we have at least one training sample
    if not combined_data:
        logger.warning("No samples of adequate length, creating artificial data")
        # Create at least one artificial sequence for training
        artificial_sequence = []
        # Repeat the first encoded sample until we reach desired length
        sample_to_repeat = encoded_data[0] if encoded_data else [0, 1, 2, 3, 4]
        while len(artificial_sequence) < model.config.block_size + 1:
            artificial_sequence.extend(sample_to_repeat)
        combined_data.append(artificial_sequence[:model.config.block_size + 1])
    
    logger.info(f"Created {len(combined_data)} training sequences of adequate length")
    
    # Make sure we have enough data for the requested iterations
    random.shuffle(combined_data)
    training_samples = combined_data[:num_iterations * batch_size]
    if len(training_samples) < num_iterations * batch_size:
        # If we don't have enough, repeat data
        logger.info(f"Repeating data to reach {num_iterations * batch_size} training samples")
        samples_needed = num_iterations * batch_size - len(training_samples)
        # Repeat some samples until we reach the desired count
        repeated_samples = [training_samples[i % len(training_samples)] for i in range(samples_needed)]
        training_samples.extend(repeated_samples)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Set up learning rate scheduler with linear decay
    def lr_lambda(current_step):
        if current_step < 20:  # Warmup for 20 steps
            return float(current_step) / float(max(1, 20))
        return max(0.0, float(num_iterations - current_step) / float(max(1, num_iterations - 20)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup for saving checkpoints
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Running {num_iterations} warmup iterations")
    model.train()
    
    progress_bar = tqdm(total=num_iterations)
    total_loss = 0
    
    # Process in batches
    for step in range(num_iterations):
        # Get batch
        batch_start = step * batch_size
        batch_end = min(batch_start + batch_size, len(training_samples))
        batch_data = training_samples[batch_start:batch_end]
        
        # Pad sequences in the batch to the same length
        max_len = max(len(seq) for seq in batch_data)
        padded_batch = []
        
        for seq in batch_data:
            # Ensure all sequences are the same length by padding or truncating
            if len(seq) < max_len:
                padding = [tokenizer.pad_token_id] * (max_len - len(seq))
                padded_seq = seq + padding
            else:
                padded_seq = seq[:max_len]  # Truncate if too long
            padded_batch.append(padded_seq)
        
        # Convert to tensors
        batch_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)
        
        # Split into input and target - match the model's expected format
        input_ids = batch_tensor[:, :-1].contiguous()  # All tokens except the last one
        targets = batch_tensor[:, 1:].contiguous()     # All tokens except the first one (shifted by 1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        try:
            # Forward pass - use the correct parameter names (idx, targets)
            logits, loss = model(idx=input_ids, targets=targets)
        except Exception as e:
            # If there's an error, add more context and try to handle it
            logger.error(f"Error during forward pass: {e}")
            # Try a fallback approach with contiguous tensors
            input_ids_cont = input_ids.contiguous()
            targets_cont = targets.contiguous()
            # Ensure shapes match what the model expects
            if input_ids_cont.size(1) > model.config.block_size:
                logger.warning(f"Truncating sequence to model block size {model.config.block_size}")
                input_ids_cont = input_ids_cont[:, :model.config.block_size]
                targets_cont = targets_cont[:, :model.config.block_size]
            logits, loss = model(idx=input_ids_cont, targets=targets_cont)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update progress
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_description(f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        progress_bar.update(1)
        
        # Save checkpoint (optional)
        if output_dir and (step + 1) % 50 == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{step+1}.pt')
            torch.save(checkpoint, checkpoint_path)
    
    # Save final checkpoint
    if output_dir:
        checkpoint = {
            'step': num_iterations,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }
        checkpoint_path = os.path.join(output_dir, 'final_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate models with different normalization techniques')
    parser.add_argument('--output-dir', default='saved_models/norm_comparison', help='Directory to save models')
    parser.add_argument('--warmup-iterations', type=int, default=100, help='Number of warmup iterations')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--warmup-data', default='data/wikitext-2/train.txt', help='Path to warmup data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--init-scale', type=float, default=0.01, help='Scaling factor for weight initialization')
    parser.add_argument('--initial-ln-scale', type=float, default=0.1, help='Scaling factor for initial normalization layer')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # Prepare warmup data
    warmup_data = load_warmup_data(args.warmup_data, tokenizer)
    logger.info(f"Loaded {len(warmup_data)} samples for warmup training")
    
    # Dictionary to store model information
    model_info = {}
    
    # List of normalization techniques to create models for
    norm_types = ['preln', 'postln', 'periln', 'mixln', 'predyt', 'postdyt', 'deepnorm']
    
    # Create model for each normalization type
    for norm_type in norm_types:
        logger.info(f"Creating model with {norm_type} normalization")
        
        # Create and save the initial model
        model = create_model_with_norm(norm_type, seed=args.seed, init_scale=args.init_scale, 
                                     initial_ln_scale=args.initial_ln_scale)
        
        # Create model directory
        model_dir = os.path.join(args.output_dir, f"{norm_type}_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save initial model (pre-warmup)
        initial_model_dir = os.path.join(model_dir, "initial_model")
        os.makedirs(initial_model_dir, exist_ok=True)
        model.save_pretrained(initial_model_dir)
        
        # Save config information
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            config_dict = model.config.__dict__.copy()
            # Remove any non-serializable items
            for key in list(config_dict.keys()):
                if not isinstance(config_dict[key], (int, float, str, bool, list, dict, type(None))):
                    del config_dict[key]
            json.dump(config_dict, f, indent=2)
        
        # Warm up the model
        warmup_model(
            model=model,
            data=warmup_data,
            tokenizer=tokenizer,
            device=get_device(),
            num_iterations=args.warmup_iterations,
            batch_size=args.batch_size,
            output_dir=os.path.join(model_dir, "checkpoints")
        )
        
        # Save final warmed-up model
        final_model_dir = os.path.join(model_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        model.save_pretrained(final_model_dir)
        
        logger.info(f"Saved initial and final models for {norm_type}")
    
    logger.info(f"All models generated and saved to {args.output_dir}")
    
    # Create a summary file
    with open(os.path.join(args.output_dir, "models_summary.json"), 'w') as f:
        summary = {
            "normalization_types": norm_types,
            "warmup_iterations": args.warmup_iterations,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_paths": {
                norm_type: {
                    "initial": os.path.join(args.output_dir, f"{norm_type}_model", "initial_model"),
                    "final": os.path.join(args.output_dir, f"{norm_type}_model", "final_model")
                } for norm_type in norm_types
            }
        }
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main() 