"""
Training script for LLaMA-style models with support for distributed training.
"""

import os
import time
import json
import random
import argparse
import logging
import numpy as np
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from datetime import datetime
# Import your custom model
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.utils.data import get_tokenizer
from src.utils.wikitext_dataset import WikiTextDataset

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def pad_sequence(batch):
    """
    Custom collate function to handle sequences of different lengths.
    Pads sequences to the maximum length in the batch.
    """
    # Convert batch to a list of tensors if it's not already
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) > 0 and isinstance(batch[0][0], torch.Tensor):
        # We have a batch of tuples/lists containing tensors
        # Let's handle each element separately
        batch_size = len(batch)
        element_count = len(batch[0])
        result = []
        
        for i in range(element_count):
            # Get all tensors for this element
            tensors = [b[i] for b in batch]
            
            # Check if all tensors have same shape
            first_shape = tensors[0].shape
            if all(t.shape == first_shape for t in tensors):
                # All same shape, just stack them
                result.append(torch.stack(tensors))
            else:
                # Find max length
                max_len = max(t.shape[0] for t in tensors)
                
                # Pad each tensor
                padded = []
                for t in tensors:
                    if t.shape[0] < max_len:
                        # Create padded tensor
                        padding = torch.zeros(max_len, dtype=t.dtype, device=t.device)
                        padding[:t.shape[0]] = t
                        padded.append(padding)
                    else:
                        padded.append(t)
                
                # Stack padded tensors
                result.append(torch.stack(padded))
                
        return tuple(result)
        
    elif isinstance(batch[0], torch.Tensor):
        # We have a batch of tensors
        # Check if all tensors have same shape
        first_shape = batch[0].shape
        if all(t.shape == first_shape for t in batch):
            # All same shape, just stack them
            return torch.stack(batch)
        else:
            # Find max length
            max_len = max(t.shape[0] for t in batch)
            
            # Pad each tensor
            padded = []
            for t in batch:
                if t.shape[0] < max_len:
                    # Create padded tensor
                    padding = torch.zeros(max_len, dtype=t.dtype, device=t.device)
                    padding[:t.shape[0]] = t
                    padded.append(padding)
                else:
                    padded.append(t)
            
            # Stack padded tensors
            return torch.stack(padded)
    
    # Default case: use PyTorch's default collate
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def get_device(args):
    """Get the appropriate device based on environment and arguments."""
    if args.single_gpu:
        # Check for CUDA first
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        
        # Check for MPS (Apple Silicon) second
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders)")
            return device
        
        # Fall back to CPU
        else:
            device = torch.device("cpu")
            print("No GPU detected, using CPU")
            return device
    
    # For distributed training
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
        print(f"Using CUDA GPU {args.local_rank}: {torch.cuda.get_device_name(args.local_rank)}")
        return device
    
    print("No CUDA available for distributed training, falling back to CPU (this isn't recommended)")
    return torch.device("cpu")


def get_world_size():
    """Get world size for distributed training."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def setup_distributed_training(args):
    """Initialize distributed training."""
    if args.single_gpu:
        # For single GPU, just use device 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1  # rank, world_size
    
    # Set up multi-GPU training
    if args.local_rank == -1:
        if "LOCAL_RANK" not in os.environ:
            logger.warning("LOCAL_RANK not set, defaulting to single GPU (device 0)")
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            return 0, 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Get rank and world size from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set device based on local rank
    torch.cuda.set_device(args.local_rank)
    
    # Initialize process group
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        dist.barrier()
        logger.info(f"Initialized distributed training with rank {rank}, world_size {world_size}")
    except Exception as e:
        logger.warning(f"Could not initialize distributed training: {e}")
        logger.info("Falling back to single GPU training")
        return 0, 1
    
    return rank, world_size


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Create a cosine learning rate scheduler."""
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_model(model, dataloader, pad_idx, device, max_eval_tokens=500000, world_size=1):
    """Evaluate model perplexity on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    for batch in dataloader:
        if total_tokens >= max_eval_tokens:
            break
        
        # Handle different dataset return types
        if isinstance(batch, (list, tuple)):
            # Tuple-style dataset
            if len(batch) == 2:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            else:
                input_ids = batch[0].to(device)
                labels = input_ids.clone()
                labels[labels == pad_idx] = -100  # Ignore padding tokens
        else:
            # Dictionary-style dataset
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            labels[labels == pad_idx] = -100  # Ignore padding tokens
        
        with torch.no_grad():
            _, loss = model(input_ids, labels)
        
        # Count non-padding tokens
        token_count = (input_ids != pad_idx).sum().item()
        total_tokens += token_count * world_size
        total_loss += loss.item() * token_count
        n_batches += 1
    
    # Calculate average loss, weighted by tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return perplexity, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description='Train a LLaMA-style model')
    
    # Model arguments
    parser.add_argument('--ln', type=str, choices=['preln', 'postln', 'periln', 'mixln', 'predyt', 'postdyt'], default='preln',
                       help='Layer normalization architecture')
    parser.add_argument('--n-layer', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--n-head', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--n-embd', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size')
    parser.add_argument('--max-position-embeddings', type=int, default=2048,
                       help='Maximum sequence length for rotary embeddings')
    parser.add_argument('--block-size', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--no-initial-ln', action='store_true',
                       help='Disable initial layer normalization')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, required=True,
                       help='Batch size per GPU')
    parser.add_argument('--total-batch-size', type=int, default=None,
                       help='Global batch size (batch_size * gradient_accumulation * world_size)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                       help='Minimum learning rate as a ratio of max lr')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Learning rate warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Number of update steps to train for')
    
    # Data and evaluation arguments
    parser.add_argument('--data-dir', type=str, default='data/wikitext-2',
                       help='Directory containing dataset')
    parser.add_argument('--tokenizer-type', type=str, default='huggingface',
                   choices=['huggingface', 'tiktoken', 'bpe', 'char'],
                   help='Tokenizer to use')
    parser.add_argument('--tokenizer-model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                   help='Model name for HuggingFace tokenizer (use open models like TinyLlama/TinyLlama-1.1B-Chat-v1.0, mistralai/Mistral-7B-v0.1, or openlm-research/open_llama_3b)')
    parser.add_argument('--local-tokenizer', type=str, default=None,
                   help='Path to local HuggingFace tokenizer directory (for offline use)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--eval-interval', type=int, default=2000,
                       help='Evaluate every N update steps')
    parser.add_argument('--eval-tokens', type=int, default=500000,
                       help='Number of tokens to use for evaluation')
    
    # Saving and loading arguments
    parser.add_argument('--save-dir', type=str, required=True,
                       help='Directory to save model')
    parser.add_argument('--save-interval', type=int, default=5000,
                       help='Save every N update steps')
    parser.add_argument('--continue-from', type=str, default=None,
                       help='Continue training from a checkpoint')
    
    # Miscellaneous arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Data type for training')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set by torchrun)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Enable gradient checkpointing to save memory')
    parser.add_argument('--single-gpu', action='store_true',
                       help='Use single GPU training (no distributed)')
    
    args = parser.parse_args()
    
    # Auto-calculate gradient_accumulation if total_batch_size is provided
    if args.total_batch_size is not None:
        # Always use world_size=1 during argument parsing, will be updated during setup
        world_size = 1
        
        if args.gradient_accumulation is None:
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            # Make sure we don't end up with zero gradient accumulation steps
            if args.gradient_accumulation == 0:
                args.gradient_accumulation = 1
            
            print(f"Setting gradient_accumulation to {args.gradient_accumulation}")
        
        # Check if the calculation works out exactly
        effective_batch_size = args.gradient_accumulation * args.batch_size * world_size
        if effective_batch_size != args.total_batch_size:
            print(f"Warning: Effective batch size ({effective_batch_size}) doesn't match requested total batch size ({args.total_batch_size})")
            print(f"Adjusting gradient_accumulation from {args.gradient_accumulation} to {args.total_batch_size // (args.batch_size * world_size)}")
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            # Verify again
            if args.gradient_accumulation * args.batch_size * world_size != args.total_batch_size:
                print(f"Warning: Cannot achieve exact total_batch_size. Using effective batch size of {args.gradient_accumulation * args.batch_size * world_size}")
    
    return args


def main():
    args = parse_args()
    
    # Setup random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize distributed training
    rank, world_size = setup_distributed_training(args) if not args.single_gpu else (0, 1)
    is_main_process = rank == 0
    
    # Only log from the main process
    if not is_main_process:
        logger.setLevel(logging.WARNING)
    
    # Setup model saving directory
    save_dir = Path(args.save_dir)
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        # Create CSV files for tracking metrics
        metrics_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Training metrics file
        train_metrics_file = os.path.join(metrics_dir, "train_metrics.csv")
        with open(train_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "learning_rate", "tokens_seen", "timestamp"])
        
        # Validation metrics file
        val_metrics_file = os.path.join(metrics_dir, "val_metrics.csv")
        with open(val_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "perplexity", "tokens_seen", "timestamp"])
    

    # Get device
    device = get_device(args)
    
    # Get tokenizer
    tokenizer = get_tokenizer(tokenizer_type=args.tokenizer_type, 
                             model_name=args.tokenizer_model,
                             local_tokenizer_path=args.local_tokenizer)
    pad_idx = tokenizer.pad_token_id  # Use the tokenizer's pad_token_id property instead of assuming last token
    
    # Create datasets
    logger.info(f"Loading datasets from {args.data_dir}")
    try:
        train_dataset = WikiTextDataset(
            os.path.join(args.data_dir, 'train.txt'),
            tokenizer,
            block_size=args.block_size,
            is_eval=False
        )
        
        val_dataset = WikiTextDataset(
            os.path.join(args.data_dir, 'validation.txt'),
            tokenizer,
            block_size=args.block_size,
            is_eval=True
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        logger.info("You may need to run 'python scripts/download_wikitext.py' first")
        return
    
    # Create dataloaders
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if not args.single_gpu and world_size > 1 else None
    
    # On Mac, especially with MPS, we need to disable multiprocessing to avoid pickle errors
    use_workers = 0 if device.type == "mps" or device.type == "cpu" else args.workers
    if use_workers != args.workers:
        logger.info(f"Detected {device.type} device, setting workers to 0 to avoid multiprocessing issues")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=use_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_sequence
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_sequence
    )
    
    # Create model
    logger.info("Creating model")
    vocab_size = args.vocab_size if args.vocab_size else getattr(tokenizer, 'vocab_size', 32000)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=False,  # LLaMA doesn't use bias
        ln=args.ln,
        use_initial_ln=not args.no_initial_ln,
        use_swiglu=True,  # Always use SwiGLU as default
        max_position_embeddings=args.max_position_embeddings,
        pad_token_id=pad_idx,  # Pass the padding token ID to the model config
    )
    
    model = GPT(config)
    
    # If we're using a HuggingFace tokenizer with a modified vocabulary
    # (like when adding a pad token to Llama), resize the model's embedding matrix
    if args.tokenizer_type == "huggingface":
        if getattr(tokenizer, "tokenizer", None) is not None and hasattr(tokenizer.tokenizer, "added_tokens_encoder"):
            # Handle case when using HuggingFaceTokenizer wrapper
            if len(tokenizer.tokenizer.added_tokens_encoder) > 0:
                logger.info(f"Resizing model embeddings to match tokenizer vocabulary size: {tokenizer.vocab_size}")
                
                # Need to resize both the embedding and lm_head
                current_vocab_size = model.transformer.wte.weight.size(0)
                if tokenizer.vocab_size != current_vocab_size:
                    model.transformer.wte = nn.Embedding(tokenizer.vocab_size, config.n_embd)
                    model.lm_head = nn.Linear(config.n_embd, tokenizer.vocab_size, bias=False)
                    # Re-tie weights
                    model.transformer.wte.weight = model.lm_head.weight
                    model.config.vocab_size = tokenizer.vocab_size
                    logger.info(f"Model vocabulary resized from {current_vocab_size} to {tokenizer.vocab_size}")
    
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            logger.warning("Model doesn't support gradient checkpointing")
    
    # Load from checkpoint if specified
    start_step = 0
    if args.continue_from is not None:
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "model.pt")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            
            # Load training state if available
            training_state_path = os.path.join(args.continue_from, "training_state.json")
            if os.path.exists(training_state_path):
                with open(training_state_path, 'r') as f:
                    training_state = json.load(f)
                    start_step = training_state.get("step", 0)
                    logger.info(f"Resuming from step {start_step}")
    
    # Set model dtype and move to device
    if args.dtype == "bfloat16" and device.type in ("cuda", "mps") and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        logger.info("Using bfloat16 precision")
    elif args.dtype == "float16" and device.type in ("cuda", "mps"):
        model_dtype = torch.float16
        logger.info("Using float16 precision")
    else:
        model_dtype = torch.float32
        logger.info("Using float32 precision")
    
    model = model.to(device=device, dtype=model_dtype)
    
    # Wrap model for distributed training
    if not args.single_gpu and world_size > 1 and device.type == "cuda":
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    
    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    
    # Load optimizer and scheduler states if resuming
    if args.continue_from is not None:
        optimizer_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=torch.device('cpu'))
            optimizer.load_state_dict(optimizer_state["optimizer"])
            scheduler.load_state_dict(optimizer_state["scheduler"])
    
    # Setup mixed precision training
    use_amp = model_dtype != torch.float32 and device.type != "cpu"
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    
    # Print model information
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of model parameters: {n_params / 1e6:.2f} Million")
    
    # Training loop
    logger.info(f"Starting training from step {start_step}")
    model.train()
    
    global_step = start_step
    update_step = start_step
    tokens_seen = 0
    best_val_loss = float('inf')
    
    progress_bar = tqdm(total=args.max_steps, disable=not is_main_process)
    progress_bar.update(start_step)
    
    # Main training loop
    while update_step < args.max_steps:
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        if train_sampler is not None:
            train_sampler.set_epoch(update_step)
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Handle both tuple-style datasets (returns x, y) and dict-style datasets (returns {"input_ids": x})
            if isinstance(batch, (list, tuple)):
                # If the batch is a tuple, assume it contains (input_ids, labels) or just input_ids
                if len(batch) == 2:
                    input_ids, labels = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                else:
                    input_ids = batch[0].to(device)
                    labels = input_ids.clone()
                    labels[labels == pad_idx] = -100  # Ignore padding tokens
            else:
                # Dictionary-style dataset
                input_ids = batch["input_ids"].to(device)
                labels = input_ids.clone()
                labels[labels == pad_idx] = -100  # Ignore padding tokens
            
            # Count non-padding tokens
            tokens_in_batch = (input_ids != pad_idx).sum().item() * world_size
            tokens_seen += tokens_in_batch
            
            # Forward pass with mixed precision
            if use_amp and device.type == "cuda":
                with autocast():
                    _, loss = model(input_ids, labels)
                    loss = loss / args.gradient_accumulation
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                # Standard forward/backward without mixed precision
                _, loss = model(input_ids, labels)
                loss = loss / args.gradient_accumulation
                loss.backward()
            
            # Track loss
            epoch_loss += loss.item() * args.gradient_accumulation
            
            if is_main_process and update_step % 10 == 0:  # Adjust frequency as needed
                with open(train_metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        update_step, 
                        epoch_loss / (batch_idx + 1),
                        scheduler.get_last_lr()[0],
                        tokens_seen,
                        datetime.now().isoformat()
                    ])
            # Only update weights and learning rate at the specified interval
            if (global_step + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                if args.grad_clip > 0:
                    if use_amp and device.type == "cuda":
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update weights
                if use_amp and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                update_step += 1
                
                # Print progress
                if is_main_process and update_step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    progress_bar.set_description(
                        f"Step {update_step} | Loss: {epoch_loss / (batch_idx + 1):.4f} | LR: {lr:.6f}"
                    )
                    progress_bar.update(1)
                
                # Evaluate
                if update_step % args.eval_interval == 0:
                    val_perplexity, eval_tokens = evaluate_model(
                        model, 
                        val_dataloader, 
                        pad_idx, 
                        device, 
                        max_eval_tokens=args.eval_tokens,
                        world_size=world_size
                    )
                    
                    logger.info(
                        f"Step {update_step} | Val Perplexity: {val_perplexity:.2f} "
                        f"| Tokens: {tokens_seen / 1e6:.2f}M"
                    )
                    
                    if is_main_process:
                        with open(val_metrics_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                update_step, 
                                val_perplexity,
                                tokens_seen,
                                datetime.now().isoformat()
                            ])
                    # Save best model
                    if is_main_process and val_perplexity < best_val_loss:
                        best_val_loss = val_perplexity
                        best_model_path = save_dir / "best_model"
                        
                        if hasattr(model, "module"):
                            model.module.save_pretrained(best_model_path)
                        else:
                            model.save_pretrained(best_model_path)
                        
                        with open(best_model_path / "val_results.json", "w") as f:
                            json.dump({
                                "step": update_step,
                                "perplexity": val_perplexity,
                                "tokens_seen": tokens_seen,
                            }, f)
                        
                        logger.info(f"Saved best model with perplexity {val_perplexity:.2f}")
                
                # Save checkpoint
                if is_main_process and update_step % args.save_interval == 0:
                    checkpoint_path = save_dir / f"checkpoint-{update_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    
                    if hasattr(model, "module"):
                        model.module.save_pretrained(checkpoint_path)
                    else:
                        model.save_pretrained(checkpoint_path)
                    
                    # Save optimizer and scheduler
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, checkpoint_path / "optimizer.pt")
                    
                    # Save training state
                    with open(checkpoint_path / "training_state.json", "w") as f:
                        json.dump({
                            "step": update_step,
                            "tokens_seen": tokens_seen,
                            "best_val_loss": best_val_loss,
                        }, f)
                    
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Check if we've reached the maximum number of steps
                if update_step >= args.max_steps:
                    break
            
            global_step += 1
    
    # Training finished
    if is_main_process:
        progress_bar.close()
        
        # Save final model
        final_path = save_dir / "final_model"
        if hasattr(model, "module"):
            model.module.save_pretrained(final_path)
        else:
            model.save_pretrained(final_path)
        
        # Final evaluation
        val_perplexity, eval_tokens = evaluate_model(
            model, 
            val_dataloader, 
            pad_idx, 
            device, 
            max_eval_tokens=args.eval_tokens * 2,  # More thorough final evaluation
            world_size=world_size
        )
        
        with open(final_path / "final_results.json", "w") as f:
            json.dump({
                "step": update_step,
                "perplexity": val_perplexity,
                "tokens_seen": tokens_seen,
                "best_val_loss": best_val_loss,
            }, f)
        
        logger.info(f"Training finished! Final perplexity: {val_perplexity:.2f}")
    
    # Clean up distributed training
    if not args.single_gpu and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()