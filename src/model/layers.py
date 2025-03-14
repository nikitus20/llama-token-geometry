"""
Neural network layers for transformer models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.model.config import GPTConfig

logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.
    
    Note: This class is kept for backward compatibility but is deprecated.
    The codebase now uses RMSNorm exclusively.
    """

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5):
        super().__init__()
        logger.warning("LayerNorm is deprecated and will be removed in a future version. Use RMSNorm instead.")
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization as used in LLaMA.
    Implementation based on the LLaMA paper.
    """
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the root mean square per feature
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # Normalize by the RMS
        x_norm = x / rms
        # Scale with learned parameters
        return self.weight * x_norm

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

class CausalSelfAttention(nn.Module):
    """
    A causal self-attention layer.
    Uses Flash Attention when available for better performance.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but KV cache is not supported
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logger.warning("Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Calculate query, key, values 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if hasattr(self, 'rotary_emb'):
            position_ids = torch.arange(T, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(v, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Use scaled_dot_product_attention when available
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.act_fn = F.silu
        self.dropout = nn.Dropout(config.dropout)
        
        # Add scaling flag for matching reference implementation
        self.scale_output = config.scale_mlp_output if hasattr(config, 'scale_mlp_output') else False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MLP(nn.Module):
    """
    MLP block in a transformer model with support for GELU or SwiGLU activation.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.use_swiglu:
            self.swiglu = SwiGLU(config)
            self.forward = self.forward_swiglu
        else:
            # Standard MLP with GELU
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
            self.forward = self.forward_gelu

    def forward_gelu(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class Block(nn.Module):
    """
    Transformer block with support for different LN architectures:
    - preln: Pre-LN architecture (normalization before attention and MLP)
    - postln: Post-LN architecture (normalization after residual connections)
    - periln: Both pre and post normalization
    - mixln: Post-LN in early layers, Pre-LN in later layers
    
    Uses RMSNorm for all normalization layers.
    """
    
    def __init__(self, config: GPTConfig, block_idx: int):
        super().__init__()
        self.ln_type = config.ln
        self.block_idx = block_idx
        self.n_layer = config.n_layer
        self.mixln_split_layer = int(config.n_layer * config.mixln_split)
        
        # Always use RMSNorm
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        
        # For periln, we need additional norm layers
        if self.ln_type == "periln":
            self.post_ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.post_ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln_type == "preln":
            # PreLN architecture (used in LLaMA and most modern transformers)
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        elif self.ln_type == "postln":
            # PostLN architecture (used in original transformer paper)
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        elif self.ln_type == "periln":
            # PeriLN architecture (both pre and post normalization)
            # Pre-norm first
            attn_output = self.attn(self.ln_1(x))
            x = x + attn_output
            # Post-norm after residual
            x = self.post_ln_1(x)
            # Pre-norm again for MLP
            mlp_output = self.mlp(self.ln_2(x))
            x = x + mlp_output
            # Post-norm after residual
            x = self.post_ln_2(x)
        elif self.ln_type == "mixln":
            # MixLN architecture (postln in first layers, preln in later layers)
            if self.block_idx < self.mixln_split_layer:
                # PostLN in early layers
                x = self.ln_1(x + self.attn(x))
                x = self.ln_2(x + self.mlp(x))
            else:
                # PreLN in later layers
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
        else:
            raise ValueError(f"Unknown layernorm type: {self.ln_type}")
            
        return x
    

