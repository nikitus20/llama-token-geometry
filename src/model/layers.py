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
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5):
        super().__init__()
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
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as used in LLaMA.
    Implementation based on the LLaMA and PaLM papers.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Determine the intermediate size (8/3 is used in LLaMA for SwiGLU)
        intermediate_size = config.intermediate_size
        
        # Gate and up-projection
        self.w1 = nn.Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, intermediate_size, bias=config.bias)
        # Down-projection
        self.w2 = nn.Linear(intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: x = (xW1) * SiLU(xW3)
        return self.dropout(self.w2(self.w1(x) * F.silu(self.w3(x))))


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
    Transformer block with support for both PreLN and PostLN architecture
    and either LayerNorm or RMSNorm.
    """
    
    def __init__(self, config: GPTConfig, block_idx: int):
        super().__init__()
        self.pre_ln = config.pre_ln
        self.block_idx = block_idx
        
        # Choose normalization type based on config
        norm_class = RMSNorm if config.use_rms_norm else LayerNorm
        
        if config.use_rms_norm:
            # RMSNorm doesn't use bias
            self.ln_1 = norm_class(config.n_embd, eps=config.norm_eps)
            self.ln_2 = norm_class(config.n_embd, eps=config.norm_eps)
        else:
            # LayerNorm with configurable bias
            self.ln_1 = norm_class(config.n_embd, bias=config.bias, eps=config.norm_eps)
            self.ln_2 = norm_class(config.n_embd, bias=config.bias, eps=config.norm_eps)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:
            # PreLN architecture (used in LLaMA and most modern transformers)
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            # PostLN architecture (used in original transformer paper)
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        return x