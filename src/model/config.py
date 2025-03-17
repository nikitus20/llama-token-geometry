"""
Configuration classes for transformer models.
"""

from dataclasses import dataclass
from typing import Literal, Union

# Update src/model/config.py

@dataclass
class GPTConfig:
    """Configuration for the GPT model with LLaMA-style architecture options."""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocabulary size (50257 rounded up)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears, False: a bit better and faster
    ln: Union[Literal["preln", "postln", "periln", "mixln", "predyt", "postdyt"], str] = "preln"  # Normalization architecture type
    use_initial_ln: bool = True  # Whether to apply normalization after embeddings
    use_swiglu: bool = True  # Use SwiGLU activation (LLaMA-style) instead of GELU
    intermediate_size: int = None  # Size of the intermediate layer in MLP, None = auto-compute based on n_embd
    norm_eps: float = 1e-6  # Epsilon value for normalization layers
    mixln_split: float = 0.25  # Fraction of layers to use postln in mixln architecture
    initializer_range: float = 0.02  # Standard deviation for weight initialization
    scale_attn_weights: bool = False
    scale_mlp_output: bool = False
    deeppost: bool = False
    
    # Added LLaMA-specific parameters
    max_position_embeddings: int = 2048  # Maximum sequence length for rotary embeddings
    rope_base: int = 10000  # Base for rotary embeddings
    
    @property
    def pre_ln(self) -> bool:
        """Backwards compatibility property for pre_ln."""
        return self.ln == "preln"
    
    def __post_init__(self):
        """Auto-compute intermediate size if not provided."""
        if self.intermediate_size is None:
            # If using SwiGLU, use LLaMA ratio of 8/3, else use 4x like in standard transformers
            self.intermediate_size = int(8/3 * self.n_embd) if self.use_swiglu else 4 * self.n_embd