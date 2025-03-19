"""
Hugging Face AutoTokenizer implementation compatible with the project's tokenizer interface.
"""

import logging
from typing import List
from src.tokenizer.base import BaseTokenizer

logger = logging.getLogger(__name__)

class HuggingFaceTokenizer(BaseTokenizer):
    """
    Tokenizer using Hugging Face's AutoTokenizer library.
    Implements the BaseTokenizer interface for compatibility.
    """
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize HuggingFace tokenizer.
        
        Args:
            model_name: Model name or path for AutoTokenizer
        """
        try:
            from transformers import AutoTokenizer
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception as e:
                if "gated repo" in str(e) or "Access to model" in str(e) or "access is restricted" in str(e):
                    error_msg = f"Cannot access {model_name} as it's a gated model requiring authentication. Try using an open model like 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' or 'mistralai/Mistral-7B-v0.1' instead."
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
                else:
                    # Re-raise other exceptions
                    raise
            
            # Add padding token if not already present (especially for Llama models)
            if self.tokenizer.pad_token is None:
                logger.info(f"No padding token found in {model_name} tokenizer, adding one")
                # Use the eos_token as the pad_token if available, otherwise create a new one
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # Add a new token for padding
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    # Note: Model embeddings would need to be resized when using this tokenizer for training
            
            self.vocab_size = len(self.tokenizer.get_vocab())
            logger.info(f"Initialized HuggingFace tokenizer with model {model_name}, vocab size: {self.vocab_size}")
        except ImportError:
            logger.error("transformers library not found. Please install it with: pip install transformers")
            raise ImportError("Please install the transformers library with: pip install transformers")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace tokenizer: {e}")
            raise
    
    @property
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        try:
            # Don't add special tokens by default to match behavior of other tokenizers
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Error encoding text with HuggingFace tokenizer: {e}")
            # Fallback to empty list with error indicator
            return [0]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        try:
            return self.tokenizer.decode(tokens)
        except Exception as e:
            logger.error(f"Error decoding tokens with HuggingFace tokenizer: {e}")
            return ""