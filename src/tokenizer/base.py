"""
Simplified tokenizer implementation using HuggingFace's AutoTokenizer.
"""

from typing import List
import logging
import os
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class Tokenizer:
    """HuggingFace tokenizer wrapper with a simple interface."""
    
    def __init__(self, model_dir: str = "tokenizer/"):
        """
        Initialize HuggingFace tokenizer from a local directory.
        
        Args:
            model_dir: Path to local tokenizer directory
        """
        try:
            logger.info(f"Loading tokenizer from local directory: {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            
            # Add padding token if not already present (especially for Llama models)
            if self.tokenizer.pad_token is None:
                logger.info(f"No padding token found in tokenizer, adding one")
                # Use the eos_token as the pad_token if available, otherwise create a new one
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # Add a new token for padding
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            self.vocab_size = len(self.tokenizer.get_vocab())
            logger.info(f"Initialized HuggingFace tokenizer, vocab size: {self.vocab_size}")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer from {model_dir}: {e}")
            raise
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        try:
            # Don't add special tokens by default to match behavior of other tokenizers
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Error encoding text with tokenizer: {e}")
            # Fallback to empty list with error indicator
            return [0]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        try:
            return self.tokenizer.decode(tokens)
        except Exception as e:
            logger.error(f"Error decoding tokens with tokenizer: {e}")
            return ""
    
    @property
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id