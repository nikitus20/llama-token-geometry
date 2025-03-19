"""
Tiktoken tokenizer implementation.
"""

import os
import logging
from typing import List

from src.tokenizer.base import BaseTokenizer

logger = logging.getLogger(__name__)

class TiktokenTokenizer(BaseTokenizer):
    """
    Tokenizer using OpenAI's tiktoken library.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize Tiktoken tokenizer.
        
        Args:
            model_name: Model name or encoding to use (e.g., "gpt2", "cl100k_base")
        """
        try:
            import tiktoken
            self.tiktoken = tiktoken
            self.encoding = tiktoken.get_encoding(model_name)
            self.vocab_size = self.encoding.n_vocab
            logger.info(f"Initialized Tiktoken tokenizer with model {model_name}, vocab size: {self.vocab_size}")
        except ImportError:
            logger.error("Tiktoken not found. Please install it with 'pip install tiktoken'")
            raise
        except Exception as e:
            logger.error(f"Error initializing Tiktoken tokenizer: {e}")
            raise
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        try:
            return self.encoding.encode(text)
        except Exception as e:
            logger.error(f"Error encoding text with Tiktoken: {e}")
            # Fallback to empty list with error indicator
            return [0]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        try:
            return self.encoding.decode(tokens)
        except Exception as e:
            logger.error(f"Error decoding tokens with Tiktoken: {e}")
            return ""
            
    @property
    def pad_token_id(self) -> int:
        """
        Return the padding token ID.
        For tiktoken, we'll use a value just past the end of the vocab as padding.
        """
        # Return a token just past the end of the vocabulary
        return self.vocab_size