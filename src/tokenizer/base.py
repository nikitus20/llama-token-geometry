"""
Base tokenizer class.
"""

from typing import List

class BaseTokenizer:
    """Base class for tokenizers."""
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        raise NotImplementedError
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        raise NotImplementedError