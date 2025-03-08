"""
Character-level tokenizer implementation.
"""

from typing import List
from src.tokenizer.base import BaseTokenizer

class CharTokenizer(BaseTokenizer):
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs - safely handle unicode by using only ASCII range."""
        return [min(ord(c), self.vocab_size-1) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        return ''.join([chr(t) for t in tokens if 0 < t < self.vocab_size])