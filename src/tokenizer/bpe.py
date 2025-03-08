"""
BPE tokenizer implementation.
"""

import os
import json
import logging
from typing import List, Dict

from src.tokenizer.base import BaseTokenizer

logger = logging.getLogger(__name__)

class BPETokenizer(BaseTokenizer):
    """
    Simple wrapper for the BPE tokenizer using encoder.json and vocab.bpe files.
    This is a simplified implementation for analysis purposes.
    """
    
    def __init__(self, encoder_json_path: str = 'data/embedding/encoder.json', 
                 vocab_bpe_path: str = 'data/embedding/vocab.bpe'):
        try:
            with open(encoder_json_path, 'r', encoding='utf-8') as f:
                self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}
            
            # Load BPE merge rules if available
            self.bpe_ranks = {}
            if os.path.exists(vocab_bpe_path):
                with open(vocab_bpe_path, 'r', encoding='utf-8') as f:
                    merges = f.read().split('\n')[1:-1]
                merges = [tuple(merge.split()) for merge in merges]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
                
            self.vocab_size = len(self.encoder)
            logger.info(f"Loaded BPE tokenizer with vocab size: {self.vocab_size}")
        except Exception as e:
            logger.error(f"Error loading BPE tokenizer: {e}")
            raise
        
    def encode(self, text: str) -> List[int]:
        """
        Simplified encoding - works for basic analysis but not perfect.
        A more robust implementation would implement proper BPE.
        """
        tokens = []
        # Add space prefix following GPT-2 convention
        text = ' ' + text.replace('\n', ' ')
        
        # Simple word-level tokenization for analysis purposes
        words = text.split()
        for word in words:
            # Try whole word first
            if word in self.encoder:
                tokens.append(self.encoder[word])
            else:
                # Fall back to character-level
                for char in word:
                    char_token = self.encoder.get(char, self.encoder.get('?', 0))
                    tokens.append(char_token)
                    
        return tokens if tokens else [0]  # Return at least one token
        
    def decode(self, token_ids: List[int]) -> str:
        """Simple decoding by joining tokens."""
        text_tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                text_tokens.append(self.decoder[token_id])
        return ''.join(text_tokens)