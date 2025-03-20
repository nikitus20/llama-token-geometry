"""
Simplified tokenizer component using HuggingFace's AutoTokenizer.
"""

from src.tokenizer.base import Tokenizer

# Provide backward compatibility for code that imports BaseTokenizer
BaseTokenizer = Tokenizer 