"""
Hugging Face AutoTokenizer implementation compatible with the project's tokenizer interface.
"""

import logging
import os
import shutil
from typing import List, Optional
from src.tokenizer.base import BaseTokenizer

logger = logging.getLogger(__name__)

class HuggingFaceTokenizer(BaseTokenizer):
    """
    Tokenizer using Hugging Face's AutoTokenizer library.
    Implements the BaseTokenizer interface for compatibility.
    """
    
    def __init__(self, model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_files_only: bool = False):
        """
        Initialize HuggingFace tokenizer.
        
        Args:
            model_name_or_path: Model name or path for AutoTokenizer. 
                                Can be a HuggingFace Hub model ID or a local path to a saved tokenizer.
            local_files_only: Whether to use only local files (no downloads).
                              Set to True when using a local tokenizer path.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("transformers library not found. Please install it with: pip install transformers")
            raise ImportError("Please install the transformers library with: pip install transformers")
        
        # Set environment variable to disable file locking (helpful for some systems)
        os.environ["HF_HUB_DISABLE_FILE_LOCKING"] = "1"
        
        # Check if the path is a local directory and exists
        is_local_path = os.path.isdir(model_name_or_path)
        if is_local_path and not local_files_only:
            logger.info(f"Detected local tokenizer path: {model_name_or_path}. Setting local_files_only=True")
            local_files_only = True
        
        if local_files_only and not is_local_path:
            logger.warning(f"local_files_only=True but path doesn't appear to be a directory: {model_name_or_path}")
            logger.warning("Will still attempt to load, but this might fail if it requires downloading from HuggingFace Hub")
            
        try:
            logger.info(f"Loading tokenizer from {'local path' if local_files_only else 'model name'}: {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                use_fast=True,
                local_files_only=local_files_only
            )
            
            # If we successfully loaded a remote model, save it locally for future offline use
            if not local_files_only and not is_local_path:
                self._save_tokenizer_locally(model_name_or_path)
                
        except Exception as e:
            if "gated repo" in str(e) or "Access to model" in str(e) or "access is restricted" in str(e):
                error_msg = f"Cannot access {model_name_or_path} as it's a gated model requiring authentication. Try using an open model like 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' or 'mistralai/Mistral-7B-v0.1' instead."
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            
            if local_files_only:
                logger.error(f"Failed to load local tokenizer from {model_name_or_path}. Check that the path exists and contains valid tokenizer files.")
            else:
                logger.error(f"Failed to load tokenizer model {model_name_or_path}. If you're working offline, try using a local tokenizer path.")
            
            logger.error(f"Error details: {e}")
            raise
            
        # Add padding token if not already present (especially for Llama models)
        if self.tokenizer.pad_token is None:
            logger.info(f"No padding token found in {model_name_or_path} tokenizer, adding one")
            # Use the eos_token as the pad_token if available, otherwise create a new one
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add a new token for padding
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Note: Model embeddings would need to be resized when using this tokenizer for training
        
        self.vocab_size = len(self.tokenizer.get_vocab())
        logger.info(f"Initialized HuggingFace tokenizer with model {model_name_or_path}, vocab size: {self.vocab_size}")
    
    def _save_tokenizer_locally(self, model_name: str) -> None:
        """
        Save the tokenizer locally for future offline use.
        
        Args:
            model_name: The name of the model used to create the tokenizer
        """
        try:
            # Create directory structure
            local_dir = os.path.join('tokenizers', model_name.split('/')[-1].lower())
            os.makedirs(local_dir, exist_ok=True)
            
            # Save tokenizer to local directory
            self.tokenizer.save_pretrained(local_dir)
            logger.info(f"Saved tokenizer to {local_dir} for future offline use")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer locally: {e}")
    
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
    
    @classmethod
    def save_remote_tokenizer_locally(cls, model_name: str, save_path: Optional[str] = None) -> str:
        """
        Utility method to download and save a remote tokenizer for offline use.
        
        Args:
            model_name: HuggingFace model name to download
            save_path: Optional custom save path
            
        Returns:
            Path where the tokenizer was saved
        """
        try:
            from transformers import AutoTokenizer
            
            # Determine save path
            if save_path is None:
                save_path = os.path.join('tokenizers', model_name.split('/')[-1].lower())
            
            # Create directory
            os.makedirs(save_path, exist_ok=True)
            
            # Download and save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            tokenizer.save_pretrained(save_path)
            
            logger.info(f"Successfully downloaded and saved tokenizer from {model_name} to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving remote tokenizer {model_name}: {e}")
            raise