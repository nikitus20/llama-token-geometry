"""
Script to download and prepare Wikitext-2 dataset for language model training.
"""

import os
import sys
import argparse
import logging

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_wikitext_huggingface(output_dir='data/wikitext-2'):
    """
    Download Wikitext-2 dataset using HuggingFace datasets library.
    
    Args:
        output_dir: Directory to save the dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("HuggingFace datasets library not found. Install it with 'pip install datasets'")
        return False
    
    logger.info("Downloading Wikitext-2 dataset from HuggingFace...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and save train, validation, and test splits
    for split in ['train', 'validation', 'test']:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            
            # Convert to text file format
            output_file = os.path.join(output_dir, f"{split}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(item['text'] + '\n')
            
            logger.info(f"Saved {split} split to {output_file}")
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")
            return False
    
    logger.info(f"Wikitext-2 dataset downloaded and saved to {output_dir}")
    return True


def download_wikitext_direct(output_dir='data/wikitext-2'):
    """
    Download Wikitext-2 dataset directly from source.
    
    Args:
        output_dir: Directory to save the dataset
    """
    import requests
    import zipfile
    from io import BytesIO
    
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    
    logger.info(f"Downloading Wikitext-2 from {url}...")
    
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('data/')
        
        # Rename and organize files
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Map the original files to our desired structure
        file_mapping = {
            'wikitext-2/wiki.train.tokens': f'{output_dir}/train.txt',
            'wikitext-2/wiki.valid.tokens': f'{output_dir}/validation.txt',
            'wikitext-2/wiki.test.tokens': f'{output_dir}/test.txt'
        }
        
        for src, dest in file_mapping.items():
            if os.path.exists(f'data/{src}'):
                # Move/rename file
                os.rename(f'data/{src}', dest)
                logger.info(f"Saved {os.path.basename(dest)} to {dest}")
        
        logger.info(f"Wikitext-2 dataset downloaded and saved to {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False


def main():
    """Main function to download and prepare the dataset."""
    parser = argparse.ArgumentParser(description='Download and prepare Wikitext-2 dataset')
    parser.add_argument('--output-dir', type=str, default='data/wikitext-2',
                       help='Directory to save the dataset (default: data/wikitext-2)')
    parser.add_argument('--method', type=str, choices=['huggingface', 'direct'], default='huggingface',
                       help='Download method (default: huggingface)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download the dataset
    if args.method == 'huggingface':
        success = download_wikitext_huggingface(args.output_dir)
    else:
        success = download_wikitext_direct(args.output_dir)
    
    if success:
        logger.info("Dataset preparation complete!")
        # Save a small metadata file with information about the dataset
        with open(os.path.join(args.output_dir, 'dataset_info.txt'), 'w') as f:
            f.write("Dataset: Wikitext-2\n")
            f.write("Source: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/\n")
            f.write("Download date: " + os.popen('date').read())
    else:
        logger.error("Dataset preparation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()