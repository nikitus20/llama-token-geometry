#!/bin/bash

# This script runs token geometry analysis with the trained model

# Ensure tiktoken is installed
pip install tiktoken

# First, check if the trained model exists
if [ ! -d "saved_models/postln_wikitext_model" ] && [ ! -d "saved_models/tiktoken_wikitext_model" ]; then
    echo "No trained models found. Please run train_postln_model.sh first."
    exit 1
fi

# Determine which model to use (prefer tiktoken model if available)
MODEL_NAME="postln_wikitext_model"
if [ -d "saved_models/tiktoken_wikitext_model" ]; then
    MODEL_NAME="tiktoken_wikitext_model"
    echo "Using tiktoken model for analysis."
fi

# Create output directory
OUTPUT_DIR="outputs/token_geometry_tiktoken"
mkdir -p "$OUTPUT_DIR"

# Run analysis
echo "Running token geometry analysis..."
python scripts/analyze.py \
    --prompts data/prompts.txt \
    --output-dir "$OUTPUT_DIR" \
    --layers 6 \
    --trained-model "$MODEL_NAME" \
    --tokenizer tiktoken

echo "Analysis complete! Results saved to $OUTPUT_DIR"

# Generate a quick overview of the results
echo "Generated files:"
ls -la "$OUTPUT_DIR"