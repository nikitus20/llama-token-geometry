#!/bin/bash

# This script runs token geometry analysis with the trained model

# First, check if the trained model exists
if [ ! -d "saved_models/postln_model_warmup" ]; then
    echo "Trained model not found. Please run train_postln_model.sh first."
    exit 1
fi

# Run analysis
echo "Running token geometry analysis..."
python scripts/analyze.py \
    --prompts data/prompts.txt \
    --output-dir outputs/token_geometry_analysis \
    --layers 6 \
    --trained-model postln_model_warmup

echo "Analysis complete! Results saved to outputs/token_geometry_analysis"