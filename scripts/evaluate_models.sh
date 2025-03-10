#!/bin/bash

# This script evaluates perplexity of trained models on Wikitext-2 test set

# Check if directories exist
if [ ! -d "saved_models" ]; then
    echo "No saved models found. Please train a model first."
    exit 1
fi

if [ ! -f "data/wikitext-2/test.txt" ]; then
    echo "Wikitext-2 test set not found. Running download script..."
    python scripts/download_wikitext.py
fi

# Determine the best device to use
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    DEVICE="cuda"
elif python -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())" | grep -q "True"; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "Using device: $DEVICE"

# Evaluate all models in the saved_models directory
echo "Evaluating models..."

for model_dir in saved_models/*; do
    if [ -d "$model_dir" ] && [ -f "$model_dir/model.pt" ]; then
        model_name=$(basename "$model_dir")
        echo "Evaluating model: $model_name"
        
        python scripts/evaluate_perplexity.py \
            --model "$model_dir" \
            --dataset wikitext \
            --data-dir data/wikitext-2 \
            --split test \
            --batch-size 16 \
            --use-bpe \
            --device $DEVICE
    fi
done

echo "Evaluation complete!"