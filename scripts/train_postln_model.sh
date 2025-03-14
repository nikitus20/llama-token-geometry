#!/bin/bash

# This script downloads Wikitext-2 and trains a PostLN model with perplexity evaluation
# With the updated codebase, it now supports different layer normalization architectures

# First, download Wikitext-2 if not already present
if [ ! -f "data/wikitext-2/train.txt" ]; then
    echo "Downloading Wikitext-2 dataset..."
    python scripts/download_wikitext.py
else
    echo "Wikitext-2 dataset already exists."
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

# Train PostLN model with Wikitext-2 and perplexity evaluation
echo "Training PostLN model with Wikitext-2..."
python scripts/train.py \
    --dataset wikitext \
    --data-dir data/wikitext-2 \
    --output-dir saved_models \
    --model-name postln_wikitext_model \
    --ln postln \
    --n-layer 6 \
    --n-head 6 \
    --n-embd 384 \
    --block-size 256 \
    --max-steps 1000 \
    --warmup-steps 100 \
    --save-interval 200 \
    --eval-interval 50 \
    --batch-size 8 \
    --learning-rate 5e-4 \
    --weight-decay 0.1 \
    --device $DEVICE

echo "Training complete!"