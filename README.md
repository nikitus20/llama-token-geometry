# Token Geometry Analyzer

This project analyzes token representations in transformer models with LLaMA-inspired architecture. It can compare Pre-Layer Normalization (PreLN) and Post-Layer Normalization (PostLN) architectures, as well as evaluate between standard transformer components and LLaMA-style components (RMSNorm, SwiGLU).

## Features

- Configurable transformer model with support for:
  - PreLN or PostLN architecture
  - RMSNorm normalization
  - SwiGLU or GELU activation functions

- Token geometry analysis showing how token representations evolve through network layers
- Advanced training capabilities:
  - Support for Wikitext-2 dataset with perplexity evaluation
  - Learning rate warmup and weight decay
  - Training progress visualization

- Visualization of token similarity matrices
- Comparative analysis between different architectures

## Requirements

```
torch>=2.0.0
numpy
matplotlib
seaborn
pandas
tqdm
tiktoken
datasets
```

## Installation

Install from source:

```bash
git clone https://github.com/yourusername/token_geometry_analyzer.git
cd token_geometry_analyzer
pip install -e .
```

## Quick Start Commands

Here are the main commands you can use to work with this project:

### Dataset Preparation

```bash
# Download and prepare the Wikitext-2 dataset
python scripts/download_wikitext.py
```
This command downloads the Wikitext-2 dataset from HuggingFace or directly from the source and prepares it for training.

### Training Models

```bash
# Train a model with default PostLN architecture
bash scripts/train_postln_model.sh
```
This script first checks if the Wikitext-2 dataset exists and downloads it if needed. Then it trains a PostLN model with the following configuration:
- 6 layers, 6 attention heads, 384 embedding dimension
- 256 block size, 1000 training steps, 100 warmup steps
- Evaluates perplexity every 50 steps and saves checkpoints every 200 steps

For more control over training parameters:

```bash
# Train with custom parameters
python scripts/train.py --dataset wikitext --data-dir data/wikitext-2 \
    --model-name my_custom_model --n-layer 8 --pre-ln --use-swiglu \
    --max-steps 2000 --warmup-steps 200
```

Available training options:
- `--pre-ln`: Use Pre-LN architecture (default is Post-LN)
- `--use-swiglu`: Use SwiGLU activation instead of GELU
- `--tokenizer`: Choose tokenizer (tiktoken, bpe, or char)

### Model Evaluation

```bash
# Evaluate perplexity of trained models on Wikitext-2 test set
bash scripts/evaluate_models.sh
```
This script evaluates the perplexity of all trained models in the saved_models directory on the Wikitext-2 test set.

### Token Geometry Analysis

```bash
# Run token geometry analysis
bash scripts/run_analysis.sh
```
This script runs token geometry analysis on both random models with different architectures and a trained model. It produces visualizations showing how token representations evolve through the network layers and creates heatmaps of token similarity.

For more control over analysis parameters:

```bash
python scripts/analyze.py \
    --prompts data/prompts.txt \
    --output-dir outputs/my_custom_analysis \
    --layers 6 \
    --trained-model my_custom_model \
    --tokenizer tiktoken
```

### Model Evolution Experiment

```bash
# Analyze how token geometry evolves during training
python scripts/model_evolution_experiment.py \
    --iterations 200 \
    --learning-rate 1e-5 \
    --output-dir outputs/my_custom_experiment
```
This script analyzes how token geometry evolves during training by:
1. Creating an initial model
2. Analyzing token geometry before training
3. Training for a specified number of iterations
4. Analyzing token geometry after training
5. Comparing the token geometry before and after training

## Model Architecture Options

The project supports several architectural variants:

- LLaMA-PreLN: RMSNorm + SwiGLU activation with Pre-LN architecture
- LLaMA-PostLN: RMSNorm + SwiGLU activation with Post-LN architecture
- Standard-PreLN: RMSNorm + GELU activation with Pre-LN architecture
- Standard-PostLN: RMSNorm + GELU activation with Post-LN architecture

## Interpreting Results

Token geometry analysis produces several outputs:

- Line plots showing average cosine similarity between token representations across layers
- Heatmaps showing pairwise token similarity matrices at different layers
- Difference plots showing how architectures differ in their token representations

Higher cosine similarity indicates tokens are becoming more similar in the representation space. The way this similarity evolves through layers provides insight into how the network processes information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.