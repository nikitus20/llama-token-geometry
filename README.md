oken Geometry Analyzer
This project analyzes token representations in transformer models with LLaMA-inspired architecture. It can compare Pre-Layer Normalization (PreLN) and Post-Layer Normalization (PostLN) architectures, as well as evaluate between standard transformer components and LLaMA-style components (RMSNorm, SwiGLU).
Show Image
Note: Example visualization showing token geometry across different model architectures.
Features

Configurable transformer model with support for:

PreLN or PostLN architecture
RMSNorm or LayerNorm
SwiGLU or GELU activation functions


Token geometry analysis showing how token representations evolve through network layers
Advanced training capabilities:

Support for Wikitext-2 dataset with perplexity evaluation
Learning rate warmup and weight decay
Training progress visualization


Visualization of token similarity matrices
Comparative analysis between different architectures

Project Structure
Copytoken_geometry_analyzer/
├── src/
│   ├── model/          - Model classes and utilities
│   ├── tokenizer/      - Tokenizer implementations 
│   ├── analyzer/       - Geometry analysis tools
│   └── utils/          - Data and utilities
├── scripts/
│   ├── train.py        - Training script
│   └── analyze.py      - Analysis script
├── data/
│   ├── embedding/      - Tokenizer files
│   └── prompts.txt     - Sample prompts
└── saved_models/       - Directory for saved models
Requirements
Copytorch>=2.0.0
numpy
matplotlib
seaborn
pandas
tqdm
Usage
Training a Model
To train a PostLN model with a small warmup stage:
bashCopypython scripts/train.py \
    --text-file data/your_corpus.txt \
    --model-name postln_model_warmup \
    --warmup-steps 50 \
    --max-steps 500 \
    --n-layer 6
Note: By default, this creates a PostLN architecture. Use --pre-ln to create a PreLN model instead.
Running Token Geometry Analysis
Analyze token geometry with both random and trained models:
bashCopypython scripts/analyze.py \
    --prompts data/prompts.txt \
    --output-dir outputs/token_geometry_analysis \
    --layers 6 \
    --trained-model postln_model_warmup
Model Architecture Options
The project supports several architectural variants:

LLaMA-PreLN: RMSNorm + SwiGLU activation with Pre-LN architecture
LLaMA-PostLN: RMSNorm + SwiGLU activation with Post-LN architecture
Standard-PreLN: LayerNorm + GELU activation with Pre-LN architecture
Standard-PostLN: LayerNorm + GELU activation with Post-LN architecture

Interpreting Results
Token geometry analysis produces several outputs:

Line plots showing average cosine similarity between token representations across layers
Heatmaps showing pairwise token similarity matrices at different layers
Difference plots showing how architectures differ in their token representations

Higher cosine similarity indicates tokens are becoming more similar in the representation space. The way this similarity evolves through layers provides insight into how the network processes information.
Example Outputs
Token Similarity Comparison
Show Image
The above graph shows how token similarity evolves across layers in different architectures. Notice how:

PreLN architectures generally maintain higher token distinctiveness
Trained models show different patterns than randomly initialized ones
LLaMA-style architectures with SwiGLU show unique patterns of token geometry

Similarity Heatmaps
Show Image
Heatmaps visualize the pairwise similarity between tokens at a specific layer. Brighter colors indicate higher similarity.
Training Progress
Show Image
During training, the system tracks both loss and perplexity, providing visualizations of how the model improves over time. Lower perplexity indicates better language modeling performance.
Installation
Install from source:
bashCopygit clone https://github.com/yourusername/token_geometry_analyzer.git
cd token_geometry_analyzer
pip install -e .
Quick Start
Training with Wikitext-2

Download the Wikitext-2 dataset and train a model:

bashCopybash scripts/train_postln_model.sh

Evaluate perplexity on the test set:

bashCopybash scripts/evaluate_models.sh
Token Geometry Analysis

Run token geometry analysis with the trained model:

bashCopybash scripts/run_analysis.sh

View results in the outputs/token_geometry_analysis directory.

Manual Training
For more control over the training process:
bashCopy# Download Wikitext-2 dataset
python scripts/download_wikitext.py

# Train with custom parameters
python scripts/train.py --dataset wikitext --data-dir data/wikitext-2 \
    --model-name my_custom_model --n-layer 8 --pre-ln --use-swiglu \
    --max-steps 2000 --warmup-steps 200
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.