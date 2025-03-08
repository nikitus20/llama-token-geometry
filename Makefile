.PHONY: setup test train analyze clean

# Default target
all: setup train analyze

# Setup project structure and data
setup:
	@echo "Setting up project..."
	python scripts/prepare_training_data.py

# Run tests
test:
	@echo "Running tests..."
	python scripts/test.py

# Train a model
train:
	@echo "Training model..."
	bash scripts/train_postln_model.sh

# Run analysis
analyze:
	@echo "Running analysis..."
	bash scripts/run_analysis.sh

# Install development requirements
dev-install:
	@echo "Installing in development mode..."
	pip install -e .

# Clean outputs and data
clean:
	@echo "Cleaning generated data..."
	rm -f data/training_corpus.txt
	rm -rf outputs/token_geometry_analysis

# Clean everything including models
clean-all: clean
	@echo "Cleaning models..."
	rm -rf saved_models/*
	touch saved_models/.gitkeep