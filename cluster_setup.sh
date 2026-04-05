#!/bin/bash
# Run once on the Wright head node to set up the conda environment.
# Usage: bash cluster_setup.sh
set -eu

ENV_NAME="arms-nlp"
PROJ="$HOME/arms-control-nlp"

echo "[1/4] Creating conda environment: $ENV_NAME"
conda create -y -n "$ENV_NAME" python=3.11

echo "[2/4] Installing PyTorch with CUDA (for L40 / CUDA 12.x)"
conda run -n "$ENV_NAME" pip install \
  torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[3/4] Installing pipeline requirements"
conda run -n "$ENV_NAME" pip install -r "$PROJ/requirements.txt"

echo "[4/4] Downloading NLTK data (vader_lexicon)"
conda run -n "$ENV_NAME" python -c "
import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
print('NLTK data ready.')
"

echo "========================================"
echo "Setup complete. Test with:"
echo "  conda activate $ENV_NAME"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo "========================================"
