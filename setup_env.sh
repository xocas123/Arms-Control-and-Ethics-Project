#!/bin/bash
#SBATCH --job-name=arms-nlp-setup
#SBATCH --partition=cmist
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xalvarez@andrew.cmu.edu

set -eu
PROJ="$HOME/arms-control-nlp"
cd "$PROJ"
mkdir -p logs

echo "=============================="
echo "Setting up arms-nlp environment"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=============================="

ENV_NAME="arms-nlp"

echo "[1/4] Creating conda environment: $ENV_NAME"
conda create -y -n "$ENV_NAME" python=3.11

echo "[2/4] Installing PyTorch with CUDA"
conda run -n "$ENV_NAME" pip install \
  torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[3/4] Installing pipeline requirements"
conda run -n "$ENV_NAME" pip install -r "$PROJ/requirements.txt"

echo "[4/4] Downloading NLTK data"
conda run -n "$ENV_NAME" python -c "
import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
print('NLTK data ready.')
"

echo "=============================="
echo "Done: $(date)"
echo "To submit the pipeline: sbatch submit.sh"
echo "=============================="
