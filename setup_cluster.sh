#!/bin/bash
# Cluster environment setup script for the Arms Control NLP pipeline.
# Run once on the cluster login node to create the conda environment.
#
# Usage:
#   bash setup_cluster.sh           # full setup with GPU dependencies
#   bash setup_cluster.sh --cpu     # CPU-only (no torch/sentence-transformers)

set -euo pipefail

ENV_NAME="arms-nlp"
CPU_ONLY=false
if [[ "${1:-}" == "--cpu" ]]; then
    CPU_ONLY=true
    echo "CPU-only mode: skipping GPU/torch dependencies"
fi

echo "=== Arms Control NLP Pipeline — Cluster Setup ==="
echo "Creating conda environment: $ENV_NAME"
date

# Create environment
conda create -n "$ENV_NAME" python=3.10 -y

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Core dependencies
echo "Installing core dependencies..."
pip install --upgrade pip
pip install \
    pandas>=2.0 \
    numpy>=1.24 \
    matplotlib>=3.7 \
    seaborn>=0.12 \
    plotly>=5.0 \
    scipy>=1.10 \
    scikit-learn>=1.3 \
    gensim>=4.3 \
    nltk>=3.8 \
    pyarrow>=12.0 \
    pycountry>=22.3 \
    pyyaml>=6.0 \
    networkx>=3.0 \
    tqdm>=4.65 \
    joblib>=1.3 \
    requests>=2.31 \
    beautifulsoup4>=4.12 \
    vaderSentiment>=3.3

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

if ! $CPU_ONLY; then
    echo "Installing GPU/heavy NLP dependencies..."

    # PyTorch (CUDA 11.8 — adjust cuda version for your cluster)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Sentence transformers
    pip install sentence-transformers>=2.2

    # UMAP + HDBSCAN for BERTopic
    pip install umap-learn>=0.5 hdbscan>=0.8

    # BERTopic
    pip install bertopic>=0.15

    echo "GPU dependencies installed."
fi

# spaCy (optional — for advanced NER)
pip install spacy>=3.6
python -m spacy download en_core_web_sm || echo "spaCy model download failed (optional)"

# Pre-download sentence transformer model
if ! $CPU_ONLY; then
    python -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model downloaded and cached.')
"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Run pipeline: python run_pipeline.py"
echo ""
echo "Environment details:"
conda env export --name "$ENV_NAME" | head -30
