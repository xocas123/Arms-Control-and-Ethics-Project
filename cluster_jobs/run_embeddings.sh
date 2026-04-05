#!/bin/bash
#SBATCH --job-name=arms-nlp-embeddings
#SBATCH --output=logs/embeddings_%j.out
#SBATCH --error=logs/embeddings_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

# Arms Control NLP Pipeline — Embedding Generation Step
# Requires sentence-transformers and a GPU for efficient encoding.

set -euo pipefail

# Activate environment
source ~/.bashrc
conda activate arms-nlp  # or: source /path/to/venv/bin/activate

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Arms Control NLP: Embedding Generation ==="
echo "Project dir: $PROJECT_DIR"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

# Check GPU availability
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "PyTorch not available"

# Run embedding generation
python - <<'EOF'
import sys
sys.path.insert(0, '.')

from src.config import load_config
from src.data.load_ungdc import load_ungdc
from src.data.load_treaties import load_treaty_anchors
from src.data.segment import segment_by_keywords
from src.analysis.embeddings import (
    embed_texts,
    embed_treaty_anchors,
    compute_country_year_anchor_scores,
)
from src.utils import save_parquet
from pathlib import Path
import numpy as np

cfg = load_config()
print(f"Config: synthetic_mode={cfg.synthetic_mode}")

# Load corpus
print("Loading corpus...")
corpus_df = load_ungdc(
    data_dir=str(cfg.data_dir),
    synthetic_mode=cfg.synthetic_mode,
    year_start=cfg.year_start,
    year_end=cfg.year_end,
)
print(f"Corpus: {len(corpus_df):,} rows")

# Segment
print("Running keyword segmentation...")
segments_df = segment_by_keywords(corpus_df, window_size=cfg.keyword_window_size)
print(f"Segments: {len(segments_df):,}")

# Embed treaty anchors
print("Embedding treaty anchors...")
treaty_anchors = load_treaty_anchors()
anchor_embeddings = embed_treaty_anchors(treaty_anchors, model_name=cfg.embedding_model)
print(f"Anchors embedded: {len(anchor_embeddings)}")

# Embed corpus and compute anchor similarity
print(f"Computing anchor similarity for {len(segments_df):,} segments...")
anchor_scores = compute_country_year_anchor_scores(
    segments_df,
    anchor_embeddings,
    model_name=cfg.embedding_model,
)
print(f"Anchor scores: {len(anchor_scores)} country-year rows")

# Save
out_dir = cfg.output_dir / 'embeddings'
out_dir.mkdir(parents=True, exist_ok=True)
anchor_scores.to_csv(out_dir / 'anchor_scores.csv', index=False)
print(f"Saved anchor scores to {out_dir / 'anchor_scores.csv'}")

# Also save full segment embeddings for BERTopic
print("Embedding all segments (this may take a while)...")
texts = segments_df['text'].fillna('').tolist()
embs = embed_texts(texts, model_name=cfg.embedding_model, batch_size=128)
np.save(out_dir / 'segment_embeddings.npy', embs)
segments_df[['country_code', 'year']].to_parquet(out_dir / 'segment_index.parquet', index=False)
print(f"Saved segment embeddings: shape {embs.shape}")
print("Embedding generation complete.")
EOF

echo "=== Embedding job finished at $(date) ==="
