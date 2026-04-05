#!/bin/bash
#SBATCH --job-name=arms-nlp-bertopic
#SBATCH --output=logs/bertopic_%j.out
#SBATCH --error=logs/bertopic_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --dependency=afterok:${EMBEDDING_JOB_ID}

# Arms Control NLP Pipeline — BERTopic Step
# Requires: bertopic, sentence-transformers, umap-learn, hdbscan, GPU 64GB RAM

set -euo pipefail

source ~/.bashrc
conda activate arms-nlp

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Arms Control NLP: BERTopic ==="
echo "Project dir: $PROJECT_DIR"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Dependency job: ${EMBEDDING_JOB_ID:-none}"
date

python - <<'EOF'
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import load_config
from src.data.load_ungdc import load_ungdc
from src.data.segment import segment_by_keywords
from src.analysis.topics_bertopic import train_bertopic, save_bertopic_model, get_topics_over_time

cfg = load_config()

# Load pre-computed embeddings if available
emb_path = cfg.output_dir / 'embeddings' / 'segment_embeddings.npy'
idx_path = cfg.output_dir / 'embeddings' / 'segment_index.parquet'

corpus_df = load_ungdc(
    data_dir=str(cfg.data_dir),
    synthetic_mode=cfg.synthetic_mode,
    year_start=cfg.year_start,
    year_end=cfg.year_end,
)
segments_df = segment_by_keywords(corpus_df, window_size=cfg.keyword_window_size)
print(f"Segments: {len(segments_df):,}")

embeddings = None
if emb_path.exists():
    print(f"Loading pre-computed embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    print(f"Embeddings shape: {embeddings.shape}")
    # Align with current segments_df if sizes differ
    if len(embeddings) != len(segments_df):
        print(f"Warning: embedding count ({len(embeddings)}) != segment count ({len(segments_df)}); recomputing...")
        embeddings = None

print("Training BERTopic...")
results = train_bertopic(
    segments_df,
    embeddings=embeddings,
    min_topic_size=cfg.bertopic_min_topic_size,
)

if results is not None:
    out_dir = cfg.output_dir / 'topics'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bertopic_model(results['model'], out_dir / 'bertopic_model')
    results['topic_info'].to_csv(out_dir / 'bertopic_topic_info.csv', index=False)
    print(f"BERTopic: {len(results['topic_info'])-1} topics found")

    # Topics over time
    tot = get_topics_over_time(results['model'], segments_df)
    if tot is not None:
        tot.to_csv(out_dir / 'bertopic_topics_over_time.csv', index=False)
        print(f"Topics over time: {len(tot)} rows")
else:
    print("BERTopic training failed or returned None")

print("BERTopic job complete.")
EOF

echo "=== BERTopic job finished at $(date) ==="
