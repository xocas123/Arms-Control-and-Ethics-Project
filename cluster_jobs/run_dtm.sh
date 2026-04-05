#!/bin/bash
#SBATCH --job-name=arms-nlp-dtm
#SBATCH --output=logs/dtm_%j.out
#SBATCH --error=logs/dtm_%j.err
#SBATCH --partition=cpu
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# Arms Control NLP Pipeline — Dynamic Topic Model (DTM) Step
# Uses gensim LdaSeqModel. CPU-only. Requires 48GB RAM for full corpus.

set -euo pipefail

source ~/.bashrc
conda activate arms-nlp

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Arms Control NLP: Dynamic Topic Model ==="
echo "Project dir: $PROJECT_DIR"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
date

python - <<'EOF'
import sys
sys.path.insert(0, '.')

from src.config import load_config
from src.data.load_ungdc import load_ungdc
from src.data.segment import segment_by_keywords
from src.analysis.topics_dtm import train_dtm, get_dtm_topic_evolution

cfg = load_config()

corpus_df = load_ungdc(
    data_dir=str(cfg.data_dir),
    synthetic_mode=cfg.synthetic_mode,
    year_start=cfg.year_start,
    year_end=cfg.year_end,
)
segments_df = segment_by_keywords(corpus_df, window_size=cfg.keyword_window_size)
print(f"Segments for DTM: {len(segments_df):,}")

dtm_results = train_dtm(
    segments_df,
    n_topics=cfg.dtm_n_topics,
    random_seed=cfg.random_seed,
    passes=5,
)

if dtm_results is not None:
    out_dir = cfg.output_dir / 'topics'
    out_dir.mkdir(parents=True, exist_ok=True)

    evolution_df = get_dtm_topic_evolution(dtm_results)
    evolution_df.to_csv(out_dir / 'dtm_topic_evolution.csv', index=False)
    print(f"DTM topic evolution: {len(evolution_df)} rows")

    # Save model
    import pickle
    model_path = out_dir / 'dtm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'time_slices': dtm_results['time_slices'], 'years': dtm_results['years']}, f)
    dtm_results['model'].save(str(out_dir / 'dtm_gensim_model'))
    print(f"DTM model saved to {out_dir}")
else:
    print("DTM training failed.")

print("DTM job complete.")
EOF

echo "=== DTM job finished at $(date) ==="
