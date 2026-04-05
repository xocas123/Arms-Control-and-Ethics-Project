#!/bin/bash
#SBATCH --job-name=arms-control-nlp
#SBATCH --partition=cmist
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xalvarez@andrew.cmu.edu

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail
PROJ="$HOME/arms-control-nlp"
cd "$PROJ"
mkdir -p logs output

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo "Working dir: $PROJ"
echo "=============================="

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arms-nlp

# Show GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Run pipeline ──────────────────────────────────────────────────────────────
# To run a single question, pass e.g.:  QUESTION=q1 sbatch submit.sh
# To run all questions, omit --question (default behaviour)
ARGS=""
[[ -n "${QUESTION:-}" ]] && ARGS="$ARGS --question $QUESTION"
[[ -n "${TREATY:-}"   ]] && ARGS="$ARGS --treaty $TREATY"
[[ -n "${FAST:-}"     ]] && ARGS="$ARGS --fast"

# Auto-resume: if shared checkpoint exists, skip completed stages
if [[ -f "output/shared/checkpoints/corpus_scored.parquet" ]]; then
    echo "[submit] Checkpoint found — running with --resume"
    ARGS="$ARGS --resume"
fi

python run.py $ARGS --year-start 1970 --year-end 2023

echo "=============================="
echo "Done : $(date)"
echo "=============================="
