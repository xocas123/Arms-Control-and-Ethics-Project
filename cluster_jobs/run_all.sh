#!/bin/bash
# Master SLURM submission script for the Arms Control NLP pipeline.
# Submits jobs with proper dependency ordering.
#
# Usage:
#   bash cluster_jobs/run_all.sh
#   bash cluster_jobs/run_all.sh --dry-run   # print commands only

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN MODE] Commands will be printed but not executed."
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"

echo "=== Arms Control NLP Pipeline — Full Cluster Submission ==="
echo "Project: $PROJECT_DIR"
date

submit() {
    local script="$1"
    local deps="${2:-}"
    local env_vars="${3:-}"

    local cmd="sbatch"
    if [[ -n "$deps" ]]; then
        cmd="$cmd --dependency=afterok:$deps"
    fi
    if [[ -n "$env_vars" ]]; then
        cmd="$cmd --export=ALL,$env_vars"
    fi
    cmd="$cmd $script"

    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
        echo "12345"  # fake job ID
        return
    fi

    local output
    output=$($cmd)
    echo "$output"
    echo "$output" | grep -oP '(?<=job )\d+'
}

# Job 1: Embeddings (GPU)
echo ""
echo "Submitting embeddings job..."
EMB_OUTPUT=$(submit "$PROJECT_DIR/cluster_jobs/run_embeddings.sh")
EMB_JOB_ID=$(echo "$EMB_OUTPUT" | tail -1)
echo "  Embeddings job ID: $EMB_JOB_ID"

# Job 2: BERTopic (GPU, depends on embeddings)
echo ""
echo "Submitting BERTopic job (depends on embeddings job $EMB_JOB_ID)..."
BT_OUTPUT=$(submit \
    "$PROJECT_DIR/cluster_jobs/run_bertopic.sh" \
    "$EMB_JOB_ID" \
    "EMBEDDING_JOB_ID=$EMB_JOB_ID")
BT_JOB_ID=$(echo "$BT_OUTPUT" | tail -1)
echo "  BERTopic job ID: $BT_JOB_ID"

# Job 3: DTM (CPU, independent of GPU jobs)
echo ""
echo "Submitting DTM job (independent)..."
DTM_OUTPUT=$(submit "$PROJECT_DIR/cluster_jobs/run_dtm.sh")
DTM_JOB_ID=$(echo "$DTM_OUTPUT" | tail -1)
echo "  DTM job ID: $DTM_JOB_ID"

# Job 4: Final pipeline (depends on BERTopic + DTM)
echo ""
echo "Submitting final pipeline job (depends on BERTopic $BT_JOB_ID and DTM $DTM_JOB_ID)..."

FINAL_SCRIPT="$PROJECT_DIR/cluster_jobs/run_final.sh"
cat > "$FINAL_SCRIPT" << SCRIPTEOF
#!/bin/bash
#SBATCH --job-name=arms-nlp-final
#SBATCH --output=$LOGS_DIR/final_%j.out
#SBATCH --error=$LOGS_DIR/final_%j.err
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

source ~/.bashrc
conda activate arms-nlp
cd $PROJECT_DIR

echo "=== Final pipeline assembly ==="
python run_pipeline.py --config config.yaml
echo "=== Pipeline complete ==="
SCRIPTEOF
chmod +x "$FINAL_SCRIPT"

FINAL_OUTPUT=$(submit "$FINAL_SCRIPT" "${BT_JOB_ID}:${DTM_JOB_ID}")
FINAL_JOB_ID=$(echo "$FINAL_OUTPUT" | tail -1)
echo "  Final pipeline job ID: $FINAL_JOB_ID"

echo ""
echo "=== Job submission summary ==="
echo "  Embeddings:    $EMB_JOB_ID"
echo "  BERTopic:      $BT_JOB_ID  (after $EMB_JOB_ID)"
echo "  DTM:           $DTM_JOB_ID  (independent)"
echo "  Final:         $FINAL_JOB_ID  (after $BT_JOB_ID + $DTM_JOB_ID)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:  scancel $EMB_JOB_ID $BT_JOB_ID $DTM_JOB_ID $FINAL_JOB_ID"
