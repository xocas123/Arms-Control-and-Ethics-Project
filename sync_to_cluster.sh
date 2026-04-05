#!/bin/bash
# Sync the project to Wright via scp.
# Run from your local machine.
#
# Usage:
#   bash sync_to_cluster.sh              # full sync
#   bash sync_to_cluster.sh --no-data    # skip data/raw (if already uploaded)

ANDREW_ID="xalvarez"
REMOTE="$ANDREW_ID@wright.hss.cmu.edu"
REMOTE_DIR="~/arms-control-nlp"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TARBALL="/tmp/arms-control-nlp.tar.gz"

EXCLUDE=(
  --exclude="output"
  --exclude="__pycache__"
  --exclude="*.pyc"
  --exclude=".DS_Store"
  --exclude="*.egg-info"
  --exclude=".git"
)

if [[ "${1:-}" == "--no-data" ]]; then
  EXCLUDE+=(--exclude="data/raw")
  echo "Skipping data/raw/ (--no-data flag)"
fi

echo "[1/3] Creating archive..."
tar czf "$TARBALL" "${EXCLUDE[@]}" -C "$(dirname "$LOCAL_DIR")" "$(basename "$LOCAL_DIR")"

echo "[2/3] Copying to $REMOTE:~/"
scp "$TARBALL" "$REMOTE:~/"

echo "[3/3] Extracting on cluster..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR && tar xzf ~/arms-control-nlp.tar.gz -C ~ && rm ~/arms-control-nlp.tar.gz"

rm "$TARBALL"
echo ""
echo "Done. Next steps:"
echo "  ssh $REMOTE"
echo "  cd arms-control-nlp"
echo "  bash cluster_setup.sh   # first time only"
echo "  sbatch submit.sh"
