#!/bin/bash
# Usage (login node):
#   module load python/3.11
#   source ~/envs/bt_bert_env/bin/activate
#   ./scripts/download_hf_cache.sh ~/hf_cache bert-base-uncased

set -euo pipefail

CACHE_DIR="${1:-$HOME/hf_cache}"
MODEL_NAME="${2:-bert-base-uncased}"

echo "Downloading ${MODEL_NAME} into ${CACHE_DIR} ..."
python -m src.experiment_manager download-cache --model "${MODEL_NAME}" --cache-dir "${CACHE_DIR}"
echo "Done."
