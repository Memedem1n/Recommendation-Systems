#!/bin/bash
#SBATCH --job-name=bt-bert-exps
#SBATCH --array=0-8
#SBATCH --output=logs/experiments_%A_%a.out
#SBATCH --error=logs/experiments_%A_%a.err
#SBATCH --partition=gpubase_bygpu_b2   # Extended GPU queue
#SBATCH --gres=gpu:h100:1              # Full H100 GPU (no MIG)
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=18:00:00

set -euo pipefail

CONFIGS=(
  "configs/scenario_baseline.yaml"
  "configs/scenario_rank_tokens.yaml"
  "configs/scenario_repeat_emphasis.yaml"
  "configs/scenario_category_prompt.yaml"
  "configs/scenario_loss_balanced.yaml"
  "configs/scenario_stopwords.yaml"
  "configs/scenario_keyword_boost.yaml"
  "configs/scenario_alt_backbone.yaml"
  "configs/scenario_pseudo_augment.yaml"
)

module load python/3.11
source ~/envs/bt_bert_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}:${HOME}/Dataset_Pipeline"
export HF_HOME="${HOME}/hf_cache"
export HF_HUB_CACHE="${HOME}/hf_cache"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="$(basename "${CONFIG%.yaml}")"

echo "[$(date --iso-8601=seconds)] Experiment ${RUN_ID} using ${CONFIG}"

# Prepare run via experiment manager
readarray -t RUN_INFO < <(python -m src.experiment_manager prepare-run --experiment "$RUN_ID" --base-config "$CONFIG")

CONFIG_PATH="${RUN_INFO[0]}"
METRICS_PATH="${RUN_INFO[1]}"
CHECKPOINT_DIR="${RUN_INFO[2]}"
DATA_DIR="${RUN_INFO[3]}"
RUN_DIR="${RUN_INFO[4]}"
BEST_CHECKPOINT="${RUN_INFO[5]}"

# Resolve train/val paths and augmentation parameters
readarray -t DATA_INFO < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
base_dir = (config_path.parent / cfg["project"]["base_dir"]).resolve()
data_dir = (base_dir / cfg["data"]["label_output_dir"]).resolve()

def resolve_path(value: str, root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()

train_csv = cfg.get("data", {}).get("train_csv", "train.csv")
val_csv = cfg.get("data", {}).get("val_csv", "val.csv")

train_path = resolve_path(train_csv, data_dir)
val_path = resolve_path(val_csv, data_dir)

print(str(train_path))
print(str(val_path))

aug = cfg.get("augmentation", {}) or {}
if not aug.get("enabled"):
    print("disabled")
else:
    base_train_csv = resolve_path(aug.get("base_train_csv", "train.csv"), data_dir)
    output_train_csv = resolve_path(aug.get("output_train_csv", train_csv), data_dir)
    pred_split = aug.get("predictions_split", "train")
    pos_th = float(aug.get("positive_threshold", 0.85))
    neg_th = float(aug.get("negative_threshold", 0.15))
    products_path = resolve_path(cfg["data"]["unified_products_path"], base_dir)
    print("enabled")
    print(str(base_train_csv))
    print(str(output_train_csv))
    print(str(pred_split))
    print(str(pos_th))
    print(str(neg_th))
    print(str(products_path))
PY
"$CONFIG_PATH")

TRAIN_PATH="${DATA_INFO[0]}"
VAL_PATH="${DATA_INFO[1]}"
AUG_FLAG="${DATA_INFO[2]:-disabled}"

if [ ! -f "$TRAIN_PATH" ] || [ ! -f "$VAL_PATH" ]; then
  python -m src.data_prep --config "$CONFIG_PATH"
fi

if [ "$AUG_FLAG" = "enabled" ]; then
  AUG_BASE="${DATA_INFO[3]}"
  AUG_OUTPUT="${DATA_INFO[4]}"
  AUG_SPLIT="${DATA_INFO[5]}"
  AUG_POS="${DATA_INFO[6]}"
  AUG_NEG="${DATA_INFO[7]}"
  AUG_PRODUCTS="${DATA_INFO[8]}"
  AUG_PRED="${RUN_DIR}/augment_predictions.csv"

  if [ -n "$BEST_CHECKPOINT" ] && [ -f "$BEST_CHECKPOINT" ]; then
    PRED_CONFIG="${RUN_DIR}/predict_config.yaml"
    python - "$CONFIG_PATH" "$PRED_CONFIG" "$AUG_BASE" "$DATA_DIR" <<'PY'
import sys
import yaml
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
base_train = Path(sys.argv[3]).resolve()
data_dir = Path(sys.argv[4]).resolve()

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
try:
    train_rel = base_train.relative_to(data_dir)
except ValueError:
    # Fall back to filename if the base train file lives elsewhere
    train_rel = Path(base_train.name)

cfg.setdefault("data", {})["train_csv"] = train_rel.as_posix()

output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
    python -m src.predict --config "$PRED_CONFIG" --checkpoint "$BEST_CHECKPOINT" --split "$AUG_SPLIT" --output "$AUG_PRED"
    python -m src.augment_dataset \
      --train "$AUG_BASE" \
      --predictions "$AUG_PRED" \
      --products "$AUG_PRODUCTS" \
      --threshold "$AUG_POS" \
      --negative-threshold "$AUG_NEG" \
      --output "$AUG_OUTPUT"
  else
    if [ "$AUG_BASE" != "$AUG_OUTPUT" ]; then
      cp "$AUG_BASE" "$AUG_OUTPUT"
    fi
  fi
fi

python -m src.train --config "$CONFIG_PATH"

CHECKPOINT=$(python - "$CHECKPOINT_DIR" <<'PY'
from pathlib import Path
import sys

ckpt_dir = Path(sys.argv[1])
candidates = sorted(ckpt_dir.glob("*.pt"))
print(str(candidates[-1]) if candidates else "", end="")
PY
)

if [ -n "$CHECKPOINT" ]; then
  python -m src.evaluate --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT" --split val --output "${RUN_DIR}/eval_val.json"
  python -m src.evaluate --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT" --split test --output "${RUN_DIR}/eval_test.json"
  python -m src.predict --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT" --split test --output "${RUN_DIR}/predictions_test.csv"
fi

COMPLETE_ARGS=(
  --experiment "$RUN_ID"
  --base-config "$CONFIG"
  --run-dir "$RUN_DIR"
  --metrics "$METRICS_PATH"
)

if [ -n "$CHECKPOINT" ]; then
  COMPLETE_ARGS+=(--checkpoint "$CHECKPOINT")
fi

python -m src.experiment_manager complete-run "${COMPLETE_ARGS[@]}"
