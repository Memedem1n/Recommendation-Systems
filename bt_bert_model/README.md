# BT-BERT Concern Classifier

This package contains everything required to label data, train, evaluate, and explain the BT-BERT implicit concern classifier on both local machines and the Rorqual HPC cluster. The directory structure matches the top-level README, so only the key commands are repeated here for quick reference.

`
bt_bert_model/
  config.yaml                 # Central configuration (shared by scripts/notebooks)
  requirements.txt            # Minimal runtime dependencies
  data/
    raw/                      # Source CSV/XLS/JSON files (ingested by data_prep.py)
    labels.csv                # Pseudo labels (generated)
    train.csv                 # Training split (generated)
    val.csv                   # Validation split (generated)
    test.csv                  # Test split (generated)
    label_summary.json        # Per-concern positive/negative counts (generated)
  outputs/
    checkpoints/              # Model checkpoints (.pt)
    attention_reports/        # Explainability artefacts
    metrics.json              # Training metrics history
  scripts/
    train.slurm               # One-shot training job (1×H100, 12 CPU cores, 48 GB RAM, 18 h)
    train_experiments.slurm   # Batch experiment launcher across 9 scenario configs
    download_hf_cache.sh      # Helper to re-populate ~/hf_cache on the cluster
  notebooks/
    bt_bert_workflow.ipynb    # End-to-end walkthrough (no heavy steps by default)
  src/
    augment_dataset.py        # Pseudo-labelled data augmentation utility
    data_prep.py              # Pseudo label generation and split creation
    dataset.py                # PyTorch Dataset for (product, concern) pairs
    evaluate.py               # Checkpoint evaluation
    explain.py                # Attention-based explanations
    experiment_manager.py     # Iterative tuning helper
    model.py                  # BT-BERT implicit model
    predict.py                # Batch scoring utility
    train.py                  # Training loop
`

## Quick Commands

`ash
# 1. Install environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r bt_bert_model/requirements.txt

# 2. Populate HF cache on the cluster (once per account)
module load python/3.11
source ~/envs/bt_bert_env/bin/activate
./scripts/download_hf_cache.sh ~/hf_cache

# 3. Generate pseudo labels
python bt_bert_model/src/data_prep.py --config bt_bert_model/config.yaml

# 4. Train locally
python bt_bert_model/src/train.py --config bt_bert_model/config.yaml

# 4b. Train on Rorqual (full GPU)
sbatch scripts/train.slurm

# 5. Launch the 9-scenario experiment sweep
sbatch scripts/train_experiments.slurm
`

Outputs for individual experiments live under outputs/experiments/<scenario>/runs/<run_id>/. Each run contains the exact configuration used, validation/test metrics, predictions, and checkpoints so you can compare scenarios side by side.
