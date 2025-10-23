#!/bin/bash
#SBATCH --job-name=bt-bert
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpubase_bygpu_b2   # Extended GPU queue
#SBATCH --gres=gpu:h100:1              # Full H100 GPU (no MIG)
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=18:00:00

module load python/3.11
source ~/envs/bt_bert_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs outputs/checkpoints outputs/attention_reports

export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}:${HOME}/Dataset_Pipeline"

# Use pre-populated HuggingFace cache (no internet on compute nodes)
export HF_HOME="${HOME}/hf_cache"
export HF_HUB_CACHE="${HOME}/hf_cache"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

python -m src.data_prep --config config.yaml
python -m src.train --config config.yaml
