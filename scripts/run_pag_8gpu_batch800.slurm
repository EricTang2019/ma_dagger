#!/bin/bash
#SBATCH --job-name=pag_8gpu_latest_800
#SBATCH --output=logs/pag_8gpu_latest_800_%j.out
#SBATCH --error=logs/pag_8gpu_latest_800_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --partition=ml.p5.48xlarge

set -euo pipefail

# Resolve repo root from this script location.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda environment
source /fsx/gstevenw/miniconda3/etc/profile.d/conda.sh
conda activate madagger

OUT_DIR=runs/pag_8gpu_batch800
mkdir -p logs "$OUT_DIR"

# Hugging Face uploads (embedded token for slurm runs).
export HF_TOKEN=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
HF_REPO_GEN="${HF_REPO_GEN:-jingwut/gen}"
HF_REPO_VER="${HF_REPO_VER:-jingwut/ver}"
HF_UPLOAD_EVERY="${HF_UPLOAD_EVERY:-10}"  # upload every N outer rounds (gen+ver pair)

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_WORKER_MULTIPROC_START_METHOD=spawn \
PYTHONPATH="$REPO_ROOT/deps/rllm:$PYTHONPATH" \
python run_pag_numina.py \
  --dataset_path nlile/NuminaMath-1.5-RL-Verifiable \
  --split train --force_register \
  --project_name pag_8gpu_batch800 --experiment_name pag_8gpu_batch800 \
  --out_dir "$OUT_DIR" \
  --hf_repo_gen "$HF_REPO_GEN" --hf_repo_ver "$HF_REPO_VER" --hf_upload_every "$HF_UPLOAD_EVERY" \
  --rounds 200 --batch_tasks 800 --parallel 800 --max_turns 2 \
  --pipeline_mode staged --rollout_mode phased \
  --train_cuda 0,1,2,3,4,5,6,7 \
  --gen_cuda 0,1,2,3,4,5,6,7 --ver_cuda 0,1,2,3,4,5,6,7 --teacher_cuda 0,1,2,3,4,5,6,7 \
  --gen_gpu_mem_util 0.9 --ver_gpu_mem_util 0.9 --teacher_gpu_mem_util 0.85 \
  --tp_s 1 --dp_s 8 \
  --tp_t 1 --dp_t 8 \
  --teacher_label_batch_size 800 \
  --train_latest_only \
  --max_model_len 32768 \
  --max_new_tokens 16384 \
  --config_override \
    data.train_batch_size=200 \
    data.micro_batch_size_per_gpu=1 \
    trainer.total_epochs=4 \
    data.rllm.tokenize_and_mask_method=stepwise \
    data.max_length=32768 \
    data.truncation=right |& tee "$OUT_DIR/run.log"
