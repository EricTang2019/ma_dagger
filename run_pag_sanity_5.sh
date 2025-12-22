#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_WORKER_MULTIPROC_START_METHOD=spawn
export PYTHONPATH=deps/rllm

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_pag_numina.py \
  --dataset_path /work5/jingwut/MADAgger/NuminaMath-1.5-RL-Verifiable-train.parquet \
  --split train \
  --force_register \
  --batch_tasks 1 \
  --parallel 1 \
  --max_model_len 16384 \
  --project_name pag_sanity_5 \
  --experiment_name pag_sanity_5 \
  --out_dir runs/pag_sanity_5 \
  --max_turns 2 \
  --collect_for auto \
  --train_cuda 0,1 \
  --gen_cuda 6 \
  --ver_cuda 7 \
  --teacher_cuda 0,1 \
  --tp_t 2 \
  --config_override \
    data.train_batch_size=2 \
    data.micro_batch_size_per_gpu=1 \
    trainer.total_epochs=1 \
    data.rllm.tokenize_and_mask_method=stepwise \
    data.max_length=16384 \
    data.truncation=right \
  --dump_transcripts_raw
