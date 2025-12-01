"""
Helper to run the madagger pipeline locally on a single 8xGPU node:
- GPU layout: 2x gen, 2x ver, 4x train; teacher uses remote TriAPI (no local GPU).
- This is intended for quick local/cluster runs without Azure job submission.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass


@dataclass
class LocalConfig:
    runid: str = "local-madagger"
    dataset_short_name: str = "aimo"
    model_full_name: str = "Qwen/Qwen3-4B"
    batch_tasks: int = 8
    teacher_backend: str = "triapi"
    teacher_instance: str = "gcr/shared"
    teacher_deployment: str = "gpt-5_2025-08-07"
    teacher_scope: str = "api://trapi"
    teacher_api_version: str = "2025-02-01-preview"
    wandb_entity: str = "kcl_coopai"
    wandb_project: str = "madagger"
    eval_dataset: str = "math500"
    eval_split: str = "test"
    eval_num_tasks: int = 100
    eval_before_train: bool = True
    eval_after_each: bool = True


def build_local_command(cfg: LocalConfig) -> str:
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not wandb_key:
        raise RuntimeError("WANDB_API_KEY/WANDB_TOKEN missing for local run.")
    if not hf_token:
        raise RuntimeError("HF_TOKEN/HUGGINGFACEHUB_API_TOKEN missing for local run.")

    teacher_flags = (
        f"--teacher_backend triapi "
        f"--teacher_triapi_instance {cfg.teacher_instance} "
        f"--teacher_triapi_deployment {cfg.teacher_deployment} "
        f"--teacher_triapi_scope {cfg.teacher_scope} "
        f"--teacher_triapi_api_version {cfg.teacher_api_version} "
        f"--teacher_triapi_max_output_tokens 64000 "
        f"--teacher_triapi_max_parallel 8 "
        f"--teacher_triapi_retries 5 "
    )

    cmd = [
        f"export WANDB_API_KEY={wandb_key} && export WANDB_TOKEN={wandb_key} && "
        f"export WANDB_ENTITY={cfg.wandb_entity} && export WANDB_PROJECT={cfg.wandb_project} &&",
        f"huggingface-cli login --token {shlex.quote(hf_token)} --add-to-git-credential &&",
        "wandb login ${WANDB_API_KEY} --host https://api.wandb.ai &&",
        "python3 register_aimo_dataset.py && python3 register_math500_dataset.py &&",
        "export PYTHONPATH=.:$PYTHONPATH &&",
        'export GEN_CUDA="0,1" VER_CUDA="2,3" TRAIN_CUDA="4,5,6,7" && export TP_PER_STUDENT=2 &&',
        f'echo "gen gpus=$GEN_CUDA ver gpus=$VER_CUDA teacher=TriAPI train gpus=$TRAIN_CUDA tp=$TP_PER_STUDENT" &&',
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_API_KEY={wandb_key} "
        f"WANDB_ENTITY={cfg.wandb_entity} WANDB_PROJECT={cfg.wandb_project} "
        f"python3 gen_ver_dagger_fullft_vllm.py iterate "
        f"--dataset_name {cfg.dataset_short_name} --split train --batch_tasks {cfg.batch_tasks} "
        f"--gen_base {cfg.model_full_name} --ver_base {cfg.model_full_name} "
        f"--gen_tokenizer {cfg.model_full_name} --ver_tokenizer {cfg.model_full_name} "
        f"--tp_s $TP_PER_STUDENT --gen_cuda $GEN_CUDA --ver_cuda $VER_CUDA --train_cuda $TRAIN_CUDA "
        f"--parallel 32 "
        f"--eval_dataset {cfg.eval_dataset} --eval_split {cfg.eval_split} --eval_num_tasks {cfg.eval_num_tasks} "
        f"{'--eval_before_train ' if cfg.eval_before_train else ''}"
        f"{'--eval_after_each ' if cfg.eval_after_each else ''}"
        f"--out_dir runs/local_madagger "
        f"--project_name madagger --experiment_name {cfg.runid} "
        f"{teacher_flags}"
    ]
    return " ".join(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local madagger command (8 GPUs: 2 gen, 2 ver, 2 teacher, 2 train)")
    p.add_argument("--runid", type=str, default="local-madagger")
    p.add_argument("--dataset_short_name", type=str, default="aimo")
    p.add_argument("--model_full_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--batch_tasks", type=int, default=8)
    p.add_argument("--teacher_backend", type=str, default="triapi", choices=["vllm", "triapi"])
    p.add_argument("--teacher_triapi_instance", type=str, default="gcr/shared")
    p.add_argument("--teacher_triapi_deployment", type=str, default="gpt-5_2025-08-07")
    p.add_argument("--teacher_triapi_scope", type=str, default="api://trapi")
    p.add_argument("--teacher_triapi_api_version", type=str, default="2025-02-01-preview")
    p.add_argument("--eval_num_tasks", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = LocalConfig(
        runid=args.runid,
        dataset_short_name=args.dataset_short_name,
        model_full_name=args.model_full_name,
        batch_tasks=args.batch_tasks,
        teacher_backend=args.teacher_backend,
        teacher_instance=args.teacher_triapi_instance,
        teacher_deployment=args.teacher_triapi_deployment,
        teacher_scope=args.teacher_triapi_scope,
        teacher_api_version=args.teacher_triapi_api_version,
        eval_num_tasks=args.eval_num_tasks,
    )
    cmd = build_local_command(cfg)
    print("\n=== Local madagger command ===")
    print(cmd)
    print("\nRun this in bash to launch locally.")


if __name__ == "__main__":
    sys.exit(main())
