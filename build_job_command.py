"""
Helper to build the AML job command for the madagger pipeline.

This keeps the shell string construction out of the notebook so it is easy to
inspect and copy/paste. Adjust the constants at the bottom as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class JobConfig:
    runid: str
    dataset_short_name: str
    model_full_name: str
    batch_tasks: int
    output_dir_placeholder: str = "${outputs.output_dir}"
    wandb_api_key: str = "5f642e1080557e1b07a844b75f8f580e7ff47791"
    teacher_instance: str = "gcr/shared"
    teacher_deployment: str = "gpt-5_2025-08-07"
    # Scope aligned to CLI token command; override if needed.
    teacher_scope: str = "api://trapi"
    # Align with the config used in Memento transforms (works in that env).
    teacher_api_version: str = "2025-02-01-preview"
    # Managed identity client id that has TriAPI access; falls back to CLI if unset.
    triapi_mi_client_id: str = "b32444ac-27e2-4f36-ab71-b664f6876f00"
    eval_dataset: str = "math500"
    eval_split: str = "test"
    eval_num_tasks: int = 100
    eval_before_train: bool = True
    eval_after_each: bool = True


def build_job_command(cfg: JobConfig) -> str:
    """Return the final shell command string."""
    job_command_list: List[str] = [
        f"export WANDB_API_KEY={cfg.wandb_api_key} && export WANDB_TOKEN={cfg.wandb_api_key} && "
        "export WANDB_ENTITY=kcl_coopai && export WANDB_PROJECT=madagger &&",
        "git config --global credential.helper store &&",
        "huggingface-cli login --token ${{inputs.hf_token}} --add-to-git-credential &&",
        "wandb login ${WANDB_API_KEY} --host https://api.wandb.ai &&",
        "free -h &&",
        "pip3 install omegaconf==2.3.0 hydra-core==1.3.2 antlr4-python3-runtime==4.9.3 &&",
        'pip3 install "math-verify[antlr4-9-3]==0.7.0" fire tensorboardX prettytable pylatexenc jsonlines &&',
        "pip3 install azure-identity azure-ai-ml &&",
        # Pre-fetch token to avoid chained credential surprises, then preflight TriAPI.
        f"export TRIAPI_SCOPE=\"{cfg.teacher_scope}\" && "
        f"export AZURE_API_VERSION=\"{cfg.teacher_api_version}\" && "
        f"export TRIAPI_MI_CLIENT_ID=\"{cfg.triapi_mi_client_id}\" && "
        # Respect a caller-provided token; otherwise fetch via MI/CLI.
        "if [ -z \"$AZURE_OPENAI_AD_TOKEN\" ]; then export AZURE_OPENAI_AD_TOKEN=$(python3 get_triapi_token.py); fi &&",
        "python3 triapi_preflight.py || exit 1 &&",
        "pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 &&",
        "pip3 install wheel psutil &&",
        "pip3 install tensordict &&",
        "pip3 install codetiming &&",
        "pip3 install torchdata==0.8.0 &&",
        "pip3 install peft==0.17.1 &&",
        "pip3 install --no-build-isolation --no-deps flash-attn==2.7.4.post1 &&",
        "pip3 install --no-deps -e ./rllm/verl &&",
        "pip3 install -e ./rllm &&",
        "pip3 install vllm==0.8.5 &&",
        "pip3 install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld &&",
        # Register datasets.
        "python3 register_aimo_dataset.py &&",
        "python3 register_math500_dataset.py &&",
        "export PYTHONPATH=.:$PYTHONPATH &&",
        "export GPU_COUNT=$(nvidia-smi -L | wc -l) &&",
        'if [ "$GPU_COUNT" -ge 8 ]; then GEN_CUDA="0,1"; VER_CUDA="2,3"; TP_PER_STUDENT=2; TRAIN_CUDA="4,5,6,7"; '
        'elif [ "$GPU_COUNT" -ge 4 ]; then GEN_CUDA="0,1"; VER_CUDA="2,3"; TP_PER_STUDENT=2; TRAIN_CUDA="2"; '
        'else GEN_CUDA="0"; VER_CUDA="0"; TP_PER_STUDENT=1; TRAIN_CUDA="0"; fi &&',
        'echo "gen gpus=$GEN_CUDA ver gpus=$VER_CUDA tp=$TP_PER_STUDENT" &&',
        f"mkdir -p {cfg.output_dir_placeholder}/madagger && "
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_API_KEY={cfg.wandb_api_key} WANDB_ENTITY=kcl_coopai WANDB_PROJECT=madagger "
        f"python3 gen_ver_dagger_fullft_vllm.py iterate "
        f"--dataset_name {cfg.dataset_short_name} --split train --batch_tasks {cfg.batch_tasks} "
        f"--teacher_backend triapi --teacher_triapi_instance {cfg.teacher_instance} "
        f"--teacher_triapi_deployment {cfg.teacher_deployment} "
        f"--teacher_triapi_scope {cfg.teacher_scope} --teacher_triapi_api_version {cfg.teacher_api_version} "
        f"--teacher_triapi_max_parallel 64 --teacher_triapi_timeout 300 "
        f"--teacher_triapi_max_output_tokens 8000 "
        f"--gen_base {cfg.model_full_name} --ver_base {cfg.model_full_name} "
        f"--gen_tokenizer {cfg.model_full_name} --ver_tokenizer {cfg.model_full_name} "
        f"--tp_s $TP_PER_STUDENT --tp_t 1 --gen_cuda $GEN_CUDA --ver_cuda $VER_CUDA --train_cuda $TRAIN_CUDA "
        f"--parallel 64 "
        f"--eval_dataset {cfg.eval_dataset} --eval_split {cfg.eval_split} --eval_num_tasks {cfg.eval_num_tasks} "
        f"{'--eval_before_train ' if cfg.eval_before_train else ''}"
        f"{'--eval_after_each ' if cfg.eval_after_each else ''}"
        f"--out_dir {cfg.output_dir_placeholder}/madagger "
        f"--project_name madagger --experiment_name {cfg.runid}",
    ]
    return " ".join(job_command_list)


if __name__ == "__main__":
    cfg = JobConfig(
        runid="p1-madagger-aimo-n8-qwen3-4b-msrresrchbasicvc-Singularity-ND96_H100_v5-shard04",
        dataset_short_name="aimo",
        model_full_name="Qwen/Qwen3-4B",
        batch_tasks=8,
    )
    print(build_job_command(cfg))
