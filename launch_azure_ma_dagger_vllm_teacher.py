"""
Submit a madagger Azure ML job with an 8xGPU layout:
- 2x generator GPUs, 2x verifier GPUs, 2x teacher GPUs (local vLLM teacher), 2x train GPUs.
- Teacher runs locally on the node via vLLM (no TriAPI).

This is a variant of launch_azure_ma_dagger.py tuned for vLLM teacher + fixed GPU split.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import AzureCliCredential

from build_job_command import JobConfig, build_job_command


def read_token_from_bashrc(key: str, path: str = "~/.bashrc") -> Optional[str]:
    try:
        content = open(os.path.expanduser(path), "r", encoding="utf-8").read()
        m = re.search(rf"export {key}=(.*)", content)
        if m:
            return m.group(1).strip().strip('"')
    except FileNotFoundError:
        return None
    return None


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    wandb_token = os.environ.get("WANDB_TOKEN") or read_token_from_bashrc("WANDB_TOKEN")
    hf_token = os.environ.get("HF_TOKEN") or read_token_from_bashrc("HF_TOKEN")
    if not wandb_token or not hf_token:
        raise RuntimeError("WANDB_TOKEN or HF_TOKEN missing (env or ~/.bashrc).")

    online_env_name = "vllm-openai-0-9-1-custom"
    candidate_vcs = [
        ("msrresrchbasicvc", "Singularity.ND96_H100_v5", "Basic", "High", 8),
    ]
    model_full_name = "Qwen/Qwen3-4B"
    dataset_short_name = "aimo"
    n = 8
    experiment_name = "ma_dagger_vllm_teacher"

    inputs = {
        "wandb_token": wandb_token,
        "hf_token": hf_token,
    }
    output_dir = (
        "azureml://subscriptions/d4fe558f-6660-4fe7-99ec-ae4716b5e03f"
        "/resourcegroups/aifrontiers/workspaces/aifrontiers_ws"
        "/datastores/ziyanwang_data/paths/"
    )
    outputs = {
        "output_dir": Output(type=AssetTypes.URI_FOLDER, path=output_dir, mode=InputOutputModes.RW_MOUNT),
    }

    subscription_id = os.getenv("SUBSCRIPTION_ID", "d4fe558f-6660-4fe7-99ec-ae4716b5e03f")
    resource_group = os.getenv("RESOURCEGROUP_NAME", "aifrontiers")
    workspace_name = os.getenv("WORKSPACE_NAME", "aifrontiers_ws")

    class VcInfo:
        def __init__(self, subscription_id: str, resource_group: str, vc: str):
            self.subscription_id = subscription_id
            self.resource_group = resource_group
            self.vc = vc
            self.compute_config = (
                "/subscriptions/" + subscription_id + "/resourceGroups/" + resource_group
                + "/providers/Microsoft.MachineLearningServices/virtualclusters/" + vc
            )

    def resolve_vc(vc_name: str) -> VcInfo:
        if vc_name in ["msrresrchbasicvc"]:
            return VcInfo("22da88f6-1210-4de2-a5a3-da4c7c2a1213", "gcr-singularity", vc_name)
        if vc_name in ["msrresrchvc"]:
            return VcInfo("22da88f6-1210-4de2-a5a3-da4c7c2a1213", "gcr-singularity-resrch", vc_name)
        if vc_name in ["msroctobasicvc"]:
            return VcInfo("d4404794-ab5b-48de-b7c7-ec1fefb0a04e", "gcr-singularity-octo", vc_name)
        raise ValueError(f"Unknown VC {vc_name}")

    ml_client = MLClient(AzureCliCredential(), subscription_id, resource_group, workspace_name)
    env = ml_client.environments.get(name=online_env_name, version="1")

    for idx, (vc, instance_type, sla_tier, priority, gpu_num) in enumerate(candidate_vcs):
        this_runid = (
            f"p1-madagger-{dataset_short_name}-n{n}-qwen3-4b-"
            f"{vc}-{instance_type}-vllmteacher-shard{idx:02d}"
        ).replace(".", "-")

        # Build the base command and then override teacher backend + GPU split via env exports.
        base_cfg = JobConfig(
            runid=this_runid,
            dataset_short_name=dataset_short_name,
            model_full_name=model_full_name,
            batch_tasks=n,
        )
        job_command = build_job_command(base_cfg)
        # Override GPU split to 2/2/2/2 and force vLLM teacher.
        job_command = job_command.replace(
            'elif [ "$GPU_COUNT" -ge 4 ]; then GEN_CUDA="0,1"; VER_CUDA="2,3"; TP_PER_STUDENT=2; TRAIN_CUDA="2"; '
            'else GEN_CUDA="0"; VER_CUDA="0"; TP_PER_STUDENT=1; TRAIN_CUDA="0"; fi &&',
            'elif [ "$GPU_COUNT" -ge 4 ]; then GEN_CUDA="0,1"; VER_CUDA="2,3"; TP_PER_STUDENT=2; TRAIN_CUDA="6,7"; '
            'else GEN_CUDA="0"; VER_CUDA="0"; TP_PER_STUDENT=1; TRAIN_CUDA="0"; fi &&',
        )
        job_command = job_command.replace(
            'if [ "$GPU_COUNT" -ge 8 ]; then GEN_CUDA="0,1"; VER_CUDA="2,3"; TP_PER_STUDENT=2; TRAIN_CUDA="4,5,6,7"; ',
            'if [ "$GPU_COUNT" -ge 8 ]; then GEN_CUDA="0,1"; VER_CUDA="2,3"; TP_PER_STUDENT=2; TRAIN_CUDA="6,7"; '
            'TEACHER_CUDA="4,5"; TP_T=2; ',
        )
        job_command = job_command.replace(
            "--teacher_backend triapi",
            "--teacher_backend vllm --teacher_cuda $TEACHER_CUDA --tp_t ${TP_T:-2} "
            f"--teacher_base {model_full_name} --teacher_tokenizer {model_full_name} "
            "--teacher_gpu_mem_util 0.9 "
        )
        # Drop TriAPI preflight; no remote teacher used in this variant.
        job_command = job_command.replace(
            "python3 triapi_preflight.py || exit 1 &&",
            "",
        )

        vc_info = resolve_vc(vc)
        vc_config = {
            "instance_type": instance_type,
            "instance_count": 1,
            "properties": {
                "AISuperComputer": {
                    "interactive": False,
                    "slaTier": sla_tier,
                    "priority": priority,
                    "tensorboardLogDirectory": "/scratch/tensorboard_logs",
                }
            },
        }
        env_vars = {
            "JOB_EXECUTION_MODE": "basic",
            "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true",
            "_AZUREML_SINGULARITY_JOB_UAI": "/subscriptions/d4fe558f-6660-4fe7-99ec-ae4716b5e03f/resourcegroups/aifrontiers/providers/Microsoft.ManagedIdentity/userAssignedIdentities/aifrontiers",
        }

        job = command(
            code=".",
            command=job_command,
            inputs=inputs,
            outputs=outputs,
            environment=env,
            environment_variables=env_vars,
            compute=vc_info.compute_config,
            resources=vc_config,
            instance_count=1,
            display_name=this_runid,
            experiment_name=experiment_name,
            distribution={"type": "PyTorch"},
        )

        returned_job = ml_client.jobs.create_or_update(job)
        print(f"[vc {idx}] Job URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
