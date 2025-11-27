"""
Submit a minimal AML job to exercise TriAPI on the target VC.

The job:
  - fetches an AD token via get_triapi_token.py (MI client id below),
  - runs triapi_preflight.py against gpt-5_2025-08-07,
  - runs check_azure_api.py (list deployments) to see if auth works,
  - targets msrresrchbasicvc / Singularity.ND96_H100_v5 (Basic/High, 8x H100).

Adjust constants below if you need a different workspace or VC.
"""

from __future__ import annotations

import datetime as dt
import os
from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential


# Workspace (compute submission) — stays in aifrontiers_ws.
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID", "d4fe558f-6660-4fe7-99ec-ae4716b5e03f")
RESOURCE_GROUP = os.getenv("RESOURCEGROUP_NAME", "aifrontiers")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "aifrontiers_ws")

# Target VC / instance (compute lives in gcr-singularity subscription).
VC = "msrresrchbasicvc"
INSTANCE_TYPE = "Singularity.ND96_H100_v5"
SLA_TIER = "Basic"
PRIORITY = "High"
NODE_COUNT = 1
VC_SUBSCRIPTION_ID = "22da88f6-1210-4de2-a5a3-da4c7c2a1213"
VC_RESOURCE_GROUP = "gcr-singularity"

# Environment to reuse (already present in the workspace).
ENV_NAME = "vllm-openai-0-9-1-custom"
ENV_VERSION = "1"

# TriAPI defaults; can be overridden via env. Scope aligns with CLI token command: api://trapi
TRIAPI_SCOPE = os.getenv("TRIAPI_SCOPE", "api://trapi")
TRIAPI_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-02-01-preview")
TRIAPI_INSTANCE = os.getenv("TRIAPI_INSTANCE", "gcr/shared")
TRIAPI_DEPLOYMENT = os.getenv("TRIAPI_DEPLOYMENT", "gpt-5_2025-08-07")
TRIAPI_MI_CLIENT_ID = os.getenv("TRIAPI_MI_CLIENT_ID", "b32444ac-27e2-4f36-ab71-b664f6876f00")
# For data access on Singularity; can override to another UAI resource id.
JOB_UAI_RESOURCE_ID = os.getenv(
    "_AZUREML_SINGULARITY_JOB_UAI",
    "/subscriptions/d4fe558f-6660-4fe7-99ec-ae4716b5e03f/resourcegroups/aifrontiers/providers/Microsoft.ManagedIdentity/userAssignedIdentities/aifrontiers",
)


def build_compute_config() -> str:
    return (
        f"/subscriptions/{VC_SUBSCRIPTION_ID}"
        f"/resourceGroups/{VC_RESOURCE_GROUP}"
        f"/providers/Microsoft.MachineLearningServices/virtualclusters/{VC}"
    )


def build_job_command() -> str:
    cmds = [
        "pip3 install azure-identity openai &&",
        "pwd && ls &&",
        f"export AZURE_API_VERSION=\"{TRIAPI_API_VERSION}\" &&",
        f"export TRIAPI_SCOPE=\"{TRIAPI_SCOPE}\" &&",
        f"export TRIAPI_INSTANCE=\"{TRIAPI_INSTANCE}\" &&",
        f"export TRIAPI_DEPLOYMENT=\"{TRIAPI_DEPLOYMENT}\" &&",
        f"export TRIAPI_MI_CLIENT_ID=\"{TRIAPI_MI_CLIENT_ID}\" &&",
        # Also expose as AZURE_CLIENT_ID to match the Amulet example.
        f"export AZURE_CLIENT_ID=\"{TRIAPI_MI_CLIENT_ID}\" &&",
        # If caller already injected AZURE_OPENAI_AD_TOKEN, keep it; else fetch via MI/CLI.
        "if [ -z \"$AZURE_OPENAI_AD_TOKEN\" ]; then export AZURE_OPENAI_AD_TOKEN=$(python3 get_triapi_token.py); fi &&",
        # 先检查部署列表（验证 scope/身份），再跑 chat 预检。
        "python3 check_azure_api.py || exit 1 &&",
        "python3 triapi_preflight.py",
    ]
    return " ".join(cmds)


def main() -> None:
    ml_client = MLClient(
        AzureCliCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )

    env = ml_client.environments.get(name=ENV_NAME, version=ENV_VERSION)

    vc_config = {
        "instance_type": INSTANCE_TYPE,
        "instance_count": NODE_COUNT,
        "properties": {
            "AISuperComputer": {
                "interactive": False,
                "slaTier": SLA_TIER,
                "priority": PRIORITY,
                "tensorboardLogDirectory": "/scratch/tensorboard_logs",
            }
        },
    }

    env_vars = {
        "JOB_EXECUTION_MODE": "basic",
        "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true",
        "_AZUREML_SINGULARITY_JOB_UAI": JOB_UAI_RESOURCE_ID,
        "TRIAPI_SCOPE": TRIAPI_SCOPE,
        "AZURE_API_VERSION": TRIAPI_API_VERSION,
    }
    host_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
    if host_token:
        env_vars["AZURE_OPENAI_AD_TOKEN"] = host_token

    job = command(
        code=".",
        command=build_job_command(),
        environment=env,
        compute=build_compute_config(),
        resources=vc_config,
        display_name=f"triapi-smoke-{dt.datetime.utcnow().strftime('%m%d-%H%M%S')}",
        experiment_name="triapi_smoke",
        environment_variables=env_vars,
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Submitted: {returned_job.name}")
    print(f"URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
