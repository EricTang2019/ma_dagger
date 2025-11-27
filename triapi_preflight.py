"""
Quick TriAPI preflight to fail fast on auth/scope issues.

Usage:
  python3 triapi_preflight.py

Environment (all optional; sensible defaults applied):
  TRIAPI_INSTANCE       gcr/shared
  TRIAPI_DEPLOYMENT     gpt-5_2025-08-07
  TRIAPI_SCOPE          api://trapi   (try https://trapi.research.microsoft.com/.default if needed)
  AZURE_API_VERSION     2025-02-01-preview
  TRIAPI_MI_CLIENT_ID   user-assigned MI client id if the default lacks permission
  AZURE_OPENAI_AD_TOKEN pre-fetched bearer token; if set, credential chain is skipped
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI


def get_token_provider(scope: str) -> Callable[[], str]:
    env_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
    if env_token:
        # Return a lambda to keep the interface identical to azure_ad_token_provider.
        return lambda: env_token

    mi_client_id = os.environ.get("TRIAPI_MI_CLIENT_ID", "b32444ac-27e2-4f36-ab71-b664f6876f00")
    cred = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(client_id=mi_client_id),
    )
    return get_bearer_token_provider(cred, scope)


def main() -> int:
    instance = os.environ.get("TRIAPI_INSTANCE", "gcr/shared").rstrip("/")
    deployment = os.environ.get("TRIAPI_DEPLOYMENT", "gpt-5.1_2025-11-13")
    scope = os.environ.get("TRIAPI_SCOPE", "api://trapi")
    api_version = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

    token_provider = get_token_provider(scope)
    token = token_provider()
    token_len = len(token or "")
    print(f"[triapi_preflight] scope={scope} token_len={token_len}")
    if token_len < 50:
        print("[triapi_preflight] ERROR: token appears empty/too short; auth will fail.")
        return 1

    endpoint = f"https://trapi.research.microsoft.com/{instance}"
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "ping"}],
            # For newer API versions, use max_completion_tokens instead of max_tokens.
            max_completion_tokens=4,
        )
        content = resp.choices[0].message.content
        print(f"[triapi_preflight] success: {content!r}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[triapi_preflight] ERROR calling TriAPI: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
