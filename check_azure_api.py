"""
Quick check to list TriAPI deployments using the current auth context.

- Prefers AZURE_OPENAI_AD_TOKEN if set; otherwise uses CLI credential.
- Scope defaults to api://trapi/.default (override via TRIAPI_SCOPE).
"""

from __future__ import annotations

import os
import sys
import requests
from azure.identity import AzureCliCredential, get_bearer_token_provider


def main() -> int:
    scope = os.environ.get("TRIAPI_SCOPE", "api://trapi/.default")
    env_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")

    if env_token:
        token = env_token
        print(f"[check_azure_api] using env token len={len(token)} scope={scope}")
    else:
        cred = AzureCliCredential()
        token_provider = get_bearer_token_provider(cred, scope)
        token = token_provider()
        print(f"[check_azure_api] fetched token via CLI len={len(token)} scope={scope}")

    if not token:
        print("[check_azure_api] no token available")
        return 1

    url = "https://trapi.research.microsoft.com/tmds/deployments"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        print("[check_azure_api] deployments listing:")
        for line in resp.iter_lines():
            print(line.decode("utf-8"))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[check_azure_api] ERROR calling {url}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
