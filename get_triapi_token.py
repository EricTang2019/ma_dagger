import os
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

# Scope and managed identity can be overridden via env for flexibility.
SCOPE = os.environ.get("TRIAPI_SCOPE", "api://trapi/.default")
MI_CLIENT_ID = os.environ.get(
    "TRIAPI_MI_CLIENT_ID", "b32444ac-27e2-4f36-ab71-b664f6876f00"
)


def main() -> None:
    cred = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(client_id=MI_CLIENT_ID),
    )
    token_provider = get_bearer_token_provider(cred, SCOPE)
    print(token_provider())


if __name__ == "__main__":
    main()
