from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)


@dataclass
class TriApiConfig:
    """Configuration for the Azure TRAPI teacher model."""

    instance: str = "gcr/shared"
    deployment: str = "gpt-5.1-chat_2025-11-13"
    scope: str = "api://trapi/.default"
    api_version: str = "2024-12-01-preview"
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 256
    timeout_s: int = 120
    max_retries: int = 3
    max_parallel: int = 32
    azure_ad_token_provider: Optional[Callable[[], str]] = None


def _default_token_provider(scope: str) -> Callable[[], str]:
    return get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )


class TriApiChatEngine:
    """Async chat engine that fans out to Azure TRAPI (ChatGPT-5.1)."""

    def __init__(self, cfg: TriApiConfig):
        self.cfg = cfg
        token_provider = cfg.azure_ad_token_provider or _default_token_provider(cfg.scope)
        endpoint = f"https://trapi.research.microsoft.com/{cfg.instance}".rstrip("/")
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=cfg.api_version,
        )
        self._sem = asyncio.Semaphore(cfg.max_parallel)

    async def get_model_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.cfg.temperature)
        top_p = kwargs.get("top_p", self.cfg.top_p)
        max_tokens = kwargs.get("max_tokens", self.cfg.max_output_tokens)
        response_format = kwargs.get("response_format") or {"type": "json_object"}
        retries = kwargs.get("retries", self.cfg.max_retries)
        timeout_s = kwargs.get("timeout", self.cfg.timeout_s)

        last_err: Optional[Exception] = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                async with self._sem:
                    t0 = time.perf_counter()
                    resp = await self.client.chat.completions.create(
                        model=self.cfg.deployment,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_completion_tokens=max_tokens,
                        response_format=response_format,
                        timeout=timeout_s,
                    )
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                return {
                    "content": content,
                    "raw": resp,
                    "usage": usage,
                    "latency_s": time.perf_counter() - t0,
                }
            except Exception as err:
                last_err = err
                logger.warning("TriAPI request failed (attempt %d/%d): %s", attempt, retries, err)
                if attempt >= retries:
                    break
                await asyncio.sleep(min(1.5 * attempt, 5.0))

        if last_err:
            raise last_err
        raise RuntimeError("TriAPI chat call failed without an exception.")
