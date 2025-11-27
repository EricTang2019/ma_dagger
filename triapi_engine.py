from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import os

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
    deployment: str = "gpt-5_2025-08-07"
    scope: str = "api://trapi"
    # Allow overriding via env to avoid code changes on jobs
    api_version: str = os.environ.get("AZURE_API_VERSION", "2025-02-01-preview")
    # Some TRAPI models (e.g., gpt-5_xxx) only allow the default temperature.
    # Use None to omit the param and stick to the service default (typically 1.0).
    temperature: Optional[float] = None
    top_p: float = 1.0
    # None/<=0 means defer to service default; set a high default to avoid truncation.
    max_output_tokens: Optional[int] = 64000
    timeout_s: int = 120
    max_retries: int = 5
    max_parallel: int = 32
    azure_ad_token_provider: Optional[Callable[[], str]] = None


def _default_token_provider(scope: str) -> Callable[[], str]:
    # Prefer an explicit token from env (as produced by get_triapi_token.py)
    env_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
    if env_token:
        return lambda: env_token
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
        # Default to plain text; JSON forcing can lead to empty outputs if the model
        # doesn't emit a valid object.
        response_format = kwargs.get("response_format")
        retries = kwargs.get("retries", self.cfg.max_retries)
        timeout_s = kwargs.get("timeout", self.cfg.timeout_s)

        # Only send temperature if explicitly set and supported. Many TRAPI
        # deployments reject non-default values (including 0.0), so omit 0/None/1.0.
        if temperature is not None and temperature <= 0:
            temperature = None
        if temperature is not None and abs(temperature - 1.0) < 1e-6:
            temperature = None

        last_err: Optional[Exception] = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                async with self._sem:
                    t0 = time.perf_counter()
                    request_kwargs = {
                        "model": self.cfg.deployment,
                        "messages": messages,
                        "top_p": top_p,
                        "timeout": timeout_s,
                    }
                    if response_format is not None:
                        request_kwargs["response_format"] = response_format
                    if temperature is not None:
                        request_kwargs["temperature"] = temperature
                    if max_tokens is not None and max_tokens > 0:
                        request_kwargs["max_completion_tokens"] = max_tokens
                    resp = await self.client.chat.completions.create(**request_kwargs)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                if not content.strip():
                    logger.warning(
                        "TriAPI returned empty content (finish_reason=%s, usage=%s)",
                        getattr(resp.choices[0], "finish_reason", None),
                        usage,
                    )
                # Optional lightweight tracing to Weights & Biases if a run is active.
                try:
                    import wandb  # type: ignore

                    if getattr(wandb, "run", None):
                        log_payload = {
                            "teacher/latency_s": time.perf_counter() - t0,
                            "teacher/finish_reason": getattr(resp.choices[0], "finish_reason", None),
                            "teacher/model": self.cfg.deployment,
                        }
                        log_payload["teacher/empty_response"] = 1 if not content.strip() else 0
                        if usage is not None:
                            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                                v = getattr(usage, k, None)
                                if v is not None:
                                    log_payload[f"teacher/{k}"] = v
                        # Avoid emitting empty dicts.
                        if log_payload:
                            wandb.log(log_payload)
                except Exception:
                    # Never fail the request because of logging issues.
                    pass
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
