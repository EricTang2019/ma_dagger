from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import torch.multiprocessing as mp

# Ensure vLLM workers use spawn to avoid CUDA re-init issues in forked procs.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_START_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method may have been set elsewhere; ignore.
    pass

from transformers import AutoTokenizer
# vLLM 0.11.0 moved LLM/SamplingParams out of the top-level package.
try:  # pragma: no cover - import shim for multiple vLLM versions
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover
    from vllm.entrypoints.llm import LLM  # type: ignore
    from vllm.sampling_params import SamplingParams  # type: ignore

try:  # pragma: no cover - best-effort version detection
    from vllm import __version__ as _VLLM_VERSION
except Exception:  # pragma: no cover
    _VLLM_VERSION = "0.0.0"


def _version_tuple(text: str) -> Tuple[int, int, int]:
    parts = text.split(".")
    result = []
    for i in range(3):
        try:
            result.append(int(parts[i]))
        except Exception:
            result.append(0)
    return tuple(result)  # type: ignore[return-value]


_VLLM_SUPPORTS_RESET_PREFIX_CACHE = _version_tuple(_VLLM_VERSION) >= (0, 12, 0)

logger = logging.getLogger(__name__)
_SP_SUPPORTED_KWARGS = set(inspect.signature(SamplingParams).parameters.keys())
_SP_UNSUPPORTED_WARNED: set[str] = set()


class _BatchingVLLM:
    """Minimal async micro-batcher that groups compatible SamplingParams requests."""

    def __init__(
        self,
        llm: LLM,
        default_sp: SamplingParams,
        max_batch: int = 64,
        flush_ms: int = 5,
        reload_lock: Optional[asyncio.Lock] = None,
    ):
        self.llm = llm
        self.default_sp = default_sp
        self.max_batch = max_batch
        self.flush_ms = flush_ms
        self.q: asyncio.Queue = asyncio.Queue()
        self._reload_lock = reload_lock
        self._stash: List[Tuple[str, SamplingParams, asyncio.Future]] = []
        self._bg = asyncio.create_task(self._loop())

    async def submit(self, prompt: str, sp: Optional[SamplingParams] = None) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.q.put((prompt, sp or self.default_sp, fut))
        return await fut

    @staticmethod
    def _bucket_key(sp: SamplingParams) -> Tuple:
        stop = tuple(getattr(sp, "stop", []) or ())
        return (
            float(getattr(sp, "temperature", 0.0) or 0.0),
            float(getattr(sp, "top_p", 1.0) or 1.0),
            int(getattr(sp, "max_tokens", 0) or 0),
            stop,
            float(getattr(sp, "repetition_penalty", 1.0) or 1.0),
            float(getattr(sp, "presence_penalty", 0.0) or 0.0),
            float(getattr(sp, "frequency_penalty", 0.0) or 0.0),
        )

    async def _loop(self):
        while True:
            item = self._stash.pop() if self._stash else await self.q.get()
            prompt, sp, fut = item
            key = self._bucket_key(sp)
            batch = [(prompt, sp, fut)]
            t0 = asyncio.get_running_loop().time()

            while len(batch) < self.max_batch:
                timeout = self.flush_ms / 1000 - (asyncio.get_running_loop().time() - t0)
                if timeout <= 0:
                    break
                if self._stash:
                    next_item = self._stash.pop()
                else:
                    try:
                        next_item = await asyncio.wait_for(self.q.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                next_key = self._bucket_key(next_item[1])
                if next_key == key:
                    batch.append(next_item)
                else:
                    self._stash.append(next_item)

            prompts, sps, futures = zip(*batch)
            sp0 = sps[0]
            try:
                loop = asyncio.get_running_loop()
                if self._reload_lock:
                    async with self._reload_lock:
                        outs = await loop.run_in_executor(None, self.llm.generate, list(prompts), sp0)
                else:
                    outs = await loop.run_in_executor(None, self.llm.generate, list(prompts), sp0)
                texts = [o.outputs[0].text for o in outs]
                for txt, future in zip(texts, futures):
                    if not future.done():
                        future.set_result(txt)
            except Exception as err:  # pragma: no cover - bubble up errors
                for future in futures:
                    if not future.done():
                        future.set_exception(err)


def _filter_sampling_kwargs(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not extra:
        return {}
    filtered: Dict[str, Any] = {}
    for key, value in extra.items():
        if key in _SP_SUPPORTED_KWARGS:
            filtered[key] = value
        elif key not in _SP_UNSUPPORTED_WARNED:
            logger.warning("SamplingParams does not support kwarg '%s'; dropping it.", key)
            _SP_UNSUPPORTED_WARNED.add(key)
    return filtered


def trim_history(messages: List[Dict[str, str]], *, limit: int) -> List[Dict[str, str]]:
    if limit <= 0 or len(messages) <= limit:
        return messages
    trimmed = [messages[0]] if messages and messages[0].get("role") == "system" else []
    body = messages[len(trimmed):]
    trimmed.extend(body[-(limit - len(trimmed)):])
    return trimmed


@dataclass
class VLLMConfig:
    model: str
    tokenizer: Optional[str] = None
    tp: int = 1
    gpu_mem_util: float = 0.9
    max_model_len: int = 32768
    dtype: Optional[str] = None
    # Enable sleep mode so we can call llm.sleep() to release GPU memory between evals.
    enable_sleep_mode: bool = True
    trust_remote_code: bool = True
    temperature: float = 0.3
    top_p: float = 0.95
    max_new_tokens: int = 512
    batch_max: int = 64
    batch_flush_ms: int = 5


class VLLMChatEngine:
    """Async chat wrapper over vLLM LLM with hot reload + micro batching."""

    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self._reload_lock = asyncio.Lock()
        tok_name = cfg.tokenizer or cfg.model
        self.tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=cfg.trust_remote_code)
        llm_kwargs = dict(
            model=cfg.model,
            tokenizer=tok_name,
            tensor_parallel_size=cfg.tp,
            gpu_memory_utilization=cfg.gpu_mem_util,
            max_model_len=cfg.max_model_len,
            enable_sleep_mode=cfg.enable_sleep_mode,
            trust_remote_code=cfg.trust_remote_code,
        )
        if cfg.dtype:
            llm_kwargs["dtype"] = cfg.dtype
        self.llm = LLM(**llm_kwargs)
        self._set_default_sampling_params()
        self._batcher = _BatchingVLLM(
            self.llm,
            self.default_sp,
            max_batch=cfg.batch_max,
            flush_ms=cfg.batch_flush_ms,
            reload_lock=self._reload_lock,
        )

    def _token_budget(self, max_new_tokens: int) -> int:
        return max(256, self.cfg.max_model_len - max_new_tokens - 32)

    def _trim_to_budget(
        self,
        messages: List[Dict[str, str]],
        budget: int,
        chat_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        if budget <= 0 or not messages:
            return messages
        chat_kwargs = chat_kwargs or {}
        token_count = len(
            self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **chat_kwargs)
        )
        if token_count <= budget:
            return messages
        head = [messages[0]] if messages[0].get("role") == "system" else []
        body = messages[len(head):]
        start = 0
        while start < len(body):
            trial = head + body[start:]
            token_count = len(
                self.tok.apply_chat_template(trial, tokenize=True, add_generation_prompt=True, **chat_kwargs)
            )
            if token_count <= budget:
                return trial
            start += 1
        return (head + body[-1:]) if body else head

    async def get_model_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.cfg.temperature)
        top_p = kwargs.get("top_p", self.cfg.top_p)
        max_new_tokens = kwargs.get("max_tokens", self.cfg.max_new_tokens)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        sp_extra = _filter_sampling_kwargs(kwargs.get("sp_extra"))
        budget = self._token_budget(max_new_tokens)
        trimmed = self._trim_to_budget(messages, budget, chat_kwargs)
        prompt = self.tok.apply_chat_template(trimmed, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, **sp_extra)
        content = await self._batcher.submit(prompt, sp)
        return {"content": content, "raw": content}

    async def generate_batch(self, list_of_messages: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        temperature = kwargs.get("temperature", self.cfg.temperature)
        top_p = kwargs.get("top_p", self.cfg.top_p)
        max_new_tokens = kwargs.get("max_tokens", self.cfg.max_new_tokens)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        sp_extra = _filter_sampling_kwargs(kwargs.get("sp_extra"))
        budget = self._token_budget(max_new_tokens)
        trimmed_msgs = [self._trim_to_budget(m, budget, chat_kwargs) for m in list_of_messages]
        prompts = [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **chat_kwargs)
            for m in trimmed_msgs
        ]
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, **sp_extra)
        loop = asyncio.get_running_loop()
        if self._reload_lock:
            async with self._reload_lock:
                outs = await loop.run_in_executor(None, self.llm.generate, prompts, sp)
        else:
            outs = await loop.run_in_executor(None, self.llm.generate, prompts, sp)
        return [o.outputs[0].text for o in outs]

    async def hot_reload_from_dir(self, new_model_dir: str, new_tokenizer_dir: Optional[str] = None):
        async with self._reload_lock:
            overrides = {
                "model_config": {"model": new_model_dir},
                "load_config": {"load_format": "auto"},
            }
            if new_tokenizer_dir:
                overrides["model_config"]["tokenizer"] = new_tokenizer_dir
            cache_reset = False
            fallback_reason: Optional[Exception] = None
            if hasattr(self.llm, "collective_rpc"):
                try:
                    try:
                        self.llm.collective_rpc("update_config", overrides=overrides)
                    except TypeError:
                        self.llm.collective_rpc("update_config", args=(overrides,))
                    self.llm.collective_rpc("reload_weights")
                    if _VLLM_SUPPORTS_RESET_PREFIX_CACHE:
                        try:
                            self.llm.collective_rpc("reset_prefix_cache")
                            cache_reset = True
                        except Exception as cache_err:  # pragma: no cover - best effort
                            logger.debug("reset_prefix_cache unavailable: %s", cache_err)
                    else:
                        logger.debug(
                            "Skipping reset_prefix_cache; vLLM %s does not expose this RPC.",
                            _VLLM_VERSION,
                        )
                    self._update_cfg_after_reload(new_model_dir, new_tokenizer_dir)
                    self._set_default_sampling_params()
                    return
                except Exception as rpc_err:
                    fallback_reason = rpc_err
                    logger.warning("collective_rpc hot reload failed, falling back: %s", rpc_err)
            else:
                logger.debug("collective_rpc not available; falling back to manual weight load.")
            try:
                model, model_config = _get_vllm_model_and_config(self.llm)
            except RuntimeError as inner_err:
                if fallback_reason is not None:
                    raise RuntimeError(
                        "collective_rpc hot reload failed and fallback is unavailable on this vLLM build."
                    ) from fallback_reason
                raise inner_err
            _load_weights_from_dir(model, model_config, new_model_dir)
            if not cache_reset:
                cache = getattr(self.llm.llm_engine, "cache_engine", None)
                if cache and hasattr(cache, "reset"):
                    try:
                        cache.reset()
                    except Exception as cache_err:  # pragma: no cover
                        logger.debug("prefix cache reset failed on fallback: %s", cache_err)
            self._update_cfg_after_reload(new_model_dir, new_tokenizer_dir)
            self._set_default_sampling_params()

    async def sleep(self, level: int = 1):
        """Offload model weights to free GPU memory (requires enable_sleep_mode)."""
        if not hasattr(self.llm, "sleep"):
            raise RuntimeError("vLLM LLM.sleep is unavailable; upgrade vLLM or disable sleep.")
        async with self._reload_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.llm.sleep, level)

    async def wake_up(self):
        """Reload model weights after sleep()."""
        if not hasattr(self.llm, "wake_up"):
            raise RuntimeError("vLLM LLM.wake_up is unavailable; upgrade vLLM or disable sleep.")
        async with self._reload_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.llm.wake_up)

    def _update_cfg_after_reload(self, new_model_dir: str, new_tokenizer_dir: Optional[str]):
        if new_tokenizer_dir:
            self.tok = AutoTokenizer.from_pretrained(
                new_tokenizer_dir, trust_remote_code=self.cfg.trust_remote_code
            )
            self.cfg = dataclasses.replace(self.cfg, model=new_model_dir, tokenizer=new_tokenizer_dir)
        else:
            self.cfg = dataclasses.replace(self.cfg, model=new_model_dir)

    def _set_default_sampling_params(self):
        self.default_sp = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )
        if hasattr(self, "_batcher"):
            self._batcher.default_sp = self.default_sp


def _get_vllm_model_and_config(llm: LLM):
    eng = llm.llm_engine
    try:
        model = eng.driver_worker.worker.model_runner.model
        model_config = getattr(eng, "model_config", None) or eng.vllm_config.model_config
        return model, model_config
    except Exception:
        pass
    try:
        model = eng.model_executor.driver_worker.worker.model_runner.model
        model_config = eng.model_config
        return model, model_config
    except Exception:
        pass
    try:
        model = eng.model_runner.model
        model_config = getattr(eng, "model_config", None)
        return model, model_config
    except Exception as err:
        raise RuntimeError("Unable to locate vLLM model internals for hot reload") from err


def _load_weights_from_dir(model, model_config, new_model_dir: str):
    try:
        from vllm.model_executor.model_loader import load_weights as vllm_load_weights
    except Exception as err:  # pragma: no cover - depends on vLLM install
        raise RuntimeError("vLLM weight loader not available in this version") from err
    vllm_load_weights(model, model_config=model_config, load_format="auto", load_dir=new_model_dir)


async def call_engine(engine: VLLMChatEngine, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    return await engine.get_model_response(messages, **kwargs)
