from __future__ import annotations

import inspect
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
_HF_LOCAL_TENSOR_SENTINEL = ".hf_local_tensors.ok"
_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters
try:  # pragma: no cover - torch optional feature
    from torch.distributed.tensor import DTensor as _TorchDTensor
except Exception:  # pragma: no cover - best effort import
    _TorchDTensor = None


def _has_hf_config_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for name in ("config.json", "params.json"):
        if (path / name).is_file():
            return True
    return False


def _hf_weights_exist(path: Path) -> bool:
    if not path.is_dir():
        return False
    if list(path.glob("*.safetensors")):
        return True
    if list(path.glob("*.bin")):
        return True
    return False


def _torch_load_state_dict(path: Path):
    kwargs = {"map_location": "cpu"}
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _maybe_dtensor_to_tensor(value: Any):
    if _TorchDTensor is not None:
        try:
            if isinstance(value, _TorchDTensor):
                return value.to_local()
        except Exception:
            pass
    return value


def _convert_checkpoint_to_local_tensors(
    source: Path,
    target: Path,
    sentinel: Path,
) -> bool:
    try:
        state = _torch_load_state_dict(source)
    except Exception as err:
        logger.warning("Failed to load checkpoint shard %s: %s", source, err)
        return False
    changed = False
    if isinstance(state, dict):
        for key in list(state.keys()):
            new_value = _maybe_dtensor_to_tensor(state[key])
            if new_value is not state[key]:
                state[key] = new_value
                changed = True
    if changed or not target.exists() or source != target:
        torch.save(state, target)
    sentinel.touch()
    return True


def maybe_materialize_hf_weights(fsdp_dir: Path) -> Path:
    hf_dir = fsdp_dir / "huggingface"
    if not hf_dir.exists() or not _has_hf_config_dir(hf_dir):
        return fsdp_dir
    target = hf_dir / "pytorch_model.bin"
    sentinel = hf_dir / _HF_LOCAL_TENSOR_SENTINEL
    shards = sorted(fsdp_dir.glob("model_world_size_*_rank_*.pt"))
    if sentinel.exists() and _hf_weights_exist(hf_dir):
        return hf_dir
    if len(shards) == 1:
        if _convert_checkpoint_to_local_tensors(shards[0], target, sentinel):
            logger.info("Materialized HuggingFace weights at %s from %s.", target, shards[0])
            return hf_dir
    elif shards:
        logger.warning(
            "Cannot auto-materialize HF weights for multi-rank checkpoint (%s shards).",
            len(shards),
        )
    if _hf_weights_exist(hf_dir):
        if not sentinel.exists():
            if _convert_checkpoint_to_local_tensors(target, target, sentinel):
                logger.info("Rewrote HuggingFace weights at %s to ensure local tensors.", target)
        return hf_dir
    return fsdp_dir


def resolve_model_dir_for_vllm(path: str) -> str:
    root = Path(path)
    if not root.exists():
        logger.debug(
            "Checkpoint path %s does not exist locally; returning raw spec for vLLM to resolve.",
            path,
        )
        return path

    def _pick_dir(candidate: Path) -> Optional[Path]:
        if _has_hf_config_dir(candidate) and _hf_weights_exist(candidate):
            return candidate
        hf_sub = candidate / "huggingface"
        if _has_hf_config_dir(hf_sub) and _hf_weights_exist(hf_sub):
            return hf_sub
        return None

    def _try_materialize(candidate: Path) -> Optional[Path]:
        resolved = maybe_materialize_hf_weights(candidate)
        if resolved is not candidate:
            pick = _pick_dir(resolved)
            if pick:
                return pick
        return _pick_dir(candidate)

    first = _try_materialize(root)
    if first:
        return str(first)

    # Search nested directories (newest first) for a valid export.
    subdirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: (p.stat().st_mtime, p.as_posix()),
        reverse=True,
    )
    for sub in subdirs:
        pick = _try_materialize(sub)
        if pick:
            return str(pick)

    raise FileNotFoundError(
        f"No HuggingFace-format checkpoint (config.json + weights) found under '{root}'."
    )


def cleanup_checkpoint_dir(candidate: Optional[Path], latest: Path, base_dir: Path):
    if candidate is None:
        return
    try:
        cand_resolved = candidate.resolve(strict=False)
        latest_resolved = latest.resolve(strict=False)
        base_resolved = base_dir.resolve(strict=False)
    except OSError:
        cand_resolved = candidate
        latest_resolved = latest
        base_resolved = base_dir
    if cand_resolved == latest_resolved:
        return
    try:
        cand_resolved.relative_to(base_resolved)
    except ValueError:
        return
    if not cand_resolved.exists():
        return
    try:
        shutil.rmtree(cand_resolved, ignore_errors=True)
        logger.info("Removed old checkpoint directory: %s", cand_resolved)
    except Exception as err:
        logger.warning("Failed to remove old checkpoint dir %s: %s", cand_resolved, err)


__all__ = [
    "resolve_model_dir_for_vllm",
    "cleanup_checkpoint_dir",
    "maybe_materialize_hf_weights",
]
