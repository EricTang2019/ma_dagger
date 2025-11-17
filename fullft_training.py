from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

logger = logging.getLogger(__name__)


def _set_cfg_field(cfg: Any, dotted: str, value: Any):
    try:
        OmegaConf.update(cfg, dotted, value, merge=True)
    except Exception:
        pass


def _coerce_single_path_str(value: Any, *, strict: bool = False, label: str = "path") -> str:
    original = value
    while True:
        if isinstance(value, (ListConfig, DictConfig)):
            try:
                value = OmegaConf.to_container(value, resolve=True)
            except Exception:
                value = list(value) if isinstance(value, ListConfig) else value
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            seq = [item for item in value if item is not None and str(item).strip()]
            if not seq:
                if strict:
                    raise ValueError(f"{label} resolved to an empty sequence from {original!r}")
                return ""
            if len(seq) > 1:
                if strict:
                    raise ValueError(f"{label} expected a single path but received {original!r}")
                logger.debug("Coercing %s sequence to its first element: %r", label, original)
            value = seq[0]
            continue
        break
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if strict and not value:
        raise ValueError(f"{label} resolved to an empty string from {original!r}")
    return value


def _normalize_path_list(value: Any) -> List[str]:
    if isinstance(value, (ListConfig, DictConfig)):
        try:
            value = OmegaConf.to_container(value, resolve=True)
        except Exception:
            value = list(value) if isinstance(value, ListConfig) else value

    def _flatten(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            return [str(v)]
        if isinstance(v, ListConfig):
            v = list(v)
        if isinstance(v, (list, tuple, set)):
            items: List[str] = []
            for item in v:
                items.extend(_flatten(item))
            return items
        return [str(v)]

    flat = [p for p in _flatten(value) if p]
    try:
        flat = list(OmegaConf.to_container(flat, resolve=True))  # type: ignore[arg-type]
    except Exception:
        pass
    return [str(p) for p in flat]


def _count_parquet_rows(paths: Sequence[str], column_hint: str = "messages") -> int:
    total = 0
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=[column_hint])
        except Exception:
            df = pd.read_parquet(path)
        total += len(df)
    return int(total)


def _ensure_torchrun_env_defaults(target_env: Optional[Dict[str, str]] = None):
    env = target_env if target_env is not None else os.environ

    def _set_if_invalid(key: str, default: str, allow_zero: bool = False):
        val = env.get(key)
        try:
            if key == "MASTER_PORT":
                port = int(str(val))
                if port <= 0 or port > 65535:
                    raise ValueError
            elif key in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}:
                num = int(str(val))
                min_allowed = 0 if allow_zero else 1
                if num < min_allowed:
                    raise ValueError
            else:
                if val is None or not str(val).strip():
                    raise ValueError
                return
        except Exception:
            env[key] = default

    if not env.get("MASTER_ADDR"):
        env["MASTER_ADDR"] = "127.0.0.1"
    else:
        addr = env.get("MASTER_ADDR", "").strip()
        if not addr:
            env["MASTER_ADDR"] = "127.0.0.1"
    _set_if_invalid("MASTER_PORT", "29500")
    _set_if_invalid("RANK", "0", allow_zero=True)
    _set_if_invalid("LOCAL_RANK", "0", allow_zero=True)
    _set_if_invalid("WORLD_SIZE", "1", allow_zero=False)


# Monkey-patch verl.utils to better handle Hydra containers.
try:  # pragma: no cover - defensive patching
    from verl.utils import fs as _verl_fs
    from verl.utils.dataset import multiturn_sft_dataset as _verl_ds

    _orig_is_non_local = _verl_fs.is_non_local
    _orig_copy_local_path_from_hdfs = _verl_fs.copy_local_path_from_hdfs
    _orig_mt_dataset_init = getattr(_verl_ds, "MultiTurnSFTDataset", None).__init__ if _verl_ds else None

    def _safe_is_non_local(path):
        try:
            coerced = _coerce_single_path_str(path, strict=False, label="path")
        except Exception:
            coerced = str(path)
        return _orig_is_non_local(coerced)

    def _safe_copy_local_path_from_hdfs(src, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False):
        coerced_src = _coerce_single_path_str(src, strict=True, label="parquet file")
        out = _orig_copy_local_path_from_hdfs(
            coerced_src,
            cache_dir=cache_dir,
            filelock=filelock,
            verbose=verbose,
            always_recopy=always_recopy,
        )
        return _coerce_single_path_str(out or coerced_src, strict=True, label="parquet file")

    _verl_fs.is_non_local = _safe_is_non_local
    _verl_fs.copy_local_path_from_hdfs = _safe_copy_local_path_from_hdfs
    if _verl_ds and hasattr(_verl_ds, "copy_local_path_from_hdfs"):
        _verl_ds.copy_local_path_from_hdfs = _safe_copy_local_path_from_hdfs

    if _orig_mt_dataset_init:
        def _safe_mt_dataset_init(self, parquet_files, tokenizer, config=None):
            files = [
                _coerce_single_path_str(p, strict=True, label="parquet file")
                for p in _normalize_path_list(parquet_files)
            ]
            return _orig_mt_dataset_init(self, files, tokenizer, config)

        _verl_ds.MultiTurnSFTDataset.__init__ = _safe_mt_dataset_init
except Exception:
    pass


def _infer_train_world_size(train_cuda: Optional[str]) -> int:
    def _safe_parse(spec: Optional[str]) -> Optional[List[str]]:
        if not spec:
            return None
        try:
            return [s.strip() for s in spec.split(",") if s.strip()]
        except ValueError:
            return None

    devices = _safe_parse(train_cuda)
    if devices:
        return len(devices)
    env_devices = _safe_parse(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if env_devices:
        return len(env_devices)
    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env and world_size_env.isdigit():
        return max(1, int(world_size_env))
    try:
        import torch as _torch

        cnt = _torch.cuda.device_count()
        if cnt > 0:
            return cnt
    except Exception:
        pass
    return 1


def run_fullft_one_round(
    which: str,
    sft_path: str,
    base_model_path: str,
    project_name: str,
    experiment_name: str,
    out_dir: str,
    config_name: str = "agent_sft_trainer",
    config_override: Optional[List[str]] = None,
    use_subprocess: bool = True,
    train_cuda: Optional[str] = None,
) -> str:
    assert which in {"gen", "ver"}
    if not os.path.exists(sft_path):
        raise FileNotFoundError(f"SFT parquet not found: {sft_path}")
    if use_subprocess:
        return _run_fullft_one_round_subprocess(
            which=which,
            sft_path=sft_path,
            base_model_path=base_model_path,
            project_name=project_name,
            experiment_name=experiment_name,
            out_dir=out_dir,
            config_name=config_name,
            config_override=config_override,
            train_cuda=train_cuda,
        )
    return _run_fullft_one_round_inner(
        which=which,
        sft_path=sft_path,
        base_model_path=base_model_path,
        project_name=project_name,
        experiment_name=experiment_name,
        out_dir=out_dir,
        config_name=config_name,
        config_override=config_override,
    )


def _run_fullft_one_round_subprocess(
    which: str,
    sft_path: str,
    base_model_path: str,
    project_name: str,
    experiment_name: str,
    out_dir: str,
    config_name: str,
    config_override: Optional[List[str]],
    train_cuda: Optional[str],
) -> str:
    parsed_train_cuda = None
    if train_cuda:
        parsed_train_cuda = [s.strip() for s in train_cuda.split(",") if s.strip()]
        train_cuda = ",".join(parsed_train_cuda)
    env = os.environ.copy()
    _ensure_torchrun_env_defaults(env)
    master_addr = env.get("MASTER_ADDR", "127.0.0.1")
    master_port = env.get("MASTER_PORT", "29500")
    torchrun_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        f"--nproc_per_node={_infer_train_world_size(train_cuda)}",
        __file__,
        "_train_once",
        "--which",
        which,
        "--sft_path",
        sft_path,
        "--base_model_path",
        base_model_path,
        "--project_name",
        project_name,
        "--experiment_name",
        experiment_name,
        "--out_dir",
        out_dir,
        "--config_name",
        config_name,
    ]
    if config_override:
        torchrun_cmd.extend(["--config_override", *config_override])
    if train_cuda:
        env["CUDA_VISIBLE_DEVICES"] = train_cuda
    else:
        env.setdefault("CUDA_DEVICE_ORDER", os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"))
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    subprocess.run(torchrun_cmd, check=True, env=env)
    return _locate_latest_ckpt(out_dir, project_name, experiment_name)


def _run_fullft_one_round_inner(
    which: str,
    sft_path: str,
    base_model_path: str,
    project_name: str,
    experiment_name: str,
    out_dir: str,
    config_name: str = "agent_sft_trainer",
    config_override: Optional[List[str]] = None,
) -> str:
    from hydra import compose, initialize_config_module

    _ensure_torchrun_env_defaults()
    path_literal = json.dumps(sft_path)
    overrides = list(config_override or [])
    overrides.extend([
        f"data.train_files=[{path_literal}]",
        f"data.val_files=[{path_literal}]",
        "data.multiturn.enable=true",
        f"model.partial_pretrain={base_model_path}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.default_local_dir={out_dir}",
    ])

    with initialize_config_module(version_base=None, config_module="rllm.trainer.config"):
        cfg = compose(config_name=config_name, overrides=overrides)

    prev_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        train_files = list(_normalize_path_list(cfg.data.train_files))
        val_files = list(_normalize_path_list(cfg.data.val_files))
        OmegaConf.update(cfg, "data.train_files", train_files, merge=False)
        OmegaConf.update(cfg, "data.val_files", val_files, merge=False)
        logger.info("SFT train parquet files (%d): %s", len(train_files), train_files)
        logger.info("SFT val parquet files   (%d): %s", len(val_files), val_files)
        if not all(isinstance(p, str) for p in train_files):
            raise TypeError(f"data.train_files must be List[str]; got {train_files!r}")
        if not all(isinstance(p, str) for p in val_files):
            raise TypeError(f"data.val_files must be List[str]; got {val_files!r}")
        train_rows = _count_parquet_rows(train_files)
        if train_rows <= 0:
            raise RuntimeError(f"SFT parquet '{train_files}' produced zero rows; cannot train.")
        val_rows = _count_parquet_rows(val_files) if val_files else 0
        logger.info("SFT row counts — train: %d, val: %d", train_rows, val_rows)
        micro_batch = int(OmegaConf.select(cfg, "data.micro_batch_size_per_gpu", default=4) or 4)
        if micro_batch <= 0:
            micro_batch = 1
        new_micro_batch = min(max(1, micro_batch), train_rows)
        if new_micro_batch != micro_batch:
            logger.warning(
                "micro_batch_size_per_gpu reduced from %d to %d to fit %d training rows.",
                micro_batch,
                new_micro_batch,
                train_rows,
            )
            OmegaConf.update(cfg, "data.micro_batch_size_per_gpu", new_micro_batch, merge=False)
        train_batch = int(OmegaConf.select(cfg, "data.train_batch_size", default=256) or 256)
        if train_batch <= 0:
            train_batch = new_micro_batch
        new_train_batch = max(new_micro_batch, min(train_batch, train_rows))
        if new_train_batch != train_batch:
            logger.warning(
                "train_batch_size reduced from %d to %d to avoid zero steps per epoch (%d rows).",
                train_batch,
                new_train_batch,
                train_rows,
            )
            OmegaConf.update(cfg, "data.train_batch_size", new_train_batch, merge=False)
        if new_train_batch % new_micro_batch != 0:
            adjusted = (train_rows // new_micro_batch) * new_micro_batch
            if adjusted <= 0:
                adjusted = new_micro_batch
            if adjusted != new_train_batch:
                logger.warning(
                    "train_batch_size adjusted from %d to %d to be a multiple of micro_batch_size_per_gpu=%d.",
                    new_train_batch,
                    adjusted,
                    new_micro_batch,
                )
                new_train_batch = adjusted
                OmegaConf.update(cfg, "data.train_batch_size", new_train_batch, merge=False)
        base_model = base_model_path
        for field in (
            "actor_rollout_ref.model.path",
            "actor_rollout_ref.actor.path",
            "actor_rollout_ref.rollout.model",
            "actor_rollout_ref.rollout.tokenizer",
        ):
            _set_cfg_field(cfg, field, base_model)
        _set_cfg_field(cfg, "actor_rollout_ref.model.use_shm", False)
        _set_cfg_field(cfg, "critic.path", base_model)

        strategy = OmegaConf.select(cfg, "model.strategy", default=None)
        if strategy:
            for field in (
                "actor_rollout_ref.model.strategy",
                "actor_rollout_ref.actor.strategy",
                "actor_rollout_ref.rollout.strategy",
                "critic.strategy",
            ):
                _set_cfg_field(cfg, field, strategy)
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        except ValueError:
            world_size = 1
        _set_cfg_field(cfg, "trainer.n_gpus_per_node", max(1, world_size))
        _set_cfg_field(cfg, "trainer.nnodes", 1)
    finally:
        OmegaConf.set_struct(cfg, prev_struct)

    trainer = AgentSFTTrainer(config=cfg)
    trainer.train()
    return _locate_latest_ckpt(out_dir, project_name, experiment_name)


def _locate_latest_ckpt(out_dir: str, project_name: str, experiment_name: str) -> str:
    exp_dir = Path(out_dir) / project_name / experiment_name
    if not exp_dir.exists():
        alt = Path(out_dir) / experiment_name
        base_dir = alt if alt.exists() else Path(out_dir)
    else:
        base_dir = exp_dir

    ckpts = [
        p
        for p in base_dir.rglob("*")
        if p.is_dir() and (re.search(r"global_step_\d+", p.name) or re.search(r"epoch_\d+", p.name))
    ]
    if not ckpts:
        return str(base_dir)
    latest = sorted(ckpts, key=lambda p: (p.stat().st_mtime, p.name))[-1]
    return str(latest)


def _train_once_cli(args):
    _run_fullft_one_round_inner(
        which=args.which,
        sft_path=args.sft_path,
        base_model_path=args.base_model_path,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        out_dir=args.out_dir,
        config_name=args.config_name,
        config_override=args.config_override,
    )


__all__ = [
    "run_fullft_one_round",
    "_train_once_cli",
    "_locate_latest_ckpt",
    "_ensure_torchrun_env_defaults",
]
