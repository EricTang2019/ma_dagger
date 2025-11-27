# -*- coding: utf-8 -*-
"""
Gen–Ver (Generator–Verifier) DAgger with full-parameter SFT and
**vLLM rollout for students (generator / verifier) + async Azure TRAPI teacher (default)**,
plus **hot in-place weight reload** after each SFT round.

This version applies the reviewer’s suggestions:
- FIX: Verifier SFT rows now use the true *generator* message (store `generator_message` in step.info).
- PERF: Add a minimal **micro-batcher** so calls coalesce into one `LLM.generate([...])`
        batch instead of many tiny calls. (Safer than multi-threading one LLM.)
- RELIABILITY: Hot-reload uses vLLM V1 `collective_rpc` with **positional args**, then
               calls `reset_prefix_cache`. Optional tokenizer sync supported.
- BEHAVIOR: Keeps your previous rLLM workflow, teacher relabeling, and alternating 2T SFT.

Quick start
-----------
# 0) Install vLLM >= 0.7 (prefer 0.8+), rLLM + VERL + Transformers.
# 1) Iterate (odd=train generator, even=train verifier):
# python gen_ver_iterative_vllm_fullft_fixed.py iterate \
#   --dataset_name deepscaler_math --split train --rounds 10 --batch_tasks 256 \
#   --teacher_base Qwen/Qwen3-4B --gen_base Qwen/Qwen3-0.6B --ver_base Qwen/Qwen3-0.6B \
#   --out_dir runs/genver_fullft --project_name genver_fullft
# 2) Collect once:
# python gen_ver_iterative_vllm_fullft_fixed.py collect \
#   --dataset_name deepscaler_math --split train --num_tasks 512 \
#   --teacher_base Qwen/Qwen3-4B --save_prefix datasets/collect
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.data.dataset import DatasetRegistry

from checkpoint_utils import cleanup_checkpoint_dir, resolve_model_dir_for_vllm
from data_utils import append_parquet
from fullft_training import _train_once_cli, run_fullft_one_round
from genver_workflow import episodes_to_sft_rows, rollout_with_workflow_engine
from gpu_utils import (
    cuda_visible_devices,
    devices_overlap,
    ensure_tp_fit,
    parse_cuda_list,
    warn_overlap,
)
from triapi_engine import TriApiChatEngine, TriApiConfig
from vllm_engine import VLLMChatEngine, VLLMConfig

logger = logging.getLogger(__name__)


def _maybe_init_wandb(args: argparse.Namespace):
    """Start a lightweight wandb run so teacher/rollout metrics are captured."""
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    if getattr(wandb, "run", None):
        run = wandb.run
        print(
            f"[wandb] active run entity={getattr(run, 'entity', None)} "
            f"project={getattr(run, 'project', None)} name={getattr(run, 'name', None)} "
            f"url={getattr(run, 'url', None)}"
        )
        logger.info(
            "wandb already active: entity=%s project=%s name=%s url=%s",
            getattr(run, "entity", None),
            getattr(run, "project", None),
            getattr(run, "name", None),
            getattr(run, "url", None),
        )
        return run
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        run = wandb.init(
            project=args.project_name or "madagger",
            name=args.experiment_name,
            config={
                "dataset": args.dataset_name,
                "eval_dataset": getattr(args, "eval_dataset", None),
                "gen_base": getattr(args, "gen_base", None),
                "ver_base": getattr(args, "ver_base", None),
                "teacher_backend": getattr(args, "teacher_backend", None),
            },
            reinit=True,
        )
        print(
            f"[wandb] init entity={getattr(run, 'entity', None)} "
            f"project={getattr(run, 'project', None)} name={getattr(run, 'name', None)} "
            f"url={getattr(run, 'url', None)}"
        )
        logger.info(
            "wandb init: entity=%s project=%s name=%s url=%s",
            getattr(run, "entity", None),
            getattr(run, "project", None),
            getattr(run, "name", None),
            getattr(run, "url", None),
        )
        return run
    except Exception as err:  # pragma: no cover - best effort
        logger.warning("wandb init failed; continuing without trace: %s", err)
        return None


def sample_tasks(dataset_name: str, split: str, n: int, seed: int = 0) -> List[Dict[str, Any]]:
    ds = DatasetRegistry.load_dataset(dataset_name, split)
    if ds is None:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found. Ensure you've registered it via rLLM examples (e.g., prepare_math_data.py)."
        )
    data = ds.get_data() if hasattr(ds, "get_data") else list(ds)
    rng = random.Random(seed)
    if n < len(data):
        idx = list(range(len(data)))
        rng.shuffle(idx)
        idx = idx[:n]
        data = [data[i] for i in idx]
    tasks: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        q = row.get("question") or row.get("query")
        gt = row.get("ground_truth")
        tasks.append({"uid": f"{split}_{i}", "question": q, "ground_truth": gt})
    return tasks


def maybe_register_eval_dataset(name: str, split: str) -> None:
    """Register a small eval dataset on the fly if it isn't already present."""
    if DatasetRegistry.load_dataset(name, split) is not None:
        return
    if name.lower() == "math500":
        from datasets import load_dataset  # lazy import

        # The HF repo only has a "test" split; fall back to it if a different
        # split is requested.
        math_split = split if split else "test"
        if math_split not in {"test"}:
            math_split = "test"
        ds = load_dataset("HuggingFaceH4/MATH-500", split=math_split)

        def to_row(ex):
            return {
                "question": ex.get("problem") or ex.get("question") or ex.get("input"),
                "ground_truth": ex.get("solution") or ex.get("answer") or ex.get("output"),
                "data_source": "math500",
            }

        ds = ds.map(to_row)
        DatasetRegistry.register_dataset(name, ds, split)
    else:
        # Leave silently if we don't know how to auto-register this dataset.
        return


async def run_eval(
    args: argparse.Namespace,
    gen_engine,
    ver_engine,
    teacher_engine,
    prefix: str,
):
    """Lightweight eval: rollout and compute accuracy (is_correct flag)."""
    maybe_register_eval_dataset(args.eval_dataset, args.eval_split)
    tasks = sample_tasks(args.eval_dataset, args.eval_split, args.eval_num_tasks, seed=args.seed)
    print(
        f"[eval:start-{prefix}] dataset={args.eval_dataset} split={args.eval_split} "
        f"n={len(tasks)}"
    )
    episodes = await rollout_with_workflow_engine(
        tasks=tasks,
        gen_engine=gen_engine,
        ver_engine=ver_engine,
        teacher_engine=teacher_engine,
        max_turns=args.max_turns,
        stop_on_verifier_fix=args.stop_on_verifier_fix,
        collect_for="none",  # skip teacher labels during eval
        parallel=min(args.parallel, 32),
        retry=1,
    )
    acc = sum(1 for ep in episodes if getattr(ep, "is_correct", False)) / max(len(episodes), 1)
    print(
        f"[eval:{prefix}] dataset={args.eval_dataset} split={args.eval_split} "
        f"n={len(episodes)} acc={acc:.3f}"
    )
    try:
        import wandb  # type: ignore

        if getattr(wandb, "run", None):
            wandb.log({f"eval/{prefix}_acc": acc, f"eval/{prefix}_n": len(episodes)})
    except Exception:
        pass


async def run_iterate(args: argparse.Namespace):
    maybe_register_eval_dataset(args.eval_dataset, args.eval_split)
    wandb_run = _maybe_init_wandb(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_sft_path = str(out_dir / "gen_sft.parquet") if not args.gen_sft_path else args.gen_sft_path
    ver_sft_path = str(out_dir / "ver_sft.parquet") if not args.ver_sft_path else args.ver_sft_path

    teacher_backend = getattr(args, "teacher_backend", "triapi").lower()
    teacher_devices = (
        parse_cuda_list(getattr(args, "teacher_cuda", None)) if teacher_backend == "vllm" else None
    )
    gen_devices = parse_cuda_list(getattr(args, "gen_cuda", None))
    ver_devices = parse_cuda_list(getattr(args, "ver_cuda", None))
    teacher_gpu_mem = (
        args.teacher_gpu_mem_util if teacher_backend == "vllm" and getattr(args, "teacher_gpu_mem_util", None) is not None else args.gpu_mem_util
    ) if teacher_backend == "vllm" else None
    gen_gpu_mem = args.gen_gpu_mem_util if getattr(args, "gen_gpu_mem_util", None) is not None else args.gpu_mem_util
    ver_gpu_mem = args.ver_gpu_mem_util if getattr(args, "ver_gpu_mem_util", None) is not None else args.gpu_mem_util

    if teacher_backend == "vllm":
        ensure_tp_fit(teacher_devices, args.tp_t, "teacher")
    ensure_tp_fit(gen_devices, args.tp_s, "generator")
    ensure_tp_fit(ver_devices, args.tp_s, "verifier")
    if teacher_backend == "vllm":
        if devices_overlap(teacher_devices, gen_devices):
            warn_overlap("teacher & generator")
        if devices_overlap(teacher_devices, ver_devices):
            warn_overlap("teacher & verifier")
    if devices_overlap(gen_devices, ver_devices):
        warn_overlap("generator & verifier")

    if teacher_backend == "vllm":
        with cuda_visible_devices(teacher_devices):
            teacher_engine = VLLMChatEngine(
                VLLMConfig(
                    model=args.teacher_base,
                    tokenizer=args.teacher_tokenizer,
                    tp=args.tp_t,
                    gpu_mem_util=teacher_gpu_mem if teacher_gpu_mem is not None else args.gpu_mem_util,
                    max_model_len=args.max_model_len,
                    temperature=args.t_temp,
                    top_p=0.95,
                )
            )
    else:
        teacher_engine = TriApiChatEngine(
            TriApiConfig(
                instance=args.teacher_triapi_instance,
                deployment=args.teacher_triapi_deployment,
                scope=args.teacher_triapi_scope,
                api_version=args.teacher_triapi_api_version,
                temperature=None,  # omit temperature so TRAPI uses its default (avoid 400s)
                top_p=1.0,
                max_output_tokens=args.teacher_triapi_max_output_tokens,
                timeout_s=args.teacher_triapi_timeout,
                max_retries=args.teacher_triapi_retries,
                max_parallel=args.teacher_triapi_max_parallel,
            )
        )
    with cuda_visible_devices(gen_devices):
        gen_engine = VLLMChatEngine(
            VLLMConfig(
                model=args.gen_base,
                tokenizer=args.gen_tokenizer,
                tp=args.tp_s,
                gpu_mem_util=gen_gpu_mem,
                max_model_len=args.max_model_len,
                temperature=0.3,
                top_p=0.95,
            )
        )
    with cuda_visible_devices(ver_devices):
        ver_engine = VLLMChatEngine(
            VLLMConfig(
                model=args.ver_base,
                tokenizer=args.ver_tokenizer,
                tp=args.tp_s,
                gpu_mem_util=ver_gpu_mem,
                max_model_len=args.max_model_len,
                temperature=0.2,
                top_p=0.9,
            )
        )

    gen_ckpt = resolve_model_dir_for_vllm(args.gen_base)
    ver_ckpt = resolve_model_dir_for_vllm(args.ver_base)
    prev_gen_ckpt_root: Optional[Path] = None
    prev_ver_ckpt_root: Optional[Path] = None

    try:
        if args.eval_before_train:
            await run_eval(args, gen_engine, ver_engine, teacher_engine, prefix="pretrain")

        rounds = args.rounds * 2
        for r in range(1, rounds + 1):
            role = "gen" if r % 2 == 1 else "ver"
            print(f"\n[round {r:03d}] role={role} — sampling, rollout, teacher relabel, full-FT, hot-reload")
            logger.info("[round %03d] start role=%s", r, role)
            print(f"[round {r:03d}] START role={role}")

            tasks = sample_tasks(args.dataset_name, args.split, args.batch_tasks, seed=args.seed + r)
            logger.info("[round %03d] sampled %d tasks", r, len(tasks))
            print(f"[round {r:03d}] sampled {len(tasks)} tasks")
            episodes = await rollout_with_workflow_engine(
                tasks=tasks,
                gen_engine=gen_engine,
                ver_engine=ver_engine,
                teacher_engine=teacher_engine,
                max_turns=args.max_turns,
                stop_on_verifier_fix=args.stop_on_verifier_fix,
                collect_for=role if args.collect_for == "auto" else args.collect_for,
                parallel=args.parallel,
                retry=1,
            )
            gen_rows, ver_rows = episodes_to_sft_rows(episodes)
            if role in {"gen", "both"} and gen_rows:
                append_parquet(gen_rows, gen_sft_path)
                print(f"[round {r:03d}] +{len(gen_rows)} gen rows -> {gen_sft_path}")
            if role in {"ver", "both"} and ver_rows:
                append_parquet(ver_rows, ver_sft_path)
                print(f"[round {r:03d}] +{len(ver_rows)} ver rows -> {ver_sft_path}")

            exp_name = f"{args.experiment_name}_{role}_r{r:03d}"
            if role == "gen" and Path(gen_sft_path).exists():
                logger.info("[round %03d] SFT training generator rows=%d", r, len(gen_rows))
                print(f"[round {r:03d}] SFT start (generator) rows={len(gen_rows)}")
                new_ckpt = run_fullft_one_round(
                    which="gen",
                    sft_path=gen_sft_path,
                    base_model_path=gen_ckpt,
                    project_name=args.project_name,
                    experiment_name=exp_name,
                    out_dir=args.out_dir,
                    config_name=args.config_name,
                    config_override=args.config_override,
                    use_subprocess=not args.train_inline,
                    train_cuda=args.train_cuda,
                )
                new_ckpt_root = Path(new_ckpt)
                gen_ckpt = resolve_model_dir_for_vllm(new_ckpt)
                logger.info("[round %03d] hot-reload generator from %s", r, gen_ckpt)
                print(f"[round {r:03d}] hot-reload generator -> {gen_ckpt}")
                await gen_engine.hot_reload_from_dir(gen_ckpt)
                cleanup_checkpoint_dir(prev_gen_ckpt_root, new_ckpt_root, out_dir)
                prev_gen_ckpt_root = new_ckpt_root
                print(f"[round {r:03d}] generator hot-reloaded -> {gen_ckpt}")
            elif role == "ver" and Path(ver_sft_path).exists():
                logger.info("[round %03d] SFT training verifier rows=%d", r, len(ver_rows))
                print(f"[round {r:03d}] SFT start (verifier) rows={len(ver_rows)}")
                new_ckpt = run_fullft_one_round(
                    which="ver",
                    sft_path=ver_sft_path,
                    base_model_path=ver_ckpt,
                    project_name=args.project_name,
                    experiment_name=exp_name,
                    out_dir=args.out_dir,
                    config_name=args.config_name,
                    config_override=args.config_override,
                    use_subprocess=not args.train_inline,
                    train_cuda=args.train_cuda,
                )
                new_ckpt_root = Path(new_ckpt)
                ver_ckpt = resolve_model_dir_for_vllm(new_ckpt)
                logger.info("[round %03d] hot-reload verifier from %s", r, ver_ckpt)
                print(f"[round {r:03d}] hot-reload verifier -> {ver_ckpt}")
                await ver_engine.hot_reload_from_dir(ver_ckpt)
                cleanup_checkpoint_dir(prev_ver_ckpt_root, new_ckpt_root, out_dir)
                prev_ver_ckpt_root = new_ckpt_root
                print(f"[round {r:03d}] verifier hot-reloaded -> {ver_ckpt}")
            else:
                print(f"[round {r:03d}] no SFT rows for {role}, skipping FT.")

            if args.eval_after_each:
                await run_eval(args, gen_engine, ver_engine, teacher_engine, prefix=f"round{r:03d}")
            logger.info("[round %03d] complete", r)
            print(f"[round {r:03d}] COMPLETE")
    finally:
        if wandb_run:
            try:
                wandb_run.finish()
            except Exception:
                pass

    print("\n[done] Alternating full-FT iterations completed.")


async def run_collect(args: argparse.Namespace):
    wandb_run = _maybe_init_wandb(args)
    teacher_backend = getattr(args, "teacher_backend", "triapi").lower()
    teacher_devices = (
        parse_cuda_list(getattr(args, "teacher_cuda", None)) if teacher_backend == "vllm" else None
    )
    gen_devices = parse_cuda_list(getattr(args, "gen_cuda", None))
    ver_devices = parse_cuda_list(getattr(args, "ver_cuda", None))
    teacher_gpu_mem = (
        args.teacher_gpu_mem_util if teacher_backend == "vllm" and getattr(args, "teacher_gpu_mem_util", None) is not None else args.gpu_mem_util
    ) if teacher_backend == "vllm" else None
    gen_gpu_mem = args.gen_gpu_mem_util if getattr(args, "gen_gpu_mem_util", None) is not None else args.gpu_mem_util
    ver_gpu_mem = args.ver_gpu_mem_util if getattr(args, "ver_gpu_mem_util", None) is not None else args.gpu_mem_util
    if teacher_backend == "vllm":
        ensure_tp_fit(teacher_devices, args.tp_t, "teacher")
    ensure_tp_fit(gen_devices, args.tp_s, "generator")
    ensure_tp_fit(ver_devices, args.tp_s, "verifier")

    if teacher_backend == "vllm":
        with cuda_visible_devices(teacher_devices):
            teacher_engine = VLLMChatEngine(
                VLLMConfig(
                    model=args.teacher_base,
                    tokenizer=args.teacher_tokenizer,
                    tp=args.tp_t,
                    gpu_mem_util=teacher_gpu_mem if teacher_gpu_mem is not None else args.gpu_mem_util,
                    max_model_len=args.max_model_len,
                    temperature=args.t_temp,
                    top_p=0.95,
                )
            )
    else:
        teacher_engine = TriApiChatEngine(
            TriApiConfig(
                instance=args.teacher_triapi_instance,
                deployment=args.teacher_triapi_deployment,
                scope=args.teacher_triapi_scope,
                api_version=args.teacher_triapi_api_version,
                temperature=None,  # omit temperature so TRAPI uses its default (avoid 400s)
                top_p=1.0,
                max_output_tokens=args.teacher_triapi_max_output_tokens,
                timeout_s=args.teacher_triapi_timeout,
                max_retries=args.teacher_triapi_retries,
                max_parallel=args.teacher_triapi_max_parallel,
            )
        )
    with cuda_visible_devices(gen_devices):
        gen_engine = VLLMChatEngine(
            VLLMConfig(
                model=args.gen_base,
                tokenizer=args.gen_tokenizer,
                tp=args.tp_s,
                gpu_mem_util=gen_gpu_mem,
                max_model_len=args.max_model_len,
            )
        )
    with cuda_visible_devices(ver_devices):
        ver_engine = VLLMChatEngine(
            VLLMConfig(
                model=args.ver_base,
                tokenizer=args.ver_tokenizer,
                tp=args.tp_s,
                gpu_mem_util=ver_gpu_mem,
                max_model_len=args.max_model_len,
            )
        )

    tasks = sample_tasks(args.dataset_name, args.split, args.num_tasks, seed=args.seed)
    episodes = await rollout_with_workflow_engine(
        tasks=tasks,
        gen_engine=gen_engine,
        ver_engine=ver_engine,
        teacher_engine=teacher_engine,
        max_turns=args.max_turns,
        stop_on_verifier_fix=args.stop_on_verifier_fix,
        collect_for=args.collect_for if args.collect_for != "auto" else "both",
        parallel=args.parallel,
    )
    gen_rows, ver_rows = episodes_to_sft_rows(episodes)

    prefix = Path(args.save_prefix)
    if gen_rows:
        path = f"{prefix}_gen.parquet"
        append_parquet(gen_rows, path)
        print(f"[collect] wrote {len(gen_rows)} gen rows -> {path}")
    if ver_rows:
        path = f"{prefix}_ver.parquet"
        append_parquet(ver_rows, path)
        print(f"[collect] wrote {len(ver_rows)} ver rows -> {path}")
    if wandb_run:
        try:
            wandb_run.finish()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gen–Ver DAgger full-FT with vLLM for ALL models + hot reload + micro-batching"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("iterate", help="Run 2T alternating rounds of rollout + full FT + hot reload")
    i.add_argument("--dataset_name", type=str, default="deepscaler_math")
    i.add_argument("--split", type=str, default="train")
    i.add_argument("--rounds", type=int, default=5)
    i.add_argument("--batch_tasks", type=int, default=256)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument(
        "--teacher_backend",
        type=str,
        default="triapi",
        choices=["triapi", "vllm"],
        help="Teacher backend: Azure TRAPI (chatgpt-5.1) or local vLLM.",
    )
    i.add_argument("--teacher_triapi_instance", type=str, default="gcr/shared")
    i.add_argument("--teacher_triapi_deployment", type=str, default="gpt-5_2025-08-07")
    i.add_argument("--teacher_triapi_scope", type=str, default="api://trapi")
    i.add_argument("--teacher_triapi_api_version", type=str, default="2025-02-01-preview")
    i.add_argument("--teacher_triapi_max_parallel", type=int, default=32)
    i.add_argument("--teacher_triapi_timeout", type=int, default=120)
    i.add_argument("--teacher_triapi_retries", type=int, default=5)
    i.add_argument(
        "--teacher_triapi_max_output_tokens",
        type=int,
        default=64000,
        help="Max tokens from TriAPI teacher; omit/<=0 to use service default.",
    )
    i.add_argument("--teacher_base", type=str, default="Qwen/Qwen3-4B")
    i.add_argument("--gen_base", type=str, default="Qwen/Qwen3-0.6B")
    i.add_argument("--ver_base", type=str, default="Qwen/Qwen3-0.6B")
    i.add_argument("--teacher_tokenizer", type=str, default=None)
    i.add_argument("--gen_tokenizer", type=str, default=None)
    i.add_argument("--ver_tokenizer", type=str, default=None)
    i.add_argument("--t_temp", type=float, default=0.2)
    i.add_argument("--max_turns", type=int, default=4)
    i.add_argument("--stop_on_verifier_fix", action="store_true")
    i.add_argument("--collect_for", type=str, default="auto", choices=["auto", "both", "gen", "ver"])
    i.add_argument("--parallel", type=int, default=64)
    i.add_argument("--tp_s", type=int, default=1)
    i.add_argument("--tp_t", type=int, default=1)
    i.add_argument("--gpu_mem_util", type=float, default=0.9)
    i.add_argument("--max_model_len", type=int, default=32768)
    i.add_argument("--teacher_gpu_mem_util", type=float, default=None,
                   help="Override gpu_mem_util for the teacher engine.")
    i.add_argument("--gen_gpu_mem_util", type=float, default=None,
                   help="Override gpu_mem_util for the generator engine.")
    i.add_argument("--ver_gpu_mem_util", type=float, default=None,
                   help="Override gpu_mem_util for the verifier engine.")
    i.add_argument("--out_dir", type=str, default="runs/genver_fullft")
    i.add_argument("--project_name", type=str, default="genver_fullft")
    i.add_argument("--experiment_name", type=str, default="exp")
    i.add_argument("--config_name", type=str, default="agent_sft_trainer")
    i.add_argument("--config_override", type=str, nargs="*", default=None)
    i.add_argument("--eval_dataset", type=str, default="math500")
    i.add_argument("--eval_split", type=str, default="test")
    i.add_argument("--eval_num_tasks", type=int, default=100)
    i.add_argument("--eval_before_train", action="store_true", help="Run a baseline eval before training.")
    i.add_argument("--eval_after_each", action="store_true", help="Run eval after each FT (gen/ver) round.")
    i.add_argument("--gen_sft_path", type=str, default=None)
    i.add_argument("--ver_sft_path", type=str, default=None)
    i.add_argument("--teacher_cuda", type=str, default=None,
                   help="Comma-separated GPU ids for the teacher engine (e.g., '2' or '2,3').")
    i.add_argument("--gen_cuda", type=str, default=None,
                   help="Comma-separated GPU ids for the generator engine.")
    i.add_argument("--ver_cuda", type=str, default=None,
                   help="Comma-separated GPU ids for the verifier engine.")
    i.add_argument("--train_inline", action="store_true",
                   help="Run SFT training in this process (default: spawn subprocess).")
    i.add_argument("--train_cuda", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES value for the training subprocess (e.g., '1').")

    c = sub.add_parser("collect", help="Just rollout & store SFT rows once with vLLM teacher+students")
    c.add_argument("--dataset_name", type=str, default="deepscaler_math")
    c.add_argument("--split", type=str, default="train")
    c.add_argument("--num_tasks", type=int, default=512)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument(
        "--teacher_backend",
        type=str,
        default="triapi",
        choices=["triapi", "vllm"],
        help="Teacher backend: Azure TRAPI (chatgpt-5.1) or local vLLM.",
    )
    c.add_argument("--teacher_triapi_instance", type=str, default="gcr/shared")
    c.add_argument("--teacher_triapi_deployment", type=str, default="gpt-5_2025-08-07")
    c.add_argument("--teacher_triapi_scope", type=str, default="api://trapi")
    c.add_argument("--teacher_triapi_api_version", type=str, default="2025-02-01-preview")
    c.add_argument("--teacher_triapi_max_parallel", type=int, default=32)
    c.add_argument("--teacher_triapi_timeout", type=int, default=120)
    c.add_argument("--teacher_triapi_retries", type=int, default=5)
    c.add_argument(
        "--teacher_triapi_max_output_tokens",
        type=int,
        default=64000,
        help="Max tokens from TriAPI teacher; omit/<=0 to use service default.",
    )
    c.add_argument("--teacher_base", type=str, default="Qwen/Qwen3-4B")
    c.add_argument("--gen_base", type=str, default="Qwen/Qwen3-0.6B")
    c.add_argument("--ver_base", type=str, default="Qwen/Qwen3-0.6B")
    c.add_argument("--teacher_tokenizer", type=str, default=None)
    c.add_argument("--gen_tokenizer", type=str, default=None)
    c.add_argument("--ver_tokenizer", type=str, default=None)
    c.add_argument("--t_temp", type=float, default=0.2)
    c.add_argument("--max_turns", type=int, default=4)
    c.add_argument("--stop_on_verifier_fix", action="store_true")
    c.add_argument("--collect_for", type=str, default="both", choices=["both", "gen", "ver"])
    c.add_argument("--parallel", type=int, default=64)
    c.add_argument("--tp_s", type=int, default=1)
    c.add_argument("--tp_t", type=int, default=1)
    c.add_argument("--gpu_mem_util", type=float, default=0.9)
    c.add_argument("--max_model_len", type=int, default=32768)
    c.add_argument("--teacher_gpu_mem_util", type=float, default=None)
    c.add_argument("--gen_gpu_mem_util", type=float, default=None)
    c.add_argument("--ver_gpu_mem_util", type=float, default=None)
    c.add_argument("--save_prefix", type=str, default="datasets/collect")
    c.add_argument("--teacher_cuda", type=str, default=None)
    c.add_argument("--gen_cuda", type=str, default=None)
    c.add_argument("--ver_cuda", type=str, default=None)

    hidden = sub.add_parser("_train_once", help=argparse.SUPPRESS)
    hidden.add_argument("--which", type=str, choices=["gen", "ver"], required=True)
    hidden.add_argument("--sft_path", type=str, required=True)
    hidden.add_argument("--base_model_path", type=str, required=True)
    hidden.add_argument("--project_name", type=str, required=True)
    hidden.add_argument("--experiment_name", type=str, required=True)
    hidden.add_argument("--out_dir", type=str, required=True)
    hidden.add_argument("--config_name", type=str, default="agent_sft_trainer")
    hidden.add_argument("--config_override", type=str, nargs="*", default=None)

    return p


def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "iterate":
        asyncio.run(run_iterate(args))
    elif args.cmd == "collect":
        asyncio.run(run_collect(args))
    elif args.cmd == "_train_once":
        _train_once_cli(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
