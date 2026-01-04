"""PAG-style alternating Gen/Ver training on Numina-style math data with per-round tests."""

from __future__ import annotations

import torch.multiprocessing as mp

# Ensure CUDA init happens under spawn instead of fork when vLLM spins up workers.
mp.set_start_method("spawn", force=True)

import argparse
import asyncio
import dataclasses
import json
import math
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from checkpoint_utils import cleanup_checkpoint_dir, resolve_model_dir_for_vllm
from data_utils import append_sft_rows
from fullft_training import run_fullft_one_round
from genver_pag_workflow import (
    pag_episodes_to_sft_rows,
    relabel_pag_episodes_with_teacher,
    rollout_with_pag_phased,
    rollout_with_pag_workflow_engine,
    dump_pag_transcripts_with_raw,
)
from rllm.rewards.math_utils.utils import extract_answer
from gpu_utils import (
    cuda_visible_devices,
    devices_overlap,
    ensure_tp_dp_fit,
    parse_cuda_list,
    split_devices_for_tp_dp,
    warn_overlap,
)
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from vllm_engine import EnginePool, VLLMChatEngine, VLLMConfig

def _coerce_question_field(row: Any) -> Any:
    if isinstance(row, dict):
        # Numina schema: columns are typically problem / solution / messages / source.
        problem = row.get("problem")
        if isinstance(problem, str) and problem.strip():
            return problem
        # Fallback: first user message content (avoid assistant/solution).
        msgs = row.get("messages")
        if isinstance(msgs, list):
            for msg in msgs:
                if isinstance(msg, dict) and (msg.get("role") or "").lower() == "user" and msg.get("content"):
                    return msg["content"]
        return None
    return row


def _coerce_ground_truth_field(row: Any) -> Any:
    if isinstance(row, dict):
        for key in ("ground_truth", "answer", "solution", "label", "gt"):
            val = row.get(key)
            if val not in (None, ""):
                return val
        # Numina CoT: derive answer from solution or assistant message if possible
        sol = row.get("solution")
        if sol:
            parsed = extract_answer(sol) or sol
            return parsed
        msgs = row.get("messages")
        if isinstance(msgs, list):
            assistant = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"), None)
            if assistant and assistant.get("content"):
                parsed = extract_answer(assistant["content"]) or assistant["content"]
                return parsed
        # Some Numina rows store problem/solution inside a nested dict under "content"
        content = row.get("content")
        if isinstance(content, dict):
            for key in ("ground_truth", "answer", "solution", "label", "gt"):
                val = content.get(key)
                if val not in (None, ""):
                    return val
            if content.get("solution"):
                parsed = extract_answer(content["solution"]) or content["solution"]
                return parsed
    return None


def _question_text_from_raw(raw_question: Any) -> str:
    """Best-effort extraction of user question from varied schemas."""
    try:
        import numpy as np

        if isinstance(raw_question, np.ndarray):
            raw_question = raw_question.tolist()
    except Exception:
        pass

    def _content_from_dict(msg: Dict[str, Any]) -> str:
        return str(
            msg.get("content")
            or msg.get("text")
            or msg.get("value")
            or msg.get("message")
            or msg.get("prompt")
            or ""
        )

    def _is_userish(role: Optional[str]) -> bool:
        if not role:
            return False
        role = role.lower()
        return role in {"user", "human", "student", "questioner", "qa.user", "customer", "client"}

    if isinstance(raw_question, list):
        last_with_content = ""
        for msg in reversed(raw_question):
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("from") or msg.get("speaker") or msg.get("name")
                content = _content_from_dict(msg)
                if _is_userish(str(role)):
                    if content:
                        return content
                if content and not last_with_content:
                    last_with_content = content
            elif isinstance(msg, str) and msg.strip():
                return msg
        return last_with_content

    if isinstance(raw_question, dict):
        role = raw_question.get("role") or raw_question.get("from") or raw_question.get("speaker") or ""
        if _is_userish(str(role)):
            return _content_from_dict(raw_question)
        return str(
            raw_question.get("content")
            or raw_question.get("question")
            or raw_question.get("prompt")
            or raw_question.get("input")
            or ""
        )

    return str(raw_question or "")


def _sanitize_split_key(split: str) -> str:
    """Make a registry-safe split key (avoid [], :) that break glob parsing."""
    safe = split.replace("/", "_")
    safe = safe.replace(":", "-")
    safe = safe.replace("[", "").replace("]", "")
    return safe


def _resolve_registered_dataset(name: str, split: str):
    """Try loading a registered dataset with raw or sanitized split keys."""
    errs: List[Exception] = []
    for key in dict.fromkeys([split, _sanitize_split_key(split)]):
        if not key:
            continue
        try:
            ds = DatasetRegistry.load_dataset(name, key)
            if ds is not None:
                return ds, key
        except Exception as err:  # catch polars/glob errors
            errs.append(err)
    if errs:
        print(f"[warn] Failed to load registered dataset {name}/{split}: {errs}")
    return None, _sanitize_split_key(split)


def _maybe_register_dataset(name: str, split: str, dataset_path: Optional[str], *, force: bool = False):
    """Register dataset with DatasetRegistry if missing (or force-refresh)."""
    existing, reg_split = _resolve_registered_dataset(name, split)
    if existing is not None and not force:
        return
    if not dataset_path:
        raise RuntimeError(
            f"Dataset '{name}' (split={split}) is not registered. Provide --dataset_path to auto-register."
        )
    # Handle bare parquet/jsonl paths that HF datasets cannot split automatically.
    if dataset_path.endswith((".parquet", ".json", ".jsonl")):
        file_key = "train" if split == "train" else split
        builder = "parquet" if dataset_path.endswith(".parquet") else "json"
        ds = load_dataset(builder, data_files={file_key: dataset_path}, split=file_key)
    else:
        ds = load_dataset(dataset_path, split=split)

    def preprocess_fn(example, idx):
        return {
            "question": _coerce_question_field(example),
            "ground_truth": _coerce_ground_truth_field(example),
            "data_source": name,
        }

    ds = ds.map(preprocess_fn, with_indices=True)
    if reg_split != split:
        print(f"[info] Registering split '{split}' under registry key '{reg_split}' to avoid glob issues.")
    DatasetRegistry.register_dataset(name, ds, reg_split)


def _dump_jsonl(records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _append_jsonl(records: List[Dict[str, Any]], path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sample_tasks(
    dataset_name: str,
    split: str,
    n: int,
    seed: int = 0,
    *,
    shard_idx: int = 0,
    num_shards: int = 1,
    mode: str = "random",
    start_offset: int = 0,
) -> List[Dict[str, Any]]:
    ds, reg_split = _resolve_registered_dataset(dataset_name, split)
    if ds is None:
        raise RuntimeError(f"Dataset '{dataset_name}' split '{split}' (key '{reg_split}') not found. Register it first.")
    data = ds.get_data() if hasattr(ds, "get_data") else list(ds)
    if not data:
        return []
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards - 1}]")
    idx = list(range(len(data)))
    if num_shards > 1:
        idx = [i for i in idx if i % num_shards == shard_idx]
    if not idx:
        return []

    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(idx)
        idx = idx[: n if n < len(idx) else len(idx)]
    elif mode == "sequential":
        # Deterministic walk over the (optionally sharded) dataset indices.
        start_offset = max(0, int(start_offset))
        if len(idx) > 0:
            start_offset = start_offset % len(idx)
        idx = idx[start_offset:] + idx[:start_offset]
    else:
        raise ValueError(f"Unknown sample mode: {mode!r} (expected 'random' or 'sequential')")

    tasks: List[Dict[str, Any]] = []
    if mode == "sequential":
        # Try to fill up to n tasks, skipping empty questions without reducing batch size.
        scanned = 0
        for i in idx:
            scanned += 1
            row = data[i]
            q = _coerce_question_field(row)
            if q in (None, ""):
                if scanned >= len(idx):
                    break
                continue
            gt = _coerce_ground_truth_field(row)
            tasks.append({"uid": f"{split}_{i}", "question": q, "ground_truth": gt})
            if len(tasks) >= n:
                break
            if scanned >= len(idx):
                break
    else:
        for i in idx:
            row = data[i]
            q = _coerce_question_field(row)
            if q in (None, ""):
                continue
            gt = _coerce_ground_truth_field(row)
            tasks.append({"uid": f"{split}_{i}", "question": q, "ground_truth": gt})
    return tasks


def _generator_accuracy(episodes) -> Tuple[float, int, int, List[dict]]:
    total = 0
    correct = 0
    details: List[dict] = []
    for ep in episodes:
        gen_traj = next((t for t in ep.trajectories if t.name == "generator"), None)
        if not gen_traj or not gen_traj.steps:
            continue
        final_step = gen_traj.steps[-1]
        g_msg = (final_step.info or {}).get("student_response", "") or ""
        parsed_pred = extract_answer(g_msg)
        if isinstance(ep.task, dict):
            raw_q = ep.task.get("question") or _coerce_question_field(ep.task)
            gt = _coerce_ground_truth_field(ep.task)
        else:
            raw_q = getattr(ep.task, "question", None)
            gt = getattr(ep.task, "ground_truth", None)
        question_text = _question_text_from_raw(raw_q)
        if not question_text and raw_q is not None:
            question_text = str(raw_q)  # fallback: at least surface the raw prompt structure
        reward = math_reward_fn({"question": question_text, "ground_truth": gt}, g_msg)
        total += 1
        if reward.is_correct:
            correct += 1
        details.append(
            {
                "episode_id": ep.id,
                "question": question_text,
                "ground_truth": gt,
                "generator_message": g_msg,
                "prediction_used": g_msg,
                "parsed_prediction": parsed_pred,
                "reward": float(getattr(reward, "reward", 0.0) or 0.0),
                "is_correct": bool(getattr(reward, "is_correct", False)),
            }
        )
    acc = (correct / total) if total else 0.0
    return acc, correct, total, details


def _dump_accuracy_details(records: List[dict], out_dir: Path, round_idx: int, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_gen_accuracy_round_{round_idx:03d}.jsonl"
    with path.open("w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _normalize_messages_for_parquet(messages: Any) -> Any:
    """Ensure messages is list of {role, content} dicts."""
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore

    def _as_content(val: Any) -> Any:
        if val is None:
            return None
        if _np is not None and isinstance(val, _np.ndarray):
            val = val.tolist()
        if isinstance(val, (list, dict)):
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        return str(val)

    def _as_dict(msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return {"role": msg.get("role"), "content": _as_content(msg.get("content"))}
        return {"role": "user", "content": _as_content(msg)}

    if messages is None:
        return [{"role": None, "content": None}]
    # Try to parse JSON strings
    if isinstance(messages, str):
        try:
            parsed = json.loads(messages)
            messages = parsed
        except Exception:
            return [{"role": "user", "content": messages}]
    if _np is not None and isinstance(messages, _np.ndarray):
        messages = messages.tolist()
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        messages = [messages]

    norm: List[Dict[str, Any]] = []
    for m in messages:
        if _np is not None and isinstance(m, _np.ndarray):
            m = m.tolist()
        if isinstance(m, list):
            # nested list; coerce each element if dict-like else join as string
            if all(isinstance(x, dict) for x in m):
                norm.extend([_as_dict(x) for x in m])  # flatten nested list of dicts
            else:
                norm.append({"role": "user", "content": " ".join(str(x) for x in m)})
            continue
        norm.append(_as_dict(m))
    if not norm:
        norm.append({"role": None, "content": None})
    return norm


def _truncate_messages_tail(
    messages: List[Dict[str, Any]],
    tokenizer,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Keep the tail of the dialogue so total tokens <= max_tokens."""
    if max_tokens <= 0 or tokenizer is None:
        return messages
    msgs = [dict(m) for m in messages if isinstance(m, dict)]
    if not msgs:
        return msgs
    sys = msgs[0] if msgs[0].get("role") == "system" else None
    body = msgs[1:] if sys else msgs

    def token_len(ms: List[Dict[str, Any]]) -> int:
        return len(tokenizer.apply_chat_template(ms, tokenize=True, add_generation_prompt=False))

    while body and token_len(([] if sys is None else [sys]) + body) > max_tokens:
        body = body[1:]
    trimmed = (([sys] if sys else []) + body) if body or sys else []
    return trimmed


def _sft_path_for_training(
    sft_path: str,
    out_dir: Path,
    label: str,
    *,
    tokenizer=None,
    max_tokens: int = 0,
) -> str:
    """Convert JSON/JSONL SFT rows to parquet for training if needed."""
    path = Path(sft_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT file not found: {sft_path}")
    if path.suffix.lower() in {".json", ".jsonl"}:
        tmp_path = out_dir / f"{label}_sft_tmp.parquet"
        df = pd.read_json(path, lines=True)
        if "messages" in df.columns:
            df["messages"] = df["messages"].apply(_normalize_messages_for_parquet)
            if max_tokens > 0 and tokenizer is not None:
                df["messages"] = df["messages"].apply(
                    lambda ms: _truncate_messages_tail(ms, tokenizer, max_tokens)
                )
        else:
            df["messages"] = [[{"role": None, "content": None}] for _ in range(len(df))]
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path, index=False)
        return str(tmp_path)
    return sft_path


def _write_sft_parquet_shard(
    rows: List[Dict[str, Any]],
    shard_dir: Path,
    *,
    label: str,
    round_idx: int,
    tokenizer=None,
    max_tokens: int = 0,
) -> str:
    if not rows:
        return ""
    shard_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if "messages" in df.columns:
        df["messages"] = df["messages"].apply(_normalize_messages_for_parquet)
        if max_tokens > 0 and tokenizer is not None:
            df["messages"] = df["messages"].apply(lambda ms: _truncate_messages_tail(ms, tokenizer, max_tokens))
    shard_name = f"{label}_r{round_idx:03d}_{uuid.uuid4().hex[:8]}.parquet"
    shard_path = shard_dir / shard_name
    df.to_parquet(shard_path, index=False)
    return str(shard_path)


def _write_latest_sft_parquet(
    rows: List[Dict[str, Any]],
    latest_dir: Path,
    *,
    label: str,
    round_idx: int,
    tokenizer=None,
    max_tokens: int = 0,
) -> str:
    if not rows:
        return ""
    latest_dir.mkdir(parents=True, exist_ok=True)
    for old in latest_dir.glob("*.parquet"):
        try:
            old.unlink()
        except Exception:
            continue
    return _write_sft_parquet_shard(
        rows,
        latest_dir,
        label=label,
        round_idx=round_idx,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )


def _infer_train_dp_size(train_cuda: Optional[str]) -> int:
    def _safe_parse(spec: Optional[str]) -> Optional[List[str]]:
        if not spec:
            return None
        parts = [s.strip() for s in spec.split(",") if s.strip()]
        if not parts or any(not part.isdigit() for part in parts):
            return None
        return parts

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


def _count_sft_rows(path: str) -> int:
    path_obj = Path(path)
    if not path_obj.exists():
        return 0
    if path_obj.is_dir():
        files = sorted((p for p in path_obj.rglob("*.parquet") if p.is_file()), key=lambda p: p.name)
        if not files:
            return 0
        total = 0
        try:
            import pyarrow.parquet as pq  # type: ignore

            for f in files:
                try:
                    total += int(pq.ParquetFile(f).metadata.num_rows)
                except Exception:
                    df = pd.read_parquet(f)
                    total += len(df)
            return int(total)
        except Exception:
            for f in files:
                try:
                    df = pd.read_parquet(f)
                    total += len(df)
                except Exception:
                    continue
            return int(total)
    suffix = path_obj.suffix.lower()
    if suffix == ".jsonl":
        try:
            with path_obj.open("r", encoding="utf-8") as fin:
                return sum(1 for line in fin if line.strip())
        except Exception as err:
            print(f"[warn] Failed to count JSONL rows in {path_obj} ({err}); treating as 0.")
            return 0
    if suffix == ".json":
        try:
            payload = json.loads(path_obj.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return len(payload)
            return 1 if payload else 0
        except Exception as err:
            print(f"[warn] Failed to count JSON rows in {path_obj} ({err}); treating as 0.")
            return 0
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            return pq.ParquetFile(path_obj).metadata.num_rows
        except Exception:
            try:
                df = pd.read_parquet(path_obj)
                return len(df)
            except Exception as err:
                print(f"[warn] Failed to count parquet rows in {path_obj} ({err}); treating as 0.")
                return 0
    try:
        with path_obj.open("r", encoding="utf-8") as fin:
            return sum(1 for line in fin if line.strip())
    except Exception:
        return 0


def _override_get_int(overrides: Optional[List[str]], key: str) -> Optional[int]:
    if not overrides:
        return None
    prefix = f"{key}="
    for item in reversed(overrides):
        if item.startswith(prefix):
            raw = item[len(prefix) :]
            if raw.isdigit():
                return int(raw)
            try:
                return int(float(raw))
            except Exception:
                return None
    return None


def _override_set(overrides: Optional[List[str]], key: str, value: int) -> List[str]:
    prefix = f"{key}="
    kept = [o for o in (overrides or []) if not o.startswith(prefix)]
    kept.append(f"{key}={int(value)}")
    return kept


def _maybe_adjust_train_batch(
    overrides: Optional[List[str]],
    *,
    dp_size: int,
    train_rows: int,
    default_train_batch_size: int = 256,
    default_micro_batch_size_per_gpu: int = 4,
) -> Tuple[List[str], bool, bool]:
    """Ensure train batch size won't crash FSDP normalization.

    rLLM's FSDP trainer requires `data.train_batch_size % dp_size == 0`.
    fullft_training may also clamp train_batch_size to <= train_rows.
    We proactively choose a safe train_batch_size that is <= train_rows and divisible.
    """
    dp_size = max(1, int(dp_size or 1))
    train_rows = max(0, int(train_rows or 0))
    if train_rows <= 0:
        return list(overrides or []), False, False

    explicit_train_batch = _override_get_int(overrides, "data.train_batch_size")
    requested_train_batch = max(1, int(explicit_train_batch or default_train_batch_size))

    micro_batch_raw = _override_get_int(overrides, "data.micro_batch_size_per_gpu")
    micro_explicit = any(
        o.startswith("data.micro_batch_size_per_gpu=") for o in (overrides or [])
    )
    if micro_batch_raw is None:
        micro_batch_raw = default_micro_batch_size_per_gpu
    micro_batch_raw = max(1, int(micro_batch_raw))

    # NOTE: FSDP normalizes global batch size by DP size, then asserts
    #   local_batch_size % micro_batch_size_per_gpu == 0
    # so global train_batch_size must be divisible by dp_size * micro_batch_size_per_gpu.
    max_feasible_micro = max(1, train_rows // dp_size)
    micro_candidates = (
        [micro_batch_raw]
        if micro_explicit
        else list(range(min(micro_batch_raw, max_feasible_micro), 0, -1))
    )
    if not micro_candidates:
        micro_candidates = [1]

    base_candidate = min(requested_train_batch, train_rows)
    for micro_batch in micro_candidates:
        divisor = dp_size * max(1, int(micro_batch))
        candidate = base_candidate - (base_candidate % divisor)
        if candidate < divisor:
            candidate = divisor

        if candidate > train_rows:
            continue

        changed = candidate != requested_train_batch
        new_overrides = _override_set(overrides, "data.train_batch_size", candidate)
        if not micro_explicit and micro_batch != micro_batch_raw:
            new_overrides = _override_set(new_overrides, "data.micro_batch_size_per_gpu", micro_batch)
            changed = True
        return new_overrides, changed, True

    return list(overrides or []), False, False


async def _run_eval_if_needed(
    args: argparse.Namespace,
    round_idx: int,
    gen_engine,
    ver_engine,
    teacher_engine,
    out_dir: Path,
    sleep_state: Optional[Dict[str, bool]] = None,
):
    if args.eval_num_tasks <= 0 or (round_idx % args.eval_every != 0):
        return
    eval_ds = args.eval_dataset_name or args.dataset_name
    eval_split = args.eval_split
    _maybe_register_dataset(eval_ds, eval_split, args.eval_dataset_path, force=args.force_register)
    eval_tasks = _sample_tasks(eval_ds, eval_split, args.eval_num_tasks, seed=args.eval_seed + round_idx)
    if not eval_tasks:
        print(f"[round {round_idx:03d}] No eval tasks found for {eval_ds}/{eval_split}; skipping eval.")
        return
    if args.rollout_mode == "phased":
        async def _gen_batch(prompts, uids):
            await _sleep_engines([("verifier", ver_engine)], sleep_state)
            await _wake_engines([("generator", gen_engine)], sleep_state)
            outs = await gen_engine.generate_batch(
                prompts,
                chat_template_kwargs={"enable_thinking": True},
                temperature=args.gen_temp,
                top_p=args.gen_top_p,
                sp_extra={"top_k": args.gen_top_k, "min_p": args.gen_min_p},
                max_tokens=args.max_new_tokens,
            )
            await _sleep_engines([("generator", gen_engine)], sleep_state)
            return outs

        async def _ver_batch(prompts, uids):
            await _sleep_engines([("generator", gen_engine)], sleep_state)
            await _wake_engines([("verifier", ver_engine)], sleep_state)
            outs = await ver_engine.generate_batch(
                prompts,
                chat_template_kwargs={"enable_thinking": True},
                temperature=args.ver_temp,
                top_p=args.ver_top_p,
                sp_extra={"top_k": args.ver_top_k, "min_p": args.ver_min_p},
                max_tokens=args.max_new_tokens,
            )
            await _sleep_engines([("verifier", ver_engine)], sleep_state)
            return outs

        await _sleep_engines([("generator", gen_engine), ("verifier", ver_engine)], sleep_state)
        eval_episodes = await rollout_with_pag_phased(
            eval_tasks,
            gen_batch=_gen_batch,
            ver_batch=_ver_batch,
            max_turns=args.max_turns,
            parallel=args.eval_parallel or args.parallel,
        )
    else:
        eval_episodes = await rollout_with_pag_workflow_engine(
            tasks=eval_tasks,
            gen_engine=gen_engine,
            ver_engine=ver_engine,
            teacher_engine=teacher_engine,
            max_turns=args.max_turns,
            collect_for="none",
            parallel=args.eval_parallel or args.parallel,
            retry=1,
        )
    acc, correct, total, records = _generator_accuracy(eval_episodes)
    print(f"[round {round_idx:03d}] eval accuracy={acc:.3f} ({correct}/{total}) on {eval_ds}/{eval_split}")
    dump_path = _dump_accuracy_details(records, out_dir, round_idx, tag="eval")
    print(f"[round {round_idx:03d}] eval accuracy data -> {dump_path}")


async def _sleep_engines(pairs, sleep_state: Optional[Dict[str, bool]] = None):
    for label, engine in pairs:
        if engine is None:
            continue
        if sleep_state is not None and sleep_state.get(label):
            continue
        try:
            await engine.sleep()
            print(f"[info] {label} engine slept to free GPU memory.")
            if sleep_state is not None:
                sleep_state[label] = True
        except Exception as err:
            print(f"[warn] Failed to sleep {label} engine: {err}")


async def _wake_engines(pairs, sleep_state: Optional[Dict[str, bool]] = None):
    for label, engine in pairs:
        if engine is None:
            continue
        if sleep_state is not None and not sleep_state.get(label, False):
            continue
        try:
            await engine.wake_up()
            print(f"[info] {label} engine woke up.")
            if sleep_state is not None:
                sleep_state[label] = False
        except Exception as err:
            print(f"[warn] Failed to wake {label} engine: {err}")


async def run(args: argparse.Namespace):
    _maybe_register_dataset(args.dataset_name, args.split, args.dataset_path, force=args.force_register)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_root = out_dir / "train_ckpts"
    train_out_dir_gen = train_root / "gen"
    train_out_dir_ver = train_root / "ver"
    train_out_dir_gen.mkdir(parents=True, exist_ok=True)
    train_out_dir_ver.mkdir(parents=True, exist_ok=True)

    if args.dump_dataset_head > 0:
        head_start = args.sample_start if args.sample_mode == "sequential" else 0
        head_tasks = _sample_tasks(
            args.dataset_name,
            args.split,
            args.dump_dataset_head,
            seed=args.seed,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            mode=args.sample_mode,
            start_offset=head_start,
        )
        _dump_jsonl(head_tasks, out_dir / "dataset_head_sample.jsonl")

    if args.sft_storage == "parquet_shards":
        gen_sft_path = str(out_dir / "gen_sft_shards") if not args.gen_sft_path else args.gen_sft_path
        ver_sft_path = str(out_dir / "ver_sft_shards") if not args.ver_sft_path else args.ver_sft_path
        if Path(gen_sft_path).suffix:
            raise ValueError("--gen_sft_path must be a directory when --sft_storage=parquet_shards")
        if Path(ver_sft_path).suffix:
            raise ValueError("--ver_sft_path must be a directory when --sft_storage=parquet_shards")
        Path(gen_sft_path).mkdir(parents=True, exist_ok=True)
        Path(ver_sft_path).mkdir(parents=True, exist_ok=True)
    else:
        gen_sft_path = str(out_dir / "gen_sft.jsonl") if not args.gen_sft_path else args.gen_sft_path
        ver_sft_path = str(out_dir / "ver_sft.jsonl") if not args.ver_sft_path else args.ver_sft_path
    latest_gen_dir = out_dir / "gen_sft_latest"
    latest_ver_dir = out_dir / "ver_sft_latest"

    teacher_devices = parse_cuda_list(args.teacher_cuda)
    gen_devices = parse_cuda_list(args.gen_cuda)
    ver_devices = parse_cuda_list(args.ver_cuda)

    ensure_tp_dp_fit(teacher_devices, args.tp_t, args.dp_t, "teacher")
    ensure_tp_dp_fit(gen_devices, args.tp_s, args.dp_s, "generator")
    ensure_tp_dp_fit(ver_devices, args.tp_s, args.dp_s, "verifier")
    if args.pipeline_mode == "inline":
        if devices_overlap(teacher_devices, gen_devices):
            warn_overlap("teacher & generator")
        if devices_overlap(teacher_devices, ver_devices):
            warn_overlap("teacher & verifier")
    if devices_overlap(gen_devices, ver_devices):
        warn_overlap("generator & verifier")
        if args.rollout_mode == "workflow":
            if args.pipeline_mode == "staged":
                print(
                    "[warn] generator & verifier share GPUs with --rollout_mode=workflow; "
                    "switching to --rollout_mode=phased so gen/ver can alternate via sleep/wake.",
                    flush=True,
                )
                args.rollout_mode = "phased"
            else:
                print(
                    "[warn] generator & verifier share GPUs with --rollout_mode=workflow; "
                    "gen/ver requests will interleave across episodes and may OOM. "
                    "Prefer --pipeline_mode=staged --rollout_mode=phased for debug on shared GPUs.",
                    flush=True,
                )

    def _build_engine_pool(*, label: str, devices: Optional[List[str]], cfg: VLLMConfig, dp: int):
        groups = split_devices_for_tp_dp(devices, cfg.tp, dp, label)
        engines: List[VLLMChatEngine] = []
        for group in groups:
            with cuda_visible_devices(group):
                engines.append(VLLMChatEngine(dataclasses.replace(cfg)))
        if len(engines) == 1:
            return engines[0]
        print(f"[info] {label} inference DP={len(engines)} TP={cfg.tp} (GPUs={len(groups) * cfg.tp})")
        return EnginePool(engines)

    sleep_state: Dict[str, bool] = {"teacher": False, "generator": False, "verifier": False}

    teacher_engine = None
    gen_engine = _build_engine_pool(
        label="generator",
        devices=gen_devices,
        dp=args.dp_s,
        cfg=VLLMConfig(
            model=args.gen_base,
            tokenizer=args.gen_tokenizer,
            tp=args.tp_s,
            gpu_mem_util=args.gen_gpu_mem_util,
            max_model_len=args.max_model_len,
            temperature=args.gen_temp,
            top_p=args.gen_top_p,
            top_k=args.gen_top_k,
            min_p=args.gen_min_p,
            max_new_tokens=args.max_new_tokens,
            batch_max=args.vllm_batch_max,
            batch_flush_ms=args.vllm_batch_flush_ms,
        ),
    )
    if devices_overlap(gen_devices, ver_devices) and args.rollout_mode == "phased":
        await _sleep_engines([("generator", gen_engine)], sleep_state)
    ver_engine = _build_engine_pool(
        label="verifier",
        devices=ver_devices,
        dp=args.dp_s,
        cfg=VLLMConfig(
            model=args.ver_base,
            tokenizer=args.ver_tokenizer,
            tp=args.tp_s,
            gpu_mem_util=args.ver_gpu_mem_util,
            max_model_len=args.max_model_len,
            temperature=args.ver_temp,
            top_p=args.ver_top_p,
            top_k=args.ver_top_k,
            min_p=args.ver_min_p,
            max_new_tokens=args.max_new_tokens,
            batch_max=args.vllm_batch_max,
            batch_flush_ms=args.vllm_batch_flush_ms,
        ),
    )
    if devices_overlap(gen_devices, ver_devices) and args.rollout_mode == "phased":
        await _sleep_engines([("verifier", ver_engine)], sleep_state)

    async def _ensure_teacher_awake():
        nonlocal teacher_engine
        if teacher_engine is None:
            teacher_engine = _build_engine_pool(
                label="teacher",
                devices=teacher_devices,
                dp=args.dp_t,
                cfg=VLLMConfig(
                    model=args.teacher_base,
                    tokenizer=args.teacher_tokenizer,
                    tp=args.tp_t,
                    gpu_mem_util=args.teacher_gpu_mem_util,
                    max_model_len=args.max_model_len,
                    temperature=args.t_temp,
                    top_p=args.t_top_p,
                    top_k=args.t_top_k,
                    min_p=args.t_min_p,
                    max_new_tokens=args.max_new_tokens,
                    batch_max=args.vllm_batch_max,
                    batch_flush_ms=args.vllm_batch_flush_ms,
                ),
            )
            sleep_state["teacher"] = False
            return
        await _wake_engines([("teacher", teacher_engine)], sleep_state)

    async def _ensure_genver_awake():
        await _wake_engines([("generator", gen_engine), ("verifier", ver_engine)], sleep_state)

    async def _ensure_genver_asleep():
        await _sleep_engines([("generator", gen_engine), ("verifier", ver_engine)], sleep_state)

    async def _ensure_teacher_asleep():
        nonlocal teacher_engine
        if teacher_engine is None:
            return
        await _sleep_engines([("teacher", teacher_engine)], sleep_state)

    if args.pipeline_mode == "inline":
        await _ensure_teacher_awake()

    async def _train_and_hot_reload(
        *,
        which: str,
        sft_path: str,
        base_model_path: str,
        train_out_dir: Path,
        engine,
        prev_ckpt: Path | None,
        experiment_name: str,
    ) -> Tuple[str, Path | None]:
        dp_size = _infer_train_dp_size(args.train_cuda)
        sft_rows = _count_sft_rows(sft_path)
        if sft_rows < dp_size:
            print(f"[round {r:03d}] {which} SFT rows={sft_rows} < dp_size={dp_size}, skipping FT.")
            return base_model_path, prev_ckpt

        train_overrides, adjusted, train_safe = _maybe_adjust_train_batch(
            args.config_override,
            dp_size=dp_size,
            train_rows=sft_rows,
        )
        if not train_safe:
            print(
                f"[round {r:03d}] No safe train batch for dp_size={dp_size} with {sft_rows} rows; skipping FT."
            )
            return base_model_path, prev_ckpt
        if adjusted:
            new_bsz = _override_get_int(train_overrides, "data.train_batch_size")
            new_micro = _override_get_int(train_overrides, "data.micro_batch_size_per_gpu")
            details = [f"data.train_batch_size={new_bsz}"]
            if new_micro is not None:
                details.append(f"data.micro_batch_size_per_gpu={new_micro}")
            print(f"[round {r:03d}] Adjusted training overrides: {', '.join(details)} (dp_size={dp_size})")

        train_sft = _sft_path_for_training(
            sft_path,
            out_dir,
            which,
            tokenizer=getattr(engine, "tok", None),
            max_tokens=args.train_truncate_tokens,
        )

        sleepers: List[Tuple[str, Any]] = []
        if args.pipeline_mode == "inline":
            await _ensure_teacher_awake()
            assert teacher_engine is not None
            sleepers = [("teacher", teacher_engine), ("generator", gen_engine), ("verifier", ver_engine)]
            await _sleep_engines(sleepers)
        else:
            await _ensure_teacher_asleep()
            await _ensure_genver_asleep()

        try:
            new_ckpt = run_fullft_one_round(
                which=which,
                sft_path=train_sft,
                base_model_path=base_model_path,
                project_name=args.project_name,
                experiment_name=experiment_name,
                out_dir=str(train_out_dir),
                config_name=args.config_name,
                config_override=train_overrides,
                use_subprocess=not args.train_inline,
                train_cuda=args.train_cuda,
            )
        finally:
            if args.pipeline_mode == "inline":
                await _wake_engines(sleepers)
            else:
                # In staged mode, only wake the engine we need for hot reload.
                label = "generator" if which == "gen" else "verifier"
                await _wake_engines([(label, engine)], sleep_state)

        new_ckpt_root = Path(new_ckpt)
        new_base_model_path = resolve_model_dir_for_vllm(new_ckpt)
        await engine.hot_reload_from_dir(new_base_model_path)
        if args.pipeline_mode == "staged" and args.rollout_mode == "phased":
            # Keep engines asleep between phases/rounds when sharing GPUs.
            label = "generator" if which == "gen" else "verifier"
            await _sleep_engines([(label, engine)], sleep_state)
        cleanup_checkpoint_dir(prev_ckpt, new_ckpt_root, train_out_dir)
        print(f"[round {r:03d}] {which} hot-reloaded -> {new_base_model_path}")
        return new_base_model_path, new_ckpt_root

    gen_ckpt = resolve_model_dir_for_vllm(args.gen_base)
    ver_ckpt = resolve_model_dir_for_vllm(args.ver_base)
    prev_gen_ckpt: Path | None = None
    prev_ver_ckpt: Path | None = None

    rounds = args.rounds * 2
    pair_tasks: Optional[List[Dict[str, Any]]] = None
    for r in range(1, rounds + 1):
        role = "gen" if r % 2 == 1 else "ver"
        pair_idx = (r - 1) // 2
        print(f"\n[round {r:03d}] role={role} — rollout + teacher relabel + full FT + hot reload")

        if r % 2 == 1 or pair_tasks is None:
            tasks = _sample_tasks(
                args.dataset_name,
                args.split,
                args.batch_tasks,
                seed=args.seed + pair_idx + 1,
                shard_idx=args.shard_idx,
                num_shards=args.num_shards,
                mode=args.sample_mode,
                start_offset=(args.sample_start + pair_idx * args.batch_tasks) if args.sample_mode == "sequential" else 0,
            )
            pair_tasks = tasks
        else:
            tasks = pair_tasks or []
        if args.dump_tasks:
            _dump_jsonl(tasks, out_dir / f"sampled_tasks_round_{r:03d}.jsonl")
        if not tasks:
            print(f"[round {r:03d}] No tasks sampled; check dataset registration.")
            continue
        collect_for = role if args.collect_for == "auto" else args.collect_for
        if args.rollout_mode == "phased":
            if args.pipeline_mode != "staged":
                raise ValueError("--rollout_mode=phased requires --pipeline_mode=staged (teacher is offline).")

            async def _gen_batch(prompts, uids):
                await _sleep_engines([("verifier", ver_engine)], sleep_state)
                await _wake_engines([("generator", gen_engine)], sleep_state)
                outs = await gen_engine.generate_batch(
                    prompts,
                    chat_template_kwargs={"enable_thinking": True},
                    temperature=args.gen_temp,
                    top_p=args.gen_top_p,
                    sp_extra={"top_k": args.gen_top_k, "min_p": args.gen_min_p},
                    max_tokens=args.max_new_tokens,
                )
                await _sleep_engines([("generator", gen_engine)], sleep_state)
                return outs

            async def _ver_batch(prompts, uids):
                await _sleep_engines([("generator", gen_engine)], sleep_state)
                await _wake_engines([("verifier", ver_engine)], sleep_state)
                outs = await ver_engine.generate_batch(
                    prompts,
                    chat_template_kwargs={"enable_thinking": True},
                    temperature=args.ver_temp,
                    top_p=args.ver_top_p,
                    sp_extra={"top_k": args.ver_top_k, "min_p": args.ver_min_p},
                    max_tokens=args.max_new_tokens,
                )
                await _sleep_engines([("verifier", ver_engine)], sleep_state)
                return outs

            await _ensure_genver_asleep()
            episodes = await rollout_with_pag_phased(
                tasks,
                gen_batch=_gen_batch,
                ver_batch=_ver_batch,
                max_turns=args.max_turns,
                parallel=args.parallel,
            )
        else:
            if args.pipeline_mode == "staged":
                await _ensure_genver_awake()
            episodes = await rollout_with_pag_workflow_engine(
                tasks=tasks,
                gen_engine=gen_engine,
                ver_engine=ver_engine,
                teacher_engine=teacher_engine,
                max_turns=args.max_turns,
                collect_for=("none" if args.pipeline_mode == "staged" else collect_for),
                parallel=args.parallel,
                retry=1,
            )
        if getattr(args, "dump_transcripts_raw", False):
            dump_pag_transcripts_with_raw(episodes, out_dir / f"transcripts_raw_round_{r:03d}.jsonl")
        acc, correct, total, acc_records = _generator_accuracy(episodes)
        print(f"[round {r:03d}] generator accuracy={acc:.3f} ({correct}/{total})")
        dump_path = _dump_accuracy_details(acc_records, out_dir, r, tag="train")
        print(f"[round {r:03d}] accuracy data -> {dump_path}")

        if args.pipeline_mode == "staged":
            await _ensure_genver_asleep()
            await _ensure_teacher_awake()
            try:
                assert teacher_engine is not None
                gen_labeled, ver_labeled = await relabel_pag_episodes_with_teacher(
                    episodes,
                    teacher_engine,
                    collect_for=collect_for,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.t_temp,
                    top_p=args.t_top_p,
                    top_k=args.t_top_k,
                    min_p=args.t_min_p,
                    batch_size=args.teacher_label_batch_size,
                )
                print(
                    f"[round {r:03d}] teacher relabel: gen={gen_labeled}, ver={ver_labeled}"
                )
            finally:
                await _ensure_teacher_asleep()
                # Keep gen/ver asleep; training typically comes next (and may use all GPUs).

        gen_rows, ver_rows = pag_episodes_to_sft_rows(
            episodes,
            collect_for=collect_for,
        )
        latest_gen_sft = ""
        latest_ver_sft = ""
        if args.sft_storage == "parquet_shards":
            if role in {"gen", "both"} and gen_rows:
                shard_path = _write_sft_parquet_shard(
                    gen_rows,
                    Path(gen_sft_path),
                    label="gen_sft",
                    round_idx=r,
                    tokenizer=getattr(gen_engine, "tok", None),
                    max_tokens=args.train_truncate_tokens,
                )
                latest_gen_sft = shard_path
                print(f"[round {r:03d}] +{len(gen_rows)} gen rows -> {shard_path}")
                if args.sft_mirror_jsonl:
                    _append_jsonl(gen_rows, out_dir / "gen_sft.jsonl")
            if role in {"ver", "both"} and ver_rows:
                shard_path = _write_sft_parquet_shard(
                    ver_rows,
                    Path(ver_sft_path),
                    label="ver_sft",
                    round_idx=r,
                    tokenizer=getattr(ver_engine, "tok", None),
                    max_tokens=args.train_truncate_tokens,
                )
                latest_ver_sft = shard_path
                print(f"[round {r:03d}] +{len(ver_rows)} ver rows -> {shard_path}")
                if args.sft_mirror_jsonl:
                    _append_jsonl(ver_rows, out_dir / "ver_sft.jsonl")
        else:
            if role in {"gen", "both"} and gen_rows:
                append_sft_rows(gen_rows, gen_sft_path)
                print(f"[round {r:03d}] +{len(gen_rows)} gen rows -> {gen_sft_path}")
                if args.train_latest_only:
                    latest_gen_sft = _write_latest_sft_parquet(
                        gen_rows,
                        latest_gen_dir,
                        label="gen_sft_latest",
                        round_idx=r,
                        tokenizer=getattr(gen_engine, "tok", None),
                        max_tokens=args.train_truncate_tokens,
                    )
            if role in {"ver", "both"} and ver_rows:
                append_sft_rows(ver_rows, ver_sft_path)
                print(f"[round {r:03d}] +{len(ver_rows)} ver rows -> {ver_sft_path}")
                if args.train_latest_only:
                    latest_ver_sft = _write_latest_sft_parquet(
                        ver_rows,
                        latest_ver_dir,
                        label="ver_sft_latest",
                        round_idx=r,
                        tokenizer=getattr(ver_engine, "tok", None),
                        max_tokens=args.train_truncate_tokens,
                    )

        gen_train_path = latest_gen_sft if args.train_latest_only else gen_sft_path
        ver_train_path = latest_ver_sft if args.train_latest_only else ver_sft_path

        exp_name = f"{args.experiment_name}_{role}_r{r:03d}"
        if role == "gen" and gen_train_path and Path(gen_train_path).exists():
            gen_ckpt, prev_gen_ckpt = await _train_and_hot_reload(
                which="gen",
                sft_path=gen_train_path,
                base_model_path=gen_ckpt,
                train_out_dir=train_out_dir_gen,
                engine=gen_engine,
                prev_ckpt=prev_gen_ckpt,
                experiment_name=exp_name,
            )
        elif role == "ver" and ver_train_path and Path(ver_train_path).exists():
            ver_ckpt, prev_ver_ckpt = await _train_and_hot_reload(
                which="ver",
                sft_path=ver_train_path,
                base_model_path=ver_ckpt,
                train_out_dir=train_out_dir_ver,
                engine=ver_engine,
                prev_ckpt=prev_ver_ckpt,
                experiment_name=exp_name,
            )
        else:
            print(f"[round {r:03d}] no SFT rows for {role}, skipping FT.")

        # If eval is enabled, ensure gen/ver are awake for the legacy workflow runner.
        if (
            args.rollout_mode == "workflow"
            and args.pipeline_mode == "staged"
            and args.eval_num_tasks > 0
            and (r % args.eval_every == 0)
        ):
            await _ensure_genver_awake()

        await _run_eval_if_needed(
            args,
            round_idx=r,
            gen_engine=gen_engine,
            ver_engine=ver_engine,
            teacher_engine=teacher_engine,
            out_dir=out_dir,
            sleep_state=sleep_state,
        )

    print("\nTraining finished.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alternating PAG Gen/Ver training on Numina math with vLLM hot reload + sanity eval."
    )
    p.add_argument("--dataset_name", type=str, default="numina_math")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--dataset_path", type=str, default=None, help="Optional HF/local path to register dataset if missing.")
    p.add_argument(
        "--rollout_mode",
        choices=["workflow", "phased"],
        default="workflow",
        help="workflow: per-episode async rollout. phased: batch all gen steps then all ver steps each turn (sleep/wake between).",
    )
    p.add_argument("--rounds", type=int, default=2, help="Number of gen/ver pairs (total iterations = 2*rounds).")
    p.add_argument("--batch_tasks", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample_mode",
        choices=["random", "sequential"],
        default="sequential",
        help="How to pick tasks each round: random (shuffle) or sequential (walk dataset in order).",
    )
    p.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Start offset for --sample_mode=sequential (offset is applied after sharding).",
    )
    p.add_argument("--num_shards", type=int, default=1, help="Split dataset into N shards (default: 1).")
    p.add_argument("--shard_idx", type=int, default=0, help="Shard index in [0, num_shards-1].")
    p.add_argument("--teacher_base", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--gen_base", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--ver_base", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher_tokenizer", type=str, default=None)
    p.add_argument("--gen_tokenizer", type=str, default=None)
    p.add_argument("--ver_tokenizer", type=str, default=None)
    p.add_argument("--t_temp", type=float, default=0.6)
    p.add_argument("--gen_temp", type=float, default=0.6)
    p.add_argument("--ver_temp", type=float, default=0.6)
    p.add_argument("--t_top_p", type=float, default=0.95)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--ver_top_p", type=float, default=0.95)
    p.add_argument("--t_top_k", type=int, default=20)
    p.add_argument("--gen_top_k", type=int, default=20)
    p.add_argument("--ver_top_k", type=int, default=20)
    p.add_argument("--t_min_p", type=float, default=0.0)
    p.add_argument("--gen_min_p", type=float, default=0.0)
    p.add_argument("--ver_min_p", type=float, default=0.0)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--stop_on_verifier_fix", action="store_true")
    p.add_argument("--collect_for", choices=["auto", "both", "gen", "ver"], default="auto")
    p.add_argument(
        "--pipeline_mode",
        choices=["inline", "staged"],
        default="staged",
        help="inline: call teacher during rollout; staged: rollout w/ gen+ver only, then teacher relabel, then train.",
    )
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--tp_s", type=int, default=1)
    p.add_argument("--tp_t", type=int, default=1)
    p.add_argument(
        "--dp_s",
        type=int,
        default=1,
        help="Inference data-parallel replicas for generator/verifier (requires tp_s*dp_s GPUs per role).",
    )
    p.add_argument(
        "--dp_t",
        type=int,
        default=1,
        help="Inference data-parallel replicas for teacher (requires tp_t*dp_t GPUs).",
    )
    p.add_argument("--max_model_len", type=int, default=40000)
    p.add_argument("--teacher_gpu_mem_util", type=float, default=0.9)
    p.add_argument("--gen_gpu_mem_util", type=float, default=0.9)
    p.add_argument("--ver_gpu_mem_util", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument(
        "--vllm_batch_max",
        type=int,
        default=128,
        help="Max prompts per vLLM micro-batch per engine (applies to gen/ver/teacher).",
    )
    p.add_argument(
        "--vllm_batch_flush_ms",
        type=int,
        default=5,
        help="Micro-batch flush window in ms per vLLM engine (applies to gen/ver/teacher).",
    )
    p.add_argument(
        "--teacher_label_batch_size",
        type=int,
        default=16,
        help="Batch size for staged teacher relabel (number of prompts per vLLM generate call).",
    )
    p.add_argument(
        "--train_truncate_tokens",
        type=int,
        default=0,
        help="If >0, truncate SFT messages from the front to keep tail within this token budget.",
    )
    p.add_argument("--out_dir", type=str, default="runs/pag_numina")
    p.add_argument("--project_name", type=str, default="pag_numina")
    p.add_argument("--experiment_name", type=str, default="exp")
    p.add_argument("--config_name", type=str, default="agent_sft_trainer")
    p.add_argument("--config_override", nargs="*", default=None)
    p.add_argument("--gen_sft_path", type=str, default=None)
    p.add_argument("--ver_sft_path", type=str, default=None)
    p.add_argument(
        "--sft_storage",
        choices=["jsonl", "parquet_shards"],
        default="jsonl",
        help="Where to store accumulated SFT rows. parquet_shards avoids rereading the full JSONL each round.",
    )
    p.add_argument(
        "--train_latest_only",
        action="store_true",
        help="Train only on newly generated SFT rows each round (keep full history on disk).",
    )
    p.add_argument(
        "--sft_mirror_jsonl",
        action="store_true",
        help="When using --sft_storage=parquet_shards, also append SFT rows to out_dir/gen_sft.jsonl and out_dir/ver_sft.jsonl (no dedup).",
    )
    p.add_argument("--teacher_cuda", type=str, default="0")
    p.add_argument("--gen_cuda", type=str, default="1")
    p.add_argument("--ver_cuda", type=str, default="1")
    p.add_argument("--train_cuda", type=str, default="1")
    p.add_argument("--train_inline", action="store_true")
    p.add_argument("--eval_num_tasks", type=int, default=0, help="If >0, run PAG eval each round on this many tasks.")
    p.add_argument("--eval_every", type=int, default=1, help="Eval frequency in rounds.")
    p.add_argument("--eval_dataset_name", type=str, default=None)
    p.add_argument("--eval_dataset_path", type=str, default=None)
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--eval_parallel", type=int, default=None)
    p.add_argument(
        "--force_register",
        action="store_true",
        help="Force re-register dataset even if already in DatasetRegistry (useful after schema fixes).",
    )
    p.add_argument(
        "--dump_dataset_head",
        type=int,
        default=0,
        help="If >0, dump this many preprocessed samples to dataset_head_sample.jsonl for sanity check.",
    )
    p.add_argument(
        "--dump_tasks",
        action="store_true",
        help="Dump sampled tasks for each round to sampled_tasks_round_xxx.jsonl to inspect question/gt.",
    )
    p.add_argument(
        "--dump_transcripts_raw",
        action="store_true",
        help="If set, dump raw/public generator/verifier transcripts per round for debugging.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
