"""PAG-style alternating Gen/Ver training on Numina-style math data with per-round tests."""

from __future__ import annotations

import torch.multiprocessing as mp

# Ensure CUDA init happens under spawn instead of fork when vLLM spins up workers.
mp.set_start_method("spawn", force=True)

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

from checkpoint_utils import cleanup_checkpoint_dir, resolve_model_dir_for_vllm
from data_utils import append_sft_rows
from fullft_training import run_fullft_one_round
from genver_pag_workflow import (
    pag_episodes_to_sft_rows,
    rollout_with_pag_workflow_engine,
    dump_pag_transcripts_with_raw,
)
from rllm.rewards.math_utils.utils import extract_answer
from gpu_utils import cuda_visible_devices, devices_overlap, ensure_tp_fit, parse_cuda_list, warn_overlap
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from vllm_engine import VLLMChatEngine, VLLMConfig

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


def _sample_tasks(
    dataset_name: str,
    split: str,
    n: int,
    seed: int = 0,
    *,
    shard_idx: int = 0,
    num_shards: int = 1,
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
    rng = random.Random(seed)
    idx = list(range(len(data)))
    if num_shards > 1:
        idx = [i for i in idx if i % num_shards == shard_idx]
    rng.shuffle(idx)
    idx = idx[: n if n < len(idx) else len(idx)]

    tasks: List[Dict[str, Any]] = []
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


async def _run_eval_if_needed(
    args: argparse.Namespace,
    round_idx: int,
    gen_engine: VLLMChatEngine,
    ver_engine: VLLMChatEngine,
    teacher_engine: VLLMChatEngine,
    out_dir: Path,
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


async def _sleep_engines(pairs):
    for label, engine in pairs:
        try:
            await engine.sleep()
            print(f"[info] {label} engine slept to free GPU memory.")
        except Exception as err:
            print(f"[warn] Failed to sleep {label} engine: {err}")


async def _wake_engines(pairs):
    for label, engine in pairs:
        try:
            await engine.wake_up()
            print(f"[info] {label} engine woke up.")
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
        head_tasks = _sample_tasks(args.dataset_name, args.split, args.dump_dataset_head, seed=args.seed)
        _dump_jsonl(head_tasks, out_dir / "dataset_head_sample.jsonl")

    gen_sft_path = str(out_dir / "gen_sft.jsonl") if not args.gen_sft_path else args.gen_sft_path
    ver_sft_path = str(out_dir / "ver_sft.jsonl") if not args.ver_sft_path else args.ver_sft_path

    teacher_devices = parse_cuda_list(args.teacher_cuda)
    gen_devices = parse_cuda_list(args.gen_cuda)
    ver_devices = parse_cuda_list(args.ver_cuda)

    ensure_tp_fit(teacher_devices, args.tp_t, "teacher")
    ensure_tp_fit(gen_devices, args.tp_s, "generator")
    ensure_tp_fit(ver_devices, args.tp_s, "verifier")
    if devices_overlap(teacher_devices, gen_devices):
        warn_overlap("teacher & generator")
    if devices_overlap(teacher_devices, ver_devices):
        warn_overlap("teacher & verifier")
    if devices_overlap(gen_devices, ver_devices):
        warn_overlap("generator & verifier")

    with cuda_visible_devices(teacher_devices):
        teacher_engine = VLLMChatEngine(
            VLLMConfig(
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
            )
        )
    with cuda_visible_devices(gen_devices):
        gen_engine = VLLMChatEngine(
            VLLMConfig(
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
            )
        )
    with cuda_visible_devices(ver_devices):
        ver_engine = VLLMChatEngine(
            VLLMConfig(
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
            )
        )

    gen_ckpt = resolve_model_dir_for_vllm(args.gen_base)
    ver_ckpt = resolve_model_dir_for_vllm(args.ver_base)
    prev_gen_ckpt: Path | None = None
    prev_ver_ckpt: Path | None = None

    rounds = args.rounds * 2
    for r in range(1, rounds + 1):
        role = "gen" if r % 2 == 1 else "ver"
        print(f"\n[round {r:03d}] role={role} — rollout + teacher relabel + full FT + hot reload")

        tasks = _sample_tasks(
            args.dataset_name,
            args.split,
            args.batch_tasks,
            seed=args.seed + r,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
        )
        if args.dump_tasks:
            _dump_jsonl(tasks, out_dir / f"sampled_tasks_round_{r:03d}.jsonl")
        if not tasks:
            print(f"[round {r:03d}] No tasks sampled; check dataset registration.")
            continue
        episodes = await rollout_with_pag_workflow_engine(
            tasks=tasks,
            gen_engine=gen_engine,
            ver_engine=ver_engine,
            teacher_engine=teacher_engine,
            max_turns=args.max_turns,
            collect_for=role if args.collect_for == "auto" else args.collect_for,
            parallel=args.parallel,
            retry=1,
        )
        if getattr(args, "dump_transcripts_raw", False):
            dump_pag_transcripts_with_raw(episodes, out_dir / f"transcripts_raw_round_{r:03d}.jsonl")
        acc, correct, total, acc_records = _generator_accuracy(episodes)
        print(f"[round {r:03d}] generator accuracy={acc:.3f} ({correct}/{total})")
        dump_path = _dump_accuracy_details(acc_records, out_dir, r, tag="train")
        print(f"[round {r:03d}] accuracy data -> {dump_path}")

        gen_rows, ver_rows = pag_episodes_to_sft_rows(
            episodes,
            collect_for=role if args.collect_for == "auto" else args.collect_for,
        )
        if role in {"gen", "both"} and gen_rows:
            append_sft_rows(gen_rows, gen_sft_path)
            print(f"[round {r:03d}] +{len(gen_rows)} gen rows -> {gen_sft_path}")
        if role in {"ver", "both"} and ver_rows:
            append_sft_rows(ver_rows, ver_sft_path)
            print(f"[round {r:03d}] +{len(ver_rows)} ver rows -> {ver_sft_path}")

        exp_name = f"{args.experiment_name}_{role}_r{r:03d}"
        if role == "gen" and Path(gen_sft_path).exists():
            train_sft = _sft_path_for_training(
                gen_sft_path,
                out_dir,
                "gen",
                tokenizer=getattr(gen_engine, "tok", None),
                max_tokens=args.train_truncate_tokens,
            )
            sleepers = [("teacher", teacher_engine), ("generator", gen_engine), ("verifier", ver_engine)]
            await _sleep_engines(sleepers)
            try:
                new_ckpt = run_fullft_one_round(
                    which="gen",
                    sft_path=train_sft,
                    base_model_path=gen_ckpt,
                    project_name=args.project_name,
                    experiment_name=exp_name,
                    out_dir=str(train_out_dir_gen),
                    config_name=args.config_name,
                    config_override=args.config_override,
                    use_subprocess=not args.train_inline,
                    train_cuda=args.train_cuda,
                )
            finally:
                await _wake_engines(sleepers)
            new_ckpt_root = Path(new_ckpt)
            gen_ckpt = resolve_model_dir_for_vllm(new_ckpt)
            await gen_engine.hot_reload_from_dir(gen_ckpt)
            cleanup_checkpoint_dir(prev_gen_ckpt, new_ckpt_root, train_out_dir_gen)
            prev_gen_ckpt = new_ckpt_root
            print(f"[round {r:03d}] generator hot-reloaded -> {gen_ckpt}")
        elif role == "ver" and Path(ver_sft_path).exists():
            train_sft = _sft_path_for_training(
                ver_sft_path,
                out_dir,
                "ver",
                tokenizer=getattr(ver_engine, "tok", None),
                max_tokens=args.train_truncate_tokens,
            )
            sleepers = [("teacher", teacher_engine), ("generator", gen_engine), ("verifier", ver_engine)]
            await _sleep_engines(sleepers)
            try:
                new_ckpt = run_fullft_one_round(
                    which="ver",
                    sft_path=train_sft,
                    base_model_path=ver_ckpt,
                    project_name=args.project_name,
                    experiment_name=exp_name,
                    out_dir=str(train_out_dir_ver),
                    config_name=args.config_name,
                    config_override=args.config_override,
                    use_subprocess=not args.train_inline,
                    train_cuda=args.train_cuda,
                )
            finally:
                await _wake_engines(sleepers)
            new_ckpt_root = Path(new_ckpt)
            ver_ckpt = resolve_model_dir_for_vllm(new_ckpt)
            await ver_engine.hot_reload_from_dir(ver_ckpt)
            cleanup_checkpoint_dir(prev_ver_ckpt, new_ckpt_root, train_out_dir_ver)
            prev_ver_ckpt = new_ckpt_root
            print(f"[round {r:03d}] verifier hot-reloaded -> {ver_ckpt}")
        else:
            print(f"[round {r:03d}] no SFT rows for {role}, skipping FT.")

        await _run_eval_if_needed(
            args,
            round_idx=r,
            gen_engine=gen_engine,
            ver_engine=ver_engine,
            teacher_engine=teacher_engine,
            out_dir=out_dir,
        )

    print("\nTraining finished.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alternating PAG Gen/Ver training on Numina math with vLLM hot reload + sanity eval."
    )
    p.add_argument("--dataset_name", type=str, default="numina_math")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--dataset_path", type=str, default=None, help="Optional HF/local path to register dataset if missing.")
    p.add_argument("--rounds", type=int, default=2, help="Number of gen/ver pairs (total iterations = 2*rounds).")
    p.add_argument("--batch_tasks", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
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
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--tp_s", type=int, default=1)
    p.add_argument("--tp_t", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--teacher_gpu_mem_util", type=float, default=0.9)
    p.add_argument("--gen_gpu_mem_util", type=float, default=0.9)
    p.add_argument("--ver_gpu_mem_util", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=8192)
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
    mp.set_start_method("spawn", force=True)
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
