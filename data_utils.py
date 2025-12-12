from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

import pandas as pd


def _messages_hash_for_row(messages: Any) -> str:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, str):
        payload = messages
    else:
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def append_parquet(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            old_df = pd.read_parquet(path)
            df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception as err:
            print(f"[warn] Failed to read existing parquet {path} ({err}); overwriting.")
            df = new_df
    else:
        df = new_df
    if "messages" in df.columns:
        df["_msg_hash"] = df["messages"].apply(_messages_hash_for_row)
        df = df.drop_duplicates(subset="_msg_hash", keep="last")
        df = df.drop(columns="_msg_hash")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def append_sft_rows(rows: List[Dict[str, Any]], path: str):
    """Append SFT rows to a json/jsonl/parquet file with dedup by messages hash."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Parquet path: delegate to append_parquet (it already dedups by messages)
    if path.lower().endswith(".parquet"):
        append_parquet(rows, path)
        return

    existing: List[Dict[str, Any]] = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except Exception:
                        continue

    # Dedup with new rows overwriting old ones by messages hash
    def _row_hash(row: Dict[str, Any]) -> str:
        msgs = row.get("messages") if isinstance(row, dict) else row
        return _messages_hash_for_row(msgs)

    merged: Dict[str, Dict[str, Any]] = {}
    for r in existing:
        try:
            merged[_row_hash(r)] = r
        except Exception:
            continue
    for r in rows:
        try:
            merged[_row_hash(r)] = r
        except Exception:
            continue

    with open(path, "w", encoding="utf-8") as fout:
        for r in merged.values():
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
