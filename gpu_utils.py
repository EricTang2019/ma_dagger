from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional


def parse_cuda_list(spec: Optional[str]) -> Optional[List[str]]:
    if not spec:
        return None
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    if not ids:
        return None
    if any(not part.isdigit() for part in ids):
        raise ValueError(f"Invalid CUDA device list: {spec}")
    return ids


def ensure_tp_fit(devices: Optional[List[str]], tp: int, name: str):
    if devices is not None and tp > len(devices):
        raise ValueError(
            f"{name}: tensor_parallel_size={tp} requires {tp} GPUs but only {len(devices)} provided ({devices})"
        )


def devices_overlap(a: Optional[List[str]], b: Optional[List[str]]) -> bool:
    if not a or not b:
        return False
    return bool(set(a) & set(b))


def warn_overlap(label: str):
    print(f"[warn] GPU sets overlap between {label}; expect possible contention/OOM.", flush=True)


@contextmanager
def cuda_visible_devices(devices: Optional[List[str]]):
    import os

    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev
