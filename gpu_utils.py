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


def ensure_tp_dp_fit(devices: Optional[List[str]], tp: int, dp: int, name: str):
    tp = max(1, int(tp or 1))
    dp = max(1, int(dp or 1))
    if devices is None:
        raise ValueError(f"{name}: no CUDA devices provided")
    required = tp * dp
    if required > len(devices):
        raise ValueError(
            f"{name}: tp={tp} dp={dp} requires {required} GPUs but only {len(devices)} provided ({devices})"
        )


def split_devices_for_tp_dp(devices: Optional[List[str]], tp: int, dp: int, name: str) -> List[List[str]]:
    """Partition a flat CUDA device list into `dp` groups of size `tp`."""
    ensure_tp_dp_fit(devices, tp, dp, name)
    assert devices is not None
    tp = max(1, int(tp or 1))
    dp = max(1, int(dp or 1))
    required = tp * dp
    selected = devices[:required]
    return [selected[i * tp : (i + 1) * tp] for i in range(dp)]


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
