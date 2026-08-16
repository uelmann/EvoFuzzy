"""Pure training-craft helpers. No panel / LightGBM import."""

from __future__ import annotations

import numpy as np

from nfn.constants_v1 import INNER_HOLDOUT_DATES, PURGE_EXTRA, SWA_K, TRAILING_K


def split_inner_holdout(
    tr_ids: np.ndarray,
    *,
    n_ho: int = INNER_HOLDOUT_DATES,
    horizon: int = 14,
    purge_extra: int = PURGE_EXTRA,
) -> tuple[np.ndarray, np.ndarray]:
    """Last `n_ho` train dates as inner holdout; purge h+3 dates between train and holdout."""
    tr_ids = np.asarray(tr_ids, dtype=np.int32)
    n = int(len(tr_ids))
    n_purge = int(horizon) + int(purge_extra)
    n_ho = int(n_ho)
    need = n_ho + n_purge + 10
    if n < need:
        n_ho = max(5, min(n_ho, max(5, n // 6)))
        n_purge = max(0, min(n_purge, max(0, n - n_ho - 10)))
    inner_ho = tr_ids[-n_ho:]
    cut = n - n_ho - n_purge
    if cut < 10:
        cut = max(10, n - n_ho)
    inner_tr = tr_ids[:cut]
    if len(inner_tr) < 10 or len(inner_ho) < 5:
        split = max(1, int(n * 0.85))
        inner_tr, inner_ho = tr_ids[:split], tr_ids[split:]
    return np.asarray(inner_tr, dtype=np.int32), np.asarray(inner_ho, dtype=np.int32)


def trailing_mean(values: list[float], k: int = TRAILING_K) -> list[float]:
    out = []
    for i in range(len(values)):
        w = values[max(0, i - int(k) + 1) : i + 1]
        finite = [float(v) for v in w if v is not None and np.isfinite(v)]
        out.append(float(np.mean(finite)) if finite else float("nan"))
    return out


def swa_average(states: list[dict]) -> dict:
    import torch

    if not states:
        raise ValueError("SWA needs at least one state")
    if len(states) == 1:
        return {k: v.clone() for k, v in states[0].items()}
    out = {}
    for k in states[0]:
        tensors = [s[k] for s in states]
        if tensors[0].is_floating_point():
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            avg = stacked.mean(dim=0).to(dtype=tensors[0].dtype)
            out[k] = avg
        else:
            out[k] = tensors[-1].clone()
    return out


def pick_swa_epochs(scores: list[float], k: int = SWA_K) -> list[int]:
    """1-based epoch indices of the top-k trailing-mean scores."""
    ranked = []
    for i, v in enumerate(scores):
        if v is not None and np.isfinite(v):
            ranked.append((float(v), i + 1))
    if not ranked:
        return list(range(1, min(int(k), len(scores)) + 1)) or [1]
    ranked.sort(key=lambda t: t[0], reverse=True)
    epochs = sorted(t[1] for t in ranked[: int(k)])
    return epochs


def init_seed(seed: int, bag: int) -> int:
    return int(seed) * 1_000_003 + int(bag) * 1_009
