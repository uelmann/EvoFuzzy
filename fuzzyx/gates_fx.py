"""FuzzyX-v1 leakage and shuffle-bias gates. Pre-registered in the addendum."""

from __future__ import annotations

import numpy as np
import torch

from .constants import SEED, SHUFFLE_SEEDS, UNIVERSE_N
from .pack import PackedPanel, slice_packed
from .torch_loss import path_loss_torch
from .torch_model import FuzzyXNet


def gate_feature_lookahead(panel) -> dict:
    from baseline.data import build_pit_topn
    from baseline.gates import gate_feature_lookahead as _g

    return _g(panel)


def gate_universe_lookahead_top30(panel) -> dict:
    from baseline.data import build_pit_topn
    from baseline.gates import gate_universe_lookahead

    return gate_universe_lookahead(
        panel, build_pit_topn, n=UNIVERSE_N, window=30, name="universe_lookahead_top30"
    )


@torch.no_grad()
def gate_seed_determinism(packed: PackedPanel, seed: int = SEED) -> dict:
    if packed.X.shape[0] < 4:
        return {"name": "seed_determinism", "passed": False, "reason": "short packed"}
    a = FuzzyXNet(seed=seed).eval()
    b = FuzzyXNet(seed=seed).eval()
    x = torch.from_numpy(packed.X[:8]).float()
    m = torch.from_numpy(packed.mask[:8])
    pa = a(x, m)["soft_pos"].numpy()
    pb = b(x, m)["soft_pos"].numpy()
    max_diff = float(np.max(np.abs(pa - pb)))
    return {
        "name": "seed_determinism",
        "passed": bool(max_diff < 1e-6),
        "max_score_diff": max_diff,
    }


@torch.no_grad()
def gate_shuffle_bias(
    model: FuzzyXNet,
    packed: PackedPanel,
    fold_start,
    fold_end,
    seeds: tuple[int, ...] = SHUFFLE_SEEDS,
) -> dict:
    """Shuffle 7-day forward returns within date; mean weekly net PnL must be centered.

    Uses this fold's own weights (caller loads them). Statistic is mean_pnl,
    not path-loss core (v1b). Positions are computed once; only labels shuffle.
    """
    sl = slice_packed(packed, fold_start, fold_end)
    if sl.X.shape[0] < 4:
        return {"name": "label_shuffle_bias", "passed": False, "reason": "short fold"}
    device = next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(sl.X).to(device=device, dtype=torch.float32)
    m = torch.from_numpy(sl.mask).to(device=device)
    pos = model(x, m)["soft_pos"]
    pnls = []
    corrs = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        rh = sl.ret_h7.copy()
        for t in range(rh.shape[0]):
            idx = np.flatnonzero(sl.mask[t])
            if idx.size < 3:
                continue
            rh[t, idx] = rng.permutation(rh[t, idx])
        rt = torch.from_numpy(rh).to(device=device, dtype=torch.float32)
        stats = path_loss_torch(pos, rt, mask=m)
        pnls.append(float(stats["mean_pnl"].cpu()))
        corrs.append(float(stats["core"].cpu()))
    arr = np.asarray(pnls, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = sd / max(np.sqrt(len(arr)), 1.0)
    passed = bool(abs(mean) <= 2.0 * se + 1e-12) if se > 0 else abs(mean) < 1e-6
    return {
        "name": "label_shuffle_bias",
        "statistic": "mean_weekly_net_pnl",
        "passed": passed,
        "mean_pnl": mean,
        "sd": sd,
        "se": float(se),
        "threshold": float(2.0 * se),
        "n": int(arr.size),
        "pnls": [float(c) for c in arr],
        "mean_corr_st_r": float(np.mean(corrs)),
        "corrs": [float(c) for c in corrs],
    }


def run_leakage_gates(panel, packed: PackedPanel) -> list[dict]:
    out = [
        gate_feature_lookahead(panel),
        gate_universe_lookahead_top30(panel),
        gate_seed_determinism(packed),
    ]
    for r in out:
        print(f"[gates] {r.get('name')}: {'PASS' if r.get('passed') else 'FAIL'} {r}", flush=True)
    return out
