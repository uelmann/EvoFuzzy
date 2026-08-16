"""Phase 7.c mechanical LIVE / PARKED / CRAFT-CONFIRMED. No post-hoc adjustment."""

from __future__ import annotations

import numpy as np

from nfn.constants import OVERLAP_DELTA, SEED_DISP_MAX, TAIL_IC_DELTA
from nfn.constants_v1 import CRAFT_MEAN_EPOCH_MIN, PHASE7C_CRAFT_LEDGER, V0_SEED_DISPERSION
from nfn.verdict import _f, mechanical_verdict as mechanical_verdict_v0


def mean_selected_epoch(hygiene: list[dict]) -> float:
    """Mean of the primary selected epoch across bagging-ensemble (seed, fold) cells.

    Hygiene rows may be per (seed, fold, bag). We average bags first, then mean
    across seed×fold so the statistic matches 'seeds at the bagging-ensemble level'.
    """
    buckets: dict[tuple[int, int], list[float]] = {}
    for h in hygiene or []:
        if h.get("status") not in (None, "ok"):
            continue
        ep = h.get("selected_epoch", h.get("best_epoch"))
        try:
            v = float(ep)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        key = (int(h.get("seed", -1)), int(h.get("fold_id", -1)))
        buckets.setdefault(key, []).append(v)
    if not buckets:
        return float("nan")
    per_cell = [float(np.mean(vs)) for vs in buckets.values()]
    return float(np.mean(per_cell))


def craft_confirmed(dispersion: float, mean_epoch: float, v0_disp: float = V0_SEED_DISPERSION) -> dict:
    d = _f(dispersion)
    e = _f(mean_epoch)
    disp_ok = bool(np.isfinite(d) and d <= float(SEED_DISP_MAX))
    epoch_ok = bool(np.isfinite(e) and e >= float(CRAFT_MEAN_EPOCH_MIN))
    confirmed = bool(disp_ok and epoch_ok)
    return {
        "craft_confirmed": confirmed,
        "disp_ok": disp_ok,
        "epoch_ok": epoch_ok,
        "dispersion": d,
        "mean_selected_epoch": e,
        "v0_dispersion": float(v0_disp),
        "dispersion_improved_to_bar": disp_ok,
    }


def mechanical_verdict_v1(grid: dict, null: dict, seed_metrics: dict[int, dict], hygiene: list[dict]) -> dict:
    base = mechanical_verdict_v0(grid, null, seed_metrics)
    base["label"] = "NFN-v1 LIVE" if base.get("live") else "NFN-v1 PARKED"
    mean_ep = mean_selected_epoch(hygiene)
    craft = craft_confirmed(base.get("seed_dispersion_tail_ic"), mean_ep)
    base.update(craft)
    ledger = None
    if craft["craft_confirmed"] and not base.get("live"):
        ledger = PHASE7C_CRAFT_LEDGER
    base["craft_ledger"] = ledger
    base["mean_selected_epoch"] = mean_ep
    base["nothing_adopted"] = True
    return base
