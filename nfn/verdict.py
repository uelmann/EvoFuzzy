"""Mechanical LIVE / PARKED verdicts. No post-hoc adjustment."""

from __future__ import annotations

import numpy as np

from nfn.constants import (
    OVERLAP_DELTA,
    PHASE7_CEILING,
    SEED_DISP_MAX,
    TAIL_IC_DELTA,
)


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def mechanical_verdict(grid: dict, null: dict, seed_metrics: dict[int, dict]) -> dict:
    base = grid.get("frozen_spread") or {}
    ens = grid.get("nfn_ensemble") or {}
    d_ic = _f(ens.get("tail_ic_top")) - _f(base.get("tail_ic_top"))
    d_ov = _f(ens.get("overlap")) - _f(base.get("overlap"))
    clause_a = bool(
        np.isfinite(d_ic)
        and np.isfinite(d_ov)
        and d_ic >= float(TAIL_IC_DELTA)
        and d_ov >= float(OVERLAP_DELTA)
    )
    ics = []
    for _s, met in sorted(seed_metrics.items()):
        v = _f((met or {}).get("tail_ic_top"))
        if np.isfinite(v):
            ics.append(v)
    disp = float(max(ics) - min(ics)) if len(ics) >= 2 else float("nan")
    clause_b = bool(np.isfinite(disp) and disp <= float(SEED_DISP_MAX))
    clause_c = bool((null or {}).get("passed"))
    live = bool(clause_a and clause_b and clause_c)
    failed = []
    if not clause_a:
        failed.append("a")
    if not clause_b:
        failed.append("b")
    if not clause_c:
        failed.append("c")
    ceiling = None
    if (not clause_a) and clause_b and clause_c:
        ceiling = PHASE7_CEILING
    label = "NFN LIVE" if live else "NFN PARKED"
    return {
        "label": label,
        "live": live,
        "clause_a": clause_a,
        "clause_b": clause_b,
        "clause_c": clause_c,
        "failed_clauses": failed,
        "delta_tail_ic": d_ic,
        "delta_overlap": d_ov,
        "seed_dispersion_tail_ic": disp,
        "null_pass": clause_c,
        "null_verdict": ((null or {}).get("tail_ic_top") or {}).get("verdict"),
        "ceiling": ceiling,
        "nothing_adopted": True,
    }
