"""Mechanical LIVE / PARKED + MAGNITUDE-GAIN. No post-hoc adjustment."""

from __future__ import annotations

import numpy as np

from nfn_va.constants import MAG_GAIN_REL, OVERLAP_DELTA, SEED_DISP_MAX, TAIL_IC_DELTA


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def magnitude_gain(va_top10: float, frozen_top10: float, rel: float = MAG_GAIN_REL) -> dict:
    a, b = _f(va_top10), _f(frozen_top10)
    if not np.isfinite(a) or not np.isfinite(b):
        return {"yes": False, "va": a, "frozen": b, "rel": float("nan"), "need": float(rel)}
    if b > 0:
        ratio = a / b
        yes = bool(ratio >= 1.0 + float(rel))
        rel_delta = ratio - 1.0
    else:
        rel_delta = (a - b) / max(abs(b), 1e-12)
        yes = bool(rel_delta >= float(rel) and a > b)
    return {"yes": yes, "va": a, "frozen": b, "rel": float(rel_delta), "need": float(rel)}


def mechanical_verdict(grid: dict, null: dict, seed_metrics: dict[int, dict], mag: dict | None = None) -> dict:
    base = grid.get("frozen_spread") or {}
    ens = grid.get("variant_a_ensemble") or grid.get("nfn_ensemble") or {}
    d_ic = _f(ens.get("tail_ic_top")) - _f(base.get("tail_ic_top"))
    d_ov = _f(ens.get("overlap")) - _f(base.get("overlap"))
    clause_a = bool(
        np.isfinite(d_ic) and np.isfinite(d_ov) and d_ic >= float(TAIL_IC_DELTA) and d_ov >= float(OVERLAP_DELTA)
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
    mag_rec = mag or {}
    label = "VARIANT-A LIVE" if live else "VARIANT-A PARKED"
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
        "magnitude_gain": bool(mag_rec.get("yes")),
        "magnitude_gain_rec": mag_rec,
        "nothing_adopted": True,
    }
