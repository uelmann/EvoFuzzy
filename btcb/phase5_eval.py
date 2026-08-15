"""Phase 5 judgment: tail-metric grid, adapted E.1b null, mechanical verdict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline.evaluate import newey_west_t
from btcb.constants import (
    ALT_BPS,
    ORACLE_LADDER2_MONSTER_K,
    ORACLE_LADDER_DECILE,
    ORACLE_LADDER_MIN_N,
    PHASE2_CYCLES,
    PHASE5_CEILING_SENTENCE,
    PHASE5_CRITERION,
    PHASE5_DELTA_OVERLAP,
    PHASE5_DELTA_TAIL_IC,
    PHASE5_NW_LAG,
    PHASE5_NULL_FOLDS,
    PHASE5_SEED_DISP_MAX,
    PHASE5_TRAIL_DAYS,
)
from btcb.oracle_ladder import (
    _as_utc,
    eligible,
    period_excess,
    run_periodic_long,
)
from btcb.oracle_ladder2 import _decile_ids, _half_ic


def _utc_idx(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert("UTC").normalize()


def cell_stats(ics: list[float]) -> dict:
    arr = np.asarray([x for x in ics if np.isfinite(x)], dtype=float)
    n = int(len(arr))
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    p95 = float(np.percentile(arr, 95)) if n else float("nan")
    se = (sd / np.sqrt(n)) if n and np.isfinite(sd) else float("nan")
    bias_lim = 2.0 * se if np.isfinite(se) else float("nan")
    bias_ok = bool(np.isfinite(mean) and np.isfinite(bias_lim) and abs(mean) <= bias_lim)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "p95": p95,
        "se": float(se) if np.isfinite(se) else float("nan"),
        "bias_lim": float(bias_lim) if np.isfinite(bias_lim) else float("nan"),
        "bias_ok": bias_ok,
        "ics": [float(x) for x in arr],
    }


def scores_from_twin(twin: pd.DataFrame, score_col: str = "spread") -> dict:
    tw = twin.copy()
    tw["date"] = pd.to_datetime(tw["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    tw["id"] = tw["id"].astype(int)
    tw = tw.sort_values(["date", "id", "fold_id"] if "fold_id" in tw.columns else ["date", "id"])
    if "fold_id" in tw.columns:
        tw = tw.drop_duplicates(["date", "id"], keep="last")
    out = {}
    for dt, g in tw.groupby("date", sort=True):
        s = pd.Series(g[score_col].to_numpy(dtype=float), index=g["id"].astype(int))
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        out[_as_utc(dt)] = s
    return out


def _window_mask(dates: pd.DatetimeIndex, start=None, end=None) -> np.ndarray:
    m = np.ones(len(dates), dtype=bool)
    if start is not None:
        m &= dates >= _as_utc(start)
    if end is not None:
        m &= dates <= _as_utc(end)
    return m


def _summarize_series(ser: pd.Series, lag: int = PHASE5_NW_LAG) -> dict:
    ser = ser.dropna()
    vals = ser.to_numpy(dtype=float) if len(ser) else np.asarray([], dtype=float)
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)) if len(vals) else float("nan"),
        "nw_t": float(newey_west_t(vals, lag=int(lag))) if len(vals) else float("nan"),
    }


def per_date_diagnostics(close, members, btc_id, score_at, pairs) -> pd.DataFrame:
    """One row per formation: overlap, monster, tail-IC halves, RankIC."""
    rows = []
    for t, t_h, _ in pairs:
        t, t_h = _as_utc(t), _as_utc(t_h)
        sc = score_at.get(t)
        ids = eligible(close, members, btc_id, t, t_h)
        ex = period_excess(close, btc_id, t, t_h, ids)
        if sc is None or len(sc) == 0 or len(ex) < int(ORACLE_LADDER_MIN_N):
            continue
        sc = pd.Series(sc, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        sc, ex_a = sc.align(ex, join="inner")
        sc = sc.dropna()
        ex_a = ex_a.reindex(sc.index).dropna()
        sc = sc.reindex(ex_a.index)
        if len(sc) < int(ORACLE_LADDER_MIN_N):
            continue
        from scipy import stats as _st

        sig = set(_decile_ids(sc))
        k_real = max(1, len(ex_a) // int(ORACLE_LADDER_DECILE))
        real = set(int(i) for i in ex_a.nlargest(k_real).index.tolist())
        k_m = min(int(ORACLE_LADDER2_MONSTER_K), len(ex_a))
        monsters = set(int(i) for i in ex_a.nlargest(k_m).index.tolist())
        from btcb.oracle_ladder import _spearman

        ric = _spearman(sc.to_numpy(), ex_a.to_numpy())
        rows.append(
            {
                "date": t,
                "overlap": float(len(sig & real) / max(len(sig), 1)),
                "monster": float(len(sig & monsters) / max(len(monsters), 1)),
                "tail_ic_top": _half_ic(sc, ex_a, "top"),
                "tail_ic_bot": _half_ic(sc, ex_a, "bottom"),
                "rankic": ric,
                "n_sig": int(len(sig)),
                "n_cs": int(len(sc)),
            }
        )
    return pd.DataFrame(rows)


def windowed_metrics(daily: pd.DataFrame, oos_start, oos_end) -> dict:
    if daily is None or daily.empty:
        empty = {k: {"n": 0, "mean": float("nan"), "nw_t": float("nan")} for k in
                 ("tail_ic_top", "tail_ic_bot", "overlap", "monster", "rankic")}
        out = {"full": empty, "trail18m": empty, "cycles": {}}
        return out
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    d = d[(d["date"] >= _as_utc(oos_start)) & (d["date"] <= _as_utc(oos_end))]
    keys = ("tail_ic_top", "tail_ic_bot", "overlap", "monster", "rankic")

    def pack(sl: pd.DataFrame) -> dict:
        out = {}
        for k in keys:
            s = pd.Series(sl[k].to_numpy(dtype=float), index=sl["date"]) if k in sl.columns else pd.Series(dtype=float)
            out[k] = _summarize_series(s)
        return out

    full = pack(d)
    end = d["date"].max() if len(d) else _as_utc(oos_end)
    trail_start = end - pd.Timedelta(days=int(PHASE5_TRAIL_DAYS))
    trail = pack(d[d["date"] >= trail_start])
    cycles = {}
    for name, a, b in PHASE2_CYCLES:
        sl = d[(d["date"] >= _as_utc(a)) & (d["date"] <= _as_utc(b))]
        cycles[name] = pack(sl)
        cycles[name]["n_formations"] = int(len(sl))
    return {"full": full, "trail18m": trail, "cycles": cycles, "n_formations": int(len(d))}


def crude_book(close, members, btc_id, score_at, pairs, label: str) -> dict:
    packed = run_periodic_long(
        close, members, btc_id, score_at, pairs, cost_bps=float(ALT_BPS), label=label
    )
    packed.setdefault("cagr", packed.get("book_cagr"))
    packed.setdefault("total", packed.get("book_total"))
    packed.setdefault("sharpe", packed.get("book_sharpe"))
    packed.setdefault("maxdd", packed.get("maxdd"))
    return packed


def fold_tail_ic_from_pred(pred: pd.DataFrame) -> float:
    if pred is None or pred.empty:
        return float("nan")
    p = pred.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True)
    score = p["spread"] if "spread" in p.columns else (p["p_top_raw"] - p["p_bot_raw"])
    from btcb.csattn import per_date_tail_ic_top

    s = per_date_tail_ic_top(p["date"], score, p["excess_h14"])
    return float(s.mean()) if len(s) else float("nan")


def null_verdict(cells: list[dict]) -> dict:
    """Adapted E.1b: 2 folds, CONTAMINATED if ≥2 bias violations; skill needs both folds."""
    n_violate = sum(1 for c in cells if not c.get("bias_ok"))
    bias_pass = n_violate < 2
    n_ex = sum(
        1
        for c in cells
        if np.isfinite(c.get("real_ic", np.nan))
        and np.isfinite(c.get("p95", np.nan))
        and float(c["real_ic"]) > float(c["p95"])
    )
    skill_pass = n_ex >= 2
    if not bias_pass:
        status = "CONTAMINATED"
        passed = False
    elif skill_pass:
        status = "PASS"
        passed = True
    else:
        status = "PARKED-NO-SKILL"
        passed = False
    return {
        "bias_pass": bool(bias_pass),
        "skill_pass": bool(skill_pass),
        "n_violate": int(n_violate),
        "n_cells": int(len(cells)),
        "n_exceed": int(n_ex),
        "need_exceed": 2,
        "verdict": status,
        "passed": bool(passed),
        "cells": cells,
    }


def mechanical_verdict(baseline: dict, seeds: dict, ensemble: dict, null: dict) -> dict:
    b_ic = float((baseline.get("full") or {}).get("tail_ic_top", {}).get("mean", float("nan")))
    b_ov = float((baseline.get("full") or {}).get("overlap", {}).get("mean", float("nan")))
    e_ic = float((ensemble.get("full") or {}).get("tail_ic_top", {}).get("mean", float("nan")))
    e_ov = float((ensemble.get("full") or {}).get("overlap", {}).get("mean", float("nan")))
    d_ic = e_ic - b_ic if np.isfinite(e_ic) and np.isfinite(b_ic) else float("nan")
    d_ov = e_ov - b_ov if np.isfinite(e_ov) and np.isfinite(b_ov) else float("nan")
    per_seed = []
    for s in sorted(seeds, key=lambda z: int(z)):
        blob = seeds[s]
        v = float((blob.get("full") or {}).get("tail_ic_top", {}).get("mean", float("nan")))
        per_seed.append(v)
    finite = [x for x in per_seed if np.isfinite(x)]
    disp = (max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
    a_ok = bool(
        np.isfinite(d_ic)
        and np.isfinite(d_ov)
        and d_ic >= float(PHASE5_DELTA_TAIL_IC)
        and d_ov >= float(PHASE5_DELTA_OVERLAP)
    )
    b_ok = bool(np.isfinite(disp) and disp <= float(PHASE5_SEED_DISP_MAX) and len(finite) == 3)
    c_ok = bool(null.get("passed"))
    live = bool(a_ok and b_ok and c_ok)
    failed = []
    if not a_ok:
        failed.append("a")
    if not b_ok:
        failed.append("b")
    if not c_ok:
        failed.append("c")
    ceiling = bool((not a_ok) and b_ok and ("a" in failed))
    return {
        "verdict": "LIVE" if live else "PARKED",
        "live": live,
        "clause_a": a_ok,
        "clause_b": b_ok,
        "clause_c": c_ok,
        "failed_clauses": failed,
        "delta_tail_ic_top": d_ic,
        "delta_overlap": d_ov,
        "seed_dispersion": disp,
        "per_seed_tail_ic_top": per_seed,
        "baseline_tail_ic_top": b_ic,
        "ensemble_tail_ic_top": e_ic,
        "baseline_overlap": b_ov,
        "ensemble_overlap": e_ov,
        "need_delta_tail_ic": float(PHASE5_DELTA_TAIL_IC),
        "need_delta_overlap": float(PHASE5_DELTA_OVERLAP),
        "need_disp": float(PHASE5_SEED_DISP_MAX),
        "null_verdict": null.get("verdict"),
        "ceiling_sentence": PHASE5_CEILING_SENTENCE if ceiling else None,
        "record_ceiling": ceiling,
        "criterion": PHASE5_CRITERION,
    }


def load_manuel_score(paths: list[Path]) -> dict | None:
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        try:
            blob = pd.read_json(p) if p.suffix == ".json" else None
        except Exception:
            blob = None
        if p.suffix == ".json":
            import json

            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            # Reading A as a benchmark row if present
            ra = data.get("reading_a") or data.get("Reading A") or data.get("manuel_score_reading_a")
            if isinstance(ra, dict) and (
                "tail_ic_top" in ra or "full" in ra or "overlap" in ra
            ):
                return {"source": str(p), "row": ra}
        text_hit = p if p.suffix == ".md" else None
        if text_hit:
            return {"source": str(p), "row": {"note": "report exists; structured Reading A not parsed"}}
    return None
