"""ORACLE LADDER 2: tail-blindness vs translation slack. ANALYSIS ONLY."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    ORACLE_LADDER2_CRITERION,
    ORACLE_LADDER2_IC_EQ,
    ORACLE_LADDER2_MAXDD_SLACK,
    ORACLE_LADDER2_MONSTER_K,
    ORACLE_LADDER2_PP_MIN,
    ORACLE_LADDER2_V1_CAP,
    ORACLE_LADDER2_V2_CAP,
    ORACLE_LADDER2_V2_N,
    ORACLE_LADDER2_V3_CAP,
    ORACLE_LADDER2_V3_Q,
    ORACLE_LADDER_DECILE,
    ORACLE_LADDER_MIN_N,
    PHASE2_CYCLES,
)
from btcb.oracle_ladder import (
    _as_utc,
    _spearman,
    eligible,
    period_excess,
)
def interp_overlap_cagr(xs: list[float], ys: list[float], x0: float) -> float:
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 2 or not np.isfinite(x0):
        return float("nan")
    pts.sort(key=lambda z: z[0])
    xp = np.asarray([p[0] for p in pts], dtype=float)
    yp = np.asarray([p[1] for p in pts], dtype=float)
    x = float(x0)
    if x <= xp[0]:
        dx = xp[1] - xp[0]
        if abs(dx) < 1e-12:
            return float(yp[0])
        return float(yp[0] + (yp[1] - yp[0]) * (x - xp[0]) / dx)
    if x >= xp[-1]:
        dx = xp[-1] - xp[-2]
        if abs(dx) < 1e-12:
            return float(yp[-1])
        return float(yp[-1] + (yp[-1] - yp[-2]) * (x - xp[-1]) / dx)
    return float(np.interp(x, xp, yp))


def _decile_ids(scores: pd.Series, decile: int = ORACLE_LADDER_DECILE) -> list[int]:
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 2:
        return []
    k = max(1, n // int(decile))
    return [int(i) for i in s.nlargest(k).index.tolist()]


def weights_v1(scores: pd.Series, cap: float = ORACLE_LADDER2_V1_CAP, decile: int = ORACLE_LADDER_DECILE) -> pd.Series:
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    ids = _decile_ids(s, decile=decile)
    if not ids:
        return pd.Series(dtype=float)
    top = s.reindex(ids).dropna()
    pct = top.rank(method="average", pct=True)
    raw = pct / float(pct.sum())
    return raw.clip(upper=float(cap))


def weights_v2(scores: pd.Series, n: int = ORACLE_LADDER2_V2_N, cap: float = ORACLE_LADDER2_V2_CAP) -> pd.Series:
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    top = s.nlargest(min(int(n), len(s)))
    k = len(top)
    if k == 0:
        return pd.Series(dtype=float)
    w = min(1.0 / float(k), float(cap))
    return pd.Series({int(i): w for i in top.index}, dtype=float)


def weights_v3(
    scores: pd.Series,
    q: float = ORACLE_LADDER2_V3_Q,
    cap: float = ORACLE_LADDER2_V3_CAP,
) -> pd.Series:
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    thr = float(s.quantile(float(q)))
    top = s[s > thr]
    if len(top) == 0:
        return pd.Series(dtype=float)
    w = min(1.0 / float(len(top)), float(cap))
    return pd.Series({int(i): w for i in top.index}, dtype=float)


def scores_to_twin(score_at: dict) -> pd.DataFrame:
    rows = []
    for t, s in score_at.items():
        if s is None or len(s) == 0:
            continue
        ser = pd.Series(s, dtype=float)
        for iid, v in ser.items():
            if np.isfinite(float(v)):
                rows.append({"date": _as_utc(t), "id": int(iid), "spread": float(v), "fold_id": 0})
    if not rows:
        return pd.DataFrame(columns=["date", "id", "spread", "fold_id"])
    return pd.DataFrame(rows)


def scores_to_daily_twin(score_at: dict, close_index) -> pd.DataFrame:
    """As-of ffill of formation scores onto the daily close calendar (no lookahead)."""
    twin = scores_to_twin(score_at)
    if twin.empty:
        return twin
    sw = twin.pivot_table(index="date", columns="id", values="spread", aggfunc="last").sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(close_index, utc=True)).tz_convert("UTC").normalize()
    sw.index = pd.DatetimeIndex(pd.to_datetime(sw.index, utc=True)).tz_convert("UTC").normalize()
    sw = sw.reindex(idx).ffill()
    out = sw.stack(dropna=True).rename("spread").reset_index()
    out.columns = ["date", "id", "spread"]
    out["id"] = out["id"].astype(int)
    out["fold_id"] = 0
    return out


def _half_ic(score: pd.Series, excess: pd.Series, half: str) -> float:
    s, e = score.align(excess, join="inner")
    s = pd.Series(s, dtype=float)
    e = pd.Series(e, dtype=float)
    m = np.isfinite(s.to_numpy()) & np.isfinite(e.to_numpy())
    s, e = s[m], e[m]
    n = len(s)
    if n < 2 * int(ORACLE_LADDER_MIN_N):
        return float("nan")
    order = s.rank(ascending=False, method="first")
    mid = n / 2.0
    mask = order <= mid if half == "top" else order > mid
    if int(mask.sum()) < int(ORACLE_LADDER_MIN_N):
        return float("nan")
    return _spearman(s[mask].to_numpy(), e[mask].to_numpy())


def _cycle_mean(dates: list, values: list) -> dict:
    out = {}
    idx = pd.DatetimeIndex([_as_utc(d) for d in dates])
    ser = pd.Series(values, index=idx, dtype=float)
    for name, a, b in PHASE2_CYCLES:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = ser[(ser.index >= t0) & (ser.index <= t1)].dropna()
        out[name] = {
            "n": int(len(sl)),
            "mean": float(sl.mean()) if len(sl) else float("nan"),
        }
    return out


def formation_diagnostics(close, members, btc_id, score_at, pairs, label: str = "") -> dict:
    overlaps, monsters, top_ics, bot_ics, full_ics = [], [], [], [], []
    dates = []
    n_sig, n_real = [], []
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
        sig = set(_decile_ids(sc))
        k_real = max(1, len(ex_a) // int(ORACLE_LADDER_DECILE))
        real = set(int(i) for i in ex_a.nlargest(k_real).index.tolist())
        k_m = min(int(ORACLE_LADDER2_MONSTER_K), len(ex_a))
        monsters_set = set(int(i) for i in ex_a.nlargest(k_m).index.tolist())
        ov = float(len(sig & real) / max(len(sig), 1))
        mon = float(len(sig & monsters_set) / max(len(monsters_set), 1))
        overlaps.append(ov)
        monsters.append(mon)
        top_ics.append(_half_ic(sc, ex_a, "top"))
        bot_ics.append(_half_ic(sc, ex_a, "bottom"))
        full_ics.append(_spearman(sc.to_numpy(), ex_a.to_numpy()))
        dates.append(t)
        n_sig.append(len(sig))
        n_real.append(len(real))

    def _mu(xs):
        a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
        return float(np.mean(a)) if len(a) else float("nan")

    return {
        "label": label,
        "n_formations": int(len(overlaps)),
        "overlap": _mu(overlaps),
        "monster": _mu(monsters),
        "tail_ic_top": _mu(top_ics),
        "tail_ic_bot": _mu(bot_ics),
        "rankic": _mu(full_ics),
        "avg_n_sig": _mu(n_sig),
        "avg_n_real": _mu(n_real),
        "overlap_cycles": _cycle_mean(dates, overlaps),
        "monster_cycles": _cycle_mean(dates, monsters),
        "tail_ic_top_cycles": _cycle_mean(dates, top_ics),
        "tail_ic_bot_cycles": _cycle_mean(dates, bot_ics),
        "bottom_minus_top": (
            _mu(bot_ics) - _mu(top_ics)
            if np.isfinite(_mu(bot_ics)) and np.isfinite(_mu(top_ics))
            else float("nan")
        ),
    }


def summarize_diag_seeds(runs: list[dict]) -> dict:
    def _arr(key):
        xs = [float(r[key]) for r in runs if r.get(key) is not None and np.isfinite(float(r[key]))]
        return np.asarray(xs, dtype=float)

    out = {"n_seeds": int(len(runs)), "label": runs[0].get("label") if runs else ""}
    for key in ("overlap", "monster", "tail_ic_top", "tail_ic_bot", "rankic", "bottom_minus_top"):
        a = _arr(key)
        if len(a) == 0:
            out[key] = float("nan")
            out[f"{key}_lo"] = float("nan")
            out[f"{key}_hi"] = float("nan")
            continue
        out[key] = float(np.mean(a))
        out[f"{key}_lo"] = float(np.min(a))
        out[f"{key}_hi"] = float(np.max(a))
    # cycle means averaged across seeds
    cyc = {}
    for name, _a, _b in PHASE2_CYCLES:
        xs = []
        for r in runs:
            blob = (r.get("overlap_cycles") or {}).get(name) or {}
            v = blob.get("mean")
            if v is not None and np.isfinite(float(v)):
                xs.append(float(v))
        cyc[name] = {"n": (runs[0].get("overlap_cycles") or {}).get(name, {}).get("n") if runs else 0, "mean": float(np.mean(xs)) if xs else float("nan")}
    out["overlap_cycles"] = cyc
    out["n_formations"] = runs[0].get("n_formations") if runs else 0
    return out


def comparable_maxdd(variant_mdd: float, base_mdd: float, slack: float = ORACLE_LADDER2_MAXDD_SLACK) -> bool:
    if not (np.isfinite(variant_mdd) and np.isfinite(base_mdd)):
        return False
    return float(variant_mdd) >= float(base_mdd) - float(slack)


def decompose_gap(
    *,
    model_cagr: float,
    model_maxdd: float,
    model_overlap: float,
    ladder_cagr: float,
    ladder_overlap: float,
    ladder16_cagr: float,
    ladder16_overlap: float,
    oracle_cagr: float,
    oracle_overlap: float,
    ladder_prod_cagr: float,
    variants: dict[str, dict],
) -> dict:
    gap = (
        float(ladder_cagr) - float(model_cagr)
        if np.isfinite(ladder_cagr) and np.isfinite(model_cagr)
        else float("nan")
    )
    c_star = interp_overlap_cagr(
        [ladder_overlap, ladder16_overlap, oracle_overlap],
        [ladder_cagr, ladder16_cagr, oracle_cagr],
        float(model_overlap),
    )
    tail_pp = (
        float(ladder_cagr) - float(c_star)
        if np.isfinite(ladder_cagr) and np.isfinite(c_star)
        else float("nan")
    )
    best_name, best_cagr, best_mdd = None, float("-inf"), float("nan")
    qualified = []
    for name, blob in variants.items():
        c = float(blob.get("cagr") if blob.get("cagr") is not None else float("nan"))
        m = float(blob.get("maxdd") if blob.get("maxdd") is not None else float("nan"))
        if not np.isfinite(c):
            continue
        if c > best_cagr:
            best_name, best_cagr, best_mdd = name, c, m
        if comparable_maxdd(m, float(model_maxdd)) and np.isfinite(c):
            qualified.append((name, c, m))
    if qualified:
        qname, qcagr, qmdd = max(qualified, key=lambda z: z[1])
    else:
        qname, qcagr, qmdd = None, float("nan"), float("nan")
    v_delta = (
        float(best_cagr) - float(model_cagr)
        if np.isfinite(best_cagr) and np.isfinite(model_cagr) and best_cagr != float("-inf")
        else float("nan")
    )
    prod_delta = (
        float(ladder_prod_cagr) - float(ladder_cagr)
        if np.isfinite(ladder_prod_cagr) and np.isfinite(ladder_cagr)
        else float("nan")
    )
    construction_pp = (
        float(v_delta) + float(prod_delta)
        if np.isfinite(v_delta) and np.isfinite(prod_delta)
        else float("nan")
    )
    unexplained = (
        float(gap) - float(tail_pp) - float(construction_pp)
        if np.isfinite(gap) and np.isfinite(tail_pp) and np.isfinite(construction_pp)
        else float("nan")
    )
    q_delta = (
        float(qcagr) - float(model_cagr)
        if np.isfinite(qcagr) and np.isfinite(model_cagr)
        else float("nan")
    )
    translation_priority = bool(np.isfinite(q_delta) and q_delta >= float(ORACLE_LADDER2_PP_MIN))
    priority = "TRANSLATION" if translation_priority else "TAIL-INFORMATION"
    return {
        "criterion": ORACLE_LADDER2_CRITERION,
        "death_convention": DEATH_CONVENTION,
        "equal_ic_target": float(ORACLE_LADDER2_IC_EQ),
        "gap_pp": gap,
        "tail_pp": tail_pp,
        "construction_pp": construction_pp,
        "unexplained_pp": unexplained,
        "cagr_at_model_overlap": c_star,
        "model_cagr": float(model_cagr) if np.isfinite(model_cagr) else float("nan"),
        "model_overlap": float(model_overlap) if np.isfinite(model_overlap) else float("nan"),
        "ladder_cagr": float(ladder_cagr) if np.isfinite(ladder_cagr) else float("nan"),
        "ladder_overlap": float(ladder_overlap) if np.isfinite(ladder_overlap) else float("nan"),
        "variant_best": best_name,
        "variant_best_cagr": float(best_cagr) if best_cagr != float("-inf") else float("nan"),
        "variant_best_maxdd": best_mdd,
        "variant_best_delta_pp": v_delta,
        "qualified_best": qname,
        "qualified_best_cagr": qcagr,
        "qualified_best_delta_pp": q_delta,
        "prod_delta_pp": prod_delta,
        "pp_min": float(ORACLE_LADDER2_PP_MIN),
        "maxdd_slack": float(ORACLE_LADDER2_MAXDD_SLACK),
        "translation_priority": translation_priority,
        "priority": priority,
        "nothing_adopted": True,
    }
