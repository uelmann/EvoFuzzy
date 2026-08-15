"""ORACLE LADDER: perfect-foresight ceiling, IC-degraded oracles, model on the curve.

ANALYSIS ONLY. Identical 14d full-rebalance long construction for every point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    ALT_BPS,
    ANNUALIZATION,
    DEATH_CONVENTION,
    NAME_CAP,
    ORACLE_LADDER_CRITERION,
    ORACLE_LADDER_DECILE,
    ORACLE_LADDER_H,
    ORACLE_LADDER_LOG_BAND,
    ORACLE_LADDER_MIN_N,
    ORACLE_LADDER_MOM_DAYS,
    ORACLE_LADDER_SEEDS,
    ORACLE_LADDER_TARGETS,
    PHASE2_CYCLES,
)


def _utc_idx(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert("UTC").normalize()


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC").normalize()
    return t.tz_convert("UTC").normalize()


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < int(ORACLE_LADDER_MIN_N):
        return float("nan")
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _cagr(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return float("nan")
    years = len(eq) / float(ANNUALIZATION)
    last = float(eq.iloc[-1])
    if last <= 0:
        return float("nan")
    return float(last ** (1.0 / max(years, 1e-6)) - 1.0)


def _maxdd(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION)) if len(x) and x.std() > 0 else 0.0


def pack_equity(rets: pd.Series) -> dict:
    rets = rets.astype(float).fillna(0.0)
    rets.index = _utc_idx(rets.index)
    eq = (1.0 + rets).cumprod()
    cycles = {}
    for name, a, b in PHASE2_CYCLES:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = rets[(rets.index >= t0) & (rets.index <= t1)]
        if sl.empty:
            continue
        eq_s = (1.0 + sl.fillna(0.0)).cumprod()
        cycles[name] = {
            "n": int(len(sl)),
            "total": float(eq_s.iloc[-1] - 1.0),
            "cagr": _cagr(eq_s),
            "maxdd": _maxdd(eq_s),
            "sharpe": _sharpe(sl),
        }
    return {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()) if len(rets) else None,
        "end": str(rets.index.max().date()) if len(rets) else None,
        "total": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
        "cagr": _cagr(eq),
        "maxdd": _maxdd(eq),
        "sharpe": _sharpe(rets),
        "cycles": cycles,
        "daily_ret": rets,
        "equity": eq,
    }


def listed_mask(close: pd.DataFrame, dt, iid: int) -> bool:
    if dt not in close.index or int(iid) not in close.columns:
        return False
    v = close.at[dt, int(iid)]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > 0)


def last_close_map(close: pd.DataFrame) -> dict[int, pd.Timestamp]:
    out: dict[int, pd.Timestamp] = {}
    for iid in close.columns:
        col = close[iid].astype(float)
        ok = col[np.isfinite(col) & (col > 0)]
        if len(ok):
            out[int(iid)] = pd.Timestamp(ok.index.max()).tz_convert("UTC").normalize()
    return out


def formation_dates(dates: list, h: int) -> list[tuple]:
    """Non-overlapping (t, t+h) pairs stepping by h on the date index."""
    out = []
    i = 0
    n = len(dates)
    while i + int(h) < n:
        out.append((dates[i], dates[i + int(h)], i))
        i += int(h)
    return out


def period_excess(close: pd.DataFrame, btc_id: int, t, t_h, ids: list[int]) -> pd.Series:
    if t not in close.index or t_h not in close.index or int(btc_id) not in close.columns:
        return pd.Series(dtype=float)
    px0 = close.loc[t]
    px1 = close.loc[t_h]
    b0 = float(px0[int(btc_id)]) if int(btc_id) in px0.index else float("nan")
    b1 = float(px1[int(btc_id)]) if int(btc_id) in px1.index else float("nan")
    if not (np.isfinite(b0) and b0 > 0 and np.isfinite(b1) and b1 > 0):
        return pd.Series(dtype=float)
    rb = b1 / b0 - 1.0
    rows = {}
    for iid in ids:
        iid = int(iid)
        if iid == int(btc_id) or iid not in px0.index or iid not in px1.index:
            continue
        a = float(px0[iid]) if px0[iid] is not None else float("nan")
        b = float(px1[iid]) if px1[iid] is not None else float("nan")
        if np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0:
            rows[iid] = (b / a - 1.0) - rb
    return pd.Series(rows, dtype=float)


def momentum_90d(close: pd.DataFrame, btc_id: int, dates: list, pos: dict, t, lookback: int) -> pd.Series:
    t = _as_utc(t)
    i = pos.get(t)
    if i is None or int(i) < int(lookback):
        return pd.Series(dtype=float)
    t0 = dates[int(i) - int(lookback)]
    ids = [int(c) for c in close.columns if int(c) != int(btc_id)]
    return period_excess(close, btc_id, t0, t, ids)


def ffill_members(members: dict, dates: list) -> dict:
    """As-of PIT membership on the close calendar (listed-at-t, no future peek)."""
    filled: dict = {}
    last: list[int] = []
    m = {_as_utc(k): [int(i) for i in v] for k, v in members.items()}
    for d in dates:
        d = _as_utc(d)
        if d in m:
            last = m[d]
        filled[d] = list(last)
    return filled


def eligible(close: pd.DataFrame, members: dict, btc_id: int, t, t_h=None) -> list[int]:
    """Binance-listed names in the PIT universe at formation t. t_h unused (no lookahead)."""
    t = _as_utc(t)
    names = [int(s) for s in members.get(t, []) if int(s) != int(btc_id)]
    return [iid for iid in names if listed_mask(close, t, iid)]


def top_decile_weights(scores: pd.Series, cap: float = NAME_CAP, decile: int = ORACLE_LADDER_DECILE) -> pd.Series:
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 2:
        return pd.Series(dtype=float)
    k = max(1, n // int(decile))
    ids = [int(i) for i in s.nlargest(k).index.tolist()]
    if not ids:
        return pd.Series(dtype=float)
    w = min(1.0 / float(len(ids)), float(cap))
    return pd.Series({i: w for i in ids}, dtype=float)


def _residualize(eps: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove linear dependence on rank(x) so large-σ noise can reach IC≈0."""
    eps = np.asarray(eps, dtype=float)
    rx = _zscore(pd.Series(np.asarray(x, dtype=float)).rank().to_numpy())
    denom = float(rx @ rx)
    if denom < 1e-12:
        return _zscore(eps)
    eps = eps - (float(eps @ rx) / denom) * rx
    return _zscore(eps)


def calibrate_sigma(x: np.ndarray, eps: np.ndarray, target: float, n_iter: int = 42) -> tuple[float, float]:
    """σ such that Spearman(x + σ·ε, x) ≈ target. x is z-scored excess."""
    x = np.asarray(x, dtype=float)
    eps = _residualize(np.asarray(eps, dtype=float), x)

    def ic(sig: float) -> float:
        return _spearman(x + float(sig) * eps, x)

    ic0 = ic(0.0)
    if not np.isfinite(ic0):
        return float("nan"), float("nan")
    if abs(float(target)) < 1e-12:
        # independent residual; RankIC vs x ≈ 0
        return 1e6, ic(1e6)
    if ic0 <= float(target) + 1e-6:
        return 0.0, float(ic0)
    lo, hi = 0.0, 8.0
    while ic(hi) > float(target) and hi < 1e6:
        hi *= 2.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if ic(mid) > float(target):
            lo = mid
        else:
            hi = mid
    sig = 0.5 * (lo + hi)
    return float(sig), ic(sig)


def run_periodic_long(
    close: pd.DataFrame,
    members: dict,
    btc_id: int,
    score_at: dict,
    pairs: list[tuple],
    *,
    cost_bps: float = ALT_BPS,
    name_cap: float = NAME_CAP,
    label: str = "",
    weight_fn=None,
) -> dict:
    """Daily-marked long book. Default: EW top-decile. score_at[t] is scores at formation t."""
    close = close.copy()
    close.index = _utc_idx(close.index)
    close = close.sort_index()
    last_map = last_close_map(close)
    dates = list(close.index)
    date_set = set(dates)
    pos = {d: i for i, d in enumerate(dates)}
    alt_c = float(cost_bps) * 1e-4

    prev_w = pd.Series(dtype=float)
    daily = {}
    ics = []
    n_names = []
    to_list = []
    forced_events = []
    n_form = 0
    _wfn = weight_fn if weight_fn is not None else (lambda sc: top_decile_weights(sc, cap=name_cap))

    for t, t_h, _i in pairs:
        t, t_h = _as_utc(t), _as_utc(t_h)
        if t not in date_set or t_h not in date_set:
            continue
        sc = score_at.get(t)
        if sc is None or len(sc) == 0:
            w = pd.Series(dtype=float)
            ic = float("nan")
        else:
            sc = pd.Series(sc, dtype=float)
            w = _wfn(sc)
            if w is None:
                w = pd.Series(dtype=float)
            else:
                w = pd.Series(w, dtype=float)
            ex = period_excess(close, btc_id, t, t_h, [int(i) for i in sc.index])
            aligned = sc.align(ex, join="inner")
            ic = _spearman(aligned[0].to_numpy(), aligned[1].to_numpy())
        n_form += 1
        if np.isfinite(ic):
            ics.append(float(ic))
        n_names.append(int(len(w)))

        i0 = pos[t]
        i1 = pos[t_h]
        for j in range(i0, i1):
            dt = dates[j]
            nxt = dates[j + 1]
            drop = []
            for iid in list(w.index):
                ld = last_map.get(int(iid))
                if ld is None or nxt > ld:
                    drop.append(int(iid))
            if drop:
                w_drop = float(w.reindex(drop).fillna(0.0).sum())
                w = w.drop(labels=drop, errors="ignore")
                if w_drop > 0:
                    forced_events.append({"date": str(dt.date()), "ids": [int(x) for x in drop], "weight": w_drop})
            idx = w.index.union(prev_w.index)
            dw = (w.reindex(idx).fillna(0.0) - prev_w.reindex(idx).fillna(0.0)).abs().sum()
            to_list.append(0.5 * float(dw))
            cost = float(dw) * alt_c
            r = 0.0
            if len(w):
                p0 = close.loc[dt]
                p1 = close.loc[nxt]
                for iid, wi in w.items():
                    if iid not in p0.index or iid not in p1.index:
                        continue
                    a = float(p0[iid]) if p0[iid] is not None else float("nan")
                    b = float(p1[iid]) if p1[iid] is not None else float("nan")
                    if np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0:
                        r += float(wi) * (b / a - 1.0)
            daily[nxt] = float(daily.get(nxt, 0.0)) + float(r - cost)
            prev_w = w.copy()

    if not daily:
        return {"error": "empty book", "label": label}
    rets = pd.Series(daily).sort_index()
    packed = pack_equity(rets)
    packed.update(
        {
            "label": label,
            "rankic": float(np.mean(ics)) if ics else float("nan"),
            "rankic_n": int(len(ics)),
            "n_formations": int(n_form),
            "avg_n_names": float(np.mean(n_names)) if n_names else float("nan"),
            "ann_turnover": float(np.mean(to_list) * ANNUALIZATION) if to_list else float("nan"),
            "forced_n": int(len(forced_events)),
            "forced_exits": {
                "n_events": int(len(forced_events)),
                "n_ids": int(len({i for e in forced_events for i in e["ids"]})),
                "weight_sum": float(sum(e["weight"] for e in forced_events)),
            },
            "cost_bps": float(cost_bps),
        }
    )
    return packed


def build_oracle_scores(close, members, btc_id, pairs) -> tuple[dict, list[float]]:
    scores = {}
    ics = []
    for t, t_h, _ in pairs:
        t, t_h = _as_utc(t), _as_utc(t_h)
        ids = eligible(close, members, btc_id, t, t_h)
        ex = period_excess(close, btc_id, t, t_h, ids)
        scores[t] = ex
        ic = _spearman(ex.to_numpy(), ex.to_numpy())
        if np.isfinite(ic):
            ics.append(float(ic))
    return scores, ics


def build_noisy_scores(
    close,
    members,
    btc_id,
    pairs,
    target: float,
    seed: int,
) -> tuple[dict, list[float], list[float]]:
    rng = np.random.default_rng(int(seed))
    scores = {}
    ics = []
    sigmas = []
    for t, t_h, _ in pairs:
        t, t_h = _as_utc(t), _as_utc(t_h)
        ids = eligible(close, members, btc_id, t, t_h)
        ex = period_excess(close, btc_id, t, t_h, ids)
        if len(ex) < int(ORACLE_LADDER_MIN_N):
            scores[t] = pd.Series(dtype=float)
            continue
        x = _zscore(ex.to_numpy())
        raw = rng.standard_normal(len(x))
        sig, ic_hit = calibrate_sigma(x, raw, float(target))
        # calibrate_sigma residualizes a copy; apply the same mix for the stored score
        eps = _residualize(raw, x)
        sc = pd.Series(x + float(sig) * eps, index=ex.index, dtype=float)
        scores[t] = sc
        ic = _spearman(sc.to_numpy(), ex.to_numpy())
        if np.isfinite(ic):
            ics.append(float(ic))
        if np.isfinite(sig):
            sigmas.append(float(sig))
    return scores, ics, sigmas


def build_spread_scores(close, members, btc_id, pairs, swide: pd.DataFrame) -> tuple[dict, list[float]]:
    scores = {}
    ics = []
    swide = swide.copy()
    swide.index = _utc_idx(swide.index)
    for t, t_h, _ in pairs:
        t, t_h = _as_utc(t), _as_utc(t_h)
        ids = eligible(close, members, btc_id, t, t_h)
        if t not in swide.index:
            scores[t] = pd.Series(dtype=float)
            continue
        row = swide.loc[t]
        sc = {}
        for iid in ids:
            if iid in row.index:
                v = float(row[iid])
                if np.isfinite(v):
                    sc[int(iid)] = v
        s = pd.Series(sc, dtype=float)
        scores[t] = s
        ex = period_excess(close, btc_id, t, t_h, list(s.index))
        ic = _spearman(s.reindex(ex.index).to_numpy(), ex.to_numpy())
        if np.isfinite(ic):
            ics.append(float(ic))
    return scores, ics


def build_mom_scores(close, members, btc_id, pairs, dates, pos, lookback: int) -> tuple[dict, list[float]]:
    scores = {}
    ics = []
    for t, t_h, _ in pairs:
        t, t_h = _as_utc(t), _as_utc(t_h)
        ids = eligible(close, members, btc_id, t, t_h)
        mom = momentum_90d(close, btc_id, dates, pos, t, lookback)
        mom = mom.reindex(ids).dropna()
        scores[t] = mom
        ex = period_excess(close, btc_id, t, t_h, list(mom.index))
        ic = _spearman(mom.reindex(ex.index).to_numpy(), ex.to_numpy())
        if np.isfinite(ic):
            ics.append(float(ic))
    return scores, ics


def summarize_seeds(runs: list[dict]) -> dict:
    def _arr(key):
        xs = [float(r[key]) for r in runs if r.get(key) is not None and np.isfinite(float(r[key]))]
        return np.asarray(xs, dtype=float)

    out = {"n_seeds": int(len(runs))}
    for key in ("rankic", "total", "cagr", "maxdd", "sharpe"):
        a = _arr(key)
        if len(a) == 0:
            out[key] = float("nan")
            out[f"{key}_lo"] = float("nan")
            out[f"{key}_hi"] = float("nan")
            continue
        out[key] = float(np.mean(a))
        out[f"{key}_lo"] = float(np.min(a))
        out[f"{key}_hi"] = float(np.max(a))
    out["rankic_n"] = int(runs[0].get("rankic_n") or 0) if runs else 0
    out["n_days"] = runs[0].get("n_days") if runs else None
    out["start"] = runs[0].get("start") if runs else None
    out["end"] = runs[0].get("end") if runs else None
    return out


def interpolate_cagr(xs: list[float], ys: list[float], x0: float) -> float:
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pts) < 2 or not np.isfinite(x0):
        return float("nan")
    pts.sort(key=lambda z: z[0])
    xp = np.asarray([p[0] for p in pts], dtype=float)
    yp = np.asarray([p[1] for p in pts], dtype=float)
    return float(np.interp(float(x0), xp, yp))


def efficiency_verdict(model_cagr: float, curve_cagr: float, oracle_cagr: float, model_ic: float) -> dict:
    band = float(ORACLE_LADDER_LOG_BAND)
    ratio = float("nan")
    log_gap = float("nan")
    label = "UNAVAILABLE"
    on_curve = False
    below = False
    above = False
    if np.isfinite(model_cagr) and np.isfinite(curve_cagr) and curve_cagr > 0 and model_cagr > 0:
        ratio = float(model_cagr / curve_cagr)
        log_gap = float(np.log(model_cagr) - np.log(curve_cagr))
        lo, hi = 1.0 / (1.0 + band), 1.0 + band
        # ±20% in log terms: |ln m - ln c| <= ln(1.20)
        lim = float(np.log(1.0 + band))
        if abs(log_gap) <= lim + 1e-12:
            label = "ON-CURVE"
            on_curve = True
        elif log_gap < 0:
            label = "BELOW-CURVE"
            below = True
        else:
            label = "ABOVE-CURVE"
            above = True
        _ = (lo, hi)
    elif np.isfinite(model_cagr) and np.isfinite(curve_cagr) and model_cagr < curve_cagr:
        label = "BELOW-CURVE"
        below = True
    capture = (
        float(model_cagr / oracle_cagr)
        if np.isfinite(model_cagr) and np.isfinite(oracle_cagr) and oracle_cagr != 0
        else float("nan")
    )
    if on_curve or above:
        info = "consistent with"
        constraint = "INFORMATION"
    elif below:
        info = "below"
        constraint = "TRANSLATION"
    else:
        info = "below"
        constraint = "UNAVAILABLE"
    return {
        "criterion": ORACLE_LADDER_CRITERION,
        "death_convention": DEATH_CONVENTION,
        "label": label,
        "on_curve": bool(on_curve),
        "below_curve": bool(below),
        "above_curve": bool(above),
        "binding_constraint": constraint,
        "consistent_word": info,
        "model_cagr": float(model_cagr) if np.isfinite(model_cagr) else float("nan"),
        "curve_cagr": float(curve_cagr) if np.isfinite(curve_cagr) else float("nan"),
        "oracle_cagr": float(oracle_cagr) if np.isfinite(oracle_cagr) else float("nan"),
        "model_rankic": float(model_ic) if np.isfinite(model_ic) else float("nan"),
        "cagr_ratio_vs_curve": ratio,
        "log_gap": log_gap,
        "band": band,
        "capture_of_oracle_cagr": capture,
        "need_lo": (curve_cagr / (1.0 + band)) if np.isfinite(curve_cagr) else float("nan"),
        "need_hi": (curve_cagr * (1.0 + band)) if np.isfinite(curve_cagr) else float("nan"),
    }
