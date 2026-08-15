"""Unconstrained D10−D1 academic factor on CMC + implementation-tax waterfall.

ANALYSIS ONLY. Signals frozen (2.c cache). No book redesign.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from baseline.evaluate import newey_west_t
from btcb.binance_replay import name_cost
from btcb.constants import (
    ACADEMIC_FACTOR_NW_LAG,
    ACADEMIC_FACTOR_NW_T_MIN,
    ACADEMIC_FACTOR_SHARPE_MIN,
    ANNUALIZATION,
    LS_TRAIL_DAYS,
    PHASE2_CYCLES,
    PHASE3C_REF_HYBRID_SHARPE,
)
from btcb.spread_ls import _as_utc, _sharpe, _utc_idx, trail_slice


def _utc_key(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()


def pit_members(pit: pd.DataFrame, btc_id: int) -> dict[pd.Timestamp, list[int]]:
    df = pit.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    out: dict[pd.Timestamp, list[int]] = {}
    for d, v in df.groupby("date")["id"]:
        ids = [int(x) for x in v if int(x) != int(btc_id)]
        out[_utc_key(d)] = ids
    return out


def spread_wide(preds: pd.DataFrame) -> pd.DataFrame:
    pr = preds.copy()
    pr["date"] = pd.to_datetime(pr["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pr["id"] = pr["id"].astype(int)
    pr = pr.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
    sw = pr.pivot(index="date", columns="id", values="spread").sort_index()
    sw.index = pd.to_datetime(sw.index, utc=True).tz_convert("UTC").normalize()
    return sw


def last_close_map(panel: pd.DataFrame) -> dict[int, pd.Timestamp]:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    return {int(i): _utc_key(d) for i, d in df.groupby("id")["date"].max().items()}


def _alive(ids: list[int], nxt: pd.Timestamp, last_map: dict[int, pd.Timestamp]) -> list[int]:
    out = []
    for iid in ids:
        ld = last_map.get(int(iid))
        if ld is not None and ld >= nxt:
            out.append(int(iid))
    return out


def _deciles(
    dt: pd.Timestamp,
    names: list[int],
    swide: pd.DataFrame,
    shortable: dict | None,
) -> tuple[list[int], list[int], list[int], int]:
    if dt not in swide.index:
        return [], [], list(names), 0
    row = swide.loc[dt]
    scored = []
    for iid in names:
        if iid not in row.index:
            continue
        sc = float(row[iid])
        if np.isfinite(sc):
            scored.append((sc, int(iid)))
    scored.sort(reverse=True)
    ordered = [i for _, i in scored]
    n = len(ordered)
    k = max(1, n // 10) if n >= 2 else 0
    if k <= 0:
        return [], [], ordered, 0
    longs = ordered[:k]
    shorts = ordered[-k:]
    if shortable is not None:
        sh_today = shortable.get(_utc_key(dt), set())
        shorts = [i for i in shorts if i in sh_today]
    return longs, shorts, ordered, k


def _leg_w(ids: list[int], sign: float, k: int) -> dict[int, float]:
    if not ids or k <= 0:
        return {}
    w = float(sign) / float(k)
    return {int(i): w for i in ids}


def _uni_w(ids: list[int]) -> dict[int, float]:
    n = len(ids)
    if n <= 0:
        return {}
    w = 1.0 / float(n)
    return {int(i): w for i in ids}


def _cost_dw(old: dict[int, float], new: dict[int, float], long_c: float, short_c: float) -> float:
    keys = set(old) | set(new)
    cost = 0.0
    for iid in keys:
        cost += _tier_cost(float(old.get(iid, 0.0)), float(new.get(iid, 0.0)), long_c, short_c)
    return cost


def _tier_cost(old_w: float, new_w: float, long_c: float, short_c: float) -> float:
    m = float(new_w) - float(old_w)
    if m == 0.0:
        return 0.0
    if old_w >= 0 and new_w >= 0:
        return abs(m) * long_c
    if old_w <= 0 and new_w <= 0:
        return abs(m) * short_c
    return abs(old_w) * (long_c if old_w > 0 else short_c) + abs(new_w) * (
        long_c if new_w > 0 else short_c
    )


def _dot(weights: dict[int, float], rrow: pd.Series) -> float:
    acc = 0.0
    if rrow is None or len(weights) == 0:
        return 0.0
    for iid, w in weights.items():
        if w == 0.0:
            continue
        if iid not in rrow.index:
            continue
        r = float(rrow[iid])
        if np.isfinite(r):
            acc += float(w) * r
    return acc


def _ew_ret(ids: list[int], rrow: pd.Series) -> float:
    xs = []
    for iid in ids:
        if iid not in rrow.index:
            continue
        r = float(rrow[iid])
        if np.isfinite(r):
            xs.append(r)
    return float(np.mean(xs)) if xs else 0.0


def factor_metrics(rets: pd.Series, *, nw_lag: int = ACADEMIC_FACTOR_NW_LAG) -> dict:
    x = rets.dropna().astype(float)
    x.index = _utc_idx(x.index)
    eq = (1.0 + x.fillna(0.0)).cumprod()
    mu = float(x.mean()) if len(x) else float("nan")
    sd = float(x.std()) if len(x) > 1 else float("nan")
    cycles = {}
    for name, a, b in PHASE2_CYCLES:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = x[(x.index >= t0) & (x.index <= t1)]
        if sl.empty:
            continue
        cycles[name] = {
            "n": int(len(sl)),
            "sharpe": _sharpe(sl),
            "ann_mean": float(sl.mean() * ANNUALIZATION) if len(sl) else float("nan"),
            "ann_vol": float(sl.std() * np.sqrt(ANNUALIZATION)) if len(sl) > 1 else float("nan"),
            "nw_t": float(newey_west_t(sl.to_numpy(), lag=int(nw_lag))),
        }
    trail = trail_slice(x, LS_TRAIL_DAYS)
    return {
        "n_days": int(len(x)),
        "start": str(x.index.min().date()) if len(x) else None,
        "end": str(x.index.max().date()) if len(x) else None,
        "sharpe": _sharpe(x),
        "sharpe_trail18m": _sharpe(trail),
        "ann_mean": float(mu * ANNUALIZATION) if np.isfinite(mu) else float("nan"),
        "ann_vol": float(sd * np.sqrt(ANNUALIZATION)) if np.isfinite(sd) and sd > 0 else float("nan"),
        "nw_t": float(newey_west_t(x.to_numpy(), lag=int(nw_lag))) if len(x) else float("nan"),
        "nw_t_trail18m": float(newey_west_t(trail.to_numpy(), lag=int(nw_lag))) if len(trail) else float("nan"),
        "total": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
        "maxdd": float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan"),
        "cycles": cycles,
        "daily_ret": x,
        "equity": eq,
    }


def run_academic_factor(
    close: pd.DataFrame,
    members: dict,
    swide: pd.DataFrame,
    last_map: dict[int, pd.Timestamp],
    btc_id: int,
    *,
    h: int = 1,
    shortable: dict | None = None,
    long_bps: float = 0.0,
    short_bps: float = 0.0,
    label: str = "",
) -> dict:
    """EW D10−D1. h=1 daily refresh; h=14 Jegadeesh-Titman overlapping.

    Leg weights are ±1/k with k = n_scored // 10 of the formation-day universe.
    Shortability (if provided) drops unlistable D1 names; remaining short weights
    are NOT renormalized. Costs apply to combined overlay weights.
    """
    close = close.sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    dates = [d for d in close.index if d in swide.index and d in members]
    if len(dates) < max(int(h), 2) + 2:
        return {"error": "not enough OOS dates", "label": label}

    long_c = float(long_bps) * 1e-4
    short_c = float(short_bps) * 1e-4
    cohorts: deque = deque()
    prev_w: dict[int, float] = {}
    recs = []

    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        names = [int(s) for s in members.get(dt, []) if s in close.columns and int(s) != int(btc_id)]
        longs, shorts, uni, k = _deciles(dt, names, swide, shortable)
        longs = _alive(longs, nxt, last_map)
        shorts = _alive(shorts, nxt, last_map)
        uni = _alive(uni, nxt, last_map)
        cohorts.append((longs, shorts, uni, k))
        while len(cohorts) > int(h):
            cohorts.popleft()

        active = [c for c in cohorts if (c[0] or c[1]) and int(c[3]) > 0]
        n_coh = max(1, len(active))
        w: dict[int, float] = {}
        w_long: dict[int, float] = {}
        w_short: dict[int, float] = {}
        w_uni: dict[int, float] = {}
        r_l_sum = r_s_sum = r_u_sum = 0.0
        n_l_acc = n_s_acc = 0
        rrow = None
        if nxt in close.index and dt in close.index:
            rrow = (close.loc[nxt] / close.loc[dt] - 1.0).replace([np.inf, -np.inf], np.nan)

        for lg, sh, un, kk in active:
            scale = 1.0 / float(n_coh)
            lw = _leg_w(lg, 1.0, kk)
            sw = _leg_w(sh, -1.0, kk)
            uw = _uni_w(un)
            for iid, wi in lw.items():
                w[iid] = w.get(iid, 0.0) + scale * wi
                w_long[iid] = w_long.get(iid, 0.0) + scale * wi
            for iid, wi in sw.items():
                w[iid] = w.get(iid, 0.0) + scale * wi
                w_short[iid] = w_short.get(iid, 0.0) + scale * wi
            for iid, wi in uw.items():
                w_uni[iid] = w_uni.get(iid, 0.0) + scale * wi
            if rrow is not None:
                r_l_sum += _ew_ret(lg, rrow)
                r_s_sum += _ew_ret(sh, rrow)
                r_u_sum += _ew_ret(un, rrow)
            n_l_acc += len(lg)
            n_s_acc += len(sh)

        r_long = r_l_sum / float(n_coh)
        r_short = r_s_sum / float(n_coh)
        r_uni = r_u_sum / float(n_coh)
        gross = r_long - r_short
        if rrow is not None:
            # weight-form must match EW-of-cohorts when all names print;
            # keep EW-of-cohorts as the academic gross (reweight missing prints).
            pass
        cost = _cost_dw(prev_w, w, long_c, short_c) if (long_c or short_c) else 0.0
        net = gross - cost
        to = 0.5 * float(sum(abs(w.get(i, 0.0) - prev_w.get(i, 0.0)) for i in set(w) | set(prev_w)))
        recs.append(
            {
                "nxt": nxt,
                "gross": float(gross),
                "net": float(net),
                "cost": float(cost),
                "r_long": float(r_long),
                "r_short": float(r_short),
                "r_uni": float(r_uni),
                "lmU": float(r_long - r_uni),
                "umS": float(r_uni - r_short),
                "n_long": float(n_l_acc / float(n_coh)),
                "n_short": float(n_s_acc / float(n_coh)),
                "n_cohorts": int(n_coh),
                "long_gross_w": float(sum(max(v, 0.0) for v in w.values())),
                "short_gross_w": float(sum(max(-v, 0.0) for v in w.values())),
                "turnover": float(to),
            }
        )
        prev_w = w
        if i % 250 == 0:
            print(
                f"[HB] factor {label} i={i}/{len(dates)-1} dt={dt.date()} "
                f"nL={recs[-1]['n_long']:.1f} nS={recs[-1]['n_short']:.1f} "
                f"gross={gross:.5f} net={net:.5f}",
                flush=True,
            )

    daily = pd.DataFrame(recs).set_index("nxt").sort_index()
    daily.index = _utc_idx(daily.index)
    packed = factor_metrics(daily["net"])
    packed["gross"] = factor_metrics(daily["gross"])
    packed["long"] = factor_metrics(daily["r_long"])
    packed["short"] = factor_metrics(daily["r_short"])
    packed["universe"] = factor_metrics(daily["r_uni"])
    packed["lmU"] = factor_metrics(daily["lmU"])
    packed["umS"] = factor_metrics(daily["umS"])
    packed["avg_n_long"] = float(daily["n_long"].mean()) if len(daily) else float("nan")
    packed["avg_n_short"] = float(daily["n_short"].mean()) if len(daily) else float("nan")
    packed["avg_long_gross"] = float(daily["long_gross_w"].mean()) if len(daily) else float("nan")
    packed["avg_short_gross"] = float(daily["short_gross_w"].mean()) if len(daily) else float("nan")
    packed["ann_turnover"] = float(daily["turnover"].mean() * ANNUALIZATION) if len(daily) else float("nan")
    packed["ann_cost_drag"] = float(daily["cost"].mean() * ANNUALIZATION) if len(daily) else float("nan")
    packed["h"] = int(h)
    packed["shortable"] = bool(shortable is not None)
    packed["long_bps"] = float(long_bps)
    packed["short_bps"] = float(short_bps)
    packed["label"] = str(label)
    packed["daily"] = daily
    # paper ceiling uses GROSS series for the 0-cost case; net==gross when bps=0
    if float(long_bps) == 0.0 and float(short_bps) == 0.0:
        packed["sharpe"] = packed["gross"]["sharpe"]
        packed["sharpe_trail18m"] = packed["gross"]["sharpe_trail18m"]
        packed["ann_mean"] = packed["gross"]["ann_mean"]
        packed["ann_vol"] = packed["gross"]["ann_vol"]
        packed["nw_t"] = packed["gross"]["nw_t"]
        packed["daily_ret"] = packed["gross"]["daily_ret"]
        packed["equity"] = packed["gross"]["equity"]
        packed["cycles"] = packed["gross"]["cycles"]
        packed["total"] = packed["gross"]["total"]
        packed["maxdd"] = packed["gross"]["maxdd"]
    return packed


def paper_alpha_verdict(jt_gross: dict) -> dict:
    sh = float(jt_gross.get("sharpe") if np.isfinite(float(jt_gross.get("sharpe") or np.nan)) else float("nan"))
    t = float(jt_gross.get("nw_t") if np.isfinite(float(jt_gross.get("nw_t") or np.nan)) else float("nan"))
    ok_s = bool(np.isfinite(sh) and sh >= float(ACADEMIC_FACTOR_SHARPE_MIN))
    ok_t = bool(np.isfinite(t) and t >= float(ACADEMIC_FACTOR_NW_T_MIN))
    exists = bool(ok_s and ok_t)
    return {
        "exists": exists,
        "label": "PAPER ALPHA EXISTS" if exists else "PAPER ALPHA DOES NOT EXIST",
        "sharpe": sh,
        "nw_t": t,
        "need_sharpe": float(ACADEMIC_FACTOR_SHARPE_MIN),
        "need_nw_t": float(ACADEMIC_FACTOR_NW_T_MIN),
        "pass_sharpe": ok_s,
        "pass_nw_t": ok_t,
        "n_days": jt_gross.get("n_days"),
        "start": jt_gross.get("start"),
        "end": jt_gross.get("end"),
    }


def waterfall_table(
    gross: dict,
    naive: dict,
    shortable: dict,
    real_costs: dict,
    hybrid: dict,
) -> dict:
    steps = [
        ("paper_gross", gross),
        ("net_naive", naive),
        ("shortability", shortable),
        ("real_costs", real_costs),
        ("hybrid_book", hybrid),
    ]
    rows = []
    prev = None
    for name, blob in steps:
        sh = float(blob.get("sharpe") if blob.get("sharpe") is not None else blob.get("net_sharpe"))
        d = float(sh - prev) if prev is not None and np.isfinite(sh) and np.isfinite(prev) else float("nan")
        rows.append(
            {
                "step": name,
                "sharpe": sh,
                "delta": d,
                "trail": float(blob.get("sharpe_trail18m") if blob.get("sharpe_trail18m") is not None else blob.get("net_sharpe_trail18m") or float("nan")),
                "nw_t": float(blob.get("nw_t") if blob.get("nw_t") is not None else float("nan")),
                "ann_mean": float(blob.get("ann_mean") if blob.get("ann_mean") is not None else float("nan")),
            }
        )
        prev = sh
    g = float(rows[0]["sharpe"])
    h = float(rows[-1]["sharpe"])
    tax = float(g - h) if np.isfinite(g) and np.isfinite(h) else float("nan")
    return {
        "rows": rows,
        "tax": tax,
        "paper_gross": g,
        "implementable": h,
        "hybrid_ref": float(PHASE3C_REF_HYBRID_SHARPE),
    }


def series_corr(a: pd.Series, b: pd.Series) -> dict:
    x, y = a.align(b, join="inner")
    x = x.astype(float)
    y = y.astype(float)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 10:
        return {"corr": float("nan"), "n": int(len(x))}
    return {"corr": float(x.corr(y)), "n": int(len(x))}


def replay_cmc_book_from_log(plog: pd.DataFrame, cmc_close: pd.DataFrame) -> pd.Series:
    """CMC-priced implementable book from the frozen 3.c position log (real costs)."""
    df = plog.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["nxt"] = pd.to_datetime(df["nxt"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    recs = []
    for (dt, nxt), g in df.groupby(["date", "nxt"], sort=True):
        dt = _utc_key(dt)
        nxt = _utc_key(nxt)
        gross = 0.0
        cost = 0.0
        rrow = None
        if dt in cmc_close.index and nxt in cmc_close.index:
            rrow = (cmc_close.loc[nxt] / cmc_close.loc[dt] - 1.0).replace([np.inf, -np.inf], np.nan)
        for row in g.itertuples(index=False):
            iid = int(row.id)
            w = float(row.w)
            dw = float(row.dw)
            old_w = w - dw
            cost += name_cost(old_w, w)
            if rrow is not None and iid in rrow.index:
                r = float(rrow[iid])
                if np.isfinite(r):
                    gross += w * r
        recs.append({"nxt": nxt, "net": gross - cost})
    s = pd.Series({r["nxt"]: r["net"] for r in recs}, dtype=float).sort_index()
    s.index = _utc_idx(s.index)
    return s


def slim_factor(blob: dict) -> dict:
    keep = (
        "n_days",
        "start",
        "end",
        "sharpe",
        "sharpe_trail18m",
        "ann_mean",
        "ann_vol",
        "nw_t",
        "nw_t_trail18m",
        "total",
        "maxdd",
        "avg_n_long",
        "avg_n_short",
        "avg_long_gross",
        "avg_short_gross",
        "ann_turnover",
        "ann_cost_drag",
        "h",
        "shortable",
        "long_bps",
        "short_bps",
        "label",
        "cycles",
    )
    out = {k: blob.get(k) for k in keep}
    for sub in ("gross", "long", "short", "universe", "lmU", "umS"):
        if isinstance(blob.get(sub), dict):
            out[sub] = {k: blob[sub].get(k) for k in ("n_days", "sharpe", "sharpe_trail18m", "ann_mean", "ann_vol", "nw_t", "total", "maxdd", "cycles")}
    return out
