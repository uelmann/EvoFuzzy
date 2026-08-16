"""Overlapping h-tranche long 10 / short 10 inside the PIT top-30."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from baseline.portfolio import _attach_aux, _inv_vol
from nasdaq_ls.constants import (
    COST_BPS,
    EXEC_TOP_N,
    GROSS_LIMIT,
    K_LONG,
    K_SHORT,
    LAG,
    MIN_CS,
)


def _simple_returns(panel: pd.DataFrame) -> pd.DataFrame:
    from nasdaq_ls.prices import close_wide

    return close_wide(panel).pct_change(fill_method=None)


def _utc_ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()


def _pick_ls(day: pd.DataFrame, k_long: int, k_short: int) -> tuple[list[str], list[str]]:
    if day.empty or "score" not in day.columns:
        return [], []
    g = day.copy()
    g["_sc"] = pd.to_numeric(g["score"], errors="coerce")
    g = g[np.isfinite(g["_sc"])].drop_duplicates("symbol")
    if len(g) < MIN_CS:
        return [], []
    g = g.sort_values("_sc", ascending=False)
    k_l = min(int(k_long), max(1, len(g) // 3))
    k_s = min(int(k_short), max(1, len(g) // 3))
    longs = g["symbol"].head(k_l).tolist()
    shorts = g["symbol"].tail(k_s).tolist()
    overlap = set(longs) & set(shorts)
    if overlap:
        shorts = [s for s in shorts if s not in overlap]
    return longs, shorts


def _size_ls(day: pd.DataFrame, longs: list[str], shorts: list[str], gross: float) -> pd.Series:
    w: dict[str, float] = {}
    if longs:
        iv = {s: _inv_vol(day, s) for s in longs}
        ssum = sum(iv.values()) or 1.0
        for s, v in iv.items():
            w[s] = 0.5 * float(gross) * v / ssum
    if shorts:
        iv = {s: _inv_vol(day, s) for s in shorts}
        ssum = sum(iv.values()) or 1.0
        for s, v in iv.items():
            w[s] = -0.5 * float(gross) * v / ssum
    return pd.Series(w, dtype=float)


def run_ls_topn(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    feat: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    k_long: int = K_LONG,
    k_short: int = K_SHORT,
    book_start: str | None = None,
    variant: str = "nasdaq_ls",
) -> dict:
    h = int(horizon)
    df = _attach_aux(preds, feat, universe)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    if book_start:
        cut = pd.Timestamp(book_start, tz="UTC").normalize()
        df = df[df["date"] >= cut]
    if "score" not in df.columns or df.empty:
        return {"error": "preds missing score or empty after book_start"}
    rets = _simple_returns(panel)
    dates = sorted(df["date"].unique(), key=_utc_ts)
    if len(dates) < h + 5:
        return {"error": "not enough dates", "variant": variant, "horizon": h}

    by_date = {pd.Timestamp(dt).normalize(): g for dt, g in df.groupby("date", sort=False)}
    tg = float(GROSS_LIMIT) / float(h)
    cost_rate = float(COST_BPS) * 1e-4
    alphas: list[pd.Series] = [pd.Series(dtype=float) for _ in range(h)]
    target_hist: list[pd.Series] = []
    prev_full = pd.Series(dtype=float)
    daily_net, daily_gross, daily_cost = [], [], []
    n_long, n_short, n_pos, flat, eq_dates = [], [], [], [], []
    daily_gross_deployed = []
    name_pnl: dict[str, float] = defaultdict(float)
    n_forced = 0

    for i, dt in enumerate(dates[:-1]):
        dt = _utc_ts(dt)
        day = by_date.get(dt, pd.DataFrame())
        k = i % h
        prev_ak = alphas[k].copy()
        longs, shorts = _pick_ls(day, k_long, k_short) if not day.empty else ([], [])
        alphas[k] = (
            _size_ls(day, longs, shorts, tg)
            if (longs or shorts)
            else pd.Series(dtype=float)
        )

        alpha = pd.Series(dtype=float)
        univ = set(day["symbol"]) if not day.empty else set()
        for tk in range(h):
            ak = alphas[tk]
            if ak.empty:
                continue
            if univ:
                ak = ak[[s for s in ak.index if s in univ]]
                alphas[tk] = ak
            alpha = alpha.add(ak, fill_value=0.0)

        if i < LAG:
            applied = pd.Series(dtype=float)
        else:
            applied = target_hist[i - LAG] if LAG else alpha
        target_hist.append(alpha)

        fidx = applied.index.union(prev_full.index)
        f = applied.reindex(fidx).fillna(0.0)
        pf = prev_full.reindex(fidx).fillna(0.0)
        dw = (f - pf).abs()
        cost = float(dw.sum()) * cost_rate

        nxt = _utc_ts(dates[i + 1])
        gross_r = 0.0
        if nxt in rets.index:
            rrow = rets.loc[nxt]
            for s, wi in applied.items():
                if s in rrow.index and np.isfinite(rrow[s]):
                    contrib = float(wi) * float(rrow[s])
                    gross_r += contrib
                    name_pnl[s] += contrib
                else:
                    n_forced += 1
        elif len(applied):
            n_forced += int((applied.abs() > 1e-12).sum())

        net = gross_r - cost
        daily_net.append(net)
        daily_gross.append(gross_r)
        daily_cost.append(cost)
        nl = int((applied > 1e-12).sum()) if len(applied) else 0
        ns = int((applied < -1e-12).sum()) if len(applied) else 0
        n_long.append(nl)
        n_short.append(ns)
        n_pos.append(nl + ns)
        flat.append(1 if nl + ns == 0 else 0)
        eq_dates.append(nxt)
        daily_gross_deployed.append(float(applied.abs().sum()) if len(applied) else 0.0)
        prev_full = applied
        if i % 250 == 0:
            print(
                f"[{variant} h={h}] day {i}/{len(dates)} L={nl} S={ns} "
                f"gross={daily_gross_deployed[-1]:.3f} net={net:.5f}",
                flush=True,
            )

    idx = pd.DatetimeIndex(eq_dates)
    daily_ret = pd.Series(daily_net, index=idx, dtype=float)
    packed = {
        "variant": variant,
        "horizon": h,
        "k_long": k_long,
        "k_short": k_short,
        "exec_top_n": EXEC_TOP_N,
        "daily_ret": daily_ret,
        "daily_gross_pnl": pd.Series(daily_gross, index=idx, dtype=float),
        "daily_cost": pd.Series(daily_cost, index=idx, dtype=float),
        "avg_n_long": float(np.mean(n_long)) if n_long else 0.0,
        "avg_n_short": float(np.mean(n_short)) if n_short else 0.0,
        "avg_n_pos": float(np.mean(n_pos)) if n_pos else 0.0,
        "pct_flat_days": float(np.mean(flat)) if flat else 1.0,
        "avg_gross_deployed": float(np.mean(daily_gross_deployed)) if daily_gross_deployed else 0.0,
        "cost_drag": float(np.sum(daily_cost)),
        "n_forced_exits": int(n_forced),
        "name_alpha_pnl": dict(name_pnl),
        "n_days": int(len(daily_ret)),
    }
    return packed
