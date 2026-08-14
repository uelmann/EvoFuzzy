"""Phase 1 naive BTC-parked rotation. Parameters frozen; no sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    ALT_BPS,
    ANNUALIZATION,
    BTC_BPS,
    CYCLES,
    HORIZON,
    LOOKBACK,
    NAME_CAP,
    N_HOLD,
)
from btcb.universe import collapse_duplicate_symbols


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION)) if len(x) and x.std() > 0 else 0.0


def _cagr(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return float("nan")
    years = len(eq) / ANNUALIZATION
    return float(eq.iloc[-1] ** (1.0 / max(years, 1e-6)) - 1.0)


def _maxdd(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def naive_rotation(
    panel: pd.DataFrame,
    pit: pd.DataFrame,
    start: pd.Timestamp,
    *,
    lookback: int = LOOKBACK,
    n_hold: int = N_HOLD,
    h: int = HORIZON,
    name_cap: float = NAME_CAP,
    alt_bps: float = ALT_BPS,
    btc_bps: float = BTC_BPS,
    degenerate_btc: bool = False,
) -> dict:
    """Equal-weight top names by 90d excess vs BTC; remainder in BTC; h overlapping slots."""
    df = collapse_duplicate_symbols(panel.copy())
    df["date"] = pd.to_datetime(df["date"], utc=True)
    start = _as_utc(start)
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    if "BTC" not in close.columns:
        raise RuntimeError("BTC missing from panel")
    logp = np.log(close.clip(lower=1e-18))
    fwd = logp.diff(lookback)
    excess = fwd.sub(fwd["BTC"], axis=0)
    pit = pit.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    members = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): list(v)
        for d, v in pit.groupby("date")["symbol"]
    }
    dates = [d for d in close.index if d >= start]
    if len(dates) < h + lookback + 5:
        return {"error": "not enough dates in usable window"}

    alt_c = alt_bps * 1e-4
    btc_c = btc_bps * 1e-4
    slots = [pd.Series(dtype=float) for _ in range(h)]
    # Product starts parked in BTC (never cash); no entry cost on the initial BTC book.
    prev_full = pd.Series({"BTC": 1.0}, dtype=float)
    daily = []
    btc_w = []
    to_list = []
    eq_dates = []

    for i, dt in enumerate(dates[:-1]):
        k = i % h
        nxt = dates[i + 1]
        if degenerate_btc:
            alpha = pd.Series(dtype=float)
        else:
            key = pd.Timestamp(dt).tz_convert("UTC").normalize() if getattr(dt, "tzinfo", None) else _as_utc(dt).normalize()
            names = members.get(key, [])
            names = [s for s in names if s in excess.columns and s != "BTC"]
            if names and dt in excess.index:
                ex = excess.loc[dt, names].astype(float)
                ex = ex[np.isfinite(ex) & (ex > 0)]
                pick = list(ex.sort_values(ascending=False).head(n_hold).index)
            else:
                pick = []
            if pick:
                w = (1.0 / h) / len(pick)
                alpha = pd.Series({s: w for s in pick}, dtype=float)
            else:
                alpha = pd.Series(dtype=float)
        slots[k] = alpha
        full = pd.Series(dtype=float)
        for sl in slots:
            if len(sl):
                full = full.add(sl, fill_value=0.0)
        # 10% cap, overflow to BTC
        if len(full):
            over = full[full > name_cap]
            if len(over):
                dumped = float((over - name_cap).sum())
                full = full.clip(upper=name_cap)
            else:
                dumped = 0.0
        else:
            dumped = 0.0
        alt_gross = float(full.abs().sum()) if len(full) else 0.0
        w_btc = max(0.0, 1.0 - alt_gross)
        applied = full.copy()
        applied.loc["BTC"] = float(applied.get("BTC", 0.0)) + w_btc

        idx = applied.index.union(prev_full.index)
        a = applied.reindex(idx).fillna(0.0)
        p = prev_full.reindex(idx).fillna(0.0)
        dw = (a - p).abs()
        cost = 0.0
        for s, mag in dw.items():
            cost += float(mag) * (btc_c if s == "BTC" else alt_c)
        turnover = 0.5 * float(dw.sum())

        r = 0.0
        if nxt in close.index:
            simple = (close.loc[nxt] / close.loc[dt] - 1.0)
            for s, wi in applied.items():
                if s in simple.index and np.isfinite(simple[s]):
                    r += float(wi) * float(simple[s])
        net = r - cost
        daily.append(net)
        btc_w.append(float(applied.get("BTC", 0.0)))
        to_list.append(turnover)
        eq_dates.append(nxt)
        prev_full = applied
        if i % 60 == 0:
            print(
                f"[HB] naive i={i}/{len(dates)} dt={dt.date()} nalts={int((full>0).sum()) if len(full) else 0} "
                f"wbtc={w_btc:.2f} net={net:.5f} deg={degenerate_btc}",
                flush=True,
            )

    rets = pd.Series(daily, index=pd.DatetimeIndex(eq_dates), dtype=float)
    btc_simple = close["BTC"].pct_change().reindex(rets.index)
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    eq_btc = (1.0 + btc_simple.fillna(0.0)).cumprod()
    rel_eq = eq / eq_btc.replace(0, np.nan)
    rel_ret = rel_eq.pct_change().fillna(0.0)
    wbtc = pd.Series(btc_w, index=rets.index)
    years = sorted({int(y) for y in rets.index.year.unique() if y >= 2018})
    by_year = {y: _sharpe(rets[rets.index.year == y]) for y in years}
    cycles = {}
    for name, a, b in CYCLES:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = rets[(rets.index >= t0) & (rets.index <= t1)]
        if sl.empty:
            continue
        eq_s = (1.0 + sl.fillna(0.0)).cumprod()
        btc_s = btc_simple.reindex(sl.index).fillna(0.0)
        eqb = (1.0 + btc_s).cumprod()
        rel = eq_s / eqb.replace(0, np.nan)
        cycles[name] = {
            "n": int(len(sl)),
            "book_total": float(eq_s.iloc[-1] - 1.0) if len(eq_s) else float("nan"),
            "btc_total": float(eqb.iloc[-1] - 1.0) if len(eqb) else float("nan"),
            "book_cagr": _cagr(eq_s),
            "btc_cagr": _cagr(eqb),
            "book_sharpe": _sharpe(sl),
            "rel_cagr": _cagr(rel),
            "rel_sharpe": _sharpe(rel.pct_change().fillna(0.0)),
            "maxdd": _maxdd(eq_s),
            "avg_w_btc": float(wbtc.reindex(sl.index).mean()) if len(sl) else float("nan"),
        }
    book_total = float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")
    btc_total = float(eq_btc.iloc[-1] - 1.0) if len(eq_btc) else float("nan")
    live = bool(np.isfinite(_sharpe(rel_ret)) and _sharpe(rel_ret) > 0 and np.isfinite(book_total) and book_total >= btc_total)
    return {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()) if len(rets) else None,
        "end": str(rets.index.max().date()) if len(rets) else None,
        "book_total": book_total,
        "btc_total": btc_total,
        "book_cagr": _cagr(eq),
        "btc_cagr": _cagr(eq_btc),
        "book_sharpe": _sharpe(rets),
        "btc_sharpe": _sharpe(btc_simple),
        "maxdd": _maxdd(eq),
        "btc_maxdd": _maxdd(eq_btc),
        "rel_cagr": _cagr(rel_eq),
        "rel_sharpe": _sharpe(rel_ret),
        "avg_w_btc": float(wbtc.mean()) if len(wbtc) else float("nan"),
        "ann_turnover": float(np.mean(to_list) * ANNUALIZATION) if to_list else float("nan"),
        "live_benchmark": live,
        "by_year_sharpe": by_year,
        "cycles": cycles,
        "daily_ret": rets,
        "btc_ret": btc_simple,
        "equity": eq,
        "equity_btc": eq_btc,
        "rel_equity": rel_eq,
        "w_btc": wbtc,
        "degenerate": bool(degenerate_btc),
    }


def naive_rotation_v3(
    panel: pd.DataFrame,
    pit: pd.DataFrame,
    start: pd.Timestamp,
    *,
    lookback: int = LOOKBACK,
    n_hold: int = N_HOLD,
    h: int = HORIZON,
    name_cap: float = NAME_CAP,
    alt_bps: float = ALT_BPS,
    btc_bps: float = BTC_BPS,
    degenerate_btc: bool = False,
) -> dict:
    """Id-keyed naive rotation with death-in-position force-exits. No symbol collapse."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    start = _as_utc(start).normalize()
    btc_rows = df[df["slug"].astype(str).str.lower().eq("bitcoin") | df["symbol"].str.upper().eq("BTC")]
    if btc_rows.empty:
        raise RuntimeError("BTC missing from panel")
    btc_id = int(btc_rows.groupby("id").size().sort_values(ascending=False).index[0])
    close = df.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    last_map = {int(i): pd.Timestamp(d).tz_convert("UTC").normalize() for i, d in df.groupby("id")["date"].max().items()}
    id_to_sym = df.sort_values("date").groupby("id")["symbol"].last().to_dict()
    logp = np.log(close.clip(lower=1e-18))
    fwd = logp.diff(lookback)
    excess = fwd.sub(fwd[btc_id], axis=0)
    pit = pit.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    members = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit.groupby("date")["id"]
    }
    dates = [d for d in close.index if d >= start]
    if len(dates) < h + lookback + 5:
        return {"error": "not enough dates in usable window"}

    alt_c = alt_bps * 1e-4
    btc_c = btc_bps * 1e-4
    slots = [pd.Series(dtype=float) for _ in range(h)]
    prev_full = pd.Series({btc_id: 1.0}, dtype=float)
    daily, btc_w, to_list, eq_dates = [], [], [], []
    forced_events = []

    for i, dt in enumerate(dates[:-1]):
        k = i % h
        nxt = dates[i + 1]
        if degenerate_btc:
            alpha = pd.Series(dtype=float)
        else:
            names = [s for s in members.get(dt, []) if s in excess.columns and s != btc_id]
            if names and dt in excess.index:
                ex = excess.loc[dt, names].astype(float)
                ex = ex[np.isfinite(ex) & (ex > 0)]
                pick = [int(x) for x in ex.sort_values(ascending=False).head(n_hold).index]
            else:
                pick = []
            if pick:
                w = (1.0 / h) / len(pick)
                alpha = pd.Series({s: w for s in pick}, dtype=float)
            else:
                alpha = pd.Series(dtype=float)
        slots[k] = alpha
        # death-in-position: drop names whose last close is before nxt
        for j in range(h):
            if not len(slots[j]):
                continue
            drop = []
            for iid in slots[j].index:
                ld = last_map.get(int(iid))
                if ld is None or nxt > ld:
                    drop.append(int(iid))
            if drop:
                w_drop = float(slots[j].reindex(drop).fillna(0.0).sum())
                slots[j] = slots[j].drop(labels=drop, errors="ignore")
                if w_drop > 0:
                    forced_events.append(
                        {
                            "date": str(dt.date()),
                            "ids": drop,
                            "weight": w_drop,
                            "slot": j,
                        }
                    )
        full = pd.Series(dtype=float)
        for sl in slots:
            if len(sl):
                full = full.add(sl, fill_value=0.0)
        if len(full):
            over = full[full > name_cap]
            dumped = float((over - name_cap).sum()) if len(over) else 0.0
            full = full.clip(upper=name_cap)
        else:
            dumped = 0.0
        alt_gross = float(full.abs().sum()) if len(full) else 0.0
        w_btc = max(0.0, 1.0 - alt_gross)
        applied = full.copy()
        applied.loc[btc_id] = float(applied.get(btc_id, 0.0)) + w_btc

        idx = applied.index.union(prev_full.index)
        a = applied.reindex(idx).fillna(0.0)
        p = prev_full.reindex(idx).fillna(0.0)
        dw = (a - p).abs()
        cost = 0.0
        for s, mag in dw.items():
            cost += float(mag) * (btc_c if int(s) == btc_id else alt_c)
        turnover = 0.5 * float(dw.sum())

        r = 0.0
        if nxt in close.index and dt in close.index:
            simple = (close.loc[nxt] / close.loc[dt] - 1.0)
            for s, wi in applied.items():
                if s in simple.index and np.isfinite(simple[s]):
                    r += float(wi) * float(simple[s])
                elif int(s) != btc_id and wi and not (s in simple.index and np.isfinite(simple.get(s, np.nan))):
                    # dead name should already have been dropped; if not, 0 return
                    pass
        net = r - cost
        daily.append(net)
        btc_w.append(float(applied.get(btc_id, 0.0)))
        to_list.append(turnover)
        eq_dates.append(nxt)
        prev_full = applied
        if i % 60 == 0:
            print(
                f"[HB] naive_v3 i={i}/{len(dates)} dt={dt.date()} nalts={int((full>0).sum()) if len(full) else 0} "
                f"wbtc={w_btc:.2f} net={net:.5f} forced={len(forced_events)} deg={degenerate_btc}",
                flush=True,
            )

    rets = pd.Series(daily, index=pd.DatetimeIndex(eq_dates), dtype=float)
    btc_simple = close[btc_id].pct_change().reindex(rets.index)
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    eq_btc = (1.0 + btc_simple.fillna(0.0)).cumprod()
    rel_eq = eq / eq_btc.replace(0, np.nan)
    rel_ret = rel_eq.pct_change().fillna(0.0)
    wbtc = pd.Series(btc_w, index=rets.index)
    years = sorted({int(y) for y in rets.index.year.unique() if y >= 2018})
    by_year = {y: _sharpe(rets[rets.index.year == y]) for y in years}
    cycles = {}
    for name, a, b in CYCLES:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = rets[(rets.index >= t0) & (rets.index <= t1)]
        if sl.empty:
            continue
        eq_s = (1.0 + sl.fillna(0.0)).cumprod()
        btc_s = btc_simple.reindex(sl.index).fillna(0.0)
        eqb = (1.0 + btc_s).cumprod()
        rel = eq_s / eqb.replace(0, np.nan)
        cycles[name] = {
            "n": int(len(sl)),
            "book_total": float(eq_s.iloc[-1] - 1.0) if len(eq_s) else float("nan"),
            "btc_total": float(eqb.iloc[-1] - 1.0) if len(eqb) else float("nan"),
            "book_cagr": _cagr(eq_s),
            "btc_cagr": _cagr(eqb),
            "book_sharpe": _sharpe(sl),
            "rel_cagr": _cagr(rel),
            "rel_sharpe": _sharpe(rel.pct_change().fillna(0.0)),
            "maxdd": _maxdd(eq_s),
            "avg_w_btc": float(wbtc.reindex(sl.index).mean()) if len(sl) else float("nan"),
        }
    book_total = float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")
    btc_total = float(eq_btc.iloc[-1] - 1.0) if len(eq_btc) else float("nan")
    live = bool(np.isfinite(_sharpe(rel_ret)) and _sharpe(rel_ret) > 0 and np.isfinite(book_total) and book_total >= btc_total)
    n_events = len(forced_events)
    ids_forced = sorted({i for e in forced_events for i in e["ids"]})
    wsum = float(sum(e["weight"] for e in forced_events))
    cost_drag = wsum * (alt_c + btc_c)
    # PnL impact vs ghost (0 subsequent return, no extra cost): recycle into BTC for h days
    impact = -cost_drag
    for e in forced_events:
        t = _as_utc(e["date"]).normalize()
        if t not in eq_btc.index:
            nxts = eq_btc.index[eq_btc.index > t]
            if not len(nxts):
                continue
            t = nxts[0]
        later = eq_btc.index[eq_btc.index >= t]
        if len(later) < 2:
            continue
        t1 = later[min(h, len(later) - 1)]
        b0, b1 = float(eq_btc.loc[t]), float(eq_btc.loc[t1])
        if b0 > 0:
            impact += float(e["weight"]) * (b1 / b0 - 1.0)
    return {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()) if len(rets) else None,
        "end": str(rets.index.max().date()) if len(rets) else None,
        "book_total": book_total,
        "btc_total": btc_total,
        "book_cagr": _cagr(eq),
        "btc_cagr": _cagr(eq_btc),
        "book_sharpe": _sharpe(rets),
        "btc_sharpe": _sharpe(btc_simple),
        "maxdd": _maxdd(eq),
        "btc_maxdd": _maxdd(eq_btc),
        "rel_cagr": _cagr(rel_eq),
        "rel_sharpe": _sharpe(rel_ret),
        "avg_w_btc": float(wbtc.mean()) if len(wbtc) else float("nan"),
        "ann_turnover": float(np.mean(to_list) * ANNUALIZATION) if to_list else float("nan"),
        "live_benchmark": live,
        "by_year_sharpe": by_year,
        "cycles": cycles,
        "daily_ret": rets,
        "btc_ret": btc_simple,
        "equity": eq,
        "equity_btc": eq_btc,
        "rel_equity": rel_eq,
        "w_btc": wbtc,
        "degenerate": bool(degenerate_btc),
        "btc_id": btc_id,
        "forced_exits": {
            "n_events": n_events,
            "n_ids": len(ids_forced),
            "ids": ids_forced[:40],
            "symbols": [id_to_sym.get(i) for i in ids_forced[:40]],
            "weight_sum": wsum,
            "cost_drag": cost_drag,
            "pnl_impact_vs_ghost": impact,
            "note": "vs ghost (0 return until slot refresh): pay exit+BTC entry costs, recycle into BTC for h days",
        },
    }
