"""Scout metrics: 252-day Sharpe, RankIC, FACTOR gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic, newey_west_t, summarize_ic
from nasdaq_ls.constants import ANNUALIZATION, HEADLINE_START, TRAIL_DAYS


def _as_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).normalize()
    return out.sort_index()


def sharpe(x: pd.Series, ann: int = ANNUALIZATION) -> float:
    x = pd.to_numeric(_as_utc(x), errors="coerce").dropna()
    if len(x) < 5 or float(x.std(ddof=1) or 0) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ann))


def cagr_maxdd(x: pd.Series, ann: int = ANNUALIZATION) -> tuple[float, float, float]:
    r = pd.to_numeric(_as_utc(x), errors="coerce").fillna(0.0)
    if r.empty:
        return float("nan"), float("nan"), float("nan")
    eq = (1.0 + r).cumprod()
    years = len(r) / float(ann)
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / max(years, 1e-9)) - 1.0) if years > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1.0).min())
    return cagr, dd, total


def window_from(s: pd.Series, start: str | None) -> pd.Series:
    s = _as_utc(s)
    if not start:
        return s
    cut = pd.Timestamp(start, tz="UTC").normalize()
    return s[s.index >= cut]


def trail18m(s: pd.Series) -> pd.Series:
    s = _as_utc(s)
    if s.empty:
        return s
    end = s.index.max()
    start = end - pd.Timedelta(days=int(TRAIL_DAYS))
    return s[(s.index >= start) & (s.index <= end)]


def by_year_sharpe(s: pd.Series) -> dict[int, float]:
    s = _as_utc(s)
    out = {}
    for y, g in s.groupby(s.index.year):
        out[int(y)] = sharpe(g)
    return out


def pooled_rankic(pred: pd.DataFrame, ycol: str, score_col: str = "score", horizon: int = 10) -> dict:
    ic = daily_rank_ic(pred, ycol, score_col)
    return summarize_ic(ic, horizon=int(horizon))


def last_fold_wins(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds
    if "fold_id" in preds.columns:
        out = preds.sort_values(["date", "symbol", "fold_id"])
    else:
        out = preds.sort_values(["date", "symbol"])
    return out.drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)


def summarize_book(raw: dict, start: str | None = None) -> dict:
    r = raw.get("daily_ret")
    if not isinstance(r, pd.Series) or r.empty:
        return {"error": raw.get("error", "empty book")}
    r = window_from(r, start)
    cagr, maxdd, total = cagr_maxdd(r)
    names = raw.get("name_alpha_pnl") or {}
    top = sorted(names.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return {
        "n_days": int(len(r)),
        "net_sharpe_full": sharpe(r),
        "net_sharpe_trail18m": sharpe(trail18m(r)),
        "net_sharpe_by_year": by_year_sharpe(r),
        "net_cagr": cagr,
        "max_drawdown": maxdd,
        "total_return": total,
        "avg_n_long": raw.get("avg_n_long"),
        "avg_n_short": raw.get("avg_n_short"),
        "avg_gross_deployed": raw.get("avg_gross_deployed"),
        "pct_flat_days": raw.get("pct_flat_days"),
        "cost_drag": raw.get("cost_drag"),
        "n_forced_exits": raw.get("n_forced_exits"),
        "top5_names": [{"symbol": s, "pnl": float(p)} for s, p in top],
        "daily_ret": r,
        "nw_t_vs_cash": newey_west_t(r.to_numpy(dtype=float), lag=10) if len(r) else float("nan"),
    }


def factor_verdict(ric_mean: float, sharpe_headline: float) -> dict:
    pass_ric = bool(np.isfinite(ric_mean) and ric_mean > 0)
    pass_sh = bool(np.isfinite(sharpe_headline) and sharpe_headline > 0)
    yes = bool(pass_ric and pass_sh)
    return {
        "verdict": "FACTOR" if yes else "NO FACTOR",
        "factor": yes,
        "pass_ric": pass_ric,
        "pass_sharpe": pass_sh,
        "ric": float(ric_mean) if np.isfinite(ric_mean) else float("nan"),
        "sharpe": float(sharpe_headline) if np.isfinite(sharpe_headline) else float("nan"),
        "headline_start": HEADLINE_START,
    }


def qqq_bh(market: pd.Series) -> pd.Series:
    s = pd.to_numeric(market, errors="coerce").sort_index()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).normalize()
    return s.pct_change(fill_method=None).dropna()


def ew_universe(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """Costless EW of PIT names held from t close to t+1 close (informational)."""
    rets = _simple_from_panel(panel)
    uni = universe.copy()
    uni["date"] = pd.to_datetime(uni["date"], utc=True).dt.normalize()
    dates = sorted(uni["date"].unique())
    rows = []
    by = {pd.Timestamp(dt).normalize(): g for dt, g in uni.groupby("date", sort=False)}
    for i, dt in enumerate(dates[:-1]):
        dt = pd.Timestamp(dt).normalize()
        nxt = pd.Timestamp(dates[i + 1]).normalize()
        names = by[dt]["symbol"].tolist()
        if nxt not in rets.index:
            continue
        row = rets.loc[nxt]
        vals = [float(row[s]) for s in names if s in row.index and np.isfinite(row[s])]
        if vals:
            rows.append((nxt, float(np.mean(vals))))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({d: v for d, v in rows})
    s.index = pd.DatetimeIndex(s.index)
    return s


def _simple_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    from nasdaq_ls.prices import close_wide

    return close_wide(panel).pct_change(fill_method=None)
