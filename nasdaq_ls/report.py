"""NASDAQ-LS report and equity chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nasdaq_ls.constants import FACTOR_CRITERION, HEADLINE_START
from nasdaq_ls.eval import _as_utc, cagr_maxdd, sharpe, trail18m, window_from


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _pct(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{100.0 * float(x):.{nd}f}%"
    except Exception:
        return str(x)


def _eq_from_rets(rets: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    r = _as_utc(rets).fillna(0.0)
    eq = (1.0 + r).cumprod()
    y = eq.to_numpy()
    if len(y) and y[0] != 0:
        y = y / y[0]
    return eq.index, y


def plot_equity(
    ls: pd.Series,
    qqq: pd.Series | None,
    ew: pd.Series | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = [(ls, "NASDAQ-LS (top10 − worst10, PIT vol-30)")]
    if isinstance(ew, pd.Series) and len(ew):
        series.append((ew, "EW PIT top-30 (costless, informational)"))
    if isinstance(qqq, pd.Series) and len(qqq):
        series.append((qqq, "QQQ B&H (informational)"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    end = None
    for rets, lab in series:
        if rets is None or not len(rets):
            continue
        d, y = _eq_from_rets(rets)
        end = d.max() if end is None else max(end, d.max())
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("NASDAQ-LS vs QQQ (long 10 / short 10, top 30 by volume)")
    axes[0].set_ylabel("Equity (rebased, log)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for rets, lab in series:
            if rets is None or not len(rets):
                continue
            d, y = _eq_from_rets(rets)
            m = np.asarray((pd.DatetimeIndex(d) >= start) & (pd.DatetimeIndex(d) <= end))
            if not m.any():
                continue
            yy = np.asarray(y)[m]
            dd = pd.DatetimeIndex(d)[m]
            if len(yy) and yy[0] != 0:
                yy = yy / yy[0]
            axes[1].plot(dd, yy, label=lab, lw=1.4)
    axes[1].set_title("Trailing-18m zoom")
    axes[1].set_ylabel("Equity (rebased)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _book_row(name: str, book: dict) -> str:
    by = book.get("net_sharpe_by_year") or {}
    years = list(range(2005, 2027))
    ycols = " | ".join(_fmt(by.get(y)) for y in years)
    top = book.get("top5_names") or []
    top_s = ", ".join(f"{t['symbol']}={_fmt(t['pnl'])}" for t in top) if top else ""
    return (
        f"| {name} | {_fmt(book.get('net_sharpe_full'))} | {_fmt(book.get('net_sharpe_trail18m'))} | "
        f"{_pct(book.get('net_cagr'))} | {_pct(book.get('max_drawdown'))} | "
        f"{_pct(book.get('total_return'))} | {_fmt(book.get('avg_n_long'))} | "
        f"{_fmt(book.get('avg_n_short'))} | {_fmt(book.get('avg_gross_deployed'))} | "
        f"{_pct(book.get('pct_flat_days'))} | {_fmt(book.get('cost_drag'))} | "
        f"{int(book.get('n_forced_exits') or 0)} | {top_s} |"
    )


def write_report(
    path: Path,
    *,
    factor: dict,
    book_2005: dict,
    book_2007: dict,
    ric_all: dict,
    ric_2007: dict,
    ric_simple: dict,
    extra: dict,
) -> str:
    by = book_2007.get("net_sharpe_by_year") or {}
    year_rows = " | ".join(str(y) for y in range(2007, 2027))
    year_vals = " | ".join(_fmt(by.get(y)) for y in range(2007, 2027))
    qqq = extra.get("qqq")
    ew = extra.get("ew")

    def _bench(s, start=None):
        if not isinstance(s, pd.Series) or not len(s):
            return "n/a"
        s = window_from(s, start) if start else s
        cagr, maxdd, total = cagr_maxdd(s)
        return (
            f"Sharpe={_fmt(sharpe(s))}, trail-18m={_fmt(sharpe(trail18m(s)))}, "
            f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(total)}."
        )

    text = f"""# NASDAQ-LS — LightGBM long 10 / short 10 on Nasdaq-100 (scout)

**BACKTEST AND ANALYSIS ONLY.** A0-style Huber LightGBM on Yahoo Nasdaq-100 bars. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**. Survivorship (today's index members) is accepted for this scout.

**Universe source:** {extra.get('ticker_source')} n={extra.get('n_symbols')}
**Price span:** {extra.get('min_date')} → {extra.get('max_date')}
**Mandate:** PIT top 30 by 30d median dollar volume; long 10 / short 10 by score; overlapping h=10; inv-vol; 5 bps one-way; no borrow; no index overlay.
**Train:** 500 trees, no early stop, Huber. Market residual = spliced ^IXIC/QQQ.

## Pre-registered factor statement

> {FACTOR_CRITERION}

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Scout: {factor.get('verdict')}** — RankIC from {HEADLINE_START}={_fmt(factor.get('ric'))} (pass={factor.get('pass_ric')}); LS net Sharpe from {HEADLINE_START}={_fmt(factor.get('sharpe'))} (pass={factor.get('pass_sharpe')}).

## Headline books (252-day Sharpe)

| book | full Sharpe | trail-18m | CAGR | MaxDD | total | avg #long | avg #short | avg |gross| | % flat | cost drag | forced | top-5 |name| PnL |
|------|-------------|-----------|------|-------|-------|-----------|------------|-------------|--------|-----------|--------|------------------|
{_book_row("NASDAQ-LS from 2005-01-01", book_2005)}
{_book_row("NASDAQ-LS from 2007-01-01 (FACTOR window)", book_2007)}

### Sharpe by year (FACTOR window)

| {' | '.join(str(y) for y in range(2007, 2027))} |
| {' | '.join(['---'] * 20)} |
| {year_vals} |

## RankIC on PIT top-30

| window | RankIC vs residual y | ICIR | NW-t | n | RankIC vs simple 10d USDT-style return |
|--------|----------------------|------|------|---|----------------------------------------|
| all OOS | {_fmt(ric_all.get('mean_ic'))} | {_fmt(ric_all.get('icir'))} | {_fmt(ric_all.get('nw_tstat'))} | {ric_all.get('n_days')} | {_fmt(ric_simple.get('mean_ic'))} |
| from {HEADLINE_START} | {_fmt(ric_2007.get('mean_ic'))} | {_fmt(ric_2007.get('icir'))} | {_fmt(ric_2007.get('nw_tstat'))} | {ric_2007.get('n_days')} | — |

### Costless benchmarks (not FACTOR inputs)

- QQQ B&H from 2007: {_bench(qqq, HEADLINE_START)}
- EW PIT top-30 from 2007: {_bench(ew, HEADLINE_START)}

## Construction notes

{extra.get('construction')}

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified. This scout does not rewrite the system card.

Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. {extra.get('train_note')}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
