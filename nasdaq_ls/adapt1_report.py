"""NASDAQ-ADAPT-1 report (does not rewrite LS / LS21 report templates)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nasdaq_ls.eval import cagr_maxdd, sharpe, trail18m, window_from
from nasdaq_ls.report import _book_row, _fmt, _pct


def _bench(s, start=None):
    if not isinstance(s, pd.Series) or not len(s):
        return "n/a"
    s = window_from(s, start) if start else s
    cagr, maxdd, total = cagr_maxdd(s)
    return (
        f"Sharpe={_fmt(sharpe(s))}, trail-18m={_fmt(sharpe(trail18m(s)))}, "
        f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(total)}."
    )


def write_adapt1_report(
    path: Path,
    *,
    factor: dict,
    ml: dict,
    book_2005: dict,
    book_2007: dict,
    ctrl_2005: dict,
    ctrl_2007: dict,
    ric_all: dict,
    ric_2007: dict,
    ric_ctrl_2007: dict,
    extra: dict,
    factor_criterion: str,
) -> str:
    headline = extra.get("headline_start") or "2007-01-01"
    by = book_2007.get("net_sharpe_by_year") or {}
    year_vals = " | ".join(_fmt(by.get(y)) for y in range(2007, 2027))
    qqq = extra.get("qqq")
    ew = extra.get("ew")
    xs = extra.get("excess_qqq")
    gain = extra.get("feature_gain_mean") or {}
    gain_s = ", ".join(f"{k}={_fmt(v, 1)}" for k, v in list(gain.items())[:10]) if gain else ""

    text = f"""# NASDAQ-ADAPT-1 — simple h=126, equity clock, long-only, 12-1 control

**BACKTEST AND ANALYSIS ONLY.** Adaptation of the Nasdaq scout after the LS/LS21 forensics. COMBO, SPREAD-LS, LONG-TIDE, NASDAQ-LS, and NASDAQ-LS21 are **untouched**. Survivorship (today's index members) is accepted for this scout.

**Universe source:** {extra.get('ticker_source')} n={extra.get('n_symbols')}
**Price span:** {extra.get('min_date')} → {extra.get('max_date')}
**Mandate:** PIT top 30 by 30d median dollar volume; long-only top 10%; overlapping h=126; inv-vol; 5 bps; no shorts; no QQQ overlay.
**Train:** 500 Huber trees, no early stop, rolling ≤1260 sessions. Label = winsorized simple 126-session forward return. Equity-clock features. Session purge/embargo.

## Pre-registered factor statement

> {factor_criterion}

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Scout: {factor.get('verdict')}** — RankIC from {headline}={_fmt(factor.get('ric'))} (pass={factor.get('pass_ric')}); long-only net Sharpe from {headline}={_fmt(factor.get('sharpe'))} (pass={factor.get('pass_sharpe')}).
- **{ml.get('verdict')}** — GBM Sharpe={_fmt(ml.get('gbm_sharpe'))} vs 12-1 control Sharpe={_fmt(ml.get('ctrl_sharpe'))} (beat={ml.get('beat_12_1_sharpe')}).

## Headline books (252-day Sharpe)

| book | full Sharpe | trail-18m | CAGR | MaxDD | total | avg #long | avg #short | avg |gross| | % flat | cost drag | forced | top-5 |name| PnL |
|------|-------------|-----------|------|-------|-------|-----------|------------|-------------|--------|-----------|--------|------------------|
{_book_row("ADAPT-1 GBM long-only from 2005-01-01", book_2005)}
{_book_row("ADAPT-1 GBM long-only from 2007-01-01 (FACTOR window)", book_2007)}
{_book_row("12-1 control long-only from 2005-01-01", ctrl_2005)}
{_book_row("12-1 control long-only from 2007-01-01", ctrl_2007)}

### GBM Sharpe by year (FACTOR window)

| {' | '.join(str(y) for y in range(2007, 2027))} |
| {' | '.join(['---'] * 20)} |
| {year_vals} |

## RankIC on PIT top-30 vs simple h=126 return

| window | LightGBM RankIC | ICIR | NW-t | n | 12-1 RankIC |
|--------|-----------------|------|------|---|-------------|
| all OOS | {_fmt(ric_all.get('mean_ic'))} | {_fmt(ric_all.get('icir'))} | {_fmt(ric_all.get('nw_tstat'))} | {ric_all.get('n_days')} | — |
| from {headline} | {_fmt(ric_2007.get('mean_ic'))} | {_fmt(ric_2007.get('icir'))} | {_fmt(ric_2007.get('nw_tstat'))} | {ric_2007.get('n_days')} | {_fmt(ric_ctrl_2007.get('mean_ic'))} |

### Costless benchmarks (not FACTOR inputs)

- QQQ B&H from 2007: {_bench(qqq, headline)}
- EW PIT top-30 from 2007: {_bench(ew, headline)}
- GBM excess vs QQQ from 2007: {_bench(xs, None)}

### Mean LightGBM gain (across ok folds)

{gain_s}

## Construction notes

{extra.get('construction')}

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, LONG-TIDE, NASDAQ-LS, and NASDAQ-LS21 are not modified. This scout does not rewrite the system card.

Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. {extra.get('train_note')}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
