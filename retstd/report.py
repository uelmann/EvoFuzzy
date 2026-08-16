"""RETSTD-LO report, JSON, and equity chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retstd.constants import IMPROVE_CRITERION, VIABILITY_CRITERION
from retstd.eval import _as_utc, _sharpe, cagr_maxdd, window_slice


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
    a0: pd.Series,
    retstd: pd.Series,
    ew: pd.Series | None,
    btc: pd.Series | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = [(a0, "A0-LO10 (frozen scores)"), (retstd, "RETSTD-LO (P top-decile ret/std)")]
    if isinstance(ew, pd.Series) and len(ew):
        series.append((ew, "EW PIT top-40 (costless, informational)"))
    if isinstance(btc, pd.Series) and len(btc):
        series.append((btc, "BTC B&H (informational, not a gate)"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    end = None
    for rets, lab in series:
        if rets is None or not len(rets):
            continue
        d, y = _eq_from_rets(rets)
        end = d.max() if end is None else max(end, d.max())
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("RETSTD-LO vs A0-LO10 (long-only top 10%)")
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
    years = [2022, 2023, 2024, 2025, 2026]
    ycols = " | ".join(_fmt(by.get(y)) for y in years)
    top = book.get("top5_names") or []
    top_s = ", ".join(f"{t['symbol']}={_fmt(t['pnl'])}" for t in top) if top else ""
    return (
        f"| {name} | {_fmt(book.get('net_sharpe_full'))} | {_fmt(book.get('net_sharpe_trail18m'))} | "
        f"{ycols} | {_pct(book.get('net_cagr'))} | {_pct(book.get('max_drawdown'))} | "
        f"{_pct(book.get('total_return'))} | {_fmt(book.get('avg_n_long'))} | "
        f"{_fmt(book.get('avg_gross_deployed'))} | {_pct(book.get('pct_flat_days'))} | "
        f"{_fmt(book.get('funding_total_pnl'))} | {_fmt(book.get('cost_drag'))} | "
        f"{_fmt(book.get('ann_turnover'))} | {int(book.get('n_forced_exits') or 0)} | "
        f"{_fmt(book.get('max_abs_btc_weight'), 6)} | {_pct(book.get('pct_days_btc_held'))} | {top_s} |"
    )


def write_report(
    path: Path,
    *,
    frozen_hash: str,
    a0_book: dict,
    retstd_book: dict,
    improve: dict,
    verdict: dict,
    null: dict,
    ric_a0_ratio: dict,
    ric_retstd_ratio: dict,
    ric_a0_simple: dict,
    ric_retstd_simple: dict,
    gap_a0: dict,
    gap_retstd: dict,
    benches: dict,
    extra: dict,
) -> str:
    cells = null.get("cells") or []
    null_rows = []
    for c in cells:
        null_rows.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'))} | {_fmt(c.get('sd'))} | "
            f"{_fmt(c.get('p95'))} | {_fmt(c.get('real'))} | {c.get('bias_ok')} | {c.get('exceeds_p95')} |"
        )
    ew = benches.get("ew")
    btc = benches.get("btc")

    def _bench(s):
        if not isinstance(s, pd.Series) or not len(s):
            return "n/a"
        cagr, maxdd, total = cagr_maxdd(s)
        return (
            f"Sharpe full={_fmt(_sharpe(s))}, trail-18m={_fmt(_sharpe(window_slice(s, 'trail18m')))}, "
            f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(total)}."
        )

    text = f"""# RETSTD-LO — P(top 10% of ret/std) vs frozen A0 (long-only)

**BACKTEST AND ANALYSIS ONLY.** Same 33 A0 features. Frozen A0 Huber scores are **not retrained**. RETSTD is a new binary LightGBM on a working copy of the labels. No schedules, no live components. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**.

**Frozen A0 SHA256 (features/config):** `{frozen_hash}`
**Mandate:** long-only top 10% of PIT top-40 by score. Residual is cash. No hedge.
**Horizon:** h=10. Label = 1 iff h=10 USDT simple return / forward path std is in that date's PIT-120 top decile.

## Pre-registered improvement statement

> {IMPROVE_CRITERION}

## Pre-registered viability statement

> {VIABILITY_CRITERION}

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Target A/B: {improve.get('verdict')}** — RankIC vs ratio RETSTD={_fmt(improve.get('ric_retstd'))} vs A0={_fmt(improve.get('ric_a0'))} (pass={improve.get('pass_ric')}); top−universe gap RETSTD={_fmt(improve.get('gap_retstd'), 5)} vs A0={_fmt(improve.get('gap_a0'), 5)} (pass={improve.get('pass_gap')}); Sharpe RETSTD={_fmt(improve.get('sharpe_retstd'))} vs A0={_fmt(improve.get('sharpe_a0'))} (pass={improve.get('pass_sharpe')}); null={improve.get('null_verdict')} pass={improve.get('pass_null')}.
- **RETSTD-LO: {verdict.get('verdict')}** — full Sharpe={_fmt(verdict.get('sharpe_full'))} (need ≥ {_fmt(verdict.get('need_full'))}, pass={verdict.get('pass_full')}); trail-18m Sharpe={_fmt(verdict.get('sharpe_trail18m'))} (need ≥ {_fmt(verdict.get('need_trail'))}, pass={verdict.get('pass_trail')}); total={_pct(verdict.get('total_return'))} (need > 0, pass={verdict.get('pass_total')}); avg gross={_fmt(verdict.get('avg_gross'))} (need ≥ {_fmt(verdict.get('need_gross'))}, pass={verdict.get('pass_gross')}); null={verdict.get('null_verdict')} pass={verdict.get('pass_null')}.

## Headline books

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | MaxDD | total | avg #longs | avg gross | % flat | funding | costs | ann TO | forced | BTC max |w| | % days BTC | top-5 name PnL |
|------|------|-----------|------|------|------|------|------|------|-------|-------|------------|-----------|--------|---------|-------|--------|--------|----------------|------------|----------------|
{_book_row("A0-LO10", a0_book)}
{_book_row("RETSTD-LO", retstd_book)}

## RankIC vs ratio and vs USDT return (PIT top-40)

| arm | RankIC vs ratio | ICIR | NW-t | n | RankIC vs USDT | top−uni USDT | % gap>0 | gap NW-t |
|-----|-----------------|------|------|---|----------------|--------------|---------|----------|
| A0-LO10 | {_fmt(ric_a0_ratio.get('mean_ic'))} | {_fmt(ric_a0_ratio.get('icir'))} | {_fmt(ric_a0_ratio.get('nw_tstat'))} | {ric_a0_ratio.get('n_days')} | {_fmt(ric_a0_simple.get('mean_ic'))} | {_fmt(gap_a0.get('mean_gap'), 5)} | {_pct(gap_a0.get('pct_gap_pos'))} | {_fmt(gap_a0.get('nw_t'))} |
| RETSTD | {_fmt(ric_retstd_ratio.get('mean_ic'))} | {_fmt(ric_retstd_ratio.get('icir'))} | {_fmt(ric_retstd_ratio.get('nw_tstat'))} | {ric_retstd_ratio.get('n_days')} | {_fmt(ric_retstd_simple.get('mean_ic'))} | {_fmt(gap_retstd.get('mean_gap'), 5)} | {_pct(gap_retstd.get('pct_gap_pos'))} | {_fmt(gap_retstd.get('nw_t'))} |

## RETSTD label-shuffle null (vs binary y)

Verdict **{null.get('verdict')}** (bias_pass={null.get('bias_pass')}, skill_pass={null.get('skill_pass')}, n_violate={null.get('n_violate')}, n_folds={null.get('n_folds')}).

| fold | n | null mean | SD | p95 | real RankIC | bias_ok | exceeds p95 |
|------|---|-----------|----|-----|-------------|---------|-------------|
{chr(10).join(null_rows) if null_rows else "| — | | | | | | | |"}

### Costless benchmarks (not viability inputs)

- EW PIT top-40 (daily rebalanced, costless): {_bench(ew)}
- BTC B&H: {_bench(btc)}

## Construction notes

{extra.get('construction')}

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified by this run. Frozen A0 Huber scores and `features_labeled.parquet` are read-only. RETSTD-LO is a parallel A/B test. No outcome here rewrites the system card.

Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. Scheduled jobs created: false. RETSTD fallback 500 trees: {extra.get('used_fixed')}. Mean label rate: {_fmt(extra.get('mean_label_rate'), 4)}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
