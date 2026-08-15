"""ALPHAMINE-LO report, JSON, and equity chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphamine.constants import IMPROVE_CRITERION, VIABILITY_CRITERION
from alphamine.eval import _as_utc, _sharpe, cagr_maxdd, window_slice


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
    mine: pd.Series,
    ew: pd.Series | None,
    btc: pd.Series | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = [(a0, "A0-LO"), (mine, "MINE-LO")]
    if isinstance(ew, pd.Series) and len(ew):
        series.append((ew, "EW PIT top-40 (costless, informational)"))
    if isinstance(btc, pd.Series) and len(btc):
        series.append((btc, "BTC B&H (not held, not a gate)"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    end = None
    for rets, lab in series:
        if rets is None or not len(rets):
            continue
        d, y = _eq_from_rets(rets)
        end = d.max() if end is None else max(end, d.max())
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("ALPHAMINE-LO vs A0-LO (long-only top-10)")
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
        f"{_fmt(book.get('max_abs_btc_weight'), 6)} | {top_s} |"
    )


def write_report(
    path: Path,
    *,
    frozen_hash: str,
    a0_book: dict,
    mine_book: dict,
    improve: dict,
    verdict: dict,
    null: dict,
    ric_a0: dict,
    ric_mine: dict,
    gap_a0: dict,
    gap_mine: dict,
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

    formulas = extra.get("formula_examples") or []
    form_lines = "\n".join(f"- `{x}`" for x in formulas[:12]) if formulas else "- (none kept)"

    text = f"""# ALPHAMINE-LO — formulaic features vs A0 (long-only)

**BACKTEST AND ANALYSIS ONLY.** Same LightGBM, same always-in top-10 long-only book. MINE adds fold-selected OHLCV formulas. No schedules, no live components. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**.

**Frozen A0 SHA256 (features/config):** `{frozen_hash}`
**Mandate:** long alts. Never BTC. Never a BTC hedge. Residual is cash.
**Horizon:** h=10. Execution universe: PIT top-40 excluding BTC. Always long top 10 by score; full tranche budget.

## Pre-registered improvement statement

> {IMPROVE_CRITERION}

## Pre-registered viability statement

> {VIABILITY_CRITERION}

Verdicts below are mechanical. No post-hoc adjustment.

## Mechanical verdict

- **Feature A/B: {improve.get('verdict')}** — RankIC MINE={_fmt(improve.get('ric_mine'))} vs A0={_fmt(improve.get('ric_a0'))} (pass={improve.get('pass_ric')}); top−universe gap MINE={_fmt(improve.get('gap_mine'), 5)} vs A0={_fmt(improve.get('gap_a0'), 5)} (pass={improve.get('pass_gap')}); Sharpe MINE={_fmt(improve.get('sharpe_mine'))} vs A0={_fmt(improve.get('sharpe_a0'))} (pass={improve.get('pass_sharpe')}); BTC0={improve.get('pass_btc0')}; null={improve.get('null_verdict')} pass={improve.get('pass_null')}.
- **ALPHAMINE-LO (MINE book): {verdict.get('verdict')}** — full Sharpe={_fmt(verdict.get('sharpe_full'))} (need ≥ {_fmt(verdict.get('need_full'))}, pass={verdict.get('pass_full')}); trail-18m Sharpe={_fmt(verdict.get('sharpe_trail18m'))} (need ≥ {_fmt(verdict.get('need_trail'))}, pass={verdict.get('pass_trail')}); total={_pct(verdict.get('total_return'))} (need > 0, pass={verdict.get('pass_total')}); avg gross={_fmt(verdict.get('avg_gross'))} (need ≥ {_fmt(verdict.get('need_gross'))}, pass={verdict.get('pass_gross')}); BTC weight ≡ 0 pass={verdict.get('pass_btc0')}; null={verdict.get('null_verdict')} pass={verdict.get('pass_null')}.

## Headline books

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | CAGR | MaxDD | total | avg #longs | avg gross | % flat | funding | costs | ann TO | forced | BTC max |w| | top-5 name PnL |
|------|------|-----------|------|------|------|------|------|------|-------|-------|------------|-----------|--------|---------|-------|--------|--------|----------------|----------------|
{_book_row("A0-LO", a0_book)}
{_book_row("MINE-LO", mine_book)}

## RankIC and long-minus-universe (PIT top-40, BTC dropped)

| arm | RankIC | ICIR | NW-t | n days | top−uni mean | % gap>0 | gap NW-t |
|-----|--------|------|------|--------|--------------|---------|----------|
| A0 | {_fmt(ric_a0.get('mean_ic'))} | {_fmt(ric_a0.get('icir'))} | {_fmt(ric_a0.get('nw_tstat'))} | {ric_a0.get('n_days')} | {_fmt(gap_a0.get('mean_gap'), 5)} | {_pct(gap_a0.get('pct_gap_pos'))} | {_fmt(gap_a0.get('nw_t'))} |
| MINE | {_fmt(ric_mine.get('mean_ic'))} | {_fmt(ric_mine.get('icir'))} | {_fmt(ric_mine.get('nw_tstat'))} | {ric_mine.get('n_days')} | {_fmt(gap_mine.get('mean_gap'), 5)} | {_pct(gap_mine.get('pct_gap_pos'))} | {_fmt(gap_mine.get('nw_t'))} |

## MINE label-shuffle null

Verdict **{null.get('verdict')}** (bias_pass={null.get('bias_pass')}, skill_pass={null.get('skill_pass')}, n_violate={null.get('n_violate')}, n_folds={null.get('n_folds')}).

| fold | n | null mean | SD | p95 | real RankIC | bias_ok | exceeds p95 |
|------|---|-----------|----|-----|-------------|---------|-------------|
{chr(10).join(null_rows) if null_rows else "| — | | | | | | | |"}

### Costless benchmarks (not viability inputs)

- EW PIT top-40 (daily rebalanced, costless): {_bench(ew)}
- BTC B&H (not held): {_bench(btc)}

## Kept formulas (examples)

{form_lines}

Mean formulas per fold: {_fmt(extra.get('mean_n_formulas'))}. Folds with 0 formulas: {extra.get('n_empty_formula_folds')}.

## Construction notes

{extra.get('construction')}

## Frozen products are unchanged

COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified by this run. ALPHAMINE-LO is a parallel A/B test. No outcome here rewrites the system card.

Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. Scheduled jobs created: false. A0 fallback 500 trees: {extra.get('used_fixed_a0')}. MINE fallback 500 trees: {extra.get('used_fixed_mine')}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
