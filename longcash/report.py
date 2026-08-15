"""LONG-CASH report, JSON, and equity chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from longcash.constants import VIABILITY_CRITERION
from longcash.eval import _as_utc, _sharpe, cagr_maxdd, window_slice


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
    book: pd.Series,
    ew: pd.Series | None,
    btc: pd.Series | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = [(book, "LONG-CASH")]
    if isinstance(ew, pd.Series) and len(ew):
        series.append((ew, "EW PIT top-40 (costless, informational)"))
    if isinstance(btc, pd.Series) and len(btc):
        series.append((btc, "BTC B&H (not held, not a gate)"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    end = None
    for rets, lab in series:
        d, y = _eq_from_rets(rets)
        end = d.max() if end is None else max(end, d.max())
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("LONG-CASH vs cash-financed benchmarks")
    axes[0].set_ylabel("Equity (rebased, log)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for rets, lab in series:
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
    axes[1].set_ylabel("Equity (rebased, log)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, which="both")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    path: Path,
    *,
    frozen_hash: str,
    book: dict,
    verdict: dict,
    null: dict,
    raw_a0: dict,
    raw_new: dict,
    benches: dict,
    extra: dict,
) -> str:
    by = book.get("net_sharpe_by_year") or {}
    years = sorted(int(y) for y in by.keys())
    yhead = " | ".join(str(y) for y in years) if years else "—"
    ycells = " | ".join(_fmt(by.get(y)) for y in years) if years else "—"
    top = book.get("top5_names") or []
    top_s = ", ".join(f"{t['symbol']}={_fmt(t['pnl'], 4)}" for t in top) if top else "—"
    lines = [
        "# LONG-CASH — cash-financed alt-long (parallel product)",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** New LightGBM heads on frozen A0 features. "
        "No schedules, no live components. COMBO, SPREAD-LS, and LONG-TIDE are **untouched**.",
        "",
        f"**Frozen A0 SHA256 (features/config):** `{frozen_hash}`",
        "**Mandate:** long alts or cash. Never BTC. Never a BTC hedge.",
        f"**Horizon:** h=10. Execution universe: PIT top-40 excluding BTC. "
        f"Enter if `er_hat > 0` and `p_up > 0.5`; min 3 / max 10 names; full tranche budget.",
        "",
        "## Pre-registered viability statement",
        "",
        f"> {VIABILITY_CRITERION}",
        "",
        "Verdicts below are mechanical. No post-hoc adjustment.",
        "",
        "## Mechanical verdict",
        "",
        f"- **LONG-CASH: {verdict.get('verdict')}** — "
        f"full Sharpe={_fmt(verdict.get('sharpe_full'))} "
        f"(need ≥ {_fmt(verdict.get('need_full'))}, pass={verdict.get('pass_full')}); "
        f"trail-18m Sharpe={_fmt(verdict.get('sharpe_trail18m'))} "
        f"(need ≥ {_fmt(verdict.get('need_trail'))}, pass={verdict.get('pass_trail')}); "
        f"total={_pct(verdict.get('total_return'))} (need > 0, pass={verdict.get('pass_total')}); "
        f"avg gross={_fmt(verdict.get('avg_gross'))} "
        f"(need ≥ {_fmt(verdict.get('need_gross'))}, pass={verdict.get('pass_gross')}); "
        f"BTC weight ≡ 0 pass={verdict.get('pass_btc0')}; "
        f"null={verdict.get('null_verdict')} pass={verdict.get('pass_null')}.",
        "",
        "## Headline book",
        "",
        f"| book | full | trail-18m | {yhead} | CAGR | MaxDD | total | avg #longs | avg gross | % flat | funding | costs | ann TO | forced | BTC max |w| | top-5 name PnL |",
        f"|------|------|-----------|{('|'.join(['------'] * max(len(years), 1)))}|------|-------|-------|------------|-----------|--------|---------|-------|--------|--------|----------------|----------------|",
        (
            f"| LONG-CASH | {_fmt(book.get('net_sharpe_full'))} | {_fmt(book.get('net_sharpe_trail18m'))} "
            f"| {ycells} | {_pct(book.get('net_cagr'))} | {_pct(book.get('max_drawdown'))} "
            f"| {_pct(book.get('total_return'))} | {_fmt(book.get('avg_n_long'), 2)} "
            f"| {_fmt(book.get('avg_gross_deployed'), 3)} | {_pct(book.get('pct_flat_days'))} "
            f"| {_fmt(book.get('funding_total_pnl'), 4)} | {_fmt(book.get('cost_drag'), 4)} "
            f"| {_fmt(book.get('ann_turnover'), 2)} | {book.get('n_forced_exits', 0)} "
            f"| {_fmt(book.get('max_abs_btc_weight'), 6)} | {top_s} |"
        ),
        "",
        f"Mean vs cash (ann.)={_fmt(book.get('mean_ann'))}; NW-t vs cash (lag=10)={_fmt(book.get('nw_t_vs_cash'))}; "
        f"trail NW-t={_fmt(book.get('nw_t_vs_cash_trail18m'))}.",
        "",
        "## Head-R label-shuffle null",
        "",
        f"Verdict **{null.get('verdict')}** (bias_pass={null.get('bias_pass')}, "
        f"skill_pass={null.get('skill_pass')}, n_violate={null.get('n_violate')}, "
        f"n_folds={null.get('n_folds')}).",
        "",
        "| fold | n | null mean | SD | p95 | real RankIC | bias_ok | exceeds p95 |",
        "|------|---|-----------|----|-----|-------------|---------|-------------|",
    ]
    for c in null.get("cells") or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'), 4)} | {_fmt(c.get('sd'), 4)} "
            f"| {_fmt(c.get('p95'), 4)} | {_fmt(c.get('real'), 4)} | {c.get('bias_ok')} | {c.get('exceeds_p95')} |"
        )
    lines += [
        "",
        "## Raw-material snapshot (informational, not a gate)",
        "",
        "Top-quintile mean of 10-day simple USDT return on PIT top-40 (BTC dropped).",
        "",
        "| signal | n days | % days top>0 | mean top | NW-t |",
        "|--------|--------|--------------|----------|------|",
        (
            f"| frozen A0 score | {raw_a0.get('n_days', '')} | {_pct(raw_a0.get('pct_top_pos'))} "
            f"| {_fmt(raw_a0.get('mean_top'), 4)} | {_fmt(raw_a0.get('nw_t'))} |"
        ),
        (
            f"| LONG-CASH er_hat | {raw_new.get('n_days', '')} | {_pct(raw_new.get('pct_top_pos'))} "
            f"| {_fmt(raw_new.get('mean_top'), 4)} | {_fmt(raw_new.get('nw_t'))} |"
        ),
        "",
        "### Costless benchmarks (not viability inputs)",
        "",
    ]
    ew = benches.get("ew")
    btc = benches.get("btc")
    if isinstance(ew, pd.Series) and len(ew):
        cagr, maxdd, tot = cagr_maxdd(ew)
        lines.append(
            f"- EW PIT top-40 (daily rebalanced, costless): Sharpe full={_fmt(_sharpe(ew))}, "
            f"trail-18m={_fmt(_sharpe(window_slice(ew, 'trail18m')))}, "
            f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(tot)}."
        )
    if isinstance(btc, pd.Series) and len(btc):
        cagr, maxdd, tot = cagr_maxdd(btc)
        lines.append(
            f"- BTC B&H (not held): Sharpe full={_fmt(_sharpe(btc))}, "
            f"trail-18m={_fmt(_sharpe(window_slice(btc, 'trail18m')))}, "
            f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(tot)}."
        )
    lines += [
        "",
        "## Construction notes",
        "",
        extra.get("construction", ""),
        "",
        "## Frozen products are unchanged",
        "",
        "COMBO v2.0-combo-final, SPREAD-LS BOOK-HYBRID, and LONG-TIDE are not modified by this run. "
        "LONG-CASH is a parallel product. No outcome here rewrites the system card.",
        "",
        f"Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. "
        f"Scheduled jobs created: false. Head-R fallback 500 trees: "
        f"{extra.get('used_fixed_trees', False)}.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
