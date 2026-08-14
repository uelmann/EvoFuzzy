"""BTC-BEATER Phase 0/1 reports and chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import PHASE0_GATE, PHASE1_LABEL


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


def write_phase0(path: Path, *, schema, graveyard, sample, quality, agree, gate, extra) -> str:
    lines = [
        "# BTC-BEATER Phase 0 — dataset audit",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** New project. Frozen COMBO v2.0-combo-final is untouched.",
        "",
        "## Pre-registered gate",
        "",
        f"> {PHASE0_GATE}",
        "",
        "## Mechanical verdict",
        "",
        f"- **{gate.get('verdict')}**",
        f"- Sample OK need ≥ {_pct(0.8)} of n={gate.get('scan', [{}])[0].get('sample_n') if gate.get('scan') else '?'} "
        f"(80% of the 30-coin historical top-200 sample).",
        f"- 2018–2020 FICTION: {gate.get('fiction_2018_2020')}. BLOCKED: {gate.get('blocked')}.",
        f"- Usable start: `{gate.get('usable_from')}`.",
        "",
        f"Source file: `{extra.get('source_path')}`. Method for PIT score: **{extra.get('pit_method')}**.",
        "",
        "## Schema",
        "",
        f"- rows={schema.get('n_rows')} ids={schema.get('n_ids')} symbols={schema.get('n_symbols')} slugs={schema.get('n_slugs')}",
        f"- date range {schema.get('date_min')} → {schema.get('date_max')}",
        f"- columns: `{schema.get('columns')}`",
        f"- null fractions: `{schema.get('null_frac')}`",
        f"- symbols mapping to >1 id: {schema.get('dup_symbol_ids')}; ids with >1 symbol: {schema.get('dup_id_symbols')}",
        "",
        "## Graveyard — named dead/collapsed assets",
        "",
        "| query | present | symbol | slug | first | last | n | terminal | event | crash window |",
        "|-------|---------|--------|------|-------|------|---|----------|-------|--------------|",
    ]
    for r in graveyard:
        lines.append(
            f"| {r.get('query')} | {r.get('present')} | {r.get('symbol')} | {r.get('slug')} "
            f"| {r.get('first')} | {r.get('last')} | {r.get('n')} | {r.get('terminal')} "
            f"| {r.get('event')} | {r.get('crash_note')} |"
        )
    n_sample = len(sample)
    n_present = n_sample  # sample drawn from present ids
    lines += [
        "",
        f"## Historical top-200 sample (n={n_sample}, seed=42)",
        "",
        f"Drawn from the union of year-end mcap top-200 in 2018/2019/2020 (BTC, stables, wrapped excluded).",
        "",
        "| id | symbol | name | slug | first | last | n | gap_frac | terminal | in_years |",
        "|----|--------|------|------|-------|------|---|----------|----------|----------|",
    ]
    data_end = extra.get("data_end")
    for r in sample:
        lines.append(
            f"| {r.get('id')} | {r.get('symbol')} | {r.get('name')} | {r.get('slug')} "
            f"| {r.get('first')} | {r.get('last')} | {r.get('n')} | {_fmt(r.get('gap_frac'), 3)} "
            f"| {r.get('terminal')} | {r.get('in_years')} |"
        )
    n_ended = sum(1 for r in sample if r.get("terminal") == "ENDED")
    n_surv = sum(1 for r in sample if r.get("terminal") == "SURVIVOR")
    lines += [
        "",
        f"Graveyard one-liner input: **{n_present}/{max(n_sample, 30)} present** in the drawn sample "
        f"(survivors={n_surv}, ended={n_ended}). Named-list misses are in the table above.",
        "",
        "## PIT reconstruction",
        "",
        f"Trailing {extra.get('dv_window')}d median dollar volume, fallback mcap. "
        f"Stables/wrapped excluded. Files: `universe/btcb_top50_pit.parquet`, `universe/btcb_top100_pit.parquet`.",
        f"PIT method: **{extra.get('pit_method')}**. "
        f"Dates with ≥50 ranked names: {extra.get('pit_n50')} / {extra.get('pit_ndates')}.",
        "",
        "## Data quality",
        "",
        f"- Stables in panel: {quality.get('stables_in_panel')}",
        f"- Wrapped in panel: {quality.get('wrapped_in_panel')}",
        f"- Duplicate tickers: {quality.get('dup_tickers')}",
        f"- Redenomination suspects (|daily ret| > 5): n={quality.get('n_redenom_suspects')}",
        f"- Gap fraction p50={_fmt(quality.get('gap_p50'), 3)} p90={_fmt(quality.get('gap_p90'), 3)} "
        f"n(gap>5%)={quality.get('n_high_gap')}",
        "",
        "### Agreement vs Binance daily closes (overlapping liquid coins)",
        "",
        f"Median return correlation = **{_fmt(agree.get('median_corr'), 4)}** "
        f"(flag suspect if < 0.99: **{agree.get('suspect')}**). Compared n={agree.get('n_compared')}.",
        "",
        "| symbol | n | corr | max\\|Δr\\| | mean\\|Δr\\| |",
        "|--------|---|------|---------|----------|",
    ]
    for r in agree.get("rows") or []:
        if not r.get("present"):
            lines.append(f"| {r.get('symbol')} | — | missing | — | — |")
            continue
        lines.append(
            f"| {r.get('symbol')} | {r.get('n')} | {_fmt(r.get('corr'), 4)} "
            f"| {_fmt(r.get('max_abs_diff'), 4)} | {_fmt(r.get('mean_abs_diff'), 5)} |"
        )
    lines += [
        "",
        "## Usable-start scan (month starts)",
        "",
        "| D | sample_ok/n | sample_frac | PIT frac | pass |",
        "|---|-------------|-------------|----------|------|",
    ]
    for r in gate.get("scan") or []:
        lines.append(
            f"| {r.get('D')} | {r.get('sample_ok')}/{r.get('sample_n')} | {_pct(r.get('sample_frac'))} "
            f"| {_pct(r.get('pit_frac'))} | {r.get('pass')} |"
        )
    lines += [
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU=false. COMBO untouched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def write_phase1(path: Path, *, naive, control, gate, extra) -> str:
    skipped = extra.get("skipped")
    lines = [
        "# BTC-BEATER Phase 1 — dumb benchmark the ML must beat",
        "",
        "**BACKTEST ONLY.** Parameters frozen a priori. No sweeps. COMBO untouched.",
        "",
        "## Pre-registered label",
        "",
        f"> {PHASE1_LABEL}",
        "",
    ]
    if skipped:
        lines += [
            "## Mechanical verdict",
            "",
            f"- Phase 1 **SKIPPED** because Phase 0 is {gate.get('verdict')}.",
            "",
        ]
        text = "\n".join(lines) + "\n"
        path.write_text(text)
        return text
    live = "LIVE BENCHMARK" if naive.get("live_benchmark") else "NOT A LIVE BENCHMARK"
    lines += [
        "## Mechanical verdict",
        "",
        f"- **NAIVE-ROTATION is a {live}**",
        f"- Relative-line Sharpe (book/BTC) = {_fmt(naive.get('rel_sharpe'))} (need > 0: {bool(naive.get('rel_sharpe', 0) > 0)})",
        f"- Book total return = {_pct(naive.get('book_total'))} vs BTC B&H {_pct(naive.get('btc_total'))} "
        f"(need book ≥ BTC: {bool(naive.get('book_total', 0) >= naive.get('btc_total', 1e9))})",
        f"- Usable window: {naive.get('start')} → {naive.get('end')} (n={naive.get('n_days')})",
        "",
        "These numbers are the floor every later ML phase must beat net of costs.",
        "",
        "## Headline vs BTC B&H",
        "",
        "| book | total | CAGR | USD Sharpe | MaxDD | rel CAGR | rel Sharpe | avg %BTC | ann TO |",
        "|------|-------|------|------------|-------|----------|------------|----------|--------|",
        f"| naive rotation | {_pct(naive.get('book_total'))} | {_pct(naive.get('book_cagr'))} "
        f"| {_fmt(naive.get('book_sharpe'))} | {_pct(naive.get('maxdd'))} | {_pct(naive.get('rel_cagr'))} "
        f"| {_fmt(naive.get('rel_sharpe'))} | {_pct(naive.get('avg_w_btc'))} | {_fmt(naive.get('ann_turnover'), 2)} |",
        f"| BTC B&H | {_pct(naive.get('btc_total'))} | {_pct(naive.get('btc_cagr'))} "
        f"| {_fmt(naive.get('btc_sharpe'))} | {_pct(naive.get('btc_maxdd'))} | 0 | 0 | 100% | 0 |",
        f"| 100% BTC control | {_pct(control.get('book_total'))} | {_pct(control.get('book_cagr'))} "
        f"| {_fmt(control.get('book_sharpe'))} | {_pct(control.get('maxdd'))} | {_fmt(control.get('rel_sharpe'), 4)} "
        f"| {_fmt(control.get('rel_cagr'), 4)} | {_pct(control.get('avg_w_btc'))} | {_fmt(control.get('ann_turnover'), 4)} |",
        "",
        "The 100% BTC control should reproduce B&H (relative line ≈ 1, rel Sharpe ≈ 0).",
        "",
        "## Per-cycle",
        "",
        "| cycle | n | book tot | BTC tot | book Sharpe | rel CAGR | rel Sharpe | MaxDD | avg %BTC |",
        "|-------|---|----------|---------|-------------|----------|------------|-------|----------|",
    ]
    for name, c in (naive.get("cycles") or {}).items():
        lines.append(
            f"| {name} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('btc_total'))} "
            f"| {_fmt(c.get('book_sharpe'))} | {_pct(c.get('rel_cagr'))} | {_fmt(c.get('rel_sharpe'))} "
            f"| {_pct(c.get('maxdd'))} | {_pct(c.get('avg_w_btc'))} |"
        )
    by = naive.get("by_year_sharpe") or {}
    lines += [
        "",
        "USD Sharpe by calendar year: " + ", ".join(f"{y}={_fmt(s)}" for y, s in sorted(by.items())),
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU=false.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_benchmark(naive: dict, start: str | None, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq = naive.get("equity")
    eqb = naive.get("equity_btc")
    rel = naive.get("rel_equity")
    if not isinstance(eq, pd.Series) or not isinstance(eqb, pd.Series):
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    axes[0].plot(eq.index, eq.values, label="naive rotation", lw=1.4)
    axes[0].plot(eqb.index, eqb.values, label="BTC B&H", lw=1.3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Equity (log, rebased)")
    axes[0].set_title("BTC-BEATER Phase 1 — naive rotation vs BTC")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)
    if isinstance(rel, pd.Series) and len(rel):
        axes[1].plot(rel.index, rel.values, color="#54A24B", lw=1.4, label="book / BTC")
        axes[1].axhline(1.0, color="black", lw=0.6)
        axes[1].set_ylabel("Relative equity")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    if start:
        t0 = pd.Timestamp(start, tz="UTC")
        for ax in axes:
            ax.axvline(t0, color="#E45756", ls="--", lw=1.0, alpha=0.8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
