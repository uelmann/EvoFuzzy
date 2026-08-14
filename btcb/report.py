"""BTC-BEATER Phase 0/1 reports and chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import DEATH_CONVENTION, PHASE0_GATE, PHASE0C_GATE, PHASE1_LABEL


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
        f"- terminal histories: survivors={extra.get('n_survivor')} ended={extra.get('n_ended')} "
        f"(an archive with ended=0 does not retain delisted names — survivorship bias).",
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
        f"Drawn from the union of year-end mcap top-200 in 2018/2019/2020 (BTC, stables, wrapped excluded). "
        f"In-file year-end pool sizes (survivors only): `{extra.get('year_end_top200_n')}`. "
        f"A 2018–2020 top-200 that never appears in this archive cannot enter the sample — the 80% present test is nearly tautological; named-list misses are the real graveyard check.",
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
    live_phrase = "a LIVE BENCHMARK" if naive.get("live_benchmark") else "NOT A LIVE BENCHMARK"
    lines += [
        "## Mechanical verdict",
        "",
        f"- **NAIVE-ROTATION is {live_phrase}**",
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
        f"| {_fmt(control.get('book_sharpe'))} | {_pct(control.get('maxdd'))} | {_fmt(control.get('rel_cagr'), 4)} "
        f"| {_fmt(control.get('rel_sharpe'), 4)} | {_pct(control.get('avg_w_btc'))} | {_fmt(control.get('ann_turnover'), 4)} |",
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


def plot_benchmark(naive: dict, start: str | None, out_path: Path, title: str | None = None) -> None:
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
    axes[0].set_title(title or "BTC-BEATER Phase 1 — naive rotation vs BTC")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)
    if isinstance(rel, pd.Series) and len(rel):
        axes[1].plot(rel.index, rel.values, color="#54A24B", lw=1.4, label="book / BTC")
        axes[1].axhline(1.0, color="black", lw=0.6)
        axes[1].set_ylabel("Relative equity")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    if start:
        t0 = pd.Timestamp(start)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        else:
            t0 = t0.tz_convert("UTC")
        for ax in axes:
            ax.axvline(t0, color="#E45756", ls="--", lw=1.0, alpha=0.8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_phase0c(path: Path, *, schema, graveyard, ended, quality, gate, extra) -> str:
    lines = [
        "# BTC-BEATER Phase 0.c — full-map rebuild + re-audit",
        "",
        "**DATA + ANALYSIS ONLY.** Frozen COMBO v2.0-combo-final is untouched. "
        "The KuCoin-filtered 828-coin archive and its benchmarks are discarded unread.",
        "",
        "## Pre-registered gate v2 (verbatim)",
        "",
        f"> {PHASE0C_GATE}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Mechanical verdict",
        "",
        f"- **{gate.get('verdict')}**",
        f"- BLOCKED: {gate.get('blocked')}. Usable start: `{gate.get('usable_from')}`.",
        f"- Ended-count: **{ended.get('n_ended')}** / {ended.get('n_ids')} "
        f"(histories ending before {ended.get('before')}). "
        + ("**FAIL: ended=0 (survivorship still present).**" if ended.get("fail") else "ended>0 (graveyard retained)."),
        "",
        "## Download provenance + credit guard",
        "",
        f"- Plan: {extra.get('plan')}",
        f"- Credits projected={extra.get('credits_projected')} available={extra.get('credits_available')} "
        f"observed_credit_count={extra.get('credit_count')}",
        f"- HTTP projected remaining={extra.get('http_remaining')} used={extra.get('http_count')} hard_stop={extra.get('hard_stop')}",
        f"- Target ids={extra.get('n_target')} (snapshot union + 828). Cached before OHLCV={extra.get('n_cached')}.",
        f"- Map: n={extra.get('n_map')} active={extra.get('n_active')} inactive={extra.get('n_inactive')} untracked={extra.get('n_untracked')}",
        "",
        "## Schema",
        "",
        f"- rows={schema.get('n_rows')} ids={schema.get('n_ids')} symbols={schema.get('n_symbols')}",
        f"- date range {schema.get('date_min')} → {schema.get('date_max')}",
        f"- extra columns: listing_status, last_available_date. Primary key=cryptocurrency_id.",
        "",
        "## Named graveyard (must be present with terminal dates)",
        "",
        "| query | present | with_terminal | id | symbol | slug | first | last | n | terminal | status | event |",
        "|-------|---------|---------------|----|--------|------|-------|------|---|----------|--------|-------|",
    ]
    for r in graveyard:
        lines.append(
            f"| {r.get('query')} | {r.get('present')} | {r.get('present_with_terminal')} "
            f"| {r.get('id')} | {r.get('symbol')} | {r.get('slug')} | {r.get('first')} | {r.get('last')} "
            f"| {r.get('n')} | {r.get('terminal')} | {r.get('listing_status')} | {r.get('event')} |"
        )
    n_ok = sum(1 for r in graveyard if r.get("present_with_terminal"))
    lines += [
        "",
        f"Graveyard one-liner input: **{n_ok}/{len(graveyard)} present-with-terminal**.",
        "",
        "## Coverage vs external snapshots (top-50/100/200)",
        "",
        f"Threshold 85% on true-top-100, sustained at all later snapshots. PIT method: **{extra.get('pit_method')}**.",
        "",
        "| D | used | top50 | top100 | top200 | pass100 |",
        "|---|------|-------|--------|--------|---------|",
    ]
    for r in gate.get("scan") or []:
        t50 = r.get("top50") or {}
        t100 = r.get("top100") or {}
        t200 = r.get("top200") or {}
        lines.append(
            f"| {r.get('quarter_end')} | {r.get('used_date')} "
            f"| {_pct(t50.get('frac'))} | {_pct(t100.get('frac'))} | {_pct(t200.get('frac'))} "
            f"| {r.get('pass100')} |"
        )
    lines += [
        "",
        f"PIT files: `universe/btcb_top50_pit.parquet`, `universe/btcb_top100_pit.parquet`.",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU=false. COMBO untouched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def write_phase1_v3(path: Path, *, naive, control, gate, extra) -> str:
    lines = [
        "# BTC-BEATER Phase 1 v3 — honest-window naive rotation",
        "",
        "**BACKTEST ONLY.** Parameters frozen a priori. No sweeps. "
        "Old-archive and 2018-circular benchmarks discarded unread. COMBO untouched.",
        "",
        "## Pre-registered label",
        "",
        f"> {PHASE1_LABEL}",
        "",
        "## Death-in-position convention",
        "",
        f"> {DEATH_CONVENTION}",
        "",
    ]
    if extra.get("skipped"):
        lines += [
            "## Mechanical verdict",
            "",
            f"- Phase 1 v3 **SKIPPED** because Phase 0.c is {gate.get('verdict')}.",
            "",
        ]
        text = "\n".join(lines) + "\n"
        path.write_text(text)
        return text
    live_phrase = "a LIVE BENCHMARK" if naive.get("live_benchmark") else "NOT A LIVE BENCHMARK"
    fe = naive.get("forced_exits") or {}
    lines += [
        "## Mechanical verdict",
        "",
        f"- **NAIVE-ROTATION is {live_phrase}**",
        f"- Relative-line Sharpe (book/BTC) = {_fmt(naive.get('rel_sharpe'))} "
        f"(need > 0: {bool((naive.get('rel_sharpe') or 0) > 0)})",
        f"- Book total return = {_pct(naive.get('book_total'))} vs BTC B&H {_pct(naive.get('btc_total'))} "
        f"(need book ≥ BTC: {bool((naive.get('book_total') or 0) >= (naive.get('btc_total') or 1e9))})",
        f"- Usable window: {naive.get('start')} → {naive.get('end')} (n={naive.get('n_days')})",
        f"- Forced exits: n_events={fe.get('n_events')} n_ids={fe.get('n_ids')} "
        f"weight_sum={_fmt(fe.get('weight_sum'), 4)} cost_drag={_fmt(fe.get('cost_drag'), 6)} "
        f"pnl_impact_vs_ghost={_fmt(fe.get('pnl_impact_vs_ghost'), 4)}",
        "",
        "These numbers are the floor every later ML phase must beat net of costs.",
        "",
        "## Headline vs BTC B&H",
        "",
        "| book | total | CAGR | USD Sharpe | MaxDD | rel CAGR | rel Sharpe | avg %BTC | ann TO |",
        "|------|-------|------|------------|-------|----------|------------|----------|--------|",
        f"| naive rotation v3 | {_pct(naive.get('book_total'))} | {_pct(naive.get('book_cagr'))} "
        f"| {_fmt(naive.get('book_sharpe'))} | {_pct(naive.get('maxdd'))} | {_pct(naive.get('rel_cagr'))} "
        f"| {_fmt(naive.get('rel_sharpe'))} | {_pct(naive.get('avg_w_btc'))} | {_fmt(naive.get('ann_turnover'), 2)} |",
        f"| BTC B&H | {_pct(naive.get('btc_total'))} | {_pct(naive.get('btc_cagr'))} "
        f"| {_fmt(naive.get('btc_sharpe'))} | {_pct(naive.get('btc_maxdd'))} | 0 | 0 | 100% | 0 |",
        f"| 100% BTC control | {_pct(control.get('book_total'))} | {_pct(control.get('book_cagr'))} "
        f"| {_fmt(control.get('book_sharpe'))} | {_pct(control.get('maxdd'))} | {_fmt(control.get('rel_cagr'), 4)} "
        f"| {_fmt(control.get('rel_sharpe'), 4)} | {_pct(control.get('avg_w_btc'))} | {_fmt(control.get('ann_turnover'), 4)} |",
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
    lines += [
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU=false.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_coverage(gate: dict, out_path: Path) -> None:
    rows = gate.get("scan") or []
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [pd.Timestamp(r["used_date"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for n, col in ((50, "#4C78A8"), (100, "#F58518"), (200, "#54A24B")):
        ys = [((r.get(f"top{n}") or {}).get("frac") or 0.0) for r in rows]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.3, color=col, label=f"top-{n}")
    ax.axhline(0.85, color="#E45756", ls="--", lw=1.0, label="85%")
    if gate.get("usable_from"):
        t0 = pd.Timestamp(gate["usable_from"])
        ax.axvline(t0, color="#E45756", ls="--", lw=1.0, alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("true-top-N coverage")
    ax.set_title("BTC-BEATER Phase 0.c — snapshot coverage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_benchmark_v3(naive: dict, start: str | None, out_path: Path) -> None:
    plot_benchmark(naive, start, out_path, title="BTC-BEATER Phase 1 v3 — honest window vs BTC")
