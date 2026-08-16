"""Phase 5 CS-ATTN reports, charts, ledger footnote."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    PHASE2_CYCLES,
    PHASE5_CRITERION,
    PHASE5_NULL_GATE,
)


def _fmt(x, nd=4):
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


def _cell(blob, window, key):
    w = (blob or {}).get(window) or {}
    k = w.get(key) or {}
    return k


def _metric_row(name, blob, window: str) -> str:
    t = _cell(blob, window, "tail_ic_top")
    b = _cell(blob, window, "tail_ic_bot")
    o = _cell(blob, window, "overlap")
    m = _cell(blob, window, "monster")
    r = _cell(blob, window, "rankic")
    return (
        f"| {name} | {_fmt(t.get('mean'))} | {_fmt(t.get('nw_t'), 2)} | {_fmt(b.get('mean'))} | "
        f"{_fmt(b.get('nw_t'), 2)} | {_fmt(o.get('mean'))} | {_fmt(m.get('mean'))} | "
        f"{_fmt(r.get('mean'))} | {t.get('n', '')} |"
    )


def write_phase5(
    path: Path,
    *,
    audit_summary: dict,
    config: dict,
    grid: dict,
    null: dict,
    verdict: dict,
    books: dict,
    gpu: dict,
    extra: dict,
) -> None:
    lines = [
        "# BTC-BEATER Phase 5 — CS-ATTN v0",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** No schedules, no live components, nothing adopted.",
        "Frozen GBM / 2.c cache / COMBO / SPREAD-LS / LONG-TIDE untouched (read-only).",
        "One architecture config, zero search. GPU = H100 (3-seed parallel after A10G start; architecture unchanged).",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE5_CRITERION}",
        "",
        "## §B null (verbatim, before results)",
        "",
        f"> {PHASE5_NULL_GATE}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Mechanical verdict",
        "",
        f"- **CS-ATTN = `{verdict.get('verdict')}`**",
        f"- failed clauses = `{verdict.get('failed_clauses')}`",
        f"- (a) Δ tail-IC(top-half) = `{_fmt(verdict.get('delta_tail_ic_top'))}` "
        f"(need ≥ `{verdict.get('need_delta_tail_ic')}`); "
        f"Δ overlap = `{_fmt(verdict.get('delta_overlap'))}` "
        f"(need ≥ `{verdict.get('need_delta_overlap')}`); pass=`{verdict.get('clause_a')}`",
        f"- (b) seed dispersion max−min = `{_fmt(verdict.get('seed_dispersion'))}` "
        f"(need ≤ `{verdict.get('need_disp')}`); per-seed = `{verdict.get('per_seed_tail_ic_top')}`; "
        f"pass=`{verdict.get('clause_b')}`",
        f"- (c) null = `{verdict.get('null_verdict')}` pass=`{verdict.get('clause_c')}`",
        "",
    ]
    if verdict.get("record_ceiling"):
        lines += [
            f"**Ledger sentence (clause a failed, dispersion passed):** `{verdict.get('ceiling_sentence')}`.",
            "",
        ]
    lines += [
        "Mechanical, no post-hoc adjustment. Nothing adopted.",
        "",
        "## Plain language",
        "",
        extra.get("plain", "See verdict."),
        "",
        "## Panel audit summary",
        "",
        f"- hourly rows=`{audit_summary.get('n_rows')}` ids=`{audit_summary.get('n_ids')}` "
        f"span=`{audit_summary.get('ts_min')}`→`{audit_summary.get('ts_max')}`",
        f"- alignment median |Δ| bps=`{audit_summary.get('alignment_median_abs_bps')}` "
        f"pass=`{audit_summary.get('alignment_pass')}` violations=`{audit_summary.get('alignment_violations_n')}`",
        f"- sources=`{audit_summary.get('source_id_counts')}` duplicates=`{audit_summary.get('n_duplicate_bars')}`",
        f"- seq cache n=`{(extra.get('seq_meta') or {}).get('n_rows')}` "
        f"nbytes=`{(extra.get('seq_meta') or {}).get('nbytes')}`",
        "",
        "## Frozen config dump",
        "",
        "```",
        str(config),
        "```",
        "",
        "## Tail-metric grid (floored PIT top-100, h=14, Binance-listed)",
        "",
        "### Full OOS",
        "",
        "| signal | tail-IC top | NW-t | tail-IC bot | NW-t | overlap | monster top-3 | RankIC | n |",
        "|--------|-------------|------|-------------|------|---------|---------------|--------|---|",
    ]
    order = ["gbm"] + [k for k in grid if k.startswith("seed")] + ["ensemble"]
    if "manuel" in grid:
        order.append("manuel")
    labels = {
        "gbm": "frozen GBM spread",
        "ensemble": "CS-ATTN 3-seed ensemble",
        "manuel": "MANUEL-SCORE Reading A",
    }
    for k in order:
        if k not in grid:
            continue
        name = labels.get(k, k.replace("seed", "CS-ATTN seed "))
        lines.append(_metric_row(name, grid[k], "full"))
    lines += [
        "",
        "### Trailing 18m",
        "",
        "| signal | tail-IC top | NW-t | tail-IC bot | NW-t | overlap | monster top-3 | RankIC | n |",
        "|--------|-------------|------|-------------|------|---------|---------------|--------|---|",
    ]
    for k in order:
        if k not in grid:
            continue
        name = labels.get(k, k.replace("seed", "CS-ATTN seed "))
        lines.append(_metric_row(name, grid[k], "trail18m"))
    lines += [
        "",
        "### Per-cycle tail-IC(top-half) mean",
        "",
        "| signal | " + " | ".join(c[0] for c in PHASE2_CYCLES) + " |",
        "|--------|" + "|".join("---" for _ in PHASE2_CYCLES) + "|",
    ]
    for k in order:
        if k not in grid:
            continue
        name = labels.get(k, k.replace("seed", "CS-ATTN seed "))
        cyc = (grid[k] or {}).get("cycles") or {}
        cells = [_fmt((cyc.get(c[0]) or {}).get("tail_ic_top", {}).get("mean")) for c in PHASE2_CYCLES]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Per-cycle top-decile overlap",
        "",
        "| signal | " + " | ".join(c[0] for c in PHASE2_CYCLES) + " |",
        "|--------|" + "|".join("---" for _ in PHASE2_CYCLES) + "|",
    ]
    for k in order:
        if k not in grid:
            continue
        name = labels.get(k, k.replace("seed", "CS-ATTN seed "))
        cyc = (grid[k] or {}).get("cycles") or {}
        cells = [_fmt((cyc.get(c[0]) or {}).get("overlap", {}).get("mean")) for c in PHASE2_CYCLES]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Crude 14d book (information only, not adopted)",
        "",
        "| signal | total | CAGR | MaxDD | Sharpe | n |",
        "|--------|-------|------|-------|--------|---|",
    ]
    for name, b in (books or {}).items():
        lines.append(
            f"| {name} | {_pct(b.get('total'))} | {_pct(b.get('cagr'))} | {_pct(b.get('maxdd'))} "
            f"| {_fmt(b.get('sharpe'), 3)} | {b.get('n_formations', b.get('n_days', ''))} |"
        )
    cells = (null or {}).get("cells") or []
    lines += [
        "",
        "## Null tables (folds {5, 21} × 10 within-date shuffles, seed 42)",
        "",
        f"Verdict: `{null.get('verdict')}` bias_pass=`{null.get('bias_pass')}` "
        f"skill_pass=`{null.get('skill_pass')}` n_violate=`{null.get('n_violate')}` "
        f"n_exceed=`{null.get('n_exceed')}` / need `{null.get('need_exceed')}`.",
        "",
        "| fold | n | null mean | SD | 2·SE | |mean| | bias | real IC | null 95th | exceeds |",
        "|------|---|-----------|----|------|--------|------|---------|-----------|---------|",
    ]
    for c in cells:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'))} | {_fmt(c.get('sd'))} | "
            f"{_fmt(c.get('bias_lim'))} | {_fmt(abs(c.get('mean')) if c.get('mean') is not None else float('nan'))} | "
            f"{'PASS' if c.get('bias_ok') else 'FAIL'} | {_fmt(c.get('real_ic'))} | {_fmt(c.get('p95'))} | "
            f"{c.get('exceeds')} |"
        )
    lines += [
        "",
        "## GPU spend log",
        "",
        f"- gpu used = `{gpu.get('gpu_used')}` type=`{gpu.get('gpu_type')}` "
        f"parallel=`{gpu.get('parallelism', gpu.get('parallel'))}`",
        f"- A10G sunk = `{_fmt(gpu.get('a10g_sunk_usd'), 2)}` "
        f"({_fmt(gpu.get('a10g_hours'), 2)} h × `{gpu.get('a10g_usd_per_hour', 1.10)}`/h)",
        f"- H100 hours = `{_fmt(gpu.get('h100_hours'), 3)}` × "
        f"`{gpu.get('usd_per_hour')}`/h = `{_fmt(gpu.get('h100_usd'), 2)}`",
        f"- wall seconds = `{gpu.get('gpu_seconds')}` total GPU-hours=`{_fmt(gpu.get('gpu_hours'), 3)}`",
        f"- USD total = `{_fmt(gpu.get('usd'), 2)}` cap=`{gpu.get('cap_usd')}` "
        f"aborted=`{gpu.get('aborted')}` reason=`{gpu.get('abort_reason')}`",
        f"- folds completed = `{gpu.get('folds_done')}` seeds done=`{gpu.get('seeds_done')}` "
        f"null jobs=`{gpu.get('null_done')}`",
        "",
        f"Elapsed total `{extra.get('elapsed_sec')}` s. Frozen products untouched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def plot_tail_ic_bars(grid: dict, out: Path) -> None:
    labels, vals = [], []
    mapping = [("gbm", "GBM")]
    for k in sorted(grid):
        if k.startswith("seed"):
            mapping.append((k, k.replace("seed", "s")))
    mapping.append(("ensemble", "ENS"))
    for k, lab in mapping:
        if k not in grid:
            continue
        v = _cell(grid[k], "full", "tail_ic_top").get("mean")
        labels.append(lab)
        vals.append(float(v) if v is not None and np.isfinite(v) else np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["0.45" if lab == "GBM" else ("steelblue" if lab == "ENS" else "0.7") for lab in labels]
    ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="0.2")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("full-OOS tail-IC (top half)")
    ax.set_title("CS-ATTN v0 — tail-IC(top) vs frozen GBM")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_overlap_cycles(grid: dict, out: Path) -> None:
    cycles = [c[0] for c in PHASE2_CYCLES]
    series = []
    names = []
    for k, lab in (("gbm", "GBM"), ("ensemble", "ENS")):
        if k not in grid:
            continue
        cyc = (grid[k] or {}).get("cycles") or {}
        series.append([float((cyc.get(c) or {}).get("overlap", {}).get("mean") or np.nan) for c in cycles])
        names.append(lab)
    if not series:
        return
    x = np.arange(len(cycles))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for i, (ys, lab) in enumerate(zip(series, names)):
        ax.bar(x + (i - 0.5) * w, ys, width=w, label=lab, edgecolor="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels(cycles)
    ax.set_ylabel("top-decile overlap")
    ax.set_title("CS-ATTN v0 — overlap per cycle")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def update_ledger_phase5(ledger: Path, verdict: dict) -> None:
    if not ledger.exists():
        return
    text = ledger.read_text()
    if "## BTC-BEATER Phase 5 CS-ATTN" in text:
        return
    block = [
        "",
        "## BTC-BEATER Phase 5 CS-ATTN v0",
        "",
        "Hourly panel + cross-sectional attention, tail-weighted twin heads. Analysis only. Nothing adopted.",
        "",
        f"**CS-ATTN = {verdict.get('verdict')}.** failed clauses `{verdict.get('failed_clauses')}`. "
        f"Δ tail-IC(top) `{verdict.get('delta_tail_ic_top')}` Δ overlap `{verdict.get('delta_overlap')}` "
        f"seed disp `{verdict.get('seed_dispersion')}` null `{verdict.get('null_verdict')}`.",
        "",
    ]
    if verdict.get("record_ceiling"):
        block.append(f"**{verdict.get('ceiling_sentence')}**")
        block.append("")
    block.append("Mechanical, no post-hoc adjustment. Frozen products untouched.")
    block.append("")
    ledger.write_text(text.rstrip() + "\n" + "\n".join(block))
