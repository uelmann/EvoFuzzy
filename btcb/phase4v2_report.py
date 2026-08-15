"""Phase 4 v2 TAIL ROUND 1 report + charts. Analysis only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btcb.constants import (
    DEATH_CONVENTION,
    PHASE2_CYCLES,
    PHASE4V2_CRITERION,
    PHASE4V2_PI_KILL,
    PHASE4V2_PI_SCOPE,
    PHASE2C_PRED_SHA256,
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


def _delta(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        v = float(x)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{nd}f}"
    except Exception:
        return str(x)


SIGNAL_ORDER = (
    ("frozen_spread", "frozen spread (baseline)"),
    ("rank", "RANK head"),
    ("spread_rank", "SPREAD+RANK"),
    ("spread_pos", "spread retrained +positioning"),
    ("spread_pos_price", "+price-additions"),
    ("full_stack", "full stack"),
)


def _null_table(cells: list, real_key: str) -> list[str]:
    lines = [
        "| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |",
        "|------|---|-----------|--------|------|---------|-----|------|-------------|",
    ]
    for c in cells or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'))} | {_fmt(c.get('center'))} "
            f"| {_fmt(c.get('bias_lim'))} | {c.get('bias_ok')} | {_fmt(c.get('p95'))} "
            f"| {_fmt(c.get(real_key))} | {c.get('exceeds_p95')} |"
        )
    return lines


def _cov_row(name: str, blob: dict) -> str:
    return (
        f"| {name} | {blob.get('n', '')} | {_pct(blob.get('perp'))} | {_pct(blob.get('funding'))} "
        f"| {_pct(blob.get('oi'))} | {_pct(blob.get('basis'))} | {_pct(blob.get('taker'))} |"
    )


def _grid_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {_fmt(m.get('tail_ic_top'))} | {_fmt(m.get('tail_ic_top_nw_t'), 2)} "
        f"| {_fmt(m.get('tail_ic_bot'))} | {_fmt(m.get('overlap'))} | {_fmt(m.get('monster'))} "
        f"| {_fmt(m.get('rankic'))} | {m.get('n_dates', '')} |"
    )


def _trail_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {_fmt(m.get('tail_ic_top_trail'))} | {_fmt(m.get('tail_ic_top_trail_nw_t'), 2)} "
        f"| {_fmt(m.get('tail_ic_bot_trail'))} | {_fmt(m.get('overlap_trail'))} "
        f"| {_fmt(m.get('monster_trail'))} | {_fmt(m.get('rankic_trail'))} |"
    )


def _cycle_overlap_rows(grid: dict) -> list[str]:
    lines = []
    header = "| cycle | " + " | ".join(lab for _, lab in SIGNAL_ORDER) + " |"
    sep = "|-------|" + "|".join(["------"] * len(SIGNAL_ORDER)) + "|"
    lines.extend([header, sep])
    for cyc, *_ in PHASE2_CYCLES:
        cells = []
        for key, _lab in SIGNAL_ORDER:
            blob = ((grid.get(key) or {}).get("overlap_cycles") or {}).get(cyc) or {}
            cells.append(_fmt(blob.get("mean")))
        lines.append(f"| {cyc} | " + " | ".join(cells) + " |")
    return lines


def _book_row(name: str, b: dict) -> str:
    if not b:
        return f"| {name} | nan | nan | nan | nan |  |"
    return (
        f"| {name} | {_pct(b.get('total'))} "
        f"| {_pct(b.get('cagr'))} | {_pct(b.get('maxdd'))} | {_fmt(b.get('sharpe'), 3)} "
        f"| {b.get('n_formations', b.get('n_days', ''))} |"
    )


def write_phase4v2(
    path: Path,
    *,
    grid: dict,
    books: dict,
    coverage: dict,
    null: dict,
    verdict: dict,
    extra: dict,
    oi_first: dict | None = None,
    download_log: list | None = None,
) -> str:
    ic = (null or {}).get("tail_ic_top") or {}
    ov = (null or {}).get("overlap") or {}
    lines = [
        "# BTC-BEATER Phase 4 v2 — TAIL ROUND 1",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. "
        "CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.",
        "",
        "Supersedes the cancelled Phase 4 v1 (unlock calendar) prompt.",
        "",
        "## Addendum notes (verbatim)",
        "",
        f"> {PHASE4V2_PI_SCOPE}",
        "",
        f"> {PHASE4V2_PI_KILL}",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE4V2_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- Window {extra.get('start')} → {extra.get('end')} n_dates={extra.get('n_eval_dates')}",
        f"- GPU used = `{extra.get('gpu_used', False)}`",
        f"- LambdaRank config = one (truncation 10, ndcg@10, 5-grade labels, h=14); no sweeps",
        f"- New Vision downloads (OI/metrics gaps) = {len(download_log or [])} symbol jobs, "
        f"{sum(int(x.get('n_new_rows') or 0) for x in (download_log or []))} new rows",
        "",
        "## 1 — RANK-head null (E.1b on per-date tail metrics)",
        "",
        f"Judged = tail-IC(top-half). Bias: original E.1b 2·SE bound; CONTAMINATED requires ≥2 fold violations "
        f"(house rule). Skill: ≥5/6 exceed p95 or Stouffer z ≥ 3.0.",
        "",
        f"tail-IC(top-half): verdict=`{ic.get('verdict')}` bias_pass={ic.get('bias_pass')} "
        f"skill_pass={ic.get('skill_pass')} exceed={ic.get('n_exceed')}/{ic.get('n_folds')} "
        f"violations={ic.get('n_violate')} Stouffer z=`{_fmt(ic.get('stouffer_z'), 3)}`.",
        "",
        *_null_table(null.get("tail_ic_cells"), "real_tail_ic_top"),
        "",
        f"Overlap (centre=0.10): verdict=`{ov.get('verdict')}` bias_pass={ov.get('bias_pass')} "
        f"skill_pass={ov.get('skill_pass')} exceed={ov.get('n_exceed')}/{ov.get('n_folds')} "
        f"violations={ov.get('n_violate')} Stouffer z=`{_fmt(ov.get('stouffer_z'), 3)}`.",
        "",
        *_null_table(null.get("overlap_cells"), "real_overlap"),
        "",
        "## 2 — Positioning coverage",
        "",
        f"Perp coverage of top-100 name-days from 2021: **{_pct(coverage.get('perp_coverage_top100_from_2021'))}** "
        f"(n={coverage.get('n_name_days_from_2021')}; live threshold 50%).",
        "",
        "| slice | n | perp | funding | ΔOI | basis | taker |",
        "|-------|---|------|---------|-----|-------|-------|",
    ]
    by_year = coverage.get("by_year") or {}
    by_tier = coverage.get("by_tier") or {}
    for y in sorted(by_year, key=lambda z: str(z)):
        lines.append(_cov_row(f"year {y}", by_year[y]))
    for t in ("1-10", "11-50", "51-100", "unknown"):
        if t in by_tier:
            lines.append(_cov_row(f"tier {t}", by_tier[t]))
    n_oi = len(oi_first or {})
    sample = ", ".join(f"{k}:{v}" for k, v in list((oi_first or {}).items())[:12])
    lines.extend(
        [
            "",
            f"First available OI date reported for {n_oi} perp symbols. Sample: {sample}.",
            "",
            "## 3 — Tail-metric ablation grid (primary, per-date, floored top-100, Binance-listed)",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster top-3 | RankIC | n |",
            "|--------|-------------|------|-------------|---------|---------------|--------|---|",
        ]
    )
    for key, lab in SIGNAL_ORDER:
        lines.append(_grid_row(lab, grid.get(key) or {}))
    lines.extend(
        [
            "",
            "Trailing-18m:",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC |",
            "|--------|-------------|------|-------------|---------|---------|--------|",
        ]
    )
    for key, lab in SIGNAL_ORDER:
        lines.append(_trail_row(lab, grid.get(key) or {}))
    lines.extend(
        [
            "",
            "Overlap by cycle:",
            "",
            *_cycle_overlap_rows(grid),
            "",
            "Tail-IC(top-half) by cycle:",
            "",
        ]
    )
    header = "| cycle | " + " | ".join(lab for _, lab in SIGNAL_ORDER) + " |"
    sep = "|-------|" + "|".join(["------"] * len(SIGNAL_ORDER)) + "|"
    lines.extend([header, sep])
    for cyc, *_ in PHASE2_CYCLES:
        cells = []
        for key, _lab in SIGNAL_ORDER:
            blob = ((grid.get(key) or {}).get("tail_ic_top_cycles") or {}).get(cyc) or {}
            cells.append(_fmt(blob.get("mean")))
        lines.append(f"| {cyc} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 4 — Secondary: crude 14d book (information check, not adopted)",
            "",
            "Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance.",
            "",
            "| book | total | CAGR | MaxDD | Sharpe | n |",
            "|------|-------|------|-------|--------|---|",
        ]
    )
    for key, lab in SIGNAL_ORDER:
        lines.append(_book_row(lab, books.get(key) or {}))
    dl_n = len(download_log or [])
    new_n = sum(int(x.get("n_new_rows") or 0) for x in (download_log or []))
    lines.extend(
        [
            "",
            "## 5 — Mechanical verdicts",
            "",
            f"- **{verdict.get('tail_loss')}** (RANK clears deltas={verdict.get('rank_clears_deltas')}: "
            f"ΔIC `{_delta(verdict.get('delta_rank_vs_base_tail_ic'))}` / "
            f"Δov `{_delta(verdict.get('delta_rank_vs_base_overlap'))}`; "
            f"blend clears={verdict.get('blend_clears_deltas')}: "
            f"ΔIC `{_delta(verdict.get('delta_blend_vs_base_tail_ic'))}` / "
            f"Δov `{_delta(verdict.get('delta_blend_vs_base_overlap'))}`; "
            f"null pass={verdict.get('null_pass')})",
            f"- **{verdict.get('positioning')}** (Δ vs best A: tail-IC `{_delta(verdict.get('delta_pos_vs_best_a_tail_ic'))}`, "
            f"overlap `{_delta(verdict.get('delta_pos_vs_best_a_overlap'))}`; "
            f"perp coverage from 2021 `{_pct(verdict.get('perp_coverage_from_2021'))}`)",
            f"- **{verdict.get('price_additions')}** (Δ vs positioning: tail-IC `{_delta(verdict.get('delta_price_vs_pos_tail_ic'))}`, "
            f"overlap `{_delta(verdict.get('delta_price_vs_pos_overlap'))}`)",
            "",
            "Mechanical, no post-hoc adjustment. Nothing adopted.",
            "",
            "## Plain language",
            "",
            extra.get("plain", ""),
            "",
            "## Notes",
            "",
            "- Frozen spread is the 2.c cache (not retrained). RANK uses one LambdaRank config.",
            "- Positioning features are 0 + `pos_missing` for non-perp names. Coverage flags enforced.",
            "- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**",
            f"- Metrics gap downloads: {dl_n} jobs, {new_n} new rows (Binance Vision only).",
            f"- Elapsed s=`{_fmt(extra.get('elapsed_sec'), 1)}`. GPU=`{extra.get('gpu_used', False)}`.",
            "",
            "COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def update_ledger_phase4v2(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER Phase 4 v2 TAIL ROUND 1"
    block = [
        "",
        marker,
        "",
        "RANK head + positioning block + price-additions. Backtest/analysis only. Nothing adopted. Binance-priced.",
        "",
        f"**{verdict.get('tail_loss')}.** **{verdict.get('positioning')}.** **{verdict.get('price_additions')}.** "
        f"Best A=`{verdict.get('best_a')}`. "
        f"Baseline tail-IC(top-half) `{extra.get('base_tail_ic')}` overlap `{extra.get('base_overlap')}`; "
        f"best tail-IC `{extra.get('best_tail_ic')}` overlap `{extra.get('best_overlap')}` (`{verdict.get('best_tail_signal')}`). "
        f"Perp coverage from 2021 `{extra.get('perp_cov')}`.",
        "",
        "Mechanical, no post-hoc adjustment. Frozen products untouched.",
        "",
    ]
    new = "\n".join(block)
    if not new.startswith("\n"):
        new = "\n" + new
    if marker in text:
        pre, rest = text.split(marker, 1)
        lines = rest.splitlines()
        cut = None
        for i, ln in enumerate(lines[1:], start=1):
            if ln.startswith("## "):
                cut = i
                break
        if cut is None:
            text = pre.rstrip() + new
        else:
            text = pre.rstrip() + new + "\n".join(lines[cut:])
            if not text.endswith("\n"):
                text += "\n"
    else:
        text = text.rstrip() + "\n" + new
    path.write_text(text)
    return text


def plot_tail_ic_bars(grid: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [lab for _, lab in SIGNAL_ORDER]
    ys = [float((grid.get(k) or {}).get("tail_ic_top") or np.nan) for k, _ in SIGNAL_ORDER]
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#72B7B2", "#B279A2"]
    xs = np.arange(len(labels))
    ax.bar(xs, ys, color=colors, width=0.72)
    ax.set_xticks(xs, labels, rotation=18, ha="right")
    ax.set_ylabel("tail-IC (top half)")
    ax.set_title("Phase 4 v2 — tail-IC(top-half) by signal")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overlap_cycles(grid: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cycles = [c[0] for c in PHASE2_CYCLES]
    x = np.arange(len(cycles))
    width = 0.13
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#72B7B2", "#B279A2"]
    fig, ax = plt.subplots(figsize=(10.4, 5.0), constrained_layout=True)
    n = len(SIGNAL_ORDER)
    for i, (key, lab) in enumerate(SIGNAL_ORDER):
        ys = []
        for cyc in cycles:
            blob = ((grid.get(key) or {}).get("overlap_cycles") or {}).get(cyc) or {}
            v = blob.get("mean")
            ys.append(float(v) if v is not None and np.isfinite(float(v)) else np.nan)
        ax.bar(x + (i - (n - 1) / 2) * width, ys, width=width, label=lab, color=colors[i])
    ax.set_xticks(x, cycles)
    ax.set_ylabel("top-decile overlap")
    ax.set_title("Phase 4 v2 — overlap by cycle")
    ax.set_ylim(0.0, 0.35)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
