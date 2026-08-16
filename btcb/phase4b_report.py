"""Phase 4.b TWIN-RANK report + charts. Analysis only."""

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
    PHASE2C_PRED_SHA256,
    PHASE4B_CEILING,
    PHASE4B_CRITERION,
    PHASE4B_DIR_RATIONALE,
    PHASE4B_NULL_REGISTRATION,
    PHASE4V2_PI_SCOPE,
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
    ("twinrank", "TWIN-RANK"),
    ("spread_twinrank", "SPREAD+TWIN-RANK"),
    ("dir_spread", "DIR-spread"),
    ("dir_twinrank", "DIR-spread+TWIN-RANK"),
)

SIGNAL_COLORS = {
    "frozen_spread": "#4C78A8",
    "twinrank": "#E45756",
    "spread_twinrank": "#F58518",
    "dir_spread": "#54A24B",
    "dir_twinrank": "#B279A2",
}

EQUITY_CHART = "charts/btcb_phase4b_equity.png"
EQUITY_CHART_LINE = (
    f"Chart: `{EQUITY_CHART}` (log equity + drawdown). Information only; nothing adopted."
)
EQUITY_NOTE = (
    f"- Crude 14d equity (log + drawdown): `{EQUITY_CHART}`. Information only; nothing adopted."
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


def _null_summary(title: str, null: dict, metric: str, real_key: str) -> list[str]:
    blob = (null or {}).get(metric) or {}
    lines = [
        f"**{title}** verdict=`{blob.get('verdict')}` bias_pass={blob.get('bias_pass')} "
        f"skill_pass={blob.get('skill_pass')} exceed={blob.get('n_exceed')}/{blob.get('n_folds')} "
        f"violations={blob.get('n_violate')} Stouffer z=`{_fmt(blob.get('stouffer_z'), 3)}`.",
        "",
        *_null_table((null or {}).get(f"{metric}_cells"), real_key),
        "",
    ]
    return lines


def _grid_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {_fmt(m.get('tail_ic_top'))} | {_fmt(m.get('tail_ic_top_nw_t'), 2)} "
        f"| {_fmt(m.get('tail_ic_bot'))} | {_fmt(m.get('overlap'))} | {_fmt(m.get('monster'))} "
        f"| {_fmt(m.get('rankic'))} | {_fmt(m.get('vol_rank_corr'))} | {m.get('n_dates', '')} |"
    )


def _trail_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {_fmt(m.get('tail_ic_top_trail'))} | {_fmt(m.get('tail_ic_top_trail_nw_t'), 2)} "
        f"| {_fmt(m.get('tail_ic_bot_trail'))} | {_fmt(m.get('overlap_trail'))} "
        f"| {_fmt(m.get('monster_trail'))} | {_fmt(m.get('rankic_trail'))} "
        f"| {_fmt(m.get('vol_rank_corr_trail'))} |"
    )


def _cycle_rows(grid: dict, field: str) -> list[str]:
    lines = []
    header = "| cycle | " + " | ".join(lab for _, lab in SIGNAL_ORDER) + " |"
    sep = "|-------|" + "|".join(["------"] * len(SIGNAL_ORDER)) + "|"
    lines.extend([header, sep])
    for cyc, *_ in PHASE2_CYCLES:
        cells = []
        for key, _lab in SIGNAL_ORDER:
            blob = ((grid.get(key) or {}).get(field) or {}).get(cyc) or {}
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


def write_phase4b(
    path: Path,
    *,
    grid: dict,
    books: dict,
    null_twin: dict,
    null_dir: dict,
    null_rank: dict,
    vol_diag: dict,
    verdict: dict,
    extra: dict,
) -> str:
    lines = [
        "# BTC-BEATER Phase 4.b — TWIN-RANK",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. "
        "CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.",
        "",
        "Positioning and price-additions remain **NOT LIVE** (Phase 4 v2, recorded). Not retested.",
        "",
        "## Addendum notes (verbatim)",
        "",
        f"> {PHASE4V2_PI_SCOPE}",
        "",
        "## Vol-matched null (NEW HOUSE STANDARD; verbatim, before results)",
        "",
        f"> {PHASE4B_NULL_REGISTRATION}",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE4B_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## DIR rationale (verbatim)",
        "",
        f"> {PHASE4B_DIR_RATIONALE}",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- Window {extra.get('start')} → {extra.get('end')} n_dates={extra.get('n_eval_dates')}",
        f"- GPU used = `{extra.get('gpu_used', False)}`",
        f"- LambdaRank config = one per head (truncation 10, ndcg@10, 5-grade labels, h=14); no sweeps",
        f"- DIR = one weight rule `w=1+2·1[top decile]` on the top classifier head; bottom = frozen 2.c",
        f"- 4v2 RANK cache reused = `{extra.get('rank_cache_reused')}`",
        "",
        "## 1 — Vol-matched null tables",
        "",
        "Centre per fold = that fold's own null mean (structural vol-matched reference). "
        "Plain-shuffle Phase 4 v2 RANK null remains on the record (CONTAMINATED vs centre 0).",
        "",
        "### TWIN-RANK",
        "",
        *_null_summary("tail-IC(top-half)", null_twin, "tail_ic_top", "real_tail_ic_top"),
        *_null_summary("overlap", null_twin, "overlap", "real_overlap"),
        *_null_summary("monster top-3", null_twin, "monster", "real_monster"),
        "### Retro Phase-4v2 RANK head (informational)",
        "",
        *_null_summary("tail-IC(top-half)", null_rank, "tail_ic_top", "real_tail_ic_top"),
        *_null_summary("overlap", null_rank, "overlap", "real_overlap"),
        *_null_summary("monster top-3", null_rank, "monster", "real_monster"),
        f"Retro answer: vol-matched RANK verdict=`{verdict.get('retro_rank_verdict')}` "
        f"bias_pass={verdict.get('retro_rank_bias_pass')} skill_pass={verdict.get('retro_rank_skill_pass')}. "
        f"Gain beyond vol: **{'YES' if verdict.get('retro_rank_vol_matched_pass') else 'NO'}**.",
        "",
        "### DIR-spread",
        "",
        *_null_summary("tail-IC(top-half)", null_dir, "tail_ic_top", "real_tail_ic_top"),
        *_null_summary("overlap", null_dir, "overlap", "real_overlap"),
        *_null_summary("monster top-3", null_dir, "monster", "real_monster"),
        "## 2 — Vol-correlation diagnostic (report only)",
        "",
        "Mean per-date cross-sectional rank-corr vs `yz_vol_30`. Twin subtraction should collapse the vol tilt.",
        "",
        f"- RANK top head: `{_fmt((vol_diag or {}).get('rank_top'))}`",
        f"- RANK bottom head: `{_fmt((vol_diag or {}).get('rank_bot'))}`",
        f"- TWIN-RANK: `{_fmt((vol_diag or {}).get('twinrank'))}`",
        f"- frozen spread: `{_fmt((vol_diag or {}).get('frozen_spread'))}`",
        f"- DIR-spread: `{_fmt((vol_diag or {}).get('dir_spread'))}`",
        "",
        "## 3 — Tail-metric judgment grid (primary, per-date, floored top-100, Binance-listed)",
        "",
        "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster top-3 | RankIC | vol-corr | n |",
        "|--------|-------------|------|-------------|---------|---------------|--------|----------|---|",
    ]
    for key, lab in SIGNAL_ORDER:
        lines.append(_grid_row(lab, grid.get(key) or {}))
    lines.extend(
        [
            "",
            "Trailing-18m:",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | vol-corr |",
            "|--------|-------------|------|-------------|---------|---------|--------|----------|",
        ]
    )
    for key, lab in SIGNAL_ORDER:
        lines.append(_trail_row(lab, grid.get(key) or {}))
    lines.extend(
        [
            "",
            "Overlap by cycle:",
            "",
            *_cycle_rows(grid, "overlap_cycles"),
            "",
            "Tail-IC(top-half) by cycle:",
            "",
            *_cycle_rows(grid, "tail_ic_top_cycles"),
            "",
            "## 4 — Secondary: crude 14d book (information check, not adopted)",
            "",
            "Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance.",
            "",
            EQUITY_CHART_LINE,
            "",
            "| book | total | CAGR | MaxDD | Sharpe | n |",
            "|------|-------|------|-------|--------|---|",
        ]
    )
    for key, lab in SIGNAL_ORDER:
        lines.append(_book_row(lab, books.get(key) or {}))
    ceil = verdict.get("ceiling")
    lines.extend(
        [
            "",
            "## 5 — Mechanical verdicts",
            "",
            f"- **{verdict.get('twinrank')}** (clears deltas={verdict.get('twin_clears_deltas')}: "
            f"ΔIC `{_delta(verdict.get('delta_twin_vs_base_tail_ic'))}` / "
            f"Δov `{_delta(verdict.get('delta_twin_vs_base_overlap'))}`; "
            f"vol-matched null pass={verdict.get('twin_null_pass')})",
            f"- **{verdict.get('dir')}** (clears deltas={verdict.get('dir_clears_deltas')}: "
            f"ΔIC `{_delta(verdict.get('delta_dir_vs_base_tail_ic'))}` / "
            f"Δov `{_delta(verdict.get('delta_dir_vs_base_overlap'))}`; "
            f"vol-matched null pass={verdict.get('dir_null_pass')})",
            f"- Retro 4v2 RANK beyond vol: **{'YES' if verdict.get('retro_rank_vol_matched_pass') else 'NO'}** "
            f"(`{verdict.get('retro_rank_verdict')}`)",
            "",
        ]
    )
    if ceil:
        lines.extend([f"- Ledger clause: **{ceil}**", ""])
    lines.extend(
        [
            "Mechanical, no post-hoc adjustment. Nothing adopted.",
            "",
            "## Plain language",
            "",
            extra.get("plain", ""),
            "",
            "## Notes",
            "",
            "- Frozen spread is the 2.c cache (not retrained). TWIN-RANK uses one LambdaRank config per head.",
            "- Vol-matched null supersedes plain within-date shuffle for tail metrics going forward.",
            "- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**",
            EQUITY_NOTE,
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


def update_ledger_phase4b(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER Phase 4.b TWIN-RANK"
    ceil = verdict.get("ceiling") or ""
    block = [
        "",
        marker,
        "",
        "TWIN-RANK + vol-matched null + DIR reweighting. Backtest/analysis only. Nothing adopted. Binance-priced.",
        "",
        f"**{verdict.get('twinrank')}.** **{verdict.get('dir')}.** "
        f"Retro RANK beyond vol=`{'YES' if verdict.get('retro_rank_vol_matched_pass') else 'NO'}` "
        f"(`{verdict.get('retro_rank_verdict')}`). "
        f"Baseline tail-IC(top-half) `{extra.get('base_tail_ic')}` overlap `{extra.get('base_overlap')}`; "
        f"TWIN-RANK tail-IC `{extra.get('twin_tail_ic')}` overlap `{extra.get('twin_overlap')}`; "
        f"DIR tail-IC `{extra.get('dir_tail_ic')}` overlap `{extra.get('dir_overlap')}`.",
        "",
    ]
    if ceil:
        block.extend([f"**{PHASE4B_CEILING}**", ""])
    block.extend(
        [
            "Mechanical, no post-hoc adjustment. Frozen products untouched.",
            "",
        ]
    )
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


def _mean_null_p95(null: dict) -> float:
    cells = (null or {}).get("tail_ic_cells") or []
    xs = [c.get("p95") for c in cells if c.get("p95") is not None and np.isfinite(float(c.get("p95")))]
    return float(np.mean(xs)) if xs else float("nan")


def plot_tail_ic_bars(grid: dict, nulls: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [lab for _, lab in SIGNAL_ORDER]
    ys = [float((grid.get(k) or {}).get("tail_ic_top") or np.nan) for k, _ in SIGNAL_ORDER]
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#B279A2"]
    xs = np.arange(len(labels))
    ax.bar(xs, ys, color=colors, width=0.72)
    overlay = {
        "twinrank": _mean_null_p95(nulls.get("twinrank")),
        "dir_spread": _mean_null_p95(nulls.get("dir_spread")),
    }
    for i, (key, _lab) in enumerate(SIGNAL_ORDER):
        p95 = overlay.get(key)
        if p95 is None or not np.isfinite(p95):
            continue
        ax.scatter([xs[i]], [p95], marker="_", s=420, color="k", zorder=5, linewidths=2)
    ax.set_xticks(xs, labels, rotation=18, ha="right")
    ax.set_ylabel("tail-IC (top half)")
    ax.set_title("Phase 4.b — tail-IC(top-half); black ticks = vol-matched null mean p95")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _book_equity(book: dict):
    if not isinstance(book, dict):
        return None
    eq = book.get("equity")
    if isinstance(eq, pd.Series) and len(eq):
        s = eq.astype(float)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).tz_convert("UTC").normalize()
        return s.sort_index()
    rets = book.get("daily_ret")
    if rets is None:
        return None
    r = pd.Series(rets, dtype=float).fillna(0.0)
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index, utc=True)).tz_convert("UTC").normalize()
    r = r.sort_index()
    if r.empty:
        return None
    return (1.0 + r).cumprod()


def plot_equity_curves(books: dict, out_path: Path) -> None:
    """Log equity + drawdown for the five crude 14d books. Information only."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax, ax2 = axes
    n_plotted = 0
    for key, lab in SIGNAL_ORDER:
        eq = _book_equity((books or {}).get(key) or {})
        if eq is None or eq.empty:
            continue
        b = (books or {}).get(key) or {}
        cagr, mdd = b.get("cagr"), b.get("maxdd")
        label = lab
        if cagr is not None and np.isfinite(float(cagr)):
            label += f"  CAGR={100.0 * float(cagr):.1f}%"
        if mdd is not None and np.isfinite(float(mdd)):
            label += f"  DD={100.0 * float(mdd):.0f}%"
        color = SIGNAL_COLORS.get(key, "#4C78A8")
        ax.plot(eq.index, eq.values, lw=1.4, color=color, label=label)
        dd = eq / eq.cummax() - 1.0
        ax2.plot(dd.index, dd.values, lw=1.05, color=color)
        n_plotted += 1
    if n_plotted == 0:
        plt.close(fig)
        raise RuntimeError("plot_equity_curves: no equity series")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("Phase 4.b — crude 14d books (information only; nothing adopted)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax2.axhline(0.0, color="0.5", lw=0.8)
    ax2.set_ylabel("drawdown")
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def ensure_equity_chart_notes(text: str) -> str:
    if EQUITY_CHART in text:
        return text
    needle = "Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance."
    if needle in text:
        text = text.replace(needle, needle + "\n\n" + EQUITY_CHART_LINE, 1)
    note_needle = "- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**"
    if note_needle in text:
        text = text.replace(note_needle, note_needle + "\n" + EQUITY_NOTE, 1)
    return text


def plot_overlap_cycles(grid: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cycles = [c[0] for c in PHASE2_CYCLES]
    x = np.arange(len(cycles))
    width = 0.15
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#B279A2"]
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
    ax.set_title("Phase 4.b — overlap by cycle")
    ax.set_ylim(0.0, 0.35)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
