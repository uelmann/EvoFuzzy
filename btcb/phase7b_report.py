"""Phase 7.b FUZZY-STACK report + charts. Analysis only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btcb.constants import (
    DEATH_CONVENTION,
    PHASE2_CYCLES,
    PHASE2C_PRED_SHA256,
    PHASE4B_NULL_REGISTRATION,
    PHASE7B_CLOSED,
    PHASE7B_CRITERION,
    PHASE7B_FIREWALL,
    PHASE7B_KEEP_K,
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
    ("arm_a", "ARM-A (product library)"),
    ("arm_b", "ARM-B (rule stack)"),
    ("arm_ab", "ARM-A+B"),
)

SIGNAL_COLORS = {
    "frozen_spread": "#4C78A8",
    "arm_a": "#E45756",
    "arm_b": "#54A24B",
    "arm_ab": "#F58518",
}


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


def _active_signals(grid: dict) -> list[tuple[str, str]]:
    out = []
    for key, lab in SIGNAL_ORDER:
        if key == "frozen_spread" or (grid.get(key) or {}).get("n_dates"):
            if key != "frozen_spread" and not (grid.get(key) or {}).get("n_dates"):
                continue
            if key in grid:
                out.append((key, lab))
    return out or list(SIGNAL_ORDER)


def _cycle_rows(grid: dict, field: str, signals: list[tuple[str, str]]) -> list[str]:
    lines = []
    header = "| cycle | " + " | ".join(lab for _, lab in signals) + " |"
    sep = "|-------|" + "|".join(["------"] * len(signals)) + "|"
    lines.extend([header, sep])
    for cyc, *_ in PHASE2_CYCLES:
        cells = []
        for key, _lab in signals:
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


def _hygiene_table(rows: list[dict]) -> list[str]:
    lines = [
        "| tag | fold | head | best_iteration | UNDERTRAINED | status | elapsed s |",
        "|-----|------|------|----------------|--------------|--------|-----------|",
    ]
    for r in rows or []:
        lines.append(
            f"| {r.get('tag')} | {r.get('fold_id')} | {r.get('head')} | {r.get('best_iteration')} "
            f"| {r.get('undertrained')} | {r.get('status')} | {_fmt(r.get('elapsed'), 1)} |"
        )
    return lines


def write_phase7b(
    path: Path,
    *,
    grid: dict,
    books: dict,
    null_best: dict | None,
    stack: dict,
    prune: dict,
    kept_formulas: list[dict],
    hygiene: dict,
    gain: dict,
    verdict: dict,
    extra: dict,
) -> str:
    signals = _active_signals(grid)
    stack_line = (
        f"STACK-SKIPPED reasons={stack.get('reasons')}"
        if stack.get("skipped")
        else (
            f"RULE-FORGE ok={stack.get('ruleforge_ok')} verdict={stack.get('ruleforge_verdict')}; "
            f"NFN ok={stack.get('nfn_ok')} verdict={stack.get('nfn_verdict')}; "
            f"n_rule_features={stack.get('n_rule_features')}"
        )
    )
    lines = [
        "# BTC-BEATER Phase 7.b — FUZZY-STACK",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. "
        "CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.",
        "",
        "## Firewall (verbatim, before results)",
        "",
        f"> {PHASE7B_FIREWALL}",
        "",
        "## Vol-matched null (house standard; verbatim, before results)",
        "",
        f"> {PHASE4B_NULL_REGISTRATION}",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE7B_CRITERION}",
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
        f"- Library = C(66,2) pairwise CDF products = `{extra.get('n_library')}` plus 33 originals",
        f"- Two-stage prune k={PHASE7B_KEEP_K} per head, union n=`{(prune or {}).get('n_union')}` (one prune, no iteration)",
        f"- Firewall = `{extra.get('firewall_passed')}`",
        "",
        "## 0 — Preconditions (Arm B)",
        "",
        stack_line,
        "",
    ]
    parents = (stack or {}).get("parents") or {}
    if parents:
        lines.extend(
            [
                "| source | verdict | n_rules | standalone tail-IC(top) | path |",
                "|--------|---------|---------|-------------------------|------|",
            ]
        )
        for src, rec in parents.items():
            lines.append(
                f"| {src} | {rec.get('verdict')} | {rec.get('n_rules')} | {_fmt(rec.get('tail_ic_top'))} "
                f"| `{rec.get('path')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## 1 — Training hygiene",
            "",
            f"ES floor `{extra.get('es_floor')}`, patience `{extra.get('es_patience')}`, cap `{extra.get('es_cap')}`. "
            f"UNDERTRAINED if best_iteration < `{extra.get('undertrained_lt')}`. "
            f"**UNDERTRAINED count = `{hygiene.get('n_undertrained', 0)}`** "
            f"(of `{hygiene.get('n_fits', 0)}` LightGBM fits in this phase, excluding cached-null hits).",
            "",
            *_hygiene_table(hygiene.get("rows") or []),
            "",
            "## 2 — Kept products (Arm A, printed formulas)",
            "",
            f"Top-150 by total gain on TOP ∪ top-150 on BOTTOM; union n=`{(prune or {}).get('n_union')}`. "
            "Top-30 of the union by (TOP+BOTTOM) total gain:",
            "",
            "| rank | feature | formula | gain TOP | gain BOTTOM | gain sum |",
            "|------|---------|---------|----------|-------------|----------|",
        ]
    )
    for i, rec in enumerate((kept_formulas or [])[:30], start=1):
        lines.append(
            f"| {i} | `{rec.get('name')}` | {rec.get('formula')} | {_fmt(rec.get('gain_top'), 1)} "
            f"| {_fmt(rec.get('gain_bot'), 1)} | {_fmt(rec.get('gain_sum'), 1)} |"
        )
    lines.extend(
        [
            "",
            "## 3 — Feature-importance gain share",
            "",
            f"- originals: `{_pct((gain or {}).get('originals'))}`",
            f"- library products: `{_pct((gain or {}).get('products'))}`",
            f"- rule features: `{_pct((gain or {}).get('rules'))}` "
            f"(RULE-FORGE `{_pct((gain or {}).get('ruleforge'))}` / NFN `{_pct((gain or {}).get('nfn'))}`)",
            f"- total gain (judged arm, both heads, all folds) = `{(gain or {}).get('total_gain')}`",
            "",
            f"Chart: `charts/btcb_phase7b_gain_share.png`.",
            "",
            "## 4 — Vol-matched null (best arm only)",
            "",
            f"Best arm = `{verdict.get('best_arm')}`. Null design = vol-matched, folds {{0,5,9,15,21,24}} × 25.",
            "",
        ]
    )
    if null_best:
        lines.extend(_null_summary("tail-IC(top-half)", null_best, "tail_ic_top", "real_tail_ic_top"))
        lines.extend(_null_summary("overlap", null_best, "overlap", "real_overlap"))
        lines.extend(_null_summary("monster top-3", null_best, "monster", "real_monster"))
    else:
        lines.append("No null (no fuzzy arm produced a score).")
        lines.append("")
    lines.extend(
        [
            "## 5 — Tail-metric judgment grid (primary, per-date, floored top-100, Binance-listed)",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster top-3 | RankIC | vol-corr | n |",
            "|--------|-------------|------|-------------|---------|---------------|--------|----------|---|",
        ]
    )
    for key, lab in signals:
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
    for key, lab in signals:
        lines.append(_trail_row(lab, grid.get(key) or {}))
    lines.extend(
        [
            "",
            "Overlap by cycle:",
            "",
            *_cycle_rows(grid, "overlap_cycles", signals),
            "",
            "Tail-IC(top-half) by cycle:",
            "",
            *_cycle_rows(grid, "tail_ic_top_cycles", signals),
            "",
            "## 6 — Secondary: crude 14d book (information check, not adopted)",
            "",
            "Ladder-1 construction: EW top decile, 10% cap, idle cash, 10 bps/side, h=14 full rebalance.",
            "",
            "| book | total | CAGR | MaxDD | Sharpe | n |",
            "|------|-------|------|-------|--------|---|",
        ]
    )
    for key, lab in signals:
        lines.append(_book_row(lab, books.get(key) or {}))
    lines.extend(
        [
            "",
            "## 7 — Mechanical verdicts",
            "",
        ]
    )
    per = (verdict or {}).get("per_arm") or {}
    for key, lab in signals:
        if key == "frozen_spread":
            continue
        rec = per.get(key) or {"label": "not run"}
        lines.append(
            f"- **{lab}: {rec.get('label')}** "
            f"(ΔIC `{_delta(rec.get('delta_tail_ic_top'))}` / Δov `{_delta(rec.get('delta_overlap'))}` / "
            f"ΔRankIC `{_delta(rec.get('delta_rankic'))}`; "
            f"clears_deltas={rec.get('clears_deltas')} null_pass={rec.get('null_pass')} "
            f"beats_parents={rec.get('beats_parents')})"
        )
    if verdict.get("stack_skipped"):
        lines.append(f"- Arm B: **STACK-SKIPPED** ({stack_line})")
    if verdict.get("closed"):
        lines.extend(["", f"- Ledger clause: **{verdict.get('closed')}**"])
    lines.extend(
        [
            "",
            "Mechanical, no post-hoc adjustment. Nothing adopted.",
            "",
            "## Plain language",
            "",
            extra.get("plain", ""),
            "",
            "## Notes",
            "",
            "- Frozen spread is the 2.c cache (not retrained). Arm A is one prune of the CDF product library.",
            "- Vol-matched null is the house standard for tail metrics; run on the best fuzzy arm only.",
            "- Crude 14d CAGR/MaxDD is an information check. **Nothing is adopted.**",
            f"- Elapsed s=`{_fmt(extra.get('elapsed_sec'), 1)}`. GPU=`{extra.get('gpu_used', False)}`.",
            f"- Charts: `charts/btcb_phase7b_tail_ic.png`, `charts/btcb_phase7b_gain_share.png`.",
            "",
            "COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def update_ledger_phase7b(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER Phase 7.b FUZZY-STACK"
    per = (verdict or {}).get("per_arm") or {}
    arm_bits = []
    for key, lab in (("arm_a", "ARM-A"), ("arm_b", "ARM-B"), ("arm_ab", "ARM-A+B")):
        rec = per.get(key)
        if rec:
            arm_bits.append(f"{lab}={rec.get('label')}")
    if verdict.get("stack_skipped") and "ARM-B" not in " ".join(arm_bits):
        arm_bits.append("ARM-B=STACK-SKIPPED")
    closed = verdict.get("closed") or ""
    block = [
        "",
        marker,
        "",
        "Learned fuzzy rules as LightGBM features + fixed-membership product library. "
        "Backtest/analysis only. Nothing adopted. Binance-priced.",
        "",
        f"**{'; '.join(arm_bits) or 'no arms'}.** Best arm=`{verdict.get('best_arm')}`. "
        f"UNDERTRAINED count=`{extra.get('n_undertrained')}`. "
        f"Gain share originals=`{extra.get('gain_originals')}` products=`{extra.get('gain_products')}` "
        f"rules=`{extra.get('gain_rules')}`. "
        f"Baseline tail-IC(top-half) `{extra.get('base_tail_ic')}` overlap `{extra.get('base_overlap')}`; "
        f"best-arm tail-IC `{extra.get('best_tail_ic')}` overlap `{extra.get('best_overlap')}`.",
        "",
    ]
    if closed:
        block.extend([f"**{PHASE7B_CLOSED}**", ""])
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
    cells = (null or {}).get("tail_ic_top_cells") or (null or {}).get("tail_ic_cells") or []
    xs = [c.get("p95") for c in cells if c.get("p95") is not None]
    xs = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def plot_tail_ic_bars(grid: dict, null_best: dict | None, best_arm: str | None, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals = [(k, lab) for k, lab in SIGNAL_ORDER if k in grid]
    labels = [lab for _, lab in signals]
    ys = [float((grid.get(k) or {}).get("tail_ic_top") or np.nan) for k, _ in signals]
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    colors = [SIGNAL_COLORS.get(k, "#4C78A8") for k, _ in signals]
    xs = np.arange(len(labels))
    ax.bar(xs, ys, color=colors, width=0.72)
    p95 = _mean_null_p95(null_best) if null_best else float("nan")
    if best_arm and np.isfinite(p95):
        for i, (key, _lab) in enumerate(signals):
            if key == best_arm:
                ax.scatter([xs[i]], [p95], marker="_", s=420, color="k", zorder=5, linewidths=2)
    ax.set_xticks(xs, labels, rotation=18, ha="right")
    ax.set_ylabel("tail-IC (top half)")
    ax.set_title("Phase 7.b — tail-IC(top-half); black tick = vol-matched null mean p95 (best arm)")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gain_share(gain: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["originals (33)", "library products", "rule activations"]
    sizes = [
        float((gain or {}).get("originals") or 0.0),
        float((gain or {}).get("products") or 0.0),
        float((gain or {}).get("rules") or 0.0),
    ]
    if not any(s > 0 for s in sizes):
        sizes = [1.0, 0.0, 0.0]
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 0.5 else "",
        startangle=90,
    )
    ax.set_title("Phase 7.b — LightGBM total-gain share")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
