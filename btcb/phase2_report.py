"""Phase 2 MODEL-V1 report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import DEATH_CONVENTION, PHASE2_CRITERION, PHASE2_PRIMARY_H


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


def _book_row(name: str, b: dict) -> str:
    fe = b.get("forced_exits") or {}
    return (
        f"| {name} | {_pct(b.get('book_total'))} | {_pct(b.get('book_cagr'))} "
        f"| {_fmt(b.get('book_sharpe'))} | {_fmt(b.get('rel_sharpe'))} | {_pct(b.get('maxdd'))} "
        f"| {_fmt(b.get('avg_n_names'), 2)} | {_pct(b.get('avg_w_btc'))} "
        f"| {_fmt(b.get('ann_turnover'), 2)} | {fe.get('n_events')} |"
    )


def write_phase2(
    path: Path,
    *,
    gates: list[dict],
    null_gate: dict,
    headline: dict,
    grid: list[dict],
    naive: dict,
    btc_ref: dict,
    verdicts: dict,
    metas: dict,
    importances: list,
    extra: dict,
    h30_headline: dict | None = None,
    h30_verdicts: dict | None = None,
) -> str:
    gates_ok = bool(extra.get("gates_ok"))
    lines = [
        "# BTC-BEATER Phase 2 — MODEL-V1 winner-tail classifier",
        "",
        "**BACKTEST ONLY.** One model. Params and guards frozen a priori. "
        "No sweeps beyond the declared 3-point p_enter grid with median convention. "
        "CPU only, zero GPU. Frozen COMBO v2.0-combo-final is untouched.",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE2_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Gates",
        "",
        f"Gates official: **{'PASS' if gates_ok else 'FAIL'}**. "
        "Results below are official only if all gates pass.",
        "",
        "| gate | passed | detail |",
        "|------|--------|--------|",
    ]
    for g in gates:
        detail = {k: v for k, v in g.items() if k not in {"name", "passed", "cells"}}
        lines.append(f"| {g.get('name')} | {g.get('passed')} | `{detail}` |")
    ng = null_gate or {}
    lines += [
        "",
        f"Label-shuffle null (E.1b design, h={PHASE2_PRIMARY_H}, 2 folds, 25 replicates): "
        f"**{ng.get('verdict')}** bias_pass={ng.get('bias_pass')} skill_pass={ng.get('skill_pass')}.",
        "",
        "| fold | n | null mean | SD | 95th | real AUC | bias_ok | exceeds_p95 |",
        "|------|---|-----------|----|------|----------|---------|-------------|",
    ]
    for c in ng.get("cells") or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'), 4)} | {_fmt(c.get('sd'), 4)} "
            f"| {_fmt(c.get('p95'), 4)} | {_fmt(c.get('real_auc'), 4)} | {c.get('bias_ok')} | {c.get('exceeds_p95')} |"
        )
    lines += [
        "",
        "## Walk-forward AUC (calibrated p, OOS)",
        "",
        "| h | fold | val_start | val_end | n_valid | auc_oos | auc_raw | best_iter |",
        "|---|------|-----------|---------|---------|---------|---------|-----------|",
    ]
    for h, ms in (metas or {}).items():
        for m in ms:
            if m.get("status") != "ok":
                continue
            lines.append(
                f"| {h} | {m.get('fold_id')} | {m.get('val_start')} | {m.get('val_end')} "
                f"| {m.get('n_valid')} | {_fmt(m.get('auc_oos'), 4)} | {_fmt(m.get('auc_oos_raw'), 4)} "
                f"| {m.get('best_iteration')} |"
            )

    hl = headline or {}
    nv = naive or {}
    lines += [
        "",
        "## Mechanical verdicts (primary h=14, median p_enter)",
        "",
        f"- Gates: {'PASS' if gates_ok else 'FAIL'}",
        f"- **MODEL-V1 is {'VIABLE' if verdicts.get('viable') and gates_ok else 'NOT VIABLE'}**",
        f"- **REPLACES-FLOOR: {bool(verdicts.get('replaces_floor') and gates_ok)}**",
        f"- median p_enter = {hl.get('p_enter')} (house median of relative-line Sharpe across the grid)",
        f"- (a) rel Sharpe = {_fmt(verdicts.get('rel_sharpe'))} > 0 → {verdicts.get('a_rel_sharpe_gt0')}",
        f"- (b) book total {_pct(verdicts.get('book_total'))} vs BTC {_pct(verdicts.get('btc_total'))} → {verdicts.get('b_total_ge_btc')}",
        f"- (c) MaxDD {_pct(verdicts.get('maxdd'))} vs BTC {_pct(verdicts.get('btc_maxdd'))} "
        f"(pass if model drawdown is no worse, i.e. not more negative) → {verdicts.get('c_maxdd_le_btc')}",
        f"- replace-floor need rel Sharpe ≥ {_fmt(verdicts.get('need_replaces'))} "
        f"(naive same-window {_fmt(verdicts.get('naive_rel_sharpe'))} + 0.15)",
        f"- OOS window: {hl.get('start')} → {hl.get('end')} (n={hl.get('n_days')})",
        f"- Forced exits: n_events={(hl.get('forced_exits') or {}).get('n_events')} "
        f"n_ids={(hl.get('forced_exits') or {}).get('n_ids')}",
        f"- % time in BTC: {_pct(hl.get('avg_w_btc'))}",
        "",
        "A verdict is not overridden by any single cycle.",
        "",
        "## Headline book vs naive v3 vs BTC B&H (same OOS window)",
        "",
        "| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % in BTC | ann TO | forced |",
        "|------|-------|------|------------|------------|-------|------------|----------|--------|--------|",
        _book_row(f"MODEL-V1 h=14 p={hl.get('p_enter')}", hl),
        _book_row("naive rotation v3 (same window)", nv),
        _book_row("BTC B&H", btc_ref or nv),
    ]
    if h30_headline:
        lines.append(_book_row(f"MODEL-V1 h=30 p={h30_headline.get('p_enter')} (robustness)", h30_headline))
    lines += [
        "",
        "## p_enter grid (h=14)",
        "",
        "| p_enter | rel Sharpe | total | MaxDD | % BTC | avg #names |",
        "|---------|------------|-------|-------|-------|------------|",
    ]
    for r in grid or []:
        mark = " ← median" if r.get("p_enter") == hl.get("p_enter") else ""
        lines.append(
            f"| {r.get('p_enter')}{mark} | {_fmt(r.get('rel_sharpe'))} | {_pct(r.get('book_total'))} "
            f"| {_pct(r.get('maxdd'))} | {_pct(r.get('avg_w_btc'))} | {_fmt(r.get('avg_n_names'), 2)} |"
        )
    lines += [
        "",
        "## Per-cycle honesty (headline h=14)",
        "",
        "| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC |",
        "|-------|---|----------|---------|------------|------------|-------|------------|-------|",
    ]
    for name, c in (hl.get("cycles") or {}).items():
        lines.append(
            f"| {name} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('btc_total'))} "
            f"| {_fmt(c.get('book_sharpe'))} | {_fmt(c.get('rel_sharpe'))} | {_pct(c.get('maxdd'))} "
            f"| {_fmt(c.get('avg_n_names'), 2)} | {_pct(c.get('avg_w_btc'))} |"
        )
    if h30_headline:
        v30 = h30_verdicts or {}
        lines += [
            "",
            "## h=30 robustness (does not override h=14)",
            "",
            f"- VIABLE={v30.get('viable')} REPLACES-FLOOR={v30.get('replaces_floor')} "
            f"median p_enter={h30_headline.get('p_enter')} rel Sharpe={_fmt(h30_headline.get('rel_sharpe'))}",
            "",
            "| cycle | n | book tot | BTC tot | rel Sharpe | MaxDD | % BTC |",
            "|-------|---|----------|---------|------------|-------|-------|",
        ]
        for name, c in (h30_headline.get("cycles") or {}).items():
            lines.append(
                f"| {name} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('btc_total'))} "
                f"| {_fmt(c.get('rel_sharpe'))} | {_pct(c.get('maxdd'))} | {_pct(c.get('avg_w_btc'))} |"
            )
    lines += [
        "",
        "## Feature importances (mean gain across h=14 folds, top 15)",
        "",
        "| rank | feature | mean gain |",
        "|------|---------|-----------|",
    ]
    for i, (feat, g) in enumerate(importances or [], start=1):
        lines.append(f"| {i} | `{feat}` | {_fmt(g, 2)} |")
    lines += [
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU={extra.get('gpu_used', False)}. "
        f"n_features={extra.get('n_features')}. n_train_rows={extra.get('n_train_rows')}.",
        "",
        "COMBO untouched (v2.0-combo-final).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_equity(model: dict, naive: dict, out_path: Path) -> None:
    eq = model.get("equity")
    eqb = model.get("equity_btc")
    rel = model.get("rel_equity")
    if eq is None or eqb is None:
        return
    n_eq = naive.get("equity")
    n_rel = naive.get("rel_equity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(eq.index, eq.values, lw=1.3, label="MODEL-V1")
    ax.plot(eqb.index, eqb.values, lw=1.2, label="BTC B&H")
    if isinstance(n_eq, pd.Series) and len(n_eq):
        ax.plot(n_eq.index, n_eq.values, lw=1.1, alpha=0.85, label="naive v3")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("BTC-BEATER Phase 2 — book vs BTC vs naive")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if rel is not None:
        ax2.plot(rel.index, rel.values, lw=1.3, label="MODEL-V1 / BTC")
    if isinstance(n_rel, pd.Series) and len(n_rel):
        ax2.plot(n_rel.index, n_rel.values, lw=1.1, alpha=0.85, label="naive / BTC")
    ax2.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax2.set_ylabel("relative equity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(metas: list[dict], out_path: Path) -> None:
    if not metas:
        return
    ok = [m for m in metas if m.get("status") == "ok" and (m.get("reliability") or {}).get("mean_p")]
    if not ok:
        return
    picks = [ok[0], ok[len(ok) // 2], ok[-1]]
    # unique by fold_id
    seen = set()
    uniq = []
    for m in picks:
        if m["fold_id"] in seen:
            continue
        seen.add(m["fold_id"])
        uniq.append(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(uniq), figsize=(4.2 * len(uniq), 3.8), constrained_layout=True)
    if len(uniq) == 1:
        axes = [axes]
    for ax, m in zip(axes, uniq):
        rel = m["reliability"]
        ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1)
        ax.plot(rel["mean_p"], rel["frac_pos"], marker="o", lw=1.2)
        ax.set_title(f"fold {m['fold_id']}  {m.get('val_start')}→{m.get('val_end')}")
        ax.set_xlabel("calibrated p")
        ax.set_ylabel("empirical P(y=1)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MODEL-V1 reliability (isotonic, OOS)", fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
