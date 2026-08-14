"""Phase 2.b report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import DEATH_CONVENTION, PHASE1_LABEL, PHASE2B_CRITERION


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


def write_phase2b(
    path: Path,
    *,
    autopsy: dict,
    clean_summary: dict,
    floor50: dict,
    floor100: dict,
    naive: dict,
    naive_contrib: dict,
    gates: list,
    null_gate: dict,
    skill: dict,
    headline: dict,
    grid: list,
    btc_ref: dict,
    verdicts: dict,
    metas: dict,
    importances: list,
    extra: dict,
) -> str:
    lines = [
        "# BTC-BEATER Phase 2.b — hygiene, naive v4, MODEL-V2",
        "",
        "**BACKTEST ONLY.** Hygiene before any backtest. Stage S is within-date only (no context). "
        "Stage T is a frozen regime gate, not learned. CPU only, zero GPU. COMBO untouched.",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE2B_CRITERION}",
        "",
        "## Naive v4 label (verbatim)",
        "",
        f"> {PHASE1_LABEL}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## 1. Naive v3 autopsy (the +9.98M% book)",
        "",
        f"Window {autopsy.get('start')} → {autopsy.get('end')}; book total {_pct(autopsy.get('book_total'))}; "
        f"rel Sharpe {_fmt(autopsy.get('rel_sharpe'))}. Additive PnL contributions (top 10):",
        "",
        "| rank | id | symbol | contrib | share | max daily ret | max |abs| daily ret | BTC? |",
        "|------|----|--------|---------|-------|---------------|----------------------|------|",
    ]
    for i, r in enumerate((autopsy.get("contrib_table") or {}).get("top") or [], start=1):
        lines.append(
            f"| {i} | {r.get('id')} | {r.get('symbol')} | {_fmt(r.get('contrib'), 4)} "
            f"| {_pct(r.get('share'))} | {_fmt(r.get('max_daily_ret'), 2)} "
            f"| {_fmt(r.get('max_abs_daily_ret'), 2)} | {r.get('is_btc')} |"
        )
    at = autopsy.get("contrib_table") or {}
    lines += [
        "",
        f"Top alt share of additive PnL = {_pct(at.get('top_alt_share'))}. "
        f"Flag >25%: **{at.get('flag_single_name_gt_25pct')}**.",
        "",
        "## 2. Redenom/split cleaning",
        "",
        f"- Jump threshold |daily ret| > 5. Ids touched={clean_summary.get('n_ids_touched')}; "
        f"splice events={clean_summary.get('n_splice')}; truncate events={clean_summary.get('n_truncate')}.",
        f"- Rows {clean_summary.get('n_rows_before')} → {clean_summary.get('n_rows_after')}.",
        f"- Remaining |daily ret|>5 after clean: {clean_summary.get('n_jumps_after')} "
        f"(BTC-exempt jumps ignored in cleaner).",
        "",
        "| id | symbol | n_jumps | truncated | actions |",
        "|----|--------|---------|-----------|---------|",
    ]
    for r in (clean_summary.get("log_head") or [])[:120]:
        acts = ",".join(a.get("action", "?") + "@" + str(a.get("date", "")) for a in (r.get("actions") or [])[:4])
        lines.append(
            f"| {r.get('id')} | {r.get('symbol')} | {r.get('n_jumps_raw')} | {r.get('truncated')} | {acts} |"
        )
    lines += [
        "",
        "## 3. Investability floor (PIT rebuild)",
        "",
        f"Floor: 30d median DV ≥ $2e6, price ≥ 1e-6, ≥60 prior sessions, no |ret|>200% in 30d.",
        f"- top-50 floored: eligible ids={floor50.get('n_eligible_ids')} med eligible/day={_fmt(floor50.get('median_eligible_per_date'), 1)} "
        f"pit rows={floor50.get('pit_rows')} vs unfloored rows={extra.get('old50_rows')} ids={extra.get('old50_ids')} med/day={_fmt(extra.get('old50_med'), 1)}",
        f"- top-100 floored: eligible ids={floor100.get('n_eligible_ids')} med eligible/day={_fmt(floor100.get('median_eligible_per_date'), 1)} "
        f"pit rows={floor100.get('pit_rows')} vs unfloored rows={extra.get('old100_rows')} ids={extra.get('old100_ids')} med/day={_fmt(extra.get('old100_med'), 1)}",
        "",
        "## 4. Naive v4 (cleaned + floored)",
        "",
        f"- **NAIVE-ROTATION v4 is {'a LIVE BENCHMARK' if naive.get('live_benchmark') else 'NOT A LIVE BENCHMARK'}**",
        f"- rel Sharpe={_fmt(naive.get('rel_sharpe'))}; book {_pct(naive.get('book_total'))} vs BTC {_pct(naive.get('btc_total'))}",
        f"- MaxDD {_pct(naive.get('maxdd'))}; %BTC {_pct(naive.get('avg_w_btc'))}; forced={(naive.get('forced_exits') or {}).get('n_events')}",
        f"- Top alt contrib share {_pct((naive_contrib or {}).get('top_alt_share'))}; "
        f"flag >25%: **{(naive_contrib or {}).get('flag_single_name_gt_25pct')}**",
        "",
        "| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC | ann TO | forced |",
        "|------|-------|------|------------|------------|-------|------------|-------|--------|--------|",
        _book_row("naive v4", naive),
        _book_row(
            "BTC B&H",
            {
                "book_total": naive.get("btc_total"),
                "book_cagr": naive.get("btc_cagr"),
                "book_sharpe": naive.get("btc_sharpe"),
                "rel_sharpe": 0.0,
                "maxdd": naive.get("btc_maxdd"),
                "avg_n_names": 0.0,
                "avg_w_btc": 1.0,
                "ann_turnover": 0.0,
                "forced_exits": {"n_events": 0},
            },
        ),
        "",
        "| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | % BTC |",
        "|-------|---|----------|---------|------------|------------|-------|-------|",
    ]
    for name, c in (naive.get("cycles") or {}).items():
        lines.append(
            f"| {name} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('btc_total'))} "
            f"| {_fmt(c.get('book_sharpe'))} | {_fmt(c.get('rel_sharpe'))} | {_pct(c.get('maxdd'))} "
            f"| {_pct(c.get('avg_w_btc'))} |"
        )
    lines += [
        "",
        "Naive v4 top-10 contributors:",
        "",
        "| rank | id | symbol | share | max daily ret |",
        "|------|----|--------|-------|---------------|",
    ]
    for i, r in enumerate((naive_contrib or {}).get("top") or [], start=1):
        lines.append(
            f"| {i} | {r.get('id')} | {r.get('symbol')} | {_pct(r.get('share'))} | {_fmt(r.get('max_daily_ret'), 2)} |"
        )
    gates_ok = bool(extra.get("gates_ok"))
    lines += [
        "",
        "## 5. Stage-S gates",
        "",
        f"Gates official: **{'PASS' if gates_ok else 'FAIL'}**.",
        "",
        "| gate | passed | detail |",
        "|------|--------|--------|",
    ]
    for g in gates or []:
        detail = {k: v for k, v in g.items() if k not in {"name", "passed", "cells"}}
        lines.append(f"| {g.get('name')} | {g.get('passed')} | `{detail}` |")
    ng = null_gate or {}
    lines += [
        "",
        f"Label-shuffle null (mean per-date AUC, 2 folds × 25): **{ng.get('verdict')}** "
        f"bias_pass={ng.get('bias_pass')} skill_pass={ng.get('skill_pass')}. "
        "Bias: |null mean − 0.5| ≤ 2·(SD/√R).",
        "",
        "| fold | n | null mean | SD | 95th | real pdauc | bias_ok | exceeds_p95 |",
        "|------|---|-----------|----|------|------------|---------|-------------|",
    ]
    for c in ng.get("cells") or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'), 4)} | {_fmt(c.get('sd'), 4)} "
            f"| {_fmt(c.get('p95'), 4)} | {_fmt(c.get('real_auc'), 4)} | {c.get('bias_ok')} | {c.get('exceeds_p95')} |"
        )
    lines += [
        "",
        "## Stage-S per-date AUC by fold",
        "",
        "| h | fold | val_start | val_end | n_valid | pdauc | rankIC | best_iter |",
        "|---|------|-----------|---------|---------|-------|--------|-----------|",
    ]
    for h, ms in (metas or {}).items():
        for m in ms:
            if m.get("status") != "ok":
                continue
            lines.append(
                f"| {h} | {m.get('fold_id')} | {m.get('val_start')} | {m.get('val_end')} "
                f"| {m.get('n_valid')} | {_fmt(m.get('pdauc_oos'), 4)} | {_fmt(m.get('rankic_oos'), 4)} "
                f"| {m.get('best_iteration')} |"
            )
    hl = headline or {}
    lines += [
        "",
        "## Mechanical verdicts",
        "",
        f"- **STAGE-S has SELECTION SKILL: {skill.get('has_skill')}** "
        f"(h=14 mean per-date AUC={_fmt(skill.get('mean_pdauc'), 4)}, need ≥ 0.52; "
        f"mean per-date RankIC={_fmt(skill.get('mean_rankic'), 4)}; "
        f"null={skill.get('null_verdict')}; gates_ok={gates_ok})",
        f"- **MODEL-V2 is {'VIABLE' if verdicts.get('viable') and gates_ok else 'NOT VIABLE'}**",
        f"- **REPLACES naive v4 floor: {bool(verdicts.get('replaces_floor') and gates_ok)}**",
        f"- median p_enter={hl.get('p_enter')}; OOS {hl.get('start')} → {hl.get('end')} n={hl.get('n_days')}",
        f"- (a) rel Sharpe {_fmt(verdicts.get('rel_sharpe'))} > 0 → {verdicts.get('a_rel_sharpe_gt0')}",
        f"- (b) book {_pct(verdicts.get('book_total'))} vs BTC {_pct(verdicts.get('btc_total'))} → {verdicts.get('b_total_ge_btc')}",
        f"- (c) MaxDD {_pct(verdicts.get('maxdd'))} vs BTC {_pct(verdicts.get('btc_maxdd'))} → {verdicts.get('c_maxdd_le_btc')}",
        f"- replace need ≥ {_fmt(verdicts.get('need_replaces'))} (naive v4 same-window {_fmt(verdicts.get('naive_rel_sharpe'))} + 0.15)",
        f"- % time in BTC {_pct(hl.get('avg_w_btc'))}; gate ON frac {_pct(hl.get('gate_on_frac'))}; "
        f"forced={(hl.get('forced_exits') or {}).get('n_events')}",
        "",
        "A verdict is not overridden by any single cycle.",
        "",
        "## MODEL-V2 book vs naive v4 vs BTC (same OOS window)",
        "",
        "| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC | ann TO | forced |",
        "|------|-------|------|------------|------------|-------|------------|-------|--------|--------|",
        _book_row(f"MODEL-V2 h=14 p={hl.get('p_enter')}", hl),
        _book_row("naive v4 (same window)", extra.get("naive_oos") or {}),
        _book_row("BTC B&H", btc_ref),
        "",
        "## p_enter grid (h=14)",
        "",
        "| p_enter | rel Sharpe | total | MaxDD | % BTC | gate ON |",
        "|---------|------------|-------|-------|-------|---------|",
    ]
    for r in grid or []:
        mark = " ← median" if r.get("p_enter") == hl.get("p_enter") else ""
        lines.append(
            f"| {r.get('p_enter')}{mark} | {_fmt(r.get('rel_sharpe'))} | {_pct(r.get('book_total'))} "
            f"| {_pct(r.get('maxdd'))} | {_pct(r.get('avg_w_btc'))} | {_pct(r.get('gate_on_frac'))} |"
        )
    lines += [
        "",
        "## Per-cycle honesty (headline h=14)",
        "",
        "| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | % BTC |",
        "|-------|---|----------|---------|------------|------------|-------|-------|",
    ]
    for name, c in (hl.get("cycles") or {}).items():
        lines.append(
            f"| {name} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('btc_total'))} "
            f"| {_fmt(c.get('book_sharpe'))} | {_fmt(c.get('rel_sharpe'))} | {_pct(c.get('maxdd'))} "
            f"| {_pct(c.get('avg_w_btc'))} |"
        )
    lines += [
        "",
        "## Feature importances (mean gain, h=14 Stage S — should be per-coin)",
        "",
        "| rank | feature | mean gain |",
        "|------|---------|-----------|",
    ]
    for i, (feat, g) in enumerate(importances or [], start=1):
        lines.append(f"| {i} | `{feat}` | {_fmt(g, 2)} |")
    lines += [
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU={extra.get('gpu_used', False)}. "
        f"n_features={extra.get('n_features')}.",
        "",
        "COMBO untouched (v2.0-combo-final).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_equity_gate(model: dict, naive: dict, gate: pd.Series | None, out_path: Path) -> None:
    eq = model.get("equity")
    eqb = model.get("equity_btc")
    rel = model.get("rel_equity")
    if eq is None:
        return
    n_eq = naive.get("equity")
    n_rel = naive.get("rel_equity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 0.5]})
    ax = axes[0]
    ax.plot(eq.index, eq.values, lw=1.3, label="MODEL-V2")
    if eqb is not None:
        ax.plot(eqb.index, eqb.values, lw=1.2, label="BTC B&H")
    if isinstance(n_eq, pd.Series) and len(n_eq):
        ax.plot(n_eq.index, n_eq.values, lw=1.1, alpha=0.85, label="naive v4")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("BTC-BEATER Phase 2.b — MODEL-V2 vs BTC vs naive v4")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if rel is not None:
        ax2.plot(rel.index, rel.values, lw=1.3, label="V2 / BTC")
    if isinstance(n_rel, pd.Series) and len(n_rel):
        ax2.plot(n_rel.index, n_rel.values, lw=1.1, alpha=0.85, label="naive v4 / BTC")
    ax2.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax2.set_ylabel("relative equity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax3 = axes[2]
    gser = model.get("gate_on") if gate is None else gate
    if isinstance(gser, pd.Series) and len(gser):
        ax3.fill_between(gser.index, 0, gser.values.astype(float), step="pre", alpha=0.7, color="#4C78A8")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("gate ON")
    ax3.set_yticks([0, 1])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pdauc_series(series: pd.Series, out_path: Path) -> None:
    if series is None or series.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.plot(series.index, series.values, lw=1.0)
    roll = series.rolling(30, min_periods=10).mean()
    ax.plot(roll.index, roll.values, lw=1.4, label="30d mean")
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.52, color="#E45756", ls="--", lw=0.8, label="0.52 skill line")
    ax.set_ylabel("per-date AUC")
    ax.set_title("Stage S — OOS per-date AUC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(metas: list[dict], out_path: Path) -> None:
    if not metas:
        return
    ok = [m for m in metas if m.get("status") == "ok" and (m.get("reliability") or {}).get("mean_p")]
    if not ok:
        return
    picks = [ok[0], ok[len(ok) // 2], ok[-1]]
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
    fig.suptitle("MODEL-V2 Stage S reliability (isotonic, OOS)", fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
