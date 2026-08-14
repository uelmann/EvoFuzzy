"""Phase 2.c report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import DEATH_CONVENTION, PHASE2C_CRITERION, PHASE2C_NULL_GATE


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
        f"| {_pct(b.get('gate_on_frac'))} | {_fmt(b.get('ann_turnover'), 2)} | {fe.get('n_events')} |"
    )


def _null_table(cells: list, real_key: str) -> list[str]:
    lines = [
        "| fold | n | null mean | SD | 95th | real | bias_ok | exceeds_p95 |",
        "|------|---|-----------|----|------|------|---------|-------------|",
    ]
    for c in cells or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'), 4)} | {_fmt(c.get('sd'), 4)} "
            f"| {_fmt(c.get('p95'), 4)} | {_fmt(c.get(real_key), 4)} | {c.get('bias_ok')} | {c.get('exceeds_p95')} |"
        )
    return lines


def write_phase2c(
    path: Path,
    *,
    skill: dict,
    null_gate: dict,
    fold_metrics: dict,
    headline: dict,
    grid: list,
    btc_ref: dict,
    naive_note: dict,
    verdicts: dict,
    importances_top: list,
    importances_bot: list,
    extra: dict,
) -> str:
    gates_ok = bool(extra.get("gates_ok"))
    ng = null_gate or {}
    ric_n = ng.get("rankic") or {}
    auc_n = ng.get("auc") or {}
    lines = [
        "# BTC-BEATER Phase 2.c — twin-head spread + repowered skill null",
        "",
        "**BACKTEST ONLY.** Cleaned+floored 2.b data reused as-is. Stage T frozen. "
        "Context excluded. CPU only, zero GPU. COMBO untouched.",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE2C_CRITERION}",
        "",
        "## Repowered skill null (verbatim, before results)",
        "",
        f"> {PHASE2C_NULL_GATE}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Mechanical verdicts",
        "",
        f"- **SPREAD has SELECTION SKILL: {skill.get('has_skill')}** "
        f"(h=14 RankIC={_fmt(skill.get('rankic_h14'), 4)} AUC={_fmt(skill.get('auc_h14'), 4)}; "
        f"h=30 RankIC={_fmt(skill.get('rankic_h30'), 4)} AUC={_fmt(skill.get('auc_h30'), 4)}; "
        f"§2 spread RankIC {ric_n.get('verdict')} {ric_n.get('n_exceed')}/6 "
        f"Stouffer z={_fmt(ric_n.get('stouffer_z'), 3)})",
        f"- **MODEL-V3 is {'VIABLE' if verdicts.get('viable') and gates_ok else 'NOT VIABLE'}**",
        f"- **MODEL-V3 is {'PRODUCT-GRADE' if verdicts.get('product_grade') and gates_ok else 'NOT PRODUCT-GRADE'}**",
        f"- median θ={headline.get('theta', headline.get('p_enter'))}; OOS {headline.get('start')} → {headline.get('end')} n={headline.get('n_days')}",
        f"- (a) rel Sharpe {_fmt(verdicts.get('rel_sharpe'))} > 0 → {verdicts.get('a_rel_sharpe_gt0')}",
        f"- (b) book {_pct(verdicts.get('book_total'))} vs BTC {_pct(verdicts.get('btc_total'))} → {verdicts.get('b_total_ge_btc')}",
        f"- (c) MaxDD {_pct(verdicts.get('maxdd'))} vs BTC {_pct(verdicts.get('btc_maxdd'))} → {verdicts.get('c_maxdd_le_btc')}",
        f"- product-grade need rel ≥ {_fmt(verdicts.get('need_product_rel'))} and alt ≥ {_pct(verdicts.get('need_product_alt'))}; "
        f"avg alt {_pct(verdicts.get('avg_alt'))}",
        f"- % time in BTC {_pct(headline.get('avg_w_btc'))}; avg #names {_fmt(headline.get('avg_n_names'), 2)}; "
        f"gate ON {_pct(headline.get('gate_on_frac'))}; forced={(headline.get('forced_exits') or {}).get('n_events')}",
        f"- uncertainty↔yz_vol_30 mean per-date RankIC = {_fmt(extra.get('uncert_vol_rankic'), 4)} "
        f"(lottery diagnostic; not used in trading)",
        "",
        "A verdict is not overridden by any single cycle. Operative floor is BTC. "
        f"Naive v4 (record only): rel Sharpe={_fmt((naive_note or {}).get('rel_sharpe'))}, "
        f"live_benchmark={(naive_note or {}).get('live_benchmark')}.",
        "",
        "## §2 null — p_top per-date AUC (h=14)",
        "",
        f"Bias pass={auc_n.get('bias_pass')}; skill {auc_n.get('verdict')}; "
        f"{auc_n.get('n_exceed')}/6 exceed p95; Stouffer z={_fmt(auc_n.get('stouffer_z'), 3)}.",
        "",
    ]
    lines += _null_table(ng.get("auc_cells") or [], "real_auc")
    lines += [
        "",
        "## §2 null — spread per-date RankIC (h=14, judged signal)",
        "",
        f"Bias pass={ric_n.get('bias_pass')}; skill {ric_n.get('verdict')}; "
        f"{ric_n.get('n_exceed')}/6 exceed p95; Stouffer z={_fmt(ric_n.get('stouffer_z'), 3)}. "
        "Failure = PARKED, no override, no retest with different folds.",
        "",
    ]
    lines += _null_table(ng.get("rankic_cells") or [], "real_rankic")
    lines += [
        "",
        "## Per-fold selection metrics (floored PIT top-100)",
        "",
        "| h | fold | RankIC spread | RankIC p_top (control) | AUC spread vs top-q | AUC p_top |",
        "|---|------|---------------|------------------------|---------------------|-----------|",
    ]
    for h, rows in (fold_metrics or {}).items():
        for r in rows:
            lines.append(
                f"| {h} | {r.get('fold_id')} | {_fmt(r.get('rankic_spread'), 4)} "
                f"| {_fmt(r.get('rankic_ptop'), 4)} | {_fmt(r.get('auc_spread'), 4)} "
                f"| {_fmt(r.get('auc_ptop'), 4)} |"
            )
    lines += [
        "",
        f"Aggregate (last-fold-wins OOS): h=14 RankIC(spread)={_fmt(skill.get('rankic_h14'), 4)} "
        f"RankIC(p_top)={_fmt(skill.get('rankic_ptop_h14'), 4)} AUC(spread)={_fmt(skill.get('auc_h14'), 4)}; "
        f"h=30 RankIC(spread)={_fmt(skill.get('rankic_h30'), 4)} "
        f"RankIC(p_top)={_fmt(skill.get('rankic_ptop_h30'), 4)} AUC(spread)={_fmt(skill.get('auc_h30'), 4)}.",
        "",
        "## MODEL-V3 book vs BTC (same OOS window)",
        "",
        "| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC | gate ON | ann TO | forced |",
        "|------|-------|------|------------|------------|-------|------------|-------|---------|--------|--------|",
        _book_row(f"MODEL-V3 h=14 θ={headline.get('theta', headline.get('p_enter'))}", headline),
        _book_row("BTC B&H", btc_ref),
        "",
        "## θ grid (h=14, median convention)",
        "",
        "| θ | rel Sharpe | total | MaxDD | % BTC | avg #names | gate ON |",
        "|---|------------|-------|-------|-------|------------|---------|",
    ]
    for r in grid or []:
        th = r.get("theta", r.get("p_enter"))
        mark = " ← median" if th == headline.get("theta", headline.get("p_enter")) else ""
        lines.append(
            f"| {th}{mark} | {_fmt(r.get('rel_sharpe'))} | {_pct(r.get('book_total'))} "
            f"| {_pct(r.get('maxdd'))} | {_pct(r.get('avg_w_btc'))} "
            f"| {_fmt(r.get('avg_n_names'), 2)} | {_pct(r.get('gate_on_frac'))} |"
        )
    lines += [
        "",
        "## Per-cycle honesty (headline h=14)",
        "",
        "| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | % BTC |",
        "|-------|---|----------|---------|------------|------------|-------|-------|",
    ]
    for name, c in (headline.get("cycles") or {}).items():
        lines.append(
            f"| {name} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('btc_total'))} "
            f"| {_fmt(c.get('book_sharpe'))} | {_fmt(c.get('rel_sharpe'))} | {_pct(c.get('maxdd'))} "
            f"| {_pct(c.get('avg_w_btc'))} |"
        )
    lines += [
        "",
        "## Feature importances (mean gain, h=14)",
        "",
        "### Head TOP",
        "",
        "| rank | feature | mean gain |",
        "|------|---------|-----------|",
    ]
    for i, (feat, g) in enumerate(importances_top or [], start=1):
        lines.append(f"| {i} | `{feat}` | {_fmt(g, 2)} |")
    lines += [
        "",
        "### Head BOTTOM",
        "",
        "| rank | feature | mean gain |",
        "|------|---------|-----------|",
    ]
    for i, (feat, g) in enumerate(importances_bot or [], start=1):
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


def plot_equity_gate(model: dict, gate: pd.Series | None, out_path: Path) -> None:
    eq = model.get("equity")
    eqb = model.get("equity_btc")
    rel = model.get("rel_equity")
    if eq is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 0.5]},
    )
    ax = axes[0]
    ax.plot(eq.index, eq.values, lw=1.3, label="MODEL-V3")
    if eqb is not None:
        ax.plot(eqb.index, eqb.values, lw=1.2, label="BTC B&H")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("BTC-BEATER Phase 2.c — MODEL-V3 vs BTC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if rel is not None:
        ax2.plot(rel.index, rel.values, lw=1.3, label="V3 / BTC")
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


def plot_rankic_series(spread: pd.Series, ptop: pd.Series, out_path: Path) -> None:
    if spread is None or spread.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.plot(spread.index, spread.values, lw=0.8, alpha=0.45, color="#4C78A8")
    rs = spread.rolling(30, min_periods=10).mean()
    ax.plot(rs.index, rs.values, lw=1.5, color="#4C78A8", label="RankIC(spread) 30d")
    if isinstance(ptop, pd.Series) and len(ptop):
        rp = ptop.rolling(30, min_periods=10).mean()
        ax.plot(rp.index, rp.values, lw=1.4, color="#E45756", label="RankIC(p_top) 30d")
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.01, color="#54A24B", ls="--", lw=0.8, label="+0.01 skill line")
    ax.set_ylabel("per-date RankIC")
    ax.set_title("Stage S — OOS RankIC of spread vs p_top control")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration_pair(metas_top: list[dict], metas_bot: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _picks(metas):
        ok = [m for m in metas if m.get("status") == "ok" and (m.get("reliability") or {}).get("mean_p")]
        if not ok:
            return []
        picks = [ok[0], ok[len(ok) // 2], ok[-1]]
        seen, uniq = set(), []
        for m in picks:
            if m["fold_id"] in seen:
                continue
            seen.add(m["fold_id"])
            uniq.append(m)
        return uniq

    t, b = _picks(metas_top), _picks(metas_bot)
    n = max(len(t), len(b), 1)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.2), constrained_layout=True)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for row, (ms, title) in enumerate(((t, "TOP head"), (b, "BOTTOM head"))):
        for i in range(n):
            ax = axes[row, i]
            if i >= len(ms):
                ax.axis("off")
                continue
            m = ms[i]
            rel = m["reliability"]
            ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1)
            ax.plot(rel["mean_p"], rel["frac_pos"], marker="o", lw=1.2)
            ax.set_title(f"{title} fold {m['fold_id']}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel("calibrated p")
            if i == 0:
                ax.set_ylabel("empirical P(y=1)")
    fig.suptitle("MODEL-V3 twin-head reliability (isotonic, OOS)", fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
