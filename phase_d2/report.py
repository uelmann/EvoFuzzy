"""Phase D.2 report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_d2.constants import ADOPTION_CRITERION, HONESTY_PREAMBLE, NOMINAL_BOOK_USD


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def plot_universe_equity(p1: dict, best: dict, best_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    e1 = p1.get("equity")
    e2 = best.get("equity") if best else None
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    ax = axes[0]
    for lab, eq, col in [("P1 A top-20", e1, "#4C78A8"), (best_name, e2, "#F58518")]:
        if not isinstance(eq, pd.DataFrame) or eq.empty:
            continue
        d = pd.to_datetime(eq["date"], utc=True)
        ax.plot(d, eq["equity"].values, label=lab, color=col, lw=1.6)
    ax.set_title("Phase D.2 equity — P1 vs best-of-P2/P4")
    ax.set_ylabel("Equity (1=start)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    end = None
    series = []
    for lab, eq, col in [("P1 A top-20", e1, "#4C78A8"), (best_name, e2, "#F58518")]:
        if not isinstance(eq, pd.DataFrame) or eq.empty:
            continue
        d = pd.to_datetime(eq["date"], utc=True)
        end = d.max() if end is None else max(end, d.max())
        series.append((lab, d, eq["equity"].values, col))
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for lab, d, y, col in series:
            m = d >= start
            if m.any():
                yy = np.asarray(y)[m.values]
                # rebase to 1 at zoom start
                if yy[0] != 0:
                    yy = yy / yy[0]
                ax2.plot(d[m], yy, label=lab, color=col, lw=1.6)
    ax2.set_title("Trailing-18m zoom (rebased)")
    ax2.set_ylabel("Equity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_hedge_bars(p1: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = p1.get("year_rows") or []
    rows = [r for r in rows if int(r.get("year", 0)) >= 2022]
    if not rows:
        return
    years = [str(r["year"]) for r in rows]
    x = np.arange(len(years))
    w = 0.25
    gross = [float(r.get("gross_total", 0.0)) for r in rows]
    hedge = [float(r.get("hedge_total", 0.0)) for r in rows]
    net = [float(r.get("net_total", 0.0)) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - w, gross, width=w, label="gross", color="#4C78A8")
    ax.bar(x, hedge, width=w, label="hedge", color="#F58518")
    ax.bar(x + w, net, width=w, label="net", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Sum of daily simple-return units")
    ax.set_title("Phase D.2 P1 hedge decomposition (per year)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_phaseD2_report(
    path: Path,
    *,
    frozen_hash: str,
    gates: list,
    tau_fix: list[dict],
    p_table: list[dict],
    ic_tables: list[dict],
    ic_nw: list[dict],
    verdict: dict,
    hedge_years: list[dict],
    oracle_years: list[dict],
    attr_2026: dict,
    extra: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase D.2 — top-40 execution, causal τ, micro ablation, hedge decomposition\n")
    lines.append(f"- Frozen A0 hash: `{frozen_hash}`")
    lines.append("- Scope: backtest/analysis only; no schedules or live components. Zero GPU.")
    lines.append(f"- Addendum (written before results): `reports/phaseD2_addendum.md`")
    lines.append(f"- Nominal book for liquidity cap: USD {NOMINAL_BOOK_USD:,.0f} for a 1.0 gross book.")
    lines.append("- Training-window τ on all D.2 portfolio runs (fold_train). Median-τ = house median of {60,70,80,90}.\n")

    lines.append("## 0. Honesty preamble (verbatim, before results)\n")
    lines.append(f"> {HONESTY_PREAMBLE}\n")

    lines.append("## Pre-registered adoption criterion (verbatim, before results)\n")
    lines.append(f"> {ADOPTION_CRITERION}\n")

    lines.append("## Gates\n")
    for g in gates or []:
        st = "PASS" if g.get("passed") else "FAIL"
        lines.append(f"- `{g.get('name')}`: **{st}** `{g}`")
    lines.append("")

    lines.append("## τ lookahead fix (A0 top-20 tranche h=7, funding on, lag 0)\n")
    lines.append("Same code path except the τ schedule. `pooled` = previous full-OOS |score| percentile (lookahead). `fold_train` = training-window only.\n")
    lines.append("| tau_mode | tau_pct | net Sharpe | gross Sharpe | cost_drag | funding | hedge | avg_npos | %flat |")
    lines.append("|----------|---------|------------|--------------|-----------|---------|-------|----------|-------|")
    for r in tau_fix or []:
        lines.append(
            f"| {r.get('tau_mode')} | {r.get('tau_pct')} | {_fmt(r.get('net_sharpe'), 3)} | "
            f"{_fmt(r.get('gross_sharpe'), 3)} | {_fmt(r.get('cost_drag'))} | {_fmt(r.get('funding_total_pnl'))} | "
            f"{_fmt(r.get('hedge_total_pnl'))} | {_fmt(r.get('avg_n_positions'), 2)} | {_fmt(r.get('pct_flat_days'), 2)} |"
        )
    lines.append("")
    tf60 = extra.get("tau_fix_one_liner", "")
    if tf60:
        lines.append(f"Isolation one-liner: {tf60}\n")

    lines.append("## P1–P4 (training-window τ, tranche, funding on, lag 0, median-τ headline, identical days)\n")
    lines.append(
        "| run | h | model | universe | τ | net Sharpe full | trail18m | y2022 | y2023 | y2024 | y2025 | y2026 | "
        "gross | cost | funding | hedge | avg_npos | %flat | ann to | avg rank |"
    )
    lines.append(
        "|-----|---|-------|----------|---|-----------------|----------|-------|-------|-------|-------|-------|"
        "-------|------|---------|-------|----------|-------|--------|----------|"
    )
    for r in p_table or []:
        by = r.get("net_sharpe_by_year") or {}
        lines.append(
            f"| {r.get('run_id')} | {r.get('horizon')} | {r.get('model')} | {r.get('universe')} | "
            f"{r.get('tau_pct')} | {_fmt(r.get('net_sharpe_full'), 3)} | {_fmt(r.get('net_sharpe_trail18m'), 3)} | "
            f"{_fmt(by.get(2022), 3)} | {_fmt(by.get(2023), 3)} | {_fmt(by.get(2024), 3)} | {_fmt(by.get(2025), 3)} | "
            f"{_fmt(by.get(2026), 3)} | {_fmt(r.get('gross_total_pnl'))} | {_fmt(r.get('cost_drag'))} | "
            f"{_fmt(r.get('funding_total_pnl'))} | {_fmt(r.get('hedge_total_pnl'))} | "
            f"{_fmt(r.get('avg_n_positions'), 2)} | {_fmt(r.get('pct_flat_days'), 2)} | "
            f"{_fmt(r.get('ann_turnover'), 2)} | {_fmt(r.get('avg_traded_rank'), 2)} |"
        )
    lines.append("")

    lines.append("## Paired ΔRankIC (A+micro vs A)\n")
    lines.append("| h | universe | window | A IC | A+micro IC | ΔIC | n_days |")
    lines.append("|---|----------|--------|------|------------|-----|--------|")
    for t in ic_tables or []:
        lines.append(
            f"| {t.get('horizon')} | {t.get('universe')} | {t.get('window')} | {_fmt(t.get('A_ic'))} | "
            f"{_fmt(t.get('B_ic'))} | {_fmt(t.get('delta_ic'))} | {t.get('n_days')} |"
        )
    lines.append("\n### Paired NW t on daily ΔIC\n")
    lines.append("| h | universe | window | mean ΔIC | NW-t | n_days |")
    lines.append("|---|----------|--------|----------|------|--------|")
    for p in ic_nw or []:
        lines.append(
            f"| {p.get('horizon')} | {p.get('universe')} | {p.get('window')} | {_fmt(p.get('mean_delta_ic'))} | "
            f"{_fmt(p.get('nw_tstat'), 2)} | {p.get('n_days')} |"
        )
    lines.append("")

    lines.append("## Mechanical verdicts\n")
    lines.append(f"> {ADOPTION_CRITERION}\n")
    lines.append(f"**Universe: {verdict.get('universe_verdict')}**")
    lines.append(f"**Micro on {verdict.get('chosen_universe')}: {verdict.get('micro_verdict')}**\n")
    lines.append("Universe comparisons (identical days):\n")
    lines.append("| candidate | h | trail18m | need (≥ P1+0.30) | full | need (≥ P1−0.20) | pass |")
    lines.append("|-----------|---|----------|------------------|------|------------------|------|")
    for r in verdict.get("universe_rows") or []:
        lines.append(
            f"| {r['candidate']} | {r['horizon']} | {_fmt(r['trail18m'], 3)} | {_fmt(r['need_trail18m'], 3)} | "
            f"{_fmt(r['full'], 3)} | {_fmt(r['need_full'], 3)} | {r['pass']} |"
        )
    lines.append("\nMicro comparisons:\n")
    lines.append("| h | universe | ΔIC trail18m | ΔIC full | ΔSharpe trail18m | pass |")
    lines.append("|---|----------|--------------|----------|------------------|------|")
    for r in verdict.get("micro_rows") or []:
        lines.append(
            f"| {r['horizon']} | {r['universe']} | {_fmt(r['delta_ic_trail18m'])} | {_fmt(r['delta_ic_full'])} | "
            f"{_fmt(r['delta_sharpe_trail18m'], 3)} | {r['pass']} |"
        )
    lines.append("")

    lines.append("## Hedge decomposition (P1, diagnostic only)\n")
    lines.append("Per calendar year, P1 (A, top-20, training-window median-τ, h=7):\n")
    lines.append("| year | gross | hedge | cost | funding | net | net Sharpe |")
    lines.append("|------|-------|-------|------|---------|-----|------------|")
    for r in hedge_years or []:
        lines.append(
            f"| {r.get('year')} | {_fmt(r.get('gross_total'))} | {_fmt(r.get('hedge_total'))} | "
            f"{_fmt(r.get('cost_drag'))} | {_fmt(r.get('funding_total'))} | {_fmt(r.get('net_total'))} | "
            f"{_fmt(r.get('net_sharpe'), 3)} |"
        )
    lines.append("\n### Oracle-beta counterfactual (LOOKAHEAD BY DESIGN, diagnostic only)\n")
    lines.append("Hedge ratio replaced with beta estimated on the forward window `[t, t+h]`. Δ(net) = oracle − actual = negative of beta-estimation cost if oracle is better.\n")
    lines.append("| year | actual net | oracle net | Δ(net) oracle−actual |")
    lines.append("|------|------------|------------|----------------------|")
    by_a = {int(r["year"]): r for r in (hedge_years or [])}
    by_o = {int(r["year"]): r for r in (oracle_years or [])}
    for y in sorted(set(by_a) | set(by_o)):
        a = by_a.get(y, {})
        o = by_o.get(y, {})
        an, on = a.get("net_total"), o.get("net_total")
        delta = (float(on) - float(an)) if an is not None and on is not None else float("nan")
        lines.append(f"| {y} | {_fmt(an)} | {_fmt(on)} | {_fmt(delta)} |")
    lines.append("")
    lines.append(f"{(attr_2026 or {}).get('sentence', '')}\n")
    lines.append("No production changes from this section.\n")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout_summary(verdict: dict, tau_line: str, attr_line: str) -> None:
    print("\n========== PHASE D.2 SUMMARY ==========", flush=True)
    print(f"UNIVERSE VERDICT: {verdict.get('universe_verdict')}", flush=True)
    print(f"MICRO VERDICT: {verdict.get('micro_verdict')} (chosen={verdict.get('chosen_universe')})", flush=True)
    print(f"TAU-FIX: {tau_line}", flush=True)
    print(f"2026 ATTRIBUTION: {attr_line}", flush=True)
    print("=======================================\n", flush=True)
