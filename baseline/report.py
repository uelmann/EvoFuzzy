"""Report + charts for Phase A0 baseline (stress-test edition)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_report(
    path: Path,
    cfg: dict,
    ic_tables: dict,
    portfolio_rows: list[dict],
    gates: list[dict],
    caveats: list[str],
    kronos_status: dict,
    best_iter_stats: dict | None = None,
    luna_report: dict | None = None,
    sensitivity: dict | None = None,
    funding_coverage: dict | None = None,
    median_tau: list[dict] | None = None,
    lag_compare: list[dict] | None = None,
    attribution: dict | None = None,
    ic_diag: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase A0 — Price-only LightGBM Baseline (stress-test pass)\n")
    lines.append("## Config\n")
    lines.append("```yaml\n" + json.dumps(cfg, indent=2) + "\n```\n")

    lines.append("## Sanity gates\n")
    for g in gates:
        lines.append(f"- **{g['name']}**: `{'PASS' if g.get('passed') else 'FAIL'}` — `{g}`\n")

    lines.append("\n## Decomposition definitions (units)\n")
    lines.append(
        "- **gross / hedge / funding / cost / net** totals in tables are **sums of daily "
        "simple-return contributions** over the OOS window (not Sharpe ratios).\n"
        "- **gross_sharpe / net_sharpe** are Sharpe ratios: "
        "`mean(daily) / std(daily) * sqrt(365)`.\n"
        "- **cost_ann_drag / funding_ann_return / …** are annualized mean return fractions: "
        "`365 * mean(daily_component)`.\n"
        "- **Identity (daily and cumulative):** "
        "`net = gross + hedge − cost + funding` "
        "(funding sign: longs pay when funding_rate>0 → contribution `−w·f`).\n"
        "- **net Sharpe can exceed gross Sharpe** when the BTC hedge reduces return variance "
        "more than it reduces mean — verified when `net_sharpe > gross_sharpe` with "
        "positive identity check (`identity_gap ≈ 0`).\n"
    )

    if funding_coverage is not None:
        lines.append("\n## Funding data coverage\n")
        lines.append(f"```json\n{json.dumps(funding_coverage, indent=2, default=str)}\n```\n")
        lines.append(
            "- Dates/symbols without a Vision funding series accrue **funding = 0**; "
            "pre-listing / missing months are in `missing_symbols_sample`.\n"
        )

    if luna_report is not None:
        lines.append("\n## LUNA presence in PIT top-20 (2021–2022)\n")
        lines.append(f"```json\n{json.dumps(luna_report, indent=2)}\n```\n")

    if best_iter_stats is not None:
        lines.append("\n## best_iteration distribution\n")
        lines.append(f"```json\n{json.dumps(best_iter_stats, indent=2)}\n```\n")

    if sensitivity is not None:
        lines.append("\n## Tree-count sensitivity (fallback)\n")
        lines.append(f"```json\n{json.dumps(sensitivity, indent=2, default=str)}\n```\n")

    lines.append("\n## RankIC summary\n")
    for key, rows in ic_tables.items():
        lines.append(f"### {key}\n")
        lines.append("| universe | mean IC | std | ICIR | NW t |\n|---|---:|---:|---:|---:|\n")
        for r in rows:
            lines.append(
                f"| {r.get('universe')} | {r.get('mean_ic', float('nan')):.4f} | "
                f"{r.get('std_ic', float('nan')):.4f} | {r.get('icir', float('nan')):.3f} | "
                f"{r.get('nw_tstat', float('nan')):.2f} |\n"
            )
        lines.append("\n")

    if ic_diag is not None:
        lines.append("## IC diagnostic (top-20 vs dispersion)\n")
        d = {k: v for k, v in ic_diag.items() if k not in ("ic_series", "disp_series")}
        lines.append(f"```json\n{json.dumps(d, indent=2)}\n```\n")
        if d.get("advantage_disappears"):
            lines.append(
                "- **Plain statement:** excluding the 10 highest cross-sectional-dispersion days, "
                "the top-20 mean IC drops sharply — the top-20 advantage is concentrated in "
                "high-dispersion days.\n"
            )
        else:
            lines.append(
                "- **Plain statement:** excluding the 10 highest dispersion days, top-20 mean IC "
                "remains comparable — the advantage is not solely a high-dispersion artifact.\n"
            )

    lines.append("## Kronos-ft export\n")
    lines.append(f"```json\n{json.dumps(kronos_status, indent=2)}\n```\n")

    if median_tau is not None:
        lines.append("## Headline: median-τ net Sharpe (funding on, lag 0 unless noted)\n")
        lines.append(
            "| variant | h | lag | funding | median Sharpe | best Sharpe | median τ | best τ | funding PnL (median τ) |\n"
            "|---|---:|---:|---|---:|---:|---:|---:|---:|\n"
        )
        for r in median_tau:
            lines.append(
                f"| {r.get('variant')} | {r.get('horizon')} | {r.get('lag')} | {r.get('funding_on')} | "
                f"{r.get('median_net_sharpe', float('nan')):.2f} | {r.get('best_net_sharpe', float('nan')):.2f} | "
                f"{r.get('median_tau')} | {r.get('best_tau')} | {r.get('median_funding_pnl', float('nan')):.3f} |\n"
            )
        lines.append(
            "\n> **Headline number = median-τ** across the τ grid. best-τ is reference only.\n"
        )

    if lag_compare is not None:
        lines.append("\n## Execution lag: lag-0 vs lag-1 (funding on)\n")
        lines.append(
            "| variant | h | τ | lag0 Sharpe | lag1 Sharpe | funding PnL lag0 |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for r in lag_compare:
            lines.append(
                f"| {r.get('variant')} | {r.get('horizon')} | {r.get('tau_pct')} | "
                f"{r.get('sharpe_lag0', float('nan')):.2f} | {r.get('sharpe_lag1', float('nan')):.2f} | "
                f"{r.get('funding_lag0', float('nan')):.3f} |\n"
            )
        lines.append(
            "\n- **Interpretation:** lag-1 is the pessimistic bound (signal at close t, trade at close t+1). "
            "Realistic 24/7 crypto execution sits near lag-0.\n"
        )

    lines.append("\n## Portfolio sweep with PnL decomposition\n")
    lines.append(
        "| variant | h | τ | lag | fund | netSh | grSh | gross | cost | hedge | funding | net | hold | annTO |\n"
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in portfolio_rows:
        if "error" in r:
            lines.append(
                f"| {r.get('variant')} | {r.get('horizon')} | {r.get('tau_pct')} | "
                f"{r.get('lag')} | {r.get('funding_on')} | ERROR | | | | | | | | |\n"
            )
            continue
        lines.append(
            f"| {r.get('variant')} | {r.get('horizon')} | {r['tau_pct']} | {r.get('lag', 0)} | "
            f"{r.get('funding_on')} | {r['net_sharpe']:.2f} | {r.get('gross_sharpe', float('nan')):.2f} | "
            f"{r.get('gross_total_pnl', float('nan')):.3f} | {r.get('cost_drag', float('nan')):.3f} | "
            f"{r.get('hedge_total_pnl', float('nan')):.3f} | {r.get('funding_total_pnl', float('nan')):.3f} | "
            f"{r.get('net_total_pnl', float('nan')):.3f} | {r.get('avg_holding_days', float('nan')):.1f} | "
            f"{r['ann_turnover']:.1f} |\n"
        )

    if attribution is not None:
        lines.append("\n## Attribution (best tranche, funding on, lag 0)\n")
        lines.append("### Per-year breakdown\n")
        lines.append(
            "| year | net Sharpe | gross | cost | funding | hedge | net |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for r in attribution.get("per_year", []):
            lines.append(
                f"| {r['year']} | {r['net_sharpe']:.2f} | {r['gross_total']:.3f} | "
                f"{r['cost_drag']:.3f} | {r['funding_total']:.3f} | {r['hedge_total']:.3f} | "
                f"{r['net_total']:.3f} |\n"
            )
        lines.append("\n### Day concentration\n")
        lines.append(f"```json\n{json.dumps(attribution.get('concentration', {}), indent=2)}\n```\n")
        lines.append("\n### Top/bottom 10 symbols (% of total PnL)\n")
        lines.append("| rank | symbol | pnl | % | side |\n|---:|---|---:|---:|---|\n")
        for i, r in enumerate(attribution.get("symbols", {}).get("top", []), 1):
            lines.append(
                f"| T{i} | {r['symbol']} | {r['pnl']:.4f} | {r['pct_of_total']:.2f}% | {r['dominant_side']} |\n"
            )
        for i, r in enumerate(attribution.get("symbols", {}).get("bottom", []), 1):
            lines.append(
                f"| B{i} | {r['symbol']} | {r['pnl']:.4f} | {r['pct_of_total']:.2f}% | {r['dominant_side']} |\n"
            )
        lines.append("\n### LUNA / FTT collapse contribution\n")
        lines.append(f"```json\n{json.dumps(attribution.get('symbols', {}).get('collapse', {}), indent=2)}\n```\n")

    lines.append("\n## Caveats / design choices\n")
    for c in caveats:
        lines.append(f"- {c}\n")
    path.write_text("".join(lines))


def plot_equity_curves(
    path: Path,
    tranche_lag0: pd.DataFrame,
    tranche_lag1: pd.DataFrame | None,
    naive_eq: pd.DataFrame | None,
    btc_eq: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    def _prep(eq: pd.DataFrame, name: str):
        e = eq.copy()
        e["date"] = pd.to_datetime(e["date"], utc=True)
        e = e.sort_values("date")
        e["equity"] = e["equity"] / e["equity"].iloc[0]
        ax1.plot(e["date"], e["equity"], label=name, lw=1.8)
        return e

    b = _prep(tranche_lag0, "tranche fund-on lag0")
    if tranche_lag1 is not None and len(tranche_lag1):
        _prep(tranche_lag1, "tranche fund-on lag1")
    if naive_eq is not None and len(naive_eq):
        _prep(naive_eq, "naive_mom28")
    _prep(btc_eq, "BTC buy&hold")
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log, start=1)")
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("OOS net equity (stress)")

    dd = b["equity"] / b["equity"].cummax() - 1.0
    ax2.fill_between(b["date"], dd, 0, color="#cf222e", alpha=0.35)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_analysis(path: Path, ic: pd.Series, quintiles: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ic = ic.sort_index()
    cum = ic.cumsum()
    roll = ic.rolling(90, min_periods=20).mean()
    ax1.plot(cum.index, cum.values, label="cum IC", color="#0969da", lw=1.8)
    ax1.plot(roll.index, roll.values, label="90d mean IC", color="#cf222e", lw=1.5)
    ax1.axhline(0, color="#d0d7de", lw=1)
    ax1.legend(frameon=False)
    ax1.set_title("Daily RankIC (top-20)")
    ax1.grid(True, alpha=0.3)

    if quintiles:
        qs = sorted(quintiles)
        vals = [quintiles[q] for q in qs]
        ax2.bar([str(q) for q in qs], vals, color="#0969da")
        ax2.set_title("Quintile mean residual return")
        ax2.set_xlabel("Quintile (1=low score)")
        ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attribution(
    path: Path,
    per_year: list[dict],
    top_symbols: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if per_year:
        years = [r["year"] for r in per_year]
        sharpes = [r["net_sharpe"] for r in per_year]
        ax1.bar([str(y) for y in years], sharpes, color="#0969da")
        ax1.axhline(0, color="#d0d7de", lw=1)
        ax1.set_title("Per-year net Sharpe")
        ax1.grid(True, axis="y", alpha=0.3)
    if top_symbols:
        # cumulative PnL of top-10 as bar of total contrib
        syms = [r["symbol"][:10] for r in top_symbols]
        vals = np.cumsum([r["pnl"] for r in top_symbols])
        ax2.plot(range(1, len(vals) + 1), vals, marker="o", color="#cf222e", lw=1.8)
        ax2.set_xticks(range(1, len(vals) + 1))
        ax2.set_xticklabels(syms, rotation=45, ha="right", fontsize=8)
        ax2.set_title("Cumulative PnL of top-10 symbols")
        ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
