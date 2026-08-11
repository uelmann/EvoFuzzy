"""Report + charts for Phase A0 baseline."""

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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase A0 — Price-only LightGBM Baseline Report\n")
    lines.append("## Config\n")
    lines.append("```yaml\n" + json.dumps(cfg, indent=2) + "\n```\n")
    lines.append("## Sanity gates\n")
    for g in gates:
        lines.append(f"- **{g['name']}**: `{'PASS' if g.get('passed') else 'FAIL'}` — `{g}`\n")
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
    lines.append("## Kronos-ft export\n")
    lines.append(f"```json\n{json.dumps(kronos_status, indent=2)}\n```\n")
    lines.append("## Portfolio sweep (OOS)\n")
    lines.append(
        "| τ pct | Sharpe | CAGR | MaxDD | avg #pos | % flat | ann turnover |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in portfolio_rows:
        if "error" in r:
            lines.append(f"| {r.get('tau_pct')} | ERROR | | | | | |\n")
            continue
        lines.append(
            f"| {r['tau_pct']} | {r['net_sharpe']:.2f} | {r['net_cagr']:.2%} | "
            f"{r['max_drawdown']:.2%} | {r['avg_n_positions']:.1f} | "
            f"{r['pct_flat_days']:.1%} | {r['ann_turnover']:.1f} |\n"
        )
    lines.append("\n## Caveats / design choices\n")
    for c in caveats:
        lines.append(f"- {c}\n")
    path.write_text("".join(lines))


def plot_equity_curves(
    path: Path,
    baseline_eq: pd.DataFrame,
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

    b = _prep(baseline_eq, "lgbm baseline (best τ)")
    if naive_eq is not None and len(naive_eq):
        _prep(naive_eq, "naive_mom28")
    _prep(btc_eq, "BTC buy&hold")
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log, start=1)")
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("OOS net equity")

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
    ax1.set_title("Daily RankIC")
    ax1.grid(True, alpha=0.3)

    if quintiles:
        qs = sorted(quintiles)
        vals = [quintiles[q] for q in qs]
        ax2.bar([str(q) for q in qs], vals, color="#8250df")
        ax2.set_title("Quintile mean residual return")
        ax2.set_xlabel("Quintile (1=low score)")
        ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
