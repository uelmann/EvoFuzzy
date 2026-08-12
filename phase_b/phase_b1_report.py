"""Phase B.1 report + charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_b.control_gates import ADOPTION_RULE, CUTOFF


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def plot_gate_equities(
    curves: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Paired post-cutoff equity curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for name, eq in curves.items():
        if eq is None or eq.empty:
            continue
        e = eq.copy()
        e["date"] = pd.to_datetime(e["date"], utc=True)
        e = e[e["date"] >= CUTOFF]
        if e.empty:
            continue
        # rebase to 1 at first post-cutoff point
        e = e.sort_values("date")
        base = float(e["equity"].iloc[0])
        y = e["equity"] / base if base > 0 else e["equity"]
        ax.plot(e["date"], y.values, label=name)
    ax.axvline(CUTOFF, color="black", ls="--", lw=1, label="cutoff")
    ax.set_title("Post-cutoff equity (rebased) — ungated vs kr_sigma vs best control")
    ax.set_ylabel("Equity (rebased)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rolling_sharpe(roll: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    r = roll.dropna().sort_index()
    ax.plot(r.index, r.values, color="#1f4e79", lw=1.2)
    ax.axvline(CUTOFF, color="black", ls="--", lw=1, label="Kronos cutoff")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_title("Ungated A0 — rolling 180-day net Sharpe")
    ax.set_ylabel("Sharpe (ann.)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_phaseB1_report(
    path: Path,
    frozen_hash: str,
    gate_rows: list[dict],
    redundancy: dict,
    adoption: dict,
    decay_ungated: list[dict],
    decay_winner: list[dict],
    trailing: dict,
    winner_label: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase B.1 — kr_sigma gate control experiment\n")
    lines.append("## 0. Frozen A0\n")
    lines.append(f"- Frozen config hash (SHA256): `{frozen_hash}`")
    lines.append("- CPU-only; Kronos features read from Volume cache (zero new GPU inference).\n")

    lines.append("## 3. Pre-registered adoption rule\n")
    lines.append(f"> {ADOPTION_RULE}\n")
    lines.append(f"**Verdict: {adoption.get('verdict')}**\n")
    lines.append(f"Details: `{adoption.get('details')}`\n")

    lines.append("## 1. Control gates (paired)\n")
    lines.append(
        "Portfolio: tranche h=7, median-τ (τ=60), funding on, lag 0. "
        "Skip NEW entries when gate column is top X% CS that day.\n"
    )
    lines.append(
        "| gate | X | Sharpe full | Sharpe pre | Sharpe post | %flat full | ann TO full | avg n_pos full | blocked |"
    )
    lines.append("|------|---|-------------|------------|-------------|------------|-------------|----------------|---------|")
    for r in gate_rows:
        lines.append(
            f"| {r.get('gate')} | {r.get('X')} | {_fmt(r.get('sharpe_full'))} | {_fmt(r.get('sharpe_pre'))} | "
            f"{_fmt(r.get('sharpe_post'))} | {_fmt(r.get('pct_flat_full'), 3)} | {_fmt(r.get('ann_turnover_full'), 2)} | "
            f"{_fmt(r.get('avg_n_pos_full'), 2)} | {r.get('n_blocked_entries', '')} |"
        )

    lines.append("\n## 2. Redundancy diagnostics\n")
    lines.append("### Mean daily CS Spearman(kr_sigma_h7, control)\n")
    lines.append("| control | full | post |")
    lines.append("|---------|------|------|")
    for name, blob in (redundancy.get("rank_corr") or {}).items():
        lines.append(
            f"| {name} | {_fmt(blob.get('full', {}).get('mean_spearman'))} | "
            f"{_fmt(blob.get('post', {}).get('mean_spearman'))} |"
        )
    lines.append("\n### Entry-skip overlap at X=20 (fraction of kr_sigma skips also skipped by control)\n")
    lines.append("| control | overlap_frac | n_ref_skips |")
    lines.append("|---------|--------------|-------------|")
    for name, blob in (redundancy.get("skip_overlap_x20") or {}).items():
        lines.append(
            f"| {name} | {_fmt(blob.get('overlap_frac'), 3)} | {blob.get('n_ref_skips')} |"
        )
    lines.append(
        f"\nPresumed redundant: **{redundancy.get('presumed_redundant')}** "
        f"(rank-corr>0.8 or skip-overlap>0.80). Reasons: `{redundancy.get('reasons')}`\n"
    )

    lines.append("## 4. Baseline decay\n")
    lines.append("### Ungated A0 — per calendar year\n")
    lines.append("| year | n_days | net Sharpe | CAGR | MaxDD |")
    lines.append("|------|--------|------------|------|-------|")
    for r in decay_ungated:
        lines.append(
            f"| {r['year']} | {r['n_days']} | {_fmt(r['net_sharpe'])} | {_fmt(r['cagr'], 3)} | {_fmt(r['max_dd'], 3)} |"
        )
    lines.append(f"\n### Winning configuration ({winner_label}) — per calendar year\n")
    lines.append("| year | n_days | net Sharpe | CAGR | MaxDD |")
    lines.append("|------|--------|------------|------|-------|")
    for r in decay_winner:
        lines.append(
            f"| {r['year']} | {r['n_days']} | {_fmt(r['net_sharpe'])} | {_fmt(r['cagr'], 3)} | {_fmt(r['max_dd'], 3)} |"
        )
    lines.append("\n### Trailing 12 months (ungated)\n")
    pos = trailing.get("positive_expectancy")
    lines.append(
        f"- Trailing-12m net Sharpe: **{_fmt(trailing.get('net_sharpe'))}** "
        f"(return {_fmt(trailing.get('total_return'), 3)}, "
        f"{trailing.get('start')}→{trailing.get('end')}, n={trailing.get('n_days')})."
    )
    lines.append(
        f"- Would the strategy currently be running at positive expectancy? "
        f"**{'YES' if pos else 'NO'}** "
        f"(mean daily net return {'>' if pos else '≤'} 0).\n"
    )

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout_summary(
    gate_rows: list[dict],
    redundancy: dict,
    adoption: dict,
    trailing: dict,
) -> None:
    print("\n========== PHASE B.1 SUMMARY ==========", flush=True)
    print(
        f"{'gate':<28} {'X':>3} {'Sh_full':>8} {'Sh_pre':>8} {'Sh_post':>8}",
        flush=True,
    )
    for r in gate_rows:
        print(
            f"{str(r.get('gate')):<28} {str(r.get('X')):>3} "
            f"{_fmt(r.get('sharpe_full')):>8} {_fmt(r.get('sharpe_pre')):>8} {_fmt(r.get('sharpe_post')):>8}",
            flush=True,
        )
    print("--- redundancy ---", flush=True)
    for name, blob in (redundancy.get("rank_corr") or {}).items():
        ov = (redundancy.get("skip_overlap_x20") or {}).get(name, {})
        print(
            f"  {name}: spearman_full={_fmt(blob.get('full', {}).get('mean_spearman'))} "
            f"post={_fmt(blob.get('post', {}).get('mean_spearman'))} "
            f"skip_ov_x20={_fmt(ov.get('overlap_frac'), 3)}",
            flush=True,
        )
    print(f"presumed_redundant={redundancy.get('presumed_redundant')}", flush=True)
    print(f"VERDICT: {adoption.get('verdict')}", flush=True)
    print(f"Rule: {ADOPTION_RULE}", flush=True)
    print(
        f"TRAILING-12M: Sharpe={_fmt(trailing.get('net_sharpe'))} "
        f"expectancy_positive={trailing.get('positive_expectancy')}",
        flush=True,
    )
    print("=======================================\n", flush=True)
