"""Phase B report + charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_b.ablation import CUTOFF, KILL_CRITERION


def plot_phaseB_ic(delta_by_h: dict[int, pd.Series], out_path: Path) -> None:
    """Cumulative daily ΔIC (B−A) with cutoff + post-cutoff zoom."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    ax = axes[0]
    for h, ser in delta_by_h.items():
        if ser is None or len(ser) == 0:
            continue
        s = ser.sort_index().fillna(0.0)
        ax.plot(s.index, s.cumsum().values, label=f"h={h}")
    ax.axvline(CUTOFF, color="black", ls="--", lw=1, label="Kronos cutoff")
    ax.set_title("Cumulative daily ΔRankIC (B − A)")
    ax.set_ylabel("Cum ΔIC")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for h, ser in delta_by_h.items():
        if ser is None or len(ser) == 0:
            continue
        s = ser.sort_index()
        s = s[s.index >= CUTOFF].fillna(0.0)
        if s.empty:
            continue
        ax2.plot(s.index, s.cumsum().values, label=f"h={h}")
    ax2.set_title("Post-cutoff zoom (evidence window)")
    ax2.set_ylabel("Cum ΔIC")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gate_equity(ungated_eq: pd.DataFrame, gated_eq: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, eq in (("ungated A0", ungated_eq), ("best gate", gated_eq)):
        if eq is None or eq.empty:
            continue
        e = eq.copy()
        e["date"] = pd.to_datetime(e["date"], utc=True)
        ax.plot(e["date"], e["equity"], label=name)
    ax.axvline(CUTOFF, color="black", ls="--", lw=1)
    ax.set_title("A0 tranche equity — uncertainty gate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_phaseB_report(
    path: Path,
    frozen_hash: str,
    budget: dict,
    coverage: dict,
    ablation: dict,
    kill: dict,
    gate: dict,
    ft_ref: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase B — Kronos frozen-feature ablation vs locked A0\n")
    lines.append("## 0. Frozen A0\n")
    lines.append(f"- Frozen config hash (SHA256): `{frozen_hash}`")
    lines.append("- Phase B adds Kronos feature columns only; A0 params untouched.\n")

    lines.append("## 1. Kronos extraction budget & coverage\n")
    lines.append("```json")
    lines.append(pd.Series(budget).to_json())
    lines.append("```")
    lines.append(f"- Coverage: {coverage}\n")

    lines.append("## 2. Ablation (Model A = frozen A0, Model B = A0 + Kronos)\n")
    lines.append("### Pre-registered kill criterion\n")
    lines.append(f"> {KILL_CRITERION}\n")
    lines.append(f"**Ablation verdict: {kill.get('verdict', 'KILL')}**\n")
    lines.append(f"Details: `{kill.get('details')}`\n")

    lines.append("### RankIC tables (A vs B)\n")
    lines.append("| h | universe | window | A IC | B IC | ΔIC | A NW-t | B NW-t | n_days |")
    lines.append("|---|----------|--------|------|------|-----|--------|--------|--------|")
    for h, blob in sorted(ablation.items()):
        for t in blob.get("tables", []):
            lines.append(
                f"| {t['horizon']} | {t['universe']} | {t['window']} | "
                f"{_fmt(t['A_mean_ic'])} | {_fmt(t['B_mean_ic'])} | {_fmt(t['delta_ic'])} | "
                f"{_fmt(t['A_nw_t'], 2)} | {_fmt(t['B_nw_t'], 2)} | {t.get('n_days')} |"
            )

    lines.append("\n### Paired Newey-West t on daily ΔIC (B−A)\n")
    lines.append("| h | window | mean ΔIC | NW-t | n_days |")
    lines.append("|---|--------|----------|------|--------|")
    for h, blob in sorted(ablation.items()):
        for w, p in blob.get("paired_nw", {}).items():
            lines.append(
                f"| {h} | {w} | {_fmt(p.get('mean_delta_ic'))} | {_fmt(p.get('nw_tstat'), 2)} | {p.get('n_days')} |"
            )

    lines.append("\n### Fold-level Δ (fraction positive)\n")
    lines.append("| h | window | n_folds | frac_positive |")
    lines.append("|---|--------|---------|---------------|")
    for h, blob in sorted(ablation.items()):
        for w, fs in blob.get("fold_stats", {}).items():
            lines.append(
                f"| {h} | {w} | {fs.get('n_folds')} | {_fmt(fs.get('frac_positive'), 3)} |"
            )

    lines.append("\n### LightGBM gain importance — Kronos features (Model B)\n")
    lines.append("| h | feature | mean_gain | median_gain |")
    lines.append("|---|---------|-----------|-------------|")
    for h, blob in sorted(ablation.items()):
        imp = blob.get("kronos_importance") or {}
        for feat, st in sorted(imp.items(), key=lambda kv: -kv[1].get("mean_gain", 0)):
            lines.append(
                f"| {h} | {feat} | {_fmt(st.get('mean_gain'), 2)} | {_fmt(st.get('median_gain'), 2)} |"
            )

    lines.append("\n## 3. Uncertainty-gate test (independent verdict)\n")
    lines.append(
        "Frozen A0 portfolio: tranche h=7, median-τ, funding on, lag 0. "
        "Rule: skip NEW entries when `kr_sigma` is in top X% CS that day.\n"
    )
    lines.append(f"**Gate verdict: {gate.get('gate_verdict')}**\n")
    lines.append("| gate | σ top% | Sharpe full | Sharpe post | ΔSh post | turnover | % flat | blocked |")
    lines.append("|------|--------|-------------|-------------|----------|----------|--------|---------|")
    for r in gate.get("rows", []):
        lines.append(
            f"| {r.get('gate')} | {r.get('sigma_top_pct')} | {_fmt(r.get('net_sharpe_full'), 3)} | "
            f"{_fmt(r.get('net_sharpe_post'), 3)} | {_fmt(r.get('delta_sharpe_post'), 3)} | "
            f"{_fmt(r.get('ann_turnover'), 2)} | {_fmt(r.get('pct_flat_days'), 3)} | "
            f"{r.get('n_blocked_entries', '')} |"
        )
    if gate.get("best"):
        lines.append(f"\nBest gate row: `{gate['best']}`\n")

    lines.append("\n## 4. CONTAMINATED REFERENCE — fine-tuned Kronos\n")
    lines.append(
        "> CONTAMINATED REFERENCE — fine-tuned on full-sample data, not comparable to walk-forward results.\n"
    )
    lines.append("```")
    lines.append(str(ft_ref))
    lines.append("```\n")

    lines.append("## 5. Final stdout summary (copy)\n")
    lines.append("See pipeline stdout table: A vs B top-20 IC pre/post, Δ, verdict; best gate row.\n")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout_summary(ablation: dict, kill: dict, gate: dict) -> None:
    print("\n========== PHASE B SUMMARY ==========", flush=True)
    print(
        f"{'h':>3} {'win':>5} {'A_top20':>10} {'B_top20':>10} {'Δ':>10} {'frac+folds':>10}",
        flush=True,
    )
    for h, blob in sorted(ablation.items()):
        for t in blob.get("tables", []):
            if t["universe"] != "top20":
                continue
            if t["window"] not in ("pre", "post"):
                continue
            frac = blob.get("fold_stats", {}).get(t["window"], {}).get("frac_positive")
            print(
                f"{h:3d} {t['window']:>5} {_fmt(t['A_mean_ic']):>10} {_fmt(t['B_mean_ic']):>10} "
                f"{_fmt(t['delta_ic']):>10} {_fmt(frac, 3):>10}",
                flush=True,
            )
    print(f"VERDICT (ablation): {kill.get('verdict')}", flush=True)
    print(f"Criterion: {KILL_CRITERION}", flush=True)
    best = gate.get("best") or {}
    print(
        f"BEST GATE: {best.get('gate')} ΔSh_post={_fmt(best.get('delta_sharpe_post'), 3)} "
        f"verdict={gate.get('gate_verdict')}",
        flush=True,
    )
    print("=====================================\n", flush=True)
