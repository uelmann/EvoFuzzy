"""Phase E.1 report + charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIRM_CRITERION = (
    "The GRU BLEND is CONFIRMED only if: (i) all §1 gates pass; (ii) EACH of the three disjoint "
    "3-seed ensembles satisfies the original BLEND KEEP criterion (trailing-18m ΔRankIC ≥ +0.005, "
    "full-OOS Δ ≥ 0, ≥60% positive trailing folds) on at least one universe at h=7 or h=10, with "
    "the SAME universe/horizon passing for all three ensembles; (iii) the grand-ensemble paired "
    "NW-t of trailing-18m ΔIC on that universe/horizon is ≥ 2.0; and (iv) the BLEND portfolio "
    "(median-τ, either τ convention) does not lose more than 0.10 net Sharpe vs A0 on the full "
    "period while improving or matching on trailing-18m. If CONFIRMED, BLEND is designated "
    "candidate baseline A1, pending the D.2 universe decision. Otherwise: NOT CONFIRMED — park, "
    "no adoption, no retuning."
)


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def plot_seeds(dist_rows: list[dict], ens_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slices = sorted({(r["horizon"], r["universe"], r["window"]) for r in dist_rows})
    n = max(len(slices), 1)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (h, uni, window) in zip(axes, slices):
        pts = [r for r in dist_rows if r["horizon"] == h and r["universe"] == uni and r["window"] == window]
        xs = [r["seed"] for r in pts]
        ys = [r["mean_ic"] for r in pts]
        ax.scatter(xs, ys, s=18, c="steelblue", label="seed")
        seen = set()
        for e in ens_rows:
            if e.get("horizon") == h and e.get("universe") == uni and e.get("window") == window:
                lab = e.get("name")
                if lab in seen:
                    continue
                seen.add(lab)
                ax.axhline(e["mean_ic"], color=e.get("color", "orange"), ls="--", lw=1, label=lab)
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"h={h} {uni} {window}")
        ax.set_xlabel("seed")
        ax.set_ylabel("RankIC")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(slices) :]:
        ax.axis("off")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_blend_equity(eq_a: pd.Series, eq_b: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    a = eq_a.sort_index()
    b = eq_b.sort_index()
    axes[0].plot(a.index, a.values, label="A0")
    axes[0].plot(b.index, b.values, label="BLEND")
    axes[0].set_title("Top-20 tranche equity (full)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    end = max(a.index.max(), b.index.max())
    start = end - pd.Timedelta(days=int(365 * 1.5))
    aa = a[(a.index >= start) & (a.index <= end)]
    bb = b[(b.index >= start) & (b.index <= end)]
    # rebase
    if len(aa):
        aa = aa / aa.iloc[0]
    if len(bb):
        bb = bb / bb.iloc[0]
    axes[1].plot(aa.index, aa.values, label="A0")
    axes[1].plot(bb.index, bb.values, label="BLEND")
    axes[1].set_title("Trailing-18m zoom (rebased)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(path: Path, **kw) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase E.1 — GRU/BLEND verification\n")
    lines.append("**Verification only.** No new ideas, no tuning. Extraordinary first-pass KEEP is guilty until confirmed.\n")
    lines.append(f"- Frozen A0 hash: `{kw.get('frozen_hash')}`")
    lines.append("- Scope: backtest/analysis only; no schedules or live components.\n")
    lines.append("## Pre-registered confirmation criterion (verbatim)\n")
    lines.append(f"> {CONFIRM_CRITERION}\n")
    lines.append(f"**Mechanical verdict: {kw.get('verdict')}**\n")
    if kw.get("verdict") == "CONFIRMED":
        lines.append("BLEND is designated **candidate baseline A1**, pending the D.2 universe decision.\n")
    else:
        lines.append("**NOT CONFIRMED — park, no adoption, no retuning.**\n")
    lines.append(f"Details: `{kw.get('verdict_details')}`\n")

    lines.append("## 1. Leakage gates\n")
    for g in kw.get("gates") or []:
        name = g.get("name")
        passed = "PASS" if g.get("passed") else "FAIL"
        lines.append(f"### {name}: **{passed}**\n")
        if name == "gru_label_shuffle":
            lines.append(f"- Threshold: `|IC| < {g.get('threshold')}` on outer-fold RankIC (mean of 3 seeds, labels shuffled within date).")
            lines.append(f"- Folds: `{g.get('folds')}`")
            lines.append("| h | fold | seed | mean IC | |IC| | pass |")
            lines.append("|---|------|------|---------|-----|------|")
            for r in g.get("rows") or []:
                lines.append(
                    f"| {r.get('horizon')} | {r.get('fold_id')} | {r.get('seed')} | "
                    f"{_fmt(r.get('mean_ic'))} | {_fmt(r.get('abs_ic'))} | {r.get('passed')} |"
                )
            lines.append("")
        elif name == "future_perturbation":
            lines.append(f"- synthetic_ok={g.get('synthetic_ok')} synthetic_score_ok={g.get('synthetic_score_ok')} real_ok={g.get('real_ok')} score_ok={g.get('score_ok')} t={g.get('t')}")
            lines.append(f"- {g.get('note')}")
            lines.append(f"- real: `{g.get('real')}`")
            lines.append("")
        elif name == "fold_isolation":
            lines.append(f"- warm_start: {g.get('warm_start')}; metas inspected={g.get('n_metas_inspected')}; warm-start files={g.get('n_warm_start_metas')}")
            lines.append("| fold | train_end | max dataloader date | max train slice | n_rows | pass |")
            lines.append("|------|-----------|---------------------|-----------------|--------|------|")
            for r in g.get("folds") or []:
                lines.append(
                    f"| {r.get('fold_id')} | {r.get('train_end')} | {r.get('max_dataloader_date')} | "
                    f"{r.get('max_train_slice_date')} | {r.get('n_train_rows')} | {r.get('passed')} |"
                )
            lines.append("")
        elif name == "prediction_alignment":
            lines.append(f"- {g.get('note', '')}")
            for h, blob in (g.get("by_h") or {}).items():
                lines.append(f"- h={h} passed={blob.get('passed')} n_rows={blob.get('n_rows')} n_bad={blob.get('n_bad')} n_unassigned={blob.get('n_unassigned')}")
            lines.append("")
        else:
            lines.append(f"- `{g}`\n")
    if not kw.get("gates_ok"):
        lines.append("Gates failed — §2–§4 not run.\n")
        text = "\n".join(lines) + "\n"
        path.write_text(text)
        return text

    lines.append("## 2. Seed robustness\n")
    lines.append(f"- Budget: `{kw.get('budget')}`")
    lines.append(f"- Horizons trained: `{kw.get('horizons_trained')}`")
    lines.append("- Disjoint ensembles: `{42,43,44}`, `{45,46,47}`, `{48,49,50}`\n")
    lines.append("### Disjoint-ensemble BLEND KEEP (mechanical)\n")
    for row in kw.get("ensemble_keep_lines") or []:
        lines.append(f"- {row}")
    lines.append("\n### Ensemble RankIC tables\n")
    lines.append("| ens | h | universe | window | A IC | S IC | BLEND IC | ΔS | ΔBLEND | NW-t ΔBLEND | frac+ | KEEP |")
    lines.append("|-----|---|----------|--------|------|------|----------|----|--------|-------------|-------|------|")
    for r in kw.get("ensemble_table") or []:
        lines.append(
            f"| {r.get('ens')} | {r.get('horizon')} | {r.get('universe')} | {r.get('window')} | "
            f"{_fmt(r.get('A_ic'))} | {_fmt(r.get('S_ic'))} | {_fmt(r.get('BLEND_ic'))} | "
            f"{_fmt(r.get('delta_S'))} | {_fmt(r.get('delta_BLEND'))} | {_fmt(r.get('nw_t_BLEND'), 2)} | "
            f"{_fmt(r.get('frac_pos'), 3)} | {r.get('keep_blend')} |"
        )
    lines.append("\n### Per-seed RankIC distribution\n")
    lines.append("| h | universe | window | min | median | max | n |")
    lines.append("|---|----------|--------|-----|--------|-----|---|")
    for r in kw.get("seed_dist") or []:
        lines.append(
            f"| {r.get('horizon')} | {r.get('universe')} | {r.get('window')} | "
            f"{_fmt(r.get('min'))} | {_fmt(r.get('median'))} | {_fmt(r.get('max'))} | {r.get('n')} |"
        )

    lines.append("\n## 3. Missing Phase E numbers\n")
    lines.append("### Paired NW t of daily ΔIC\n")
    lines.append("| ens | model | h | universe | window | mean ΔIC | NW-t | n_days |")
    lines.append("|-----|-------|---|----------|--------|----------|------|--------|")
    for r in kw.get("nw_table") or []:
        lines.append(
            f"| {r.get('ens')} | {r.get('model')} | {r.get('horizon')} | {r.get('universe')} | {r.get('window')} | "
            f"{_fmt(r.get('mean_delta_ic'))} | {_fmt(r.get('nw_tstat'), 2)} | {r.get('n_days')} |"
        )
    lines.append("\n### Daily A0↔S Spearman\n")
    lines.append("| ens | h | universe | window | mean Spearman | n_days |")
    lines.append("|-----|---|----------|--------|---------------|--------|")
    for r in kw.get("corr_table") or []:
        lines.append(
            f"| {r.get('ens')} | {r.get('horizon')} | {r.get('universe')} | {r.get('window')} | "
            f"{_fmt(r.get('mean_spearman'))} | {r.get('n_days')} |"
        )
    lines.append("\n### Per-year RankIC\n")
    lines.append("| ens | h | universe | year | A IC | S IC | BLEND IC |")
    lines.append("|-----|---|----------|------|------|------|----------|")
    for r in kw.get("year_table") or []:
        lines.append(
            f"| {r.get('ens')} | {r.get('horizon')} | {r.get('universe')} | {r.get('year')} | "
            f"{_fmt(r.get('A_ic'))} | {_fmt(r.get('S_ic'))} | {_fmt(r.get('BLEND_ic'))} |"
        )
    lines.append("\n### Lag-1 score autocorrelation (mean across symbols)\n")
    lines.append("| ens | h | model | mean lag-1 autocorr | n_symbols |")
    lines.append("|-----|---|-------|---------------------|-----------|")
    for r in kw.get("acf_table") or []:
        lines.append(
            f"| {r.get('ens')} | {r.get('horizon')} | {r.get('model')} | {_fmt(r.get('mean_acf'))} | {r.get('n_symbols')} |"
        )
    lines.append("\n### Portfolio translation (top-20 tranche, median-τ, funding on)\n")
    lines.append("| ens | h | τ mode | window | A Sharpe | BLEND Sharpe | ΔSharpe | A TO | B TO | A npos | B npos | A %flat | B %flat |")
    lines.append("|-----|---|--------|--------|----------|--------------|---------|------|------|--------|--------|---------|---------|")
    for r in kw.get("port_table") or []:
        lines.append(
            f"| {r.get('ens')} | {r.get('horizon')} | {r.get('tau_mode')} | {r.get('window')} | "
            f"{_fmt(r.get('A_sharpe'), 3)} | {_fmt(r.get('B_sharpe'), 3)} | {_fmt(r.get('delta_sharpe'), 3)} | "
            f"{_fmt(r.get('A_to'), 2)} | {_fmt(r.get('B_to'), 2)} | {_fmt(r.get('A_npos'), 2)} | "
            f"{_fmt(r.get('B_npos'), 2)} | {_fmt(r.get('A_flat'), 3)} | {_fmt(r.get('B_flat'), 3)} |"
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text
