"""Phase D report + charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_d.ablation import KEEP_CRITERION
from phase_d.decay import plot_decay


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def plot_ablation_ic(delta_by_h: dict[int, pd.Series], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    ax = axes[0]
    end = None
    for h, ser in delta_by_h.items():
        if ser is None or len(ser) == 0:
            continue
        s = ser.sort_index().fillna(0.0)
        end = s.index.max() if end is None else max(end, s.index.max())
        ax.plot(s.index, s.cumsum().values, label=f"h={h}")
    ax.set_title("Cumulative daily ΔRankIC (D − A)")
    ax.set_ylabel("Cum ΔIC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for h, ser in delta_by_h.items():
            if ser is None or len(ser) == 0:
                continue
            s = ser.sort_index()
            s = s[(s.index >= start) & (s.index <= end)].fillna(0.0)
            if s.empty:
                continue
            ax2.plot(s.index, s.cumsum().values, label=f"h={h}")
    ax2.set_title("Trailing-18m zoom")
    ax2.set_ylabel("Cum ΔIC")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_phaseD_report(
    path: Path,
    frozen_hash: str,
    decay: dict,
    coverage: dict,
    ablation: dict,
    keep: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase D — decay diagnostic + microstructure ablation\n")
    lines.append(f"- Frozen A0 hash: `{frozen_hash}`")
    lines.append("- Scope: backtest/analysis only; no schedules or live components.\n")

    lines.append("## 1. Decay diagnostic (frozen A0, tranche h=7, τ=60, funding on, no gate)\n")
    lines.append(f"Gross alpha proxy formula: `{decay.get('formula')}`\n")
    lines.append("| year | RankIC | dispersion | score_disp | avg_npos | %nonempty | proxy | gross | cost | cost_share | funding | fund_share | net | Sharpe |")
    lines.append("|------|--------|------------|------------|----------|-----------|-------|-------|------|------------|---------|------------|-----|--------|")
    for r in decay.get("by_year") or []:
        lines.append(
            f"| {r['year']} | {_fmt(r['rank_ic'])} | {_fmt(r['dispersion'])} | {_fmt(r['score_dispersion'])} | "
            f"{_fmt(r['avg_n_positions'], 2)} | {_fmt(r['pct_nonempty_book'], 2)} | {_fmt(r['gross_proxy'], 5)} | "
            f"{_fmt(r['gross_pnl'])} | {_fmt(r['cost_drag'])} | {_fmt(r.get('cost_share_of_abs_gross'), 3)} | "
            f"{_fmt(r['funding_pnl'])} | {_fmt(r.get('funding_share_of_abs_gross'), 3)} | "
            f"{_fmt(r['net_pnl'])} | {_fmt(r['net_sharpe'], 3)} |"
        )
    lines.append(f"\nProxy↔gross corr (years): **{_fmt(decay.get('proxy_vs_gross_corr'), 3)}**\n")
    lines.append(f"**Diagnostic verdict: {decay.get('verdict')}**\n")
    lines.append(f"{decay.get('justification')}\n")

    lines.append("## 2. Microstructure coverage\n")
    lines.append("| source/field | n_sym | cov_sym | min_date | max_date | n_rows | note |")
    lines.append("|--------------|-------|---------|----------|----------|--------|------|")
    for name, blob in (coverage or {}).items():
        if not isinstance(blob, dict):
            continue
        lines.append(
            f"| {name} | {blob.get('n_symbols_with_data')} | {_fmt(blob.get('coverage_pct_symbol'), 3)} | "
            f"{blob.get('min_date')} | {blob.get('max_date')} | {blob.get('n_rows')} | {blob.get('note', '')} |"
        )
    lines.append("\nNaN handling: unavailable fields left as NaN (no zero-imputation); LightGBM native NaN.\n")

    lines.append("## 3. Microstructure feature block (12 features)\n")
    lines.append("Per (symbol, date), data ≤ close of t only, then cross-sectional z-score per date, clip ±5:\n")
    lines.append("- `funding_now`, `funding_z_30`, `funding_cum_7`, `funding_cs_rank`")
    lines.append("- `basis_z_30`")
    lines.append("- `oi_chg_1`, `oi_chg_7`, `oi_turnover`")
    lines.append("- `liq_imb_1`, `liq_imb_7` (always NaN — UM liquidationSnapshot absent on Vision)")
    lines.append("- `taker_imb_z`, `ls_ratio_z`\n")

    lines.append("## 4. Ablation criterion (pre-registered)\n")
    lines.append(f"> {KEEP_CRITERION}\n")
    lines.append(f"**Ablation verdict: {keep.get('verdict')}**\n")
    lines.append(f"Details: `{keep.get('details')}`\n")

    lines.append("## Ablation tables (A = A0, D = A0+micro)\n")
    lines.append("| h | universe | window | A IC | D IC | ΔIC | n_days |")
    lines.append("|---|----------|--------|------|------|-----|--------|")
    for h, blob in sorted((ablation or {}).items(), key=lambda kv: int(kv[0])):
        for t in blob.get("tables") or []:
            if t["window"] not in ("full", "trail18m") and not str(t["window"]).startswith("y"):
                continue
            lines.append(
                f"| {t['horizon']} | {t['universe']} | {t['window']} | {_fmt(t['A_ic'])} | {_fmt(t['B_ic'])} | "
                f"{_fmt(t['delta_ic'])} | {t.get('n_days')} |"
            )

    lines.append("\n### Paired NW t on daily ΔIC\n")
    lines.append("| h | window | mean ΔIC | NW-t | n_days |")
    lines.append("|---|--------|----------|------|--------|")
    for h, blob in sorted((ablation or {}).items(), key=lambda kv: int(kv[0])):
        for w, p in (blob.get("paired_nw") or {}).items():
            lines.append(
                f"| {h} | {w} | {_fmt(p.get('mean_delta_ic'))} | {_fmt(p.get('nw_tstat'), 2)} | {p.get('n_days')} |"
            )

    lines.append("\n### Coverage-conditional ΔIC (≥80% book micro coverage)\n")
    lines.append("| h | window | mean ΔIC | NW-t | n_days |")
    lines.append("|---|--------|----------|------|--------|")
    for h, blob in sorted((ablation or {}).items(), key=lambda kv: int(kv[0])):
        for w, p in (blob.get("coverage_conditional_delta") or {}).items():
            lines.append(
                f"| {h} | {w} | {_fmt(p.get('mean_delta_ic'))} | {_fmt(p.get('nw_tstat'), 2)} | {p.get('n_days')} |"
            )

    lines.append("\n### Δ median-τ net Sharpe (tranche, funding on, paired days)\n")
    lines.append("| h | A Sharpe | D Sharpe | Δ |")
    lines.append("|---|----------|----------|---|")
    for h, blob in sorted((ablation or {}).items(), key=lambda kv: int(kv[0])):
        s = blob.get("sharpe_delta") or {}
        lines.append(
            f"| {h} | {_fmt(s.get('A_sharpe'), 3)} | {_fmt(s.get('B_sharpe'), 3)} | {_fmt(s.get('delta_sharpe'), 3)} |"
        )

    lines.append("\n### Microstructure LightGBM gain importances\n")
    lines.append("| h | feature | mean_gain | median_gain |")
    lines.append("|---|---------|-----------|-------------|")
    for h, blob in sorted((ablation or {}).items(), key=lambda kv: int(kv[0])):
        imp = blob.get("micro_importance") or {}
        for feat, st in sorted(imp.items(), key=lambda kv: -kv[1].get("mean_gain", 0)):
            lines.append(f"| {h} | {feat} | {_fmt(st.get('mean_gain'), 2)} | {_fmt(st.get('median_gain'), 2)} |")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout_summary(decay: dict, keep: dict, ablation: dict) -> None:
    print("\n========== PHASE D SUMMARY ==========", flush=True)
    print(f"DIAGNOSTIC VERDICT: {decay.get('verdict')}", flush=True)
    print(f"  {decay.get('justification')}", flush=True)
    print(f"CRITERION: {KEEP_CRITERION}", flush=True)
    print(f"ABLATION VERDICT: {keep.get('verdict')}", flush=True)
    for h, blob in sorted((ablation or {}).items(), key=lambda kv: int(kv[0])):
        print(
            f"  h={h} ΔIC trail18m={_fmt(blob.get('delta_top20_trail18m'))} "
            f"full={_fmt(blob.get('delta_top20_full'))} "
            f"frac+folds18m={_fmt(blob.get('frac_pos_folds_trail18m'), 3)}",
            flush=True,
        )
    # top-5 importances pooled
    pooled = {}
    for h, blob in (ablation or {}).items():
        for f, st in (blob.get("micro_importance") or {}).items():
            pooled[f] = pooled.get(f, 0.0) + float(st.get("mean_gain") or 0.0)
    top5 = sorted(pooled.items(), key=lambda kv: -kv[1])[:5]
    print("TOP-5 NEW FEATURES BY GAIN:", flush=True)
    for f, g in top5:
        print(f"  {f}: {g:.2f}", flush=True)
    print("=====================================\n", flush=True)
