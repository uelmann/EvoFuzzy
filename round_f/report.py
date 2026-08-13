"""Round F report + charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from round_f.constants import COMBO_CRITERION, KEEP_CRITERION


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def plot_ic(delta_by: dict, out_path: Path) -> None:
    """delta_by[block][h] = daily ΔIC series (mean of universes or specified)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    ax = axes[0]
    end = None
    for block, by_h in delta_by.items():
        for h, ser in by_h.items():
            if ser is None or len(ser) == 0:
                continue
            s = ser.sort_index().fillna(0.0)
            end = s.index.max() if end is None else max(end, s.index.max())
            ax.plot(s.index, s.cumsum().values, label=f"{block} h={h}", lw=1.2)
    ax.set_title("Round F cumulative ΔRankIC vs A0 (top-20 & top-40 mean of plotted series)")
    ax.set_ylabel("Cum ΔIC")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for block, by_h in delta_by.items():
            for h, ser in by_h.items():
                if ser is None or len(ser) == 0:
                    continue
                s = ser.sort_index()
                s = s[(s.index >= start) & (s.index <= end)].fillna(0.0)
                if s.empty:
                    continue
                ax2.plot(s.index, s.cumsum().values, label=f"{block} h={h}", lw=1.2)
    ax2.set_title("Trailing-18m zoom")
    ax2.set_ylabel("Cum ΔIC")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_combo(combo: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    def _eq(blob, col, lab):
        eq = blob.get("equity") if isinstance(blob, dict) else None
        if not isinstance(eq, pd.DataFrame) or eq.empty:
            return
        d = pd.to_datetime(eq["date"], utc=True)
        y = eq["equity"].astype(float).values
        if y[0] != 0:
            y = y / y[0]
        ax.plot(d, y, label=lab, lw=1.5)

    _eq(combo.get("p1_plot") or {}, "equity", "P1 A0 top-20 h=7")
    _eq(combo.get("p2_plot") or {}, "equity", "P2 A0 top-40 h=10")
    ce = combo.get("equity")
    if isinstance(ce, pd.DataFrame) and not ce.empty:
        d = pd.to_datetime(ce["date"], utc=True)
        y = ce["equity"].astype(float).values
        if len(y) and y[0] != 0:
            y = y / y[0]
        ax.plot(d, y, label="COMBO 50/50", lw=1.8, color="black")
    ax.set_title("Round F combo — causal τ")
    ax.set_ylabel("Equity (rebased)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_roundF_report(
    path: Path,
    *,
    frozen_hash: str,
    gates: list,
    pruned: list,
    ic_tables: list,
    ic_nw: list,
    port_rows: list,
    keep: dict,
    combo: dict,
    combo_v: dict,
    extra: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Round F — context, complexity, pruning, two-sleeve combo\n")
    lines.append(f"- Frozen A0 hash: `{frozen_hash}`")
    lines.append("- Scope: backtest only; zero GPU; causal (training-window) τ house standard.")
    lines.append("- Ledger: `reports/numbers_ledger.md`")
    lines.append("- Addendum (criteria frozen before results): `reports/roundF_addendum.md`")
    lines.append("- Hurst: single-scale R/S, H = log(R/S)/log(n) on the 90d residual window.\n")

    lines.append("## Pre-registered KEEP criterion (verbatim, before results)\n")
    lines.append(f"> {KEEP_CRITERION}\n")
    lines.append("## Pre-registered COMBO criterion (verbatim, before results)\n")
    lines.append(f"> {COMBO_CRITERION}\n")

    lines.append("## Gates\n")
    for g in gates or []:
        st = "PASS" if g.get("passed") else "FAIL"
        lines.append(f"- `{g.get('name')}`: **{st}**")
    lines.append("")

    lines.append("## F4 pruned features (8 lowest mean A0 gain)\n")
    lines.append(f"A0 metas used: {extra.get('n_a0_metas', '?')}\n")
    lines.append("| rank | feature | mean gain |")
    lines.append("|------|---------|-----------|")
    for i, (feat, g) in enumerate(pruned or [], start=1):
        lines.append(f"| {i} | `{feat}` | {_fmt(g, 2)} |")
    lines.append("")
    if extra.get("a0_gain_bottom_to_top"):
        lines.append("All A0 features by rising mean gain:\n")
        lines.append(", ".join(f"`{c}` ({_fmt(g, 1)})" for c, g in extra["a0_gain_bottom_to_top"]))
        lines.append("")

    lines.append("## Ablation RankIC vs A0\n")
    lines.append("| block | h | universe | window | A0 IC | F IC | ΔIC | n_days |")
    lines.append("|-------|---|----------|--------|-------|------|-----|--------|")
    for t in ic_tables or []:
        lines.append(
            f"| {t.get('block')} | {t.get('horizon')} | {t.get('universe')} | {t.get('window')} | "
            f"{_fmt(t.get('A_ic'))} | {_fmt(t.get('B_ic'))} | {_fmt(t.get('delta_ic'))} | {t.get('n_days')} |"
        )
    lines.append("\n### Paired NW t and fold fraction\n")
    lines.append("| block | h | universe | window | mean ΔIC | NW-t | n | frac+ trail18m folds |")
    lines.append("|-------|---|----------|--------|----------|------|---|----------------------|")
    for p in ic_nw or []:
        lines.append(
            f"| {p.get('block')} | {p.get('horizon')} | {p.get('universe')} | {p.get('window')} | "
            f"{_fmt(p.get('mean_delta_ic'))} | {_fmt(p.get('nw_tstat'), 2)} | {p.get('n_days')} | "
            f"{_fmt(p.get('frac_pos_trail18m'), 3)} |"
        )
    lines.append("")

    lines.append("## Portfolio Δ (causal median-τ) on adopted books\n")
    lines.append("P1 book = A0 top-20 h=7; P2 book = A0 top-40 h=10 (tiered costs + ADV cap).\n")
    lines.append("| block | book | τ_A0 | τ_F | A0 Sharpe full | F full | A0 trail18m | F trail18m | Δ trail18m |")
    lines.append("|-------|------|------|-----|----------------|--------|-------------|------------|------------|")
    for r in port_rows or []:
        lines.append(
            f"| {r.get('block')} | {r.get('book')} | {r.get('tau_a0')} | {r.get('tau_f')} | "
            f"{_fmt(r.get('a0_full'), 3)} | {_fmt(r.get('f_full'), 3)} | {_fmt(r.get('a0_trail18m'), 3)} | "
            f"{_fmt(r.get('f_trail18m'), 3)} | {_fmt(r.get('delta_trail18m'), 3)} |"
        )
    lines.append("")

    lines.append("## Mechanical KEEP verdicts\n")
    lines.append(f"> {KEEP_CRITERION}\n")
    for block, blob in (keep or {}).items():
        lines.append(f"### {block}\n")
        for uni, u in (blob.get("by_universe") or {}).items():
            lines.append(
                f"- **{uni}: {u.get('verdict')}** (IC any-h={u.get('ic_any_h')}, "
                f"port Δtrail18m={_fmt(u.get('port_delta_trail18m'), 3)}, port_ok={u.get('port_ok')})"
            )
        lines.append("")
        lines.append("| universe | h | ΔIC trail18m | ΔIC full | frac+ folds | IC pass |")
        lines.append("|----------|---|--------------|----------|-------------|---------|")
        for r in blob.get("rows") or []:
            lines.append(
                f"| {r['universe']} | {r['horizon']} | {_fmt(r['delta_ic_trail18m'])} | "
                f"{_fmt(r['delta_ic_full'])} | {_fmt(r['frac_pos_trail18m'], 3)} | {r['ic_pass_h']} |"
            )
        lines.append("")

    lines.append("## COMBO 50/50 P1+P2\n")
    lines.append(f"> {COMBO_CRITERION}\n")
    lines.append(f"**COMBO verdict: {combo_v.get('verdict')}**\n")
    lines.append(
        f"need trail18m ≥ {_fmt(combo_v.get('need_trail18m'), 3)} (max P={_fmt(combo_v.get('max_p_trail18m'), 3)}−0.10); "
        f"need full ≥ {_fmt(combo_v.get('need_full'), 3)} (max P={_fmt(combo_v.get('max_p_full'), 3)}−0.10).\n"
    )
    by = combo.get("net_sharpe_by_year") or {}
    lines.append("| book | full | trail18m | 2022 | 2023 | 2024 | 2025 | 2026 | MaxDD | corr sleeves | ann to |")
    lines.append("|------|------|----------|------|------|------|------|------|-------|--------------|--------|")
    lines.append(
        f"| COMBO | {_fmt(combo.get('net_sharpe_full'), 3)} | {_fmt(combo.get('net_sharpe_trail18m'), 3)} | "
        f"{_fmt(by.get(2022), 3)} | {_fmt(by.get(2023), 3)} | {_fmt(by.get(2024), 3)} | {_fmt(by.get(2025), 3)} | "
        f"{_fmt(by.get(2026), 3)} | {_fmt(combo.get('max_drawdown'), 3)} | {_fmt(combo.get('sleeve_corr'), 3)} | "
        f"{_fmt(combo.get('ann_turnover'), 2)} |"
    )
    lines.append(
        f"| P1 | {_fmt(combo.get('p1_full'), 3)} | {_fmt(combo.get('p1_trail18m'), 3)} |  |  |  |  |  |  |  |  |"
    )
    lines.append(
        f"| P2 | {_fmt(combo.get('p2_full'), 3)} | {_fmt(combo.get('p2_trail18m'), 3)} |  |  |  |  |  |  |  |  |"
    )
    lines.append("")
    lines.append("Ledger confirmation: all Round F portfolios used `tau_mode=fold_train` (causal).\n")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout(keep: dict, combo_v: dict) -> None:
    print("\n========== ROUND F SUMMARY ==========", flush=True)
    print("LEDGER: causal τ house standard; pooled-τ deprecated (see reports/numbers_ledger.md)", flush=True)
    for block, blob in (keep or {}).items():
        for uni, u in (blob.get("by_universe") or {}).items():
            print(f"VERDICT {block} {uni}: {u.get('verdict')}", flush=True)
    print(f"COMBO VERDICT: {combo_v.get('verdict')}", flush=True)
    print("=====================================\n", flush=True)
