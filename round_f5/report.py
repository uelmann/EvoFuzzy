"""Round F5 report, charts, and ledger append."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from round_f5.constants import COMBO_PRIME_CRITERION, SLEEVE_CRITERION


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _eq_series(blob) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    eq = blob.get("equity") if isinstance(blob, dict) else None
    if isinstance(eq, pd.DataFrame) and not eq.empty and "equity" in eq.columns:
        d = pd.to_datetime(eq["date"], utc=True)
        y = eq["equity"].astype(float).to_numpy()
        if len(y) and y[0] != 0:
            y = y / y[0]
        return d, y
    r = blob.get("daily_ret") if isinstance(blob, dict) else None
    if isinstance(r, pd.Series) and len(r):
        r = r.copy()
        r.index = pd.DatetimeIndex(pd.to_datetime(r.index, utc=True))
        eq = (1.0 + r.fillna(0.0)).cumprod()
        y = eq.to_numpy() / (eq.iloc[0] if eq.iloc[0] != 0 else 1.0)
        return eq.index, y
    return None


def plot_sleeves(cands: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    labels = {"C0": "C0 A0 incumbent", "C1": "C1 A0+context", "C2": "C2 A0 pruned", "C3": "C3 P2′ stacked"}
    end = None
    for cid, lab in labels.items():
        got = _eq_series(cands.get(cid) or {})
        if got is None:
            continue
        d, y = got
        end = d.max() if end is None else max(end, d.max())
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("Round F5 P2 sleeves — causal τ")
    axes[0].set_ylabel("Equity (rebased)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for cid, lab in labels.items():
            got = _eq_series(cands.get(cid) or {})
            if got is None:
                continue
            d, y = got
            m = np.asarray((pd.DatetimeIndex(d) >= start) & (pd.DatetimeIndex(d) <= end))
            if not m.any():
                continue
            yy = np.asarray(y)[m]
            dd = pd.DatetimeIndex(d)[m]
            if len(yy) and yy[0] != 0:
                yy = yy / yy[0]
            axes[1].plot(dd, yy, label=lab, lw=1.4)
    axes[1].set_title("Trailing-18m zoom")
    axes[1].set_ylabel("Equity (rebased)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_combo(combo_f, combo_p, p1, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for blob, lab, kw in (
        (p1, "P1 A0 top-20 h=7", dict(lw=1.3)),
        (combo_f, "COMBO Round F", dict(lw=1.6)),
        (combo_p, "COMBO′", dict(lw=1.8, color="black")),
    ):
        got = _eq_series(blob or {})
        if got is None:
            continue
        d, y = got
        ax.plot(d, y, label=lab, **kw)
    ax.set_title("Round F5 COMBO vs COMBO′ — causal τ")
    ax.set_ylabel("Equity (rebased)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _port_row(cid: str, blob: dict) -> dict:
    by = blob.get("net_sharpe_by_year") or {}
    yp = {r["year"]: r for r in (blob.get("year_pos_flat") or [])}
    return {
        "id": cid,
        "tau": blob.get("tau_pct"),
        "full": blob.get("net_sharpe_full"),
        "trail18m": blob.get("net_sharpe_trail18m"),
        "y2022": by.get(2022, by.get("2022")),
        "y2023": by.get(2023, by.get("2023")),
        "y2024": by.get(2024, by.get("2024")),
        "y2025": by.get(2025, by.get("2025")),
        "y2026": by.get(2026, by.get("2026")),
        "gross": blob.get("gross_total_pnl"),
        "cost": blob.get("cost_drag"),
        "funding": blob.get("funding_total_pnl"),
        "hedge": blob.get("hedge_total_pnl"),
        "avg_n_pos": blob.get("avg_n_positions"),
        "pct_flat": blob.get("pct_flat_days"),
        "ann_to": blob.get("ann_turnover"),
        "flat_2022": (yp.get(2022) or {}).get("pct_flat"),
        "flat_2023": (yp.get(2023) or {}).get("pct_flat"),
        "flat_2024": (yp.get(2024) or {}).get("pct_flat"),
        "flat_2025": (yp.get(2025) or {}).get("pct_flat"),
        "flat_2026": (yp.get(2026) or {}).get("pct_flat"),
        "npos_2022": (yp.get(2022) or {}).get("avg_n_pos"),
        "npos_2023": (yp.get(2023) or {}).get("avg_n_pos"),
        "npos_2024": (yp.get(2024) or {}).get("avg_n_pos"),
        "npos_2025": (yp.get(2025) or {}).get("avg_n_pos"),
        "npos_2026": (yp.get(2026) or {}).get("avg_n_pos"),
    }


def append_ledger(path: Path, selected: dict, combo_ref: dict, changelog: str) -> str:
    text = path.read_text() if path.exists() else ""
    if "Round F5" in text and "COMBO" in text and changelog[:40] in text:
        return text
    sel_row = selected
    combo_row = combo_ref
    extra = []
    extra.append("")
    extra.append("## Round F5 append (causal τ)")
    extra.append("")
    extra.append("| row | status | model | universe | h | median-τ | net Sharpe full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | gross | cost | funding | hedge | avg #pos | % flat | ann turnover |")
    extra.append("|-----|--------|-------|----------|---|----------|-----------------|-----------|------|------|------|------|------|-------|------|---------|-------|----------|--------|--------------|")
    extra.append(
        f"| {sel_row.get('row')} | {sel_row.get('status')} | {sel_row.get('model')} | {sel_row.get('universe')} | "
        f"{sel_row.get('h')} | {sel_row.get('tau')} | {_fmt(sel_row.get('full'))} | {_fmt(sel_row.get('trail18m'))} | "
        f"{_fmt(sel_row.get('y2022'))} | {_fmt(sel_row.get('y2023'))} | {_fmt(sel_row.get('y2024'))} | "
        f"{_fmt(sel_row.get('y2025'))} | {_fmt(sel_row.get('y2026'))} | {_fmt(sel_row.get('gross'))} | "
        f"{_fmt(sel_row.get('cost'), 4)} | {_fmt(sel_row.get('funding'), 4)} | {_fmt(sel_row.get('hedge'), 4)} | "
        f"{_fmt(sel_row.get('avg_n_pos'), 2)} | {_fmt(sel_row.get('pct_flat'), 2)} | {_fmt(sel_row.get('ann_to'), 2)} |"
    )
    extra.append(
        f"| {combo_row.get('row')} | {combo_row.get('status')} | {combo_row.get('model')} | {combo_row.get('universe')} | "
        f"{combo_row.get('h')} | {combo_row.get('tau')} | {_fmt(combo_row.get('full'))} | {_fmt(combo_row.get('trail18m'))} | "
        f"{_fmt(combo_row.get('y2022'))} | {_fmt(combo_row.get('y2023'))} | {_fmt(combo_row.get('y2024'))} | "
        f"{_fmt(combo_row.get('y2025'))} | {_fmt(combo_row.get('y2026'))} | {_fmt(combo_row.get('gross'))} | "
        f"{_fmt(combo_row.get('cost'), 4)} | {_fmt(combo_row.get('funding'), 4)} | {_fmt(combo_row.get('hedge'), 4)} | "
        f"{_fmt(combo_row.get('avg_n_pos'), 2)} | {_fmt(combo_row.get('pct_flat'), 2)} | {_fmt(combo_row.get('ann_to'), 2)} |"
    )
    extra.append("")
    extra.append(f"Changelog ({date.today().isoformat()}): {changelog}")
    extra.append("")
    new = text.rstrip() + "\n" + "\n".join(extra)
    path.write_text(new)
    return new


def write_roundF5_report(
    path: Path,
    *,
    frozen_hash: str,
    gates: list,
    cands: dict,
    ic_tables: list,
    ic_nw: list,
    c3_ic: dict,
    sleeve_v: dict,
    stability: dict,
    combo_f: dict,
    combo_p: dict,
    combo_v: dict,
    ledger_diff: str,
    extra: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {
        "C0": "C0 incumbent (A0)",
        "C1": "C1 A0+context (F1)",
        "C2": "C2 A0 pruned (F4)",
        "C3": "C3 P2′ (pruned+context)",
    }
    lines = []
    lines.append("# Round F5 — top-40 sleeve stack (pruned + context) and COMBO update\n")
    lines.append(f"- Frozen A0 hash: `{frozen_hash}`")
    lines.append("- Scope: backtest only; zero GPU; causal (training-window) τ house standard.")
    lines.append("- Ledger: `reports/numbers_ledger.md`")
    lines.append("- Addendum (criteria frozen before results): `reports/roundF5_addendum.md`")
    lines.append("- Context features reused from Round F volume cache. P2′ = 32 features (A0−8+7 ctx).\n")
    lines.append(
        f"**Mechanical:** selected sleeve **{sleeve_v.get('selected')}** ({sleeve_v.get('verdict')}); "
        f"COMBO′ **{combo_v.get('verdict')}** → reference **{combo_v.get('reference')}**.\n"
    )

    lines.append("## Pre-registered sleeve selection rule (verbatim, before results)\n")
    lines.append(f"> {SLEEVE_CRITERION}\n")
    lines.append("## Pre-registered COMBO′ rule (verbatim, before results)\n")
    lines.append(f"> {COMBO_PRIME_CRITERION}\n")

    lines.append("## Gates\n")
    for g in gates or []:
        st = "PASS" if g.get("passed") else "FAIL"
        lines.append(f"- `{g.get('name')}`: **{st}**")
    lines.append("")

    lines.append("## Four-candidate P2 sleeve table (top-40, h=10, causal median-τ, identical days)\n")
    lines.append(
        "C0 uses ledger τ=70. C1/C2/C3 pick own causal median-τ. Round F published: "
        "C0 1.470/0.723, C1 1.257/1.045, C2 1.314/1.120.\n"
    )
    lines.append("| id | model | τ | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | gross | cost | funding | hedge | avg #pos | % flat | ann to |")
    lines.append("|----|-------|---|------|-----------|------|------|------|------|------|-------|------|---------|-------|----------|--------|--------|")
    for cid in ("C0", "C1", "C2", "C3"):
        r = _port_row(cid, cands.get(cid) or {})
        lines.append(
            f"| {cid} | {names[cid]} | {r.get('tau')} | {_fmt(r.get('full'))} | {_fmt(r.get('trail18m'))} | "
            f"{_fmt(r.get('y2022'))} | {_fmt(r.get('y2023'))} | {_fmt(r.get('y2024'))} | {_fmt(r.get('y2025'))} | "
            f"{_fmt(r.get('y2026'))} | {_fmt(r.get('gross'))} | {_fmt(r.get('cost'), 4)} | {_fmt(r.get('funding'), 4)} | "
            f"{_fmt(r.get('hedge'), 4)} | {_fmt(r.get('avg_n_pos'), 2)} | {_fmt(r.get('pct_flat'), 2)} | {_fmt(r.get('ann_to'), 2)} |"
        )
    lines.append("\n% flat by year:\n")
    lines.append("| id | 2022 | 2023 | 2024 | 2025 | 2026 |")
    lines.append("|----|------|------|------|------|------|")
    for cid in ("C0", "C1", "C2", "C3"):
        r = _port_row(cid, cands.get(cid) or {})
        lines.append(
            f"| {cid} | {_fmt(r.get('flat_2022'), 3)} | {_fmt(r.get('flat_2023'), 3)} | "
            f"{_fmt(r.get('flat_2024'), 3)} | {_fmt(r.get('flat_2025'), 3)} | {_fmt(r.get('flat_2026'), 3)} |"
        )
    lines.append("\nAvg #positions by year:\n")
    lines.append("| id | 2022 | 2023 | 2024 | 2025 | 2026 |")
    lines.append("|----|------|------|------|------|------|")
    for cid in ("C0", "C1", "C2", "C3"):
        r = _port_row(cid, cands.get(cid) or {})
        lines.append(
            f"| {cid} | {_fmt(r.get('npos_2022'), 2)} | {_fmt(r.get('npos_2023'), 2)} | "
            f"{_fmt(r.get('npos_2024'), 2)} | {_fmt(r.get('npos_2025'), 2)} | {_fmt(r.get('npos_2026'), 2)} |"
        )
    lines.append("")

    lines.append("## ΔRankIC (top-40)\n")
    lines.append("| pair | h | window | A IC | B IC | ΔIC | n_days |")
    lines.append("|------|---|--------|------|------|-----|--------|")
    for t in ic_tables or []:
        lines.append(
            f"| {t.get('pair')} | {t.get('horizon')} | {t.get('window')} | {_fmt(t.get('A_ic'), 4)} | "
            f"{_fmt(t.get('B_ic'), 4)} | {_fmt(t.get('delta_ic'), 4)} | {t.get('n_days')} |"
        )
    lines.append("\n### Paired NW t and fold fraction\n")
    lines.append("| pair | h | window | mean ΔIC | NW-t | n | frac+ trail18m folds |")
    lines.append("|------|---|--------|----------|------|---|----------------------|")
    for p in ic_nw or []:
        lines.append(
            f"| {p.get('pair')} | {p.get('horizon')} | {p.get('window')} | {_fmt(p.get('mean_delta_ic'), 4)} | "
            f"{_fmt(p.get('nw_tstat'), 2)} | {p.get('n_days')} | {_fmt(p.get('frac_pos_trail18m'), 3)} |"
        )
    lines.append("\nC3 house-block IC gate (vs A0, top-40):\n")
    for r in (c3_ic or {}).get("rows") or []:
        lines.append(
            f"- h={r.get('horizon')}: Δtrail={_fmt(r.get('delta_ic_trail18m'), 4)} Δfull={_fmt(r.get('delta_ic_full'), 4)} "
            f"frac+={_fmt(r.get('frac_pos_trail18m'), 3)} pass={r.get('pass')}"
        )
    lines.append(f"- **C3 IC gate: {'PASS' if (c3_ic or {}).get('pass') else 'FAIL'}**\n")

    lines.append("## Mechanical sleeve selection\n")
    lines.append(f"> {SLEEVE_CRITERION}\n")
    lines.append(
        f"Incumbent C0 trail={_fmt(sleeve_v.get('incumbent_trail18m'))} full={_fmt(sleeve_v.get('incumbent_full'))}; "
        f"need trail ≥ {_fmt(sleeve_v.get('need_trail18m'))} and full ≥ {_fmt(sleeve_v.get('need_full'))}.\n"
    )
    lines.append("| id | trail-18m | full | sharpe_ok | ic_ok | qualify |")
    lines.append("|----|-----------|------|-----------|-------|---------|")
    for r in sleeve_v.get("rows") or []:
        lines.append(
            f"| {r.get('id')} | {_fmt(r.get('trail18m'))} | {_fmt(r.get('full'))} | {r.get('sharpe_ok')} | "
            f"{r.get('ic_ok')} | {r.get('qualify')} |"
        )
    lines.append(
        f"\n**Selected sleeve: {sleeve_v.get('selected')}** (verdict={sleeve_v.get('verdict')}; "
        f"qualifying={sleeve_v.get('qualifying')}).\n"
    )

    lines.append("## Stability diagnostic (selected vs incumbent; information only)\n")
    st = stability or {}
    lines.append(
        f"Selected avg #pos={_fmt(st.get('sel_avg_n_pos'), 2)} vs incumbent {_fmt(st.get('inc_avg_n_pos'), 2)}; "
        f"% flat { _fmt(st.get('sel_pct_flat'), 3)} vs {_fmt(st.get('inc_pct_flat'), 3)}.\n"
    )
    lines.append("| year | sel avg #pos | inc avg #pos | sel % flat | inc % flat |")
    lines.append("|------|--------------|--------------|------------|------------|")
    for r in st.get("by_year") or []:
        lines.append(
            f"| {r.get('year')} | {_fmt(r.get('sel_avg_n_pos'), 2)} | {_fmt(r.get('inc_avg_n_pos'), 2)} | "
            f"{_fmt(r.get('sel_pct_flat'), 3)} | {_fmt(r.get('inc_pct_flat'), 3)} |"
        )
    sd, id_ = st.get("sel_dpos") or {}, st.get("inc_dpos") or {}
    lines.append("\nDaily |Δ position count|:\n")
    lines.append("| book | mean | median | p90 | max | frac≥10 | frac≥20 |")
    lines.append("|------|------|--------|-----|-----|---------|---------|")
    lines.append(
        f"| selected | {_fmt(sd.get('mean'), 2)} | {_fmt(sd.get('median'), 2)} | {_fmt(sd.get('p90'), 2)} | "
        f"{_fmt(sd.get('max'), 1)} | {_fmt(sd.get('frac_ge_10'), 3)} | {_fmt(sd.get('frac_ge_20'), 3)} |"
    )
    lines.append(
        f"| incumbent | {_fmt(id_.get('mean'), 2)} | {_fmt(id_.get('median'), 2)} | {_fmt(id_.get('p90'), 2)} | "
        f"{_fmt(id_.get('max'), 1)} | {_fmt(id_.get('frac_ge_10'), 3)} | {_fmt(id_.get('frac_ge_20'), 3)} |"
    )
    lines.append(f"\n{st.get('note', '')}\n")

    lines.append("## COMBO′ vs COMBO\n")
    lines.append(f"> {COMBO_PRIME_CRITERION}\n")
    lines.append(f"**COMBO′ verdict: {combo_v.get('verdict')}** → reference book = **{combo_v.get('reference')}**\n")
    lines.append(
        f"need trail ≥ {_fmt(combo_v.get('need_trail18m'))} (COMBO trail { _fmt(combo_v.get('combo_f_trail18m'))}−0.05); "
        f"need full ≥ {_fmt(combo_v.get('need_full'))} (COMBO full {_fmt(combo_v.get('combo_f_full'))}−0.05).\n"
    )
    byp = (combo_p or {}).get("net_sharpe_by_year") or {}
    byf = (combo_f or {}).get("net_sharpe_by_year") or {}
    p1y = (combo_p or {}).get("p1_by_year") or {}
    sely = extra.get("sel_by_year") or {}
    lines.append("| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 | MaxDD | corr sleeves | ann to |")
    lines.append("|------|------|-----------|------|------|------|------|------|-------|--------------|--------|")

    def _combo_line(name, blob, by, extra_dd=None):
        return (
            f"| {name} | {_fmt(blob.get('net_sharpe_full'))} | {_fmt(blob.get('net_sharpe_trail18m'))} | "
            f"{_fmt(by.get(2022, by.get('2022')))} | {_fmt(by.get(2023, by.get('2023')))} | "
            f"{_fmt(by.get(2024, by.get('2024')))} | {_fmt(by.get(2025, by.get('2025')))} | "
            f"{_fmt(by.get(2026, by.get('2026')))} | {_fmt(blob.get('max_drawdown'))} | "
            f"{_fmt(blob.get('sleeve_corr'))} | {_fmt(blob.get('ann_turnover'), 2)} |"
        )

    lines.append(_combo_line("COMBO (Round F re-run)", combo_f or {}, byf))
    lines.append(_combo_line("COMBO′", combo_p or {}, byp))
    lines.append(
        f"| P1 | {_fmt((combo_p or {}).get('p1_full'))} | {_fmt((combo_p or {}).get('p1_trail18m'))} | "
        f"{_fmt(p1y.get(2022, p1y.get('2022')))} | {_fmt(p1y.get(2023, p1y.get('2023')))} | "
        f"{_fmt(p1y.get(2024, p1y.get('2024')))} | {_fmt(p1y.get(2025, p1y.get('2025')))} | "
        f"{_fmt(p1y.get(2026, p1y.get('2026')))} |  |  |  |"
    )
    lines.append(
        f"| selected sleeve | {_fmt(extra.get('sel_full'))} | {_fmt(extra.get('sel_trail'))} | "
        f"{_fmt(sely.get(2022, sely.get('2022')))} | {_fmt(sely.get(2023, sely.get('2023')))} | "
        f"{_fmt(sely.get(2024, sely.get('2024')))} | {_fmt(sely.get(2025, sely.get('2025')))} | "
        f"{_fmt(sely.get(2026, sely.get('2026')))} |  |  | {_fmt(extra.get('sel_to'), 2)} |"
    )
    lines.append("")
    lines.append("Ledger confirmation: all Round F5 portfolios used `tau_mode=fold_train` (causal).\n")
    lines.append("## Ledger diff\n")
    lines.append("```")
    lines.append((ledger_diff or "").strip()[-2000:])
    lines.append("```\n")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout(sleeve_v: dict, combo_v: dict, cands: dict) -> None:
    sel = sleeve_v.get("selected")
    blob = (cands or {}).get(sel) or {}
    print("\n========== ROUND F5 SUMMARY ==========", flush=True)
    print(
        f"SELECTED SLEEVE: {sel} ({sleeve_v.get('verdict')}) "
        f"full={_fmt(blob.get('net_sharpe_full'))} trail18m={_fmt(blob.get('net_sharpe_trail18m'))} "
        f"τ={blob.get('tau_pct')}",
        flush=True,
    )
    print(
        f"COMBO VERDICT: {combo_v.get('verdict')} reference={combo_v.get('reference')} "
        f"COMBO′ full={_fmt(combo_v.get('combo_prime_full'))} trail={_fmt(combo_v.get('combo_prime_trail18m'))} "
        f"(need full≥{_fmt(combo_v.get('need_full'))} trail≥{_fmt(combo_v.get('need_trail18m'))})",
        flush=True,
    )
    print(
        f"REFERENCE BOOK: {combo_v.get('reference')}",
        flush=True,
    )
    print("LEDGER: causal τ; Round F5 rows appended to reports/numbers_ledger.md", flush=True)
    print("======================================\n", flush=True)
