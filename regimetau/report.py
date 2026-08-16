"""REGIME-TAU report, JSON, and equity chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from longonly.eval import _as_utc
from regimetau.constants import DEATH_CONVENTION, VIABILITY_CRITERION


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _eq_from_rets(rets: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    r = _as_utc(rets).fillna(0.0)
    eq = (1.0 + r).cumprod()
    y = eq.to_numpy()
    if len(y) and y[0] != 0:
        y = y / y[0]
    return eq.index, y


def plot_equity(reg: pd.Series, ref: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for rets, lab in ((reg, "COMBO-REGIME-TAU"), (ref, "Reference COMBO")):
        if not isinstance(rets, pd.Series) or len(rets) == 0:
            continue
        d, y = _eq_from_rets(rets)
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("REGIME-TAU vs reference COMBO")
    axes[0].set_ylabel("Equity (rebased, log)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")
    end = None
    for rets, lab in ((reg, "COMBO-REGIME-TAU"), (ref, "Reference COMBO")):
        if not isinstance(rets, pd.Series) or len(rets) == 0:
            continue
        d, y = _eq_from_rets(rets)
        end = d.max() if end is None else max(end, d.max())
        start = end - pd.Timedelta(days=int(365 * 1.5))
        m = np.asarray((pd.DatetimeIndex(d) >= start) & (pd.DatetimeIndex(d) <= end))
        if m.any():
            axes[1].plot(np.asarray(d)[m], y[m], label=lab, lw=1.4)
    axes[1].set_title("Trailing 18m")
    axes[1].set_ylabel("Equity (rebased, log)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, which="both")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _year_cell(book: dict, y: int) -> str:
    v = (book.get("net_sharpe_by_year") or {}).get(y)
    return _fmt(v)


def write_report(
    path: Path,
    *,
    frozen_hash: str,
    pred_hashes: dict,
    books: dict,
    verdict: dict,
    slices: dict,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = books["Reference COMBO"]
    reg = books["COMBO-REGIME-TAU"]
    years = sorted(set(ref.get("net_sharpe_by_year", {})) | set(reg.get("net_sharpe_by_year", {})))
    year_hdr = " | ".join(str(y) for y in years)
    year_sep = " | ".join(["------"] * len(years)) if years else "------"

    def row(name, b):
        ys = " | ".join(_year_cell(b, y) for y in years) if years else "nan"
        return (
            f"| {name} | {_fmt(b.get('net_sharpe_full'))} | {_fmt(b.get('net_sharpe_trail18m'))} "
            f"| {ys} | {_fmt(b.get('net_cagr'), 3)} | {_fmt(b.get('max_drawdown'), 3)} "
            f"| {_fmt(b.get('avg_n_long'), 2)} | {_fmt(b.get('ann_turnover'), 2)} "
            f"| {_fmt(b.get('corr_vs_ref_combo'), 3)} |"
        )

    sl_reg = slices.get("COMBO-REGIME-TAU") or {}
    sl_ref = slices.get("Reference COMBO") or {}

    lines = [
        "# REGIME-TAU — causal CS-corr τ overlay",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Portfolio layer only. Frozen A0 scores reused. "
        "No schedules, no live components, no product changes. CPU only, zero GPU.",
        "",
        f"**Frozen A0 SHA256:** `{frozen_hash}`",
        f"**Prediction files (reused):** h7=`{pred_hashes.get('h7')}` h10=`{pred_hashes.get('h10')}`",
        "**Reference book:** COMBO v2.0-combo-final. **UNCHANGED.**",
        "",
        DEATH_CONVENTION,
        "",
        f"Forced exits: n=`{extra.get('n_forced_exits', 0)}` "
        f"PnL units=`{_fmt(extra.get('forced_exit_pnl'))}`.",
        "",
        "## Pre-registered viability",
        "",
        f"> {VIABILITY_CRITERION}",
        "",
        "Verdict is mechanical. No post-hoc adjustment.",
        "",
        "## Mechanical verdict",
        "",
        f"- **COMBO-REGIME-TAU:** **{verdict.get('label')}** — "
        f"full Sharpe={_fmt(verdict.get('reg_full'))} "
        f"(need ≥ {_fmt(verdict.get('need_full'))}, pass={verdict.get('full_ok')}); "
        f"trail-18m Sharpe={_fmt(verdict.get('reg_trail'))} "
        f"(need ≥ {_fmt(verdict.get('need_trail'))}, pass={verdict.get('trail_ok')}). "
        f"Δ full={_fmt(verdict.get('delta_full'))} Δ trail={_fmt(verdict.get('delta_trail'))}.",
        "",
        f"Identical days n=`{extra.get('identical_days')}`. "
        f"HIGH frac=`{_fmt(extra.get('high_frac'))}` "
        f"LOW=`{_fmt(extra.get('low_frac'))}` BASE=`{_fmt(extra.get('base_frac'))}`.",
        "",
        "## Headline books",
        "",
        f"| book | full | trail-18m | {year_hdr} | CAGR | MaxDD | avg #longs | ann TO | corr vs ref |",
        f"|------|------|-----------|{year_sep}|------|-------|------------|--------|-------------|",
        row("COMBO-REGIME-TAU", reg),
        row("REGIME Sleeve A", books.get("REGIME Sleeve A") or {}),
        row("REGIME Sleeve B", books.get("REGIME Sleeve B") or {}),
        row("Reference COMBO", ref),
        row("Reference Sleeve A", books.get("Reference Sleeve A") or {}),
        row("Reference Sleeve B", books.get("Reference Sleeve B") or {}),
        "",
        "## HIGH / LOW day Sharpes (diagnostic)",
        "",
        "| book | HIGH n | HIGH Sharpe | LOW n | LOW Sharpe | BASE n | BASE Sharpe |",
        "|------|--------|-------------|-------|------------|--------|-------------|",
    ]
    for name, sl in (("COMBO-REGIME-TAU", sl_reg), ("Reference COMBO", sl_ref)):
        hi, lo, ba = sl.get("HIGH") or {}, sl.get("LOW") or {}, sl.get("BASE") or {}
        lines.append(
            f"| {name} | {hi.get('n')} | {_fmt(hi.get('sharpe'))} | "
            f"{lo.get('n')} | {_fmt(lo.get('sharpe'))} | "
            f"{ba.get('n')} | {_fmt(ba.get('sharpe'))} |"
        )
    lines += [
        "",
        "## Frozen τ map",
        "",
        "| sleeve | BASE | HIGH | LOW |",
        "|--------|------|------|-----|",
        "| A (top-20, h=7) | 80 | 90 | 70 |",
        "| B (top-40, h=10) | 70 | 80 | 60 |",
        "",
        f"Elapsed sec=`{_fmt(extra.get('elapsed_sec'), 1)}`. GPU used = False.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"[HB] wrote {path}", flush=True)


def print_stdout(verdict: dict, extra: dict) -> None:
    print("[HB] BACKTEST ONLY; REGIME-TAU; frozen products untouched", flush=True)
    print(f"[HB] {verdict.get('label')} full={verdict.get('reg_full')} "
          f"trail={verdict.get('reg_trail')} Δfull={verdict.get('delta_full')} "
          f"Δtrail={verdict.get('delta_trail')}", flush=True)
    print(
        f"[HB] HIGH frac={extra.get('high_frac')} LOW={extra.get('low_frac')} "
        f"forced={extra.get('n_forced_exits')}",
        flush=True,
    )
