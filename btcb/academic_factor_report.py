"""Academic-factor report and charts. Analysis only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import ACADEMIC_FACTOR_CRITERION, DEATH_CONVENTION, PHASE2C_PRED_SHA256


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _pct(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{100.0 * float(x):.{nd}f}%"
    except Exception:
        return str(x)


def _cycle_cell(blob: dict, name: str, key: str = "sharpe") -> str:
    c = (blob.get("cycles") or {}).get(name) or {}
    return _fmt(c.get(key))


def _factor_row(name: str, b: dict) -> str:
    return (
        f"| {name} | {_fmt(b.get('sharpe'))} | {_fmt(b.get('sharpe_trail18m'))} "
        f"| {_fmt(b.get('ann_mean'))} | {_fmt(b.get('ann_vol'))} | {_fmt(b.get('nw_t'), 2)} "
        f"| {_cycle_cell(b, '2019-20')} | {_cycle_cell(b, '2021')} | {_cycle_cell(b, '2022')} "
        f"| {_cycle_cell(b, '2023-24')} | {_cycle_cell(b, '2025-26')} "
        f"| {_pct(b.get('maxdd'))} | {_fmt(b.get('avg_n_long'), 1)} | {_fmt(b.get('avg_n_short'), 1)} "
        f"| {_fmt(b.get('ann_turnover'), 2)} |"
    )


def write_academic_factor(
    path: Path,
    *,
    tables: dict,
    legs: dict,
    corr: dict,
    waterfall: dict,
    verdict: dict,
    extra: dict,
) -> str:
    rows = waterfall.get("rows") or []
    wf_lines = [
        "| step | Sharpe | ΔSharpe | trail-18m | NW-t | ann mean |",
        "|------|--------|---------|-----------|------|----------|",
    ]
    labels = {
        "paper_gross": "1. paper GROSS (FACTOR-JT top-100)",
        "net_naive": "2. NET-NAIVE (10 bps/side)",
        "shortability": "3. + shortability filter (CMC, no funding)",
        "real_costs": "4. + real costs/tiering (10 / 5+3 bps)",
        "hybrid_book": "5. + Binance prices + funding (3.c hybrid book)",
    }
    for r in rows:
        d = r.get("delta")
        dcell = "—" if d is None or (isinstance(d, float) and not np.isfinite(d)) else _fmt(d)
        wf_lines.append(
            f"| {labels.get(r.get('step'), r.get('step'))} | {_fmt(r.get('sharpe'))} "
            f"| {dcell} | {_fmt(r.get('trail'))} | {_fmt(r.get('nw_t'), 2)} | {_fmt(r.get('ann_mean'))} |"
        )

    exists = bool(verdict.get("exists"))
    vlab = verdict.get("label") or ("PAPER ALPHA EXISTS" if exists else "PAPER ALPHA DOES NOT EXIST")

    def _blk(title: str, items: list[tuple[str, dict]]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "| series | Sharpe | trail-18m | ann mean | ann vol | NW-t | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 | MaxDD | avg nL | avg nS | ann TO |",
            "|--------|--------|-----------|----------|---------|------|---------|------|------|---------|---------|-------|--------|--------|--------|",
        ]
        for name, b in items:
            lines.append(_factor_row(name, b or {}))
        lines.append("")
        return lines

    lm = legs.get("lmU") or {}
    um = legs.get("umS") or {}
    lng = legs.get("long") or {}
    sh = legs.get("short") or {}
    uni = legs.get("universe") or {}
    mean_f = float((legs.get("factor") or {}).get("ann_mean") or float("nan"))
    share_l = (
        float(lm.get("ann_mean")) / mean_f
        if np.isfinite(mean_f) and mean_f != 0 and np.isfinite(float(lm.get("ann_mean") or np.nan))
        else float("nan")
    )
    share_s = (
        float(um.get("ann_mean")) / mean_f
        if np.isfinite(mean_f) and mean_f != 0 and np.isfinite(float(um.get("ann_mean") or np.nan))
        else float("nan")
    )

    lines = [
        "# BTC-BEATER — Academic factor (unconstrained D10−D1) + implementation tax",
        "",
        "**ANALYSIS ONLY.** Frozen 2.c spread, CMC prices. No shortability / anti-blowoff / hysteresis "
        "on the paper factor. CPU only, zero GPU. COMBO untouched. 3.c suspension unchanged. Nothing adopted.",
        "",
        "## Pre-registered labels (verbatim, frozen before results)",
        "",
        f"> {ACADEMIC_FACTOR_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Mechanical verdicts",
        "",
        f"- **{vlab}**",
        f"- FACTOR-JT top-100 GROSS Sharpe = `{_fmt(verdict.get('sharpe'))}` "
        f"(need ≥ `{_fmt(verdict.get('need_sharpe'))}`; pass={verdict.get('pass_sharpe')})",
        f"- NW-t (lag 14) = `{_fmt(verdict.get('nw_t'), 2)}` "
        f"(need ≥ `{_fmt(verdict.get('need_nw_t'), 1)}`; pass={verdict.get('pass_nw_t')})",
        f"- n_days = {verdict.get('n_days')} ({verdict.get('start')} → {verdict.get('end')})",
        f"- **IMPLEMENTATION TAX** = paper GROSS Sharpe − 3.c hybrid Sharpe = `{_fmt(waterfall.get('tax'))}`",
        "",
        "Mechanical, no post-hoc adjustment. Diagnostic only; nothing is adopted or changed.",
        "",
        "## Construction notes",
        "",
        "- Universe: floored PIT top-N (dollar-volume primary; mcap informational). BTC excluded.",
        "- Rank: last-fold-wins 2.c spread, top/bottom decile (`k = n_scored // 10`).",
        "- FACTOR-DAILY: refresh every day. FACTOR-JT: 14 overlapping cohorts, each held 14 OOS steps.",
        "- Academic legs are 1.0 long / 1.0 short (D10−D1). Sharpe is scale-invariant vs the 0.5/0.5 book.",
        "- GROSS = no costs. NET-NAIVE = 10 bps/side flat on combined overlay traded notional, both legs.",
        "- Shortability step: bottom-decile ∩ live USDT-M perp; remaining short weights not renormalized; CMC; funding=0.",
        "- Real costs: longs 10 bps, shorts 5+3 bps, same overlay.",
        "- Waterfall terminal = frozen 3.c BOOK-HYBRID (β-matched h=14 book, Binance+funding), not a JT variant.",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- Position-log sha256 = `{extra.get('position_sha256')}`",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- GPU used = `{extra.get('gpu_used')}`",
        "",
    ]
    lines += _blk(
        "FACTOR-DAILY (dollar-volume)",
        [
            ("top-100 GROSS", tables.get("daily_dv100_gross")),
            ("top-100 NET-NAIVE", tables.get("daily_dv100_naive")),
            ("top-50 GROSS", tables.get("daily_dv50_gross")),
            ("top-50 NET-NAIVE", tables.get("daily_dv50_naive")),
        ],
    )
    lines += _blk(
        "FACTOR-JT (dollar-volume)",
        [
            ("top-100 GROSS", tables.get("jt_dv100_gross")),
            ("top-100 NET-NAIVE", tables.get("jt_dv100_naive")),
            ("top-50 GROSS", tables.get("jt_dv50_gross")),
            ("top-50 NET-NAIVE", tables.get("jt_dv50_naive")),
        ],
    )
    lines += _blk(
        "FACTOR-JT (market-cap, informational)",
        [
            ("mcap top-100 GROSS", tables.get("jt_mcap100_gross")),
            ("mcap top-100 NET-NAIVE", tables.get("jt_mcap100_naive")),
            ("mcap top-50 GROSS", tables.get("jt_mcap50_gross")),
            ("mcap top-50 NET-NAIVE", tables.get("jt_mcap50_naive")),
        ],
    )
    lines += [
        "## Leg decomposition (FACTOR-JT top-100 GROSS, unconstrained)",
        "",
        "Academic identity: D10−D1 = (long − universe) + (universe − short).",
        "",
        "| piece | Sharpe | trail-18m | ann mean | NW-t | share of factor mean |",
        "|-------|--------|-----------|----------|------|----------------------|",
        f"| long leg | {_fmt(lng.get('sharpe'))} | {_fmt(lng.get('sharpe_trail18m'))} | {_fmt(lng.get('ann_mean'))} | {_fmt(lng.get('nw_t'), 2)} | — |",
        f"| short leg | {_fmt(sh.get('sharpe'))} | {_fmt(sh.get('sharpe_trail18m'))} | {_fmt(sh.get('ann_mean'))} | {_fmt(sh.get('nw_t'), 2)} | — |",
        f"| universe EW | {_fmt(uni.get('sharpe'))} | {_fmt(uni.get('sharpe_trail18m'))} | {_fmt(uni.get('ann_mean'))} | {_fmt(uni.get('nw_t'), 2)} | — |",
        f"| long − universe | {_fmt(lm.get('sharpe'))} | {_fmt(lm.get('sharpe_trail18m'))} | {_fmt(lm.get('ann_mean'))} | {_fmt(lm.get('nw_t'), 2)} | {_fmt(share_l, 3)} |",
        f"| universe − short | {_fmt(um.get('sharpe'))} | {_fmt(um.get('sharpe_trail18m'))} | {_fmt(um.get('ann_mean'))} | {_fmt(um.get('nw_t'), 2)} | {_fmt(share_s, 3)} |",
        "",
        "## Correlation vs implementable SPREAD-LS",
        "",
        f"- FACTOR-JT top-100 GROSS vs BOOK-CMC (same prices, frozen positions): "
        f"`{_fmt((corr.get('vs_cmc') or {}).get('corr'), 4)}` (n={(corr.get('vs_cmc') or {}).get('n')})",
        f"- FACTOR-JT top-100 GROSS vs BOOK-HYBRID (3.c implementable): "
        f"`{_fmt((corr.get('vs_hybrid') or {}).get('corr'), 4)}` (n={(corr.get('vs_hybrid') or {}).get('n')})",
        "",
        "## Implementation-tax waterfall (FACTOR-JT top-100 spine)",
        "",
        *wf_lines,
        "",
        f"IMPLEMENTATION TAX (paper GROSS − hybrid) = `{_fmt(waterfall.get('tax'))}`.",
        "",
        "Charts: `charts/btcb_academic_factor_equity.png`, `charts/btcb_academic_factor_waterfall.png`.",
        "",
        "## 3.c suspension (unchanged)",
        "",
        "Official SPREAD-LS record remains SUSPENDED. This phase does not adopt the paper factor, "
        "does not change the production book, and does not lift the pricing-gap freeze.",
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def plot_factor_equity(
    factor_eq: pd.Series,
    book_eq: pd.Series | None,
    out_path: Path,
    *,
    factor_label: str = "FACTOR-JT top-100 GROSS",
    book_label: str = "SPREAD-LS hybrid",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fe = factor_eq.copy()
    fe.index = pd.DatetimeIndex(pd.to_datetime(fe.index, utc=True)).tz_convert("UTC").normalize()
    fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    ax.plot(fe.index, fe.values, lw=1.5, color="#4C78A8", label=factor_label)
    if book_eq is not None and len(book_eq):
        be = book_eq.copy()
        be.index = pd.DatetimeIndex(pd.to_datetime(be.index, utc=True)).tz_convert("UTC").normalize()
        ax.plot(be.index, be.values, lw=1.2, color="#54A24B", ls="--", label=book_label)
    ax.set_yscale("log")
    ax.set_ylabel("cumulative (log)")
    ax.set_title("Academic D10−D1 (GROSS) vs implementable SPREAD-LS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_waterfall(rows: list[dict], out_path: Path, *, tax: float | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = []
    vals = []
    short = {
        "paper_gross": "paper GROSS",
        "net_naive": "NET-NAIVE",
        "shortability": "+ shortability",
        "real_costs": "+ real costs",
        "hybrid_book": "3.c hybrid",
    }
    for r in rows:
        names.append(short.get(r.get("step"), r.get("step")))
        vals.append(float(r.get("sharpe") if r.get("sharpe") is not None else np.nan))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B"]
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors[: len(names)], width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Sharpe")
    title = "Implementation-tax waterfall — FACTOR-JT top-100"
    if tax is not None and np.isfinite(tax):
        title += f"  (tax={tax:.2f})"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
