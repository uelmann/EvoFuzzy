"""Universe sensitivity report for SPREAD-LS."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btcb.constants import (
    PHASE3_FUNDING_CAVEAT,
    UNIVERSE_FUNDING_ON,
    UNIVERSE_SENS_CRITERION,
)


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


def _row(tag: str, b: dict) -> str:
    c = b.get("concentration") or {}
    return (
        f"| {tag} | {_fmt(b.get('net_sharpe'))} | {_fmt(b.get('net_sharpe_trail18m'))} "
        f"| {_pct(b.get('maxdd'))} | {_fmt(b.get('avg_n_long'), 2)} | {_fmt(b.get('avg_n_short'), 2)} "
        f"| {_fmt(b.get('avg_shortable'), 1)} | {_pct(b.get('pct_incomplete_short'))} "
        f"| {_fmt(b.get('ann_turnover'), 2)} | {_fmt(b.get('squeeze_mean'))} "
        f"| {_fmt(b.get('realized_beta_full'))} | {_pct(c.get('top5_pnl_share'))} "
        f"| {_fmt(b.get('rankic'))} |"
    )


def _cycle_block(books: dict) -> list[str]:
    lines = [
        "| U | cycle | n | net Sharpe | MaxDD | #long | #short |",
        "|---|-------|---|------------|-------|-------|--------|",
    ]
    for n in (30, 50, 100):
        b = books.get(n) or {}
        for cyc, c in (b.get("cycles") or {}).items():
            lines.append(
                f"| {n} | {cyc} | {c.get('n')} | {_fmt(c.get('net_sharpe'))} | {_pct(c.get('maxdd'))} "
                f"| {_fmt(c.get('avg_n_long'), 2)} | {_fmt(c.get('avg_n_short'), 2)} |"
            )
    return lines


def write_universe_sensitivity(
    path: Path,
    *,
    choice: dict,
    ranking: dict,
    dv: dict,
    mcap: dict,
    extra: dict,
) -> str:
    ch = choice or {}
    rk = ranking or {}
    lines = [
        "# BTC-BEATER SPREAD-LS — universe sensitivity (top-30 / top-50 / top-100)",
        "",
        "**BACKTEST ONLY.** Portfolio layer only. 2.c spread cache byte-identical. "
        f"FUNDING={'ON' if UNIVERSE_FUNDING_ON else 'OFF'} (3.b has not run). "
        "β-matched. CPU only, zero GPU. COMBO untouched.",
        "",
        "## Pre-registered reading (verbatim, before results)",
        "",
        f"> {UNIVERSE_SENS_CRITERION}",
        "",
        "## Funding caveat (verbatim)",
        "",
        f"> {PHASE3_FUNDING_CAVEAT}",
        "",
        "## Mechanical choice",
        "",
        f"- **Chosen production U = top-{ch.get('chosen_u')}** "
        f"(fallback={bool(ch.get('fallback'))}; ranking={rk.get('ranking')})",
        f"- best full={_fmt(ch.get('best_full'))} so need ≥ {_fmt(ch.get('need_full'))}; "
        f"best trail-18m={_fmt(ch.get('best_trail'))} so need ≥ {_fmt(ch.get('need_trail'))}",
        f"- mcap beats volume by ≥ 0.20 on both windows for chosen U: "
        f"**{bool(rk.get('mcap_beats_volume'))}** "
        f"(DV {_fmt(rk.get('dv_full'))}/{_fmt(rk.get('dv_trail'))} vs "
        f"mcap {_fmt(rk.get('mcap_full'))}/{_fmt(rk.get('mcap_trail'))})",
        "",
        "## Dollar-volume universes (house standard, β-matched, funding-off, h=14)",
        "",
        "| U | net Sharpe | trail-18m | MaxDD | #long | #short | shortable | % inc. short | ann TO | squeeze mean | β vs BTC | top-5 PnL | RankIC |",
        "|---|------------|-----------|-------|-------|--------|-----------|--------------|--------|--------------|----------|-----------|--------|",
    ]
    for n in (30, 50, 100):
        lines.append(_row(f"DV top-{n}", dv[n]))
    lines += [
        "",
        "## Per-cycle (dollar-volume)",
        "",
    ]
    lines += _cycle_block(dv)
    lines += [
        "",
        "## Market-cap universes (informational, same mechanics)",
        "",
        "| U | net Sharpe | trail-18m | MaxDD | #long | #short | shortable | % inc. short | ann TO | squeeze mean | β vs BTC | top-5 PnL | RankIC |",
        "|---|------------|-----------|-------|-------|--------|-----------|--------------|--------|--------------|----------|-----------|--------|",
    ]
    for n in (30, 50, 100):
        lines.append(_row(f"mcap top-{n}", mcap[n]))
    lines += [
        "",
        "## Within-universe RankIC (spread vs excess h=14, last-fold-wins)",
        "",
        "| ranking | U=30 | U=50 | U=100 |",
        "|---------|------|------|-------|",
        f"| dollar-volume | {_fmt(dv[30].get('rankic'), 4)} | {_fmt(dv[50].get('rankic'), 4)} | {_fmt(dv[100].get('rankic'), 4)} |",
        f"| market-cap | {_fmt(mcap[30].get('rankic'), 4)} | {_fmt(mcap[50].get('rankic'), 4)} | {_fmt(mcap[100].get('rankic'), 4)} |",
        "",
        "## Cache / reuse",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` n_files={extra.get('pred_n_files')}",
        f"- BTC in book hits (all runs) = {extra.get('btc_hits_total')}",
        f"- GPU={extra.get('gpu_used', False)}. Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}.",
        "",
        "COMBO untouched (v2.0-combo-final).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_three_equity(
    books: dict,
    out_path: Path,
    *,
    ranking_label: str = "DV",
    title: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    colors = {30: "#4C78A8", 50: "#F58518", 100: "#54A24B"}
    for n in (30, 50, 100):
        b = books.get(n) or {}
        eq = b.get("equity")
        if eq is None or len(eq) == 0:
            continue
        sh = b.get("net_sharpe")
        tr = b.get("net_sharpe_trail18m")
        lab = f"{ranking_label} top-{n}"
        if sh is not None and np.isfinite(float(sh)):
            lab += f"  Sh={float(sh):.2f}"
        if tr is not None and np.isfinite(float(tr)):
            lab += f"  trail={float(tr):.2f}"
        ax.plot(eq.index, eq.values, lw=1.4, color=colors[n], label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title(title or "SPREAD-LS universe sensitivity — β-matched, funding-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
