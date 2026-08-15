"""Phase 3 SPREAD-LS report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import DEATH_CONVENTION, PHASE3_CRITERION, PHASE3_FUNDING_CAVEAT


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


def _book_row(name: str, b: dict) -> str:
    fe = b.get("forced_exits") or {}
    fc = b.get("forced_covers") or {}
    return (
        f"| {name} | {_fmt(b.get('net_sharpe'))} | {_fmt(b.get('net_sharpe_trail18m'))} "
        f"| {_pct(b.get('book_total'))} | {_pct(b.get('book_cagr'))} | {_pct(b.get('maxdd'))} "
        f"| {_fmt(b.get('avg_n_long'), 2)} | {_fmt(b.get('avg_n_short'), 2)} "
        f"| {_fmt(b.get('avg_shortable'), 1)} | {_pct(b.get('pct_incomplete_short'))} "
        f"| {_fmt(b.get('ann_turnover'), 2)} | {fe.get('n_events')} | {fc.get('n_events')} "
        f"| {_fmt(b.get('realized_beta_full'))} |"
    )


def write_phase3(
    path: Path,
    *,
    verdicts: dict,
    books: dict,
    overlap: dict,
    squeeze: list,
    extra: dict,
) -> str:
    h14 = books.get("dn_h14") or {}
    lines = [
        "# BTC-BEATER Phase 3 — SPREAD-LS challenger",
        "",
        "**BACKTEST ONLY.** Portfolio layer only. 2.c spread scores reused byte-identical. "
        "No retraining. No BTC in either leg. CPU only, zero GPU. COMBO untouched.",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE3_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "Shorts are force-covered on the same convention. Forced-exit and forced-cover counts are reported separately.",
        "",
        "## Funding caveat (verbatim)",
        "",
        f"> {PHASE3_FUNDING_CAVEAT}",
        "",
        "## Mechanical verdicts (dollar-neutral h=14 headline)",
        "",
        f"- **SPREAD-LS is VIABLE: {bool(verdicts.get('viable'))}** "
        f"(full {_fmt(verdicts.get('net_sharpe_full'))} need ≥ {_fmt(verdicts.get('need_full'))}; "
        f"trail-18m {_fmt(verdicts.get('net_sharpe_trail18m'))} need ≥ {_fmt(verdicts.get('need_trail'))})",
        f"- **SPREAD-LS is SLEEVE-GRADE: {bool(verdicts.get('sleeve_grade'))}** "
        f"(corr={_fmt(verdicts.get('corr_combo'))} need < {_fmt(verdicts.get('need_corr_lt'))}; "
        f"same-window {_fmt(verdicts.get('same_window_ls'))} vs COMBO {_fmt(verdicts.get('same_window_combo'))} "
        f"need ≥ {_fmt(verdicts.get('need_sleeve_sharpe'))})",
        f"- **SPREAD-LS is REPLACEMENT CANDIDATE: {bool(verdicts.get('replacement_candidate'))}** "
        f"(need ≥ {_fmt(verdicts.get('need_replace_sharpe'))})",
        f"- OOS {h14.get('start')} → {h14.get('end')} n={h14.get('n_days')}",
        f"- realized beta vs BTC (full OLS) = {_fmt(h14.get('realized_beta_full'))}",
        f"- avg shortable (PIT-100 ex-BTC) = {_fmt(h14.get('avg_shortable'), 1)}; "
        f"% incomplete short = {_pct(h14.get('pct_incomplete_short'))}",
        f"- forced exits={ (h14.get('forced_exits') or {}).get('n_events') } "
        f"covers={ (h14.get('forced_covers') or {}).get('n_events') }",
        f"- BTC in book hits = {h14.get('btc_in_book_hits')}",
        "",
        "The beta-matched variant is reported, not judged. No post-hoc adjustment.",
        "",
        "## Books",
        "",
        "| book | net Sharpe | trail-18m | total | CAGR | MaxDD | #long | #short | shortable | % inc. short | ann TO | exits | covers | β vs BTC |",
        "|------|------------|-----------|-------|------|-------|-------|--------|-----------|--------------|--------|-------|--------|----------|",
    ]
    order = [
        ("SPREAD-LS DN h=14 (headline)", "dn_h14"),
        ("SPREAD-LS β-match h=14", "bm_h14"),
        ("SPREAD-LS DN h=30", "dn_h30"),
        ("SPREAD-LS β-match h=30", "bm_h30"),
    ]
    for name, key in order:
        b = books.get(key)
        if b:
            lines.append(_book_row(name, b))
    lines += [
        "",
        "## Per-cycle honesty (headline DN h=14)",
        "",
        "| cycle | n | total | CAGR | net Sharpe | MaxDD | #long | #short |",
        "|-------|---|-------|------|------------|-------|-------|--------|",
    ]
    for cyc, c in (h14.get("cycles") or {}).items():
        lines.append(
            f"| {cyc} | {c.get('n')} | {_pct(c.get('book_total'))} | {_pct(c.get('book_cagr'))} "
            f"| {_fmt(c.get('net_sharpe'))} | {_pct(c.get('maxdd'))} "
            f"| {_fmt(c.get('avg_n_long'), 2)} | {_fmt(c.get('avg_n_short'), 2)} |"
        )
    ov = overlap or {}
    lines += [
        "",
        "## vs frozen COMBO (overlap 2022-01 →)",
        "",
        f"Window {ov.get('start')} → {ov.get('end')} n={ov.get('n_days')}. COMBO replayed from frozen A0 scores; product untouched.",
        "",
        f"- SPREAD-LS same-window net Sharpe = {_fmt(ov.get('ls_sharpe'))} (total {_pct(ov.get('ls_total'))})",
        f"- COMBO same-window net Sharpe = {_fmt(ov.get('combo_sharpe'))} (total {_pct(ov.get('combo_total'))})",
        f"- daily PnL correlation = {_fmt(ov.get('corr'))}",
        "",
        "## Squeeze-days (20 largest EW floored top-100 up-days)",
        "",
        "| date | EW top-100 | SPREAD-LS net |",
        "|------|------------|---------------|",
    ]
    for r in squeeze or []:
        lines.append(f"| {r.get('date')} | {_pct(r.get('ew_top100'), 2)} | {_pct(r.get('spread_ls'), 2)} |")
    if squeeze:
        sl = [r.get("spread_ls") for r in squeeze if r.get("spread_ls") is not None]
        sl = [float(x) for x in sl if x is not None and np.isfinite(float(x))]
        lines.append("")
        lines.append(
            f"Squeeze-day mean SPREAD-LS PnL = {_pct(float(np.mean(sl)) if sl else float('nan'), 2)}; "
            f"sum = {_pct(float(np.sum(sl)) if sl else float('nan'), 2)}."
        )
    extra = extra or {}
    lines += [
        "",
        "## Cache / reuse",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` n_files={extra.get('pred_n_files')}",
        f"- BTC id = {extra.get('btc_id')}; BTC in book hits = {extra.get('btc_in_book_hits')}",
        f"- shortable mapped Binance perps = {extra.get('n_binance_symbols')}",
        f"- GPU={extra.get('gpu_used', False)}. Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}.",
        "",
        "COMBO untouched (v2.0-combo-final).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_equity_dd(book: dict, out_path: Path) -> None:
    eq = book.get("equity")
    if eq is None or len(eq) == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dd = eq / eq.cummax() - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    ax.plot(eq.index, eq.values, lw=1.3, color="#4C78A8", label="SPREAD-LS DN")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("BTC-BEATER Phase 3 — SPREAD-LS (dollar-neutral, vs cash)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.fill_between(dd.index, dd.values, 0, color="#E45756", alpha=0.7)
    ax2.set_ylabel("drawdown")
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rolling_beta(book: dict, out_path: Path) -> None:
    b = book.get("realized_beta_90d")
    if b is None or (isinstance(b, pd.Series) and b.dropna().empty):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.plot(b.index, b.values, lw=1.2, color="#4C78A8", label="90d OLS β vs BTC")
    full = book.get("realized_beta_full")
    if full is not None and np.isfinite(full):
        ax.axhline(float(full), color="#E45756", ls="--", lw=1.0, label=f"full β={float(full):.2f}")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("beta")
    ax.set_title("SPREAD-LS realized beta vs BTC (residual market bet)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overlap(overlap: dict, out_path: Path) -> None:
    a = overlap.get("ls_daily")
    b = overlap.get("combo_daily")
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series) or a.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq_a = (1.0 + a.fillna(0.0)).cumprod()
    eq_b = (1.0 + b.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    ax.plot(eq_a.index, eq_a.values, lw=1.3, label="SPREAD-LS DN")
    ax.plot(eq_b.index, eq_b.values, lw=1.3, label="COMBO (frozen)")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("Overlap 2022-01 → — SPREAD-LS vs frozen COMBO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
