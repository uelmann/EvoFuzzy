"""MANUEL-2 report + charts. Analysis only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    MANUEL2_BEST_RULE,
    MANUEL2_BTC_COMPLETION,
    MANUEL2_CRITERION,
    MANUEL2_FORMULA,
    MANUEL2_GAUSS_COMPLETION,
    MANUEL2_MAXDD_RULE,
    MANUEL2_STABLE_COMPLETION,
    PHASE2_CYCLES,
    PHASE2C_PRED_SHA256,
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


BOOK_ORDER = (
    ("daily_btc_in", "daily btc-in"),
    ("daily_btc_ex", "daily btc-ex"),
    ("weekly_btc_in", "weekly btc-in"),
    ("weekly_btc_ex", "weekly btc-ex"),
)


def _book_row(name: str, b: dict) -> str:
    if not b:
        return f"| {name} | nan | nan | nan | nan | nan | nan |  |  |  |"
    fe = b.get("forced_exits") or {}
    return (
        f"| {name} | {_pct(b.get('total', b.get('book_total')))} "
        f"| {_pct(b.get('cagr', b.get('book_cagr')))} "
        f"| {_fmt(b.get('sharpe', b.get('book_sharpe')))} "
        f"| {_fmt(b.get('net_sharpe_trail18m'))} "
        f"| {_pct(b.get('maxdd'))} "
        f"| {_fmt(b.get('rel_sharpe'))} "
        f"| {_pct(b.get('rel_total'))} "
        f"| {_fmt(b.get('corr_btc'))} "
        f"| {_fmt(b.get('avg_n_names'), 2)} "
        f"| {_fmt(b.get('ann_turnover'), 1)} "
        f"| {fe.get('n_events', b.get('forced_n', ''))} |"
    )


def _cycle_table(books: dict) -> list[str]:
    lines = [
        "| cycle | " + " | ".join(lab for _, lab in BOOK_ORDER) + " | BTC |",
        "|-------|" + "|".join(["------"] * (len(BOOK_ORDER) + 1)) + "|",
    ]
    for cyc, *_ in PHASE2_CYCLES:
        cells = []
        for key, _lab in BOOK_ORDER:
            blob = ((books.get(key) or {}).get("cycles") or {}).get(cyc) or {}
            tot = blob.get("book_total")
            sh = blob.get("book_sharpe")
            cells.append(f"{_pct(tot)} / {_fmt(sh)}")
        btc_blob = ((books.get("daily_btc_ex") or {}).get("cycles") or {}).get(cyc) or {}
        cells.append(_pct(btc_blob.get("btc_total")))
        lines.append(f"| {cyc} | " + " | ".join(cells) + " |")
    return lines


def write_manuel2(
    path: Path,
    *,
    books: dict,
    bench: dict,
    verdict: dict,
    peg: dict,
    extra: dict,
) -> str:
    best_key = verdict.get("best_book")
    best = books.get(best_key) or {}
    btc = bench.get("btc_bh") or {}
    ew = bench.get("ew_mcap50") or {}
    spr = bench.get("frozen_spread") or {}
    dv = books.get("daily_dv100_btc_ex") or {}
    lines = [
        "# BTC-BEATER MANUEL-2 — gauss-momentum falsification",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. "
        "CPU only, zero GPU. Frozen products untouched. Pricing = Binance-hybrid where listed (3.e canonical). Master only.",
        "",
        "## Formula (verbatim, frozen before results)",
        "",
        f"> {MANUEL2_FORMULA}",
        "",
        "## Spec completions (verbatim, frozen before results)",
        "",
        f"> {MANUEL2_STABLE_COMPLETION}",
        "",
        f"> {MANUEL2_BTC_COMPLETION}",
        "",
        f"> {MANUEL2_GAUSS_COMPLETION}",
        "",
        "## Best-book rule (verbatim, frozen)",
        "",
        f"> {MANUEL2_BEST_RULE}",
        "",
        "## MaxDD convention (verbatim, frozen)",
        "",
        f"> {MANUEL2_MAXDD_RULE}",
        "",
        "## Pre-registered verdicts (verbatim, before results)",
        "",
        f"> {MANUEL2_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Identity",
        "",
        f"- Window {extra.get('start')} → {extra.get('end')} n_days={extra.get('n_days')}",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- GPU used = `{extra.get('gpu_used', False)}`",
        f"- Hybrid name-days Binance share = `{_fmt(extra.get('hybrid_bn_share'))}`",
        "",
        "## Pegged-asset exclusion (logged)",
        "",
        f"- Tagged ids = `{peg.get('n_tagged')}` (STABLE_OR_WRAP ∪ extra stables ∪ USD-suffix ∪ name/slug needles)",
        f"- Heuristic `|90d total return| < 2%`: median flagged/date = `{_fmt(peg.get('median_flagged_per_date'), 1)}` "
        f"mean = `{_fmt(peg.get('mean_flagged_per_date'), 1)}`",
        f"- Sample tagged: {', '.join(str(s) for s in (peg.get('sample_symbols') or [])[:24])}",
        "",
        "Rationale: `/gauss(std63)` would otherwise buy near-zero-vol pegs.",
        "",
        "## 1 — Four books (4-way look)",
        "",
        "Ladder construction: EW top 5% (`ceil` → 3 on a full 50), no name cap, always invested, 10 bps/side, "
        "death-in-position. Daily = headline; weekly Mondays = secondary.",
        "",
        "| book | total | CAGR | USD Sharpe | trail-18m | MaxDD | rel Sharpe | rel total | corr vs BTC | avg n | ann TO | forced |",
        "|------|-------|------|------------|-----------|-------|------------|-----------|-------------|-------|--------|--------|",
    ]
    for key, lab in BOOK_ORDER:
        lines.append(_book_row(lab, books.get(key) or {}))
    lines.extend(
        [
            "",
            f"Best book (highest total) = **{best_key}**.",
            "",
            "Informational DV row (floored PIT top-100, daily, btc-ex, same formula):",
            "",
            "| book | total | CAGR | USD Sharpe | trail-18m | MaxDD | rel Sharpe | rel total | corr vs BTC | avg n | ann TO | forced |",
            "|------|-------|------|------------|-----------|-------|------------|-----------|-------------|-------|--------|--------|",
            _book_row("daily DV100 btc-ex", dv),
            "",
            "## 2 — Benchmarks (identical window)",
            "",
            "| book | total | CAGR | USD Sharpe | trail-18m | MaxDD | rel Sharpe | rel total | corr vs BTC | avg n | ann TO | forced |",
            "|------|-------|------|------------|-----------|-------|------------|-----------|-------------|-------|--------|--------|",
            _book_row("BTC B&H", btc),
            _book_row("EW mcap-50 ex-pegged", ew),
            _book_row("frozen-spread crude (ref)", spr),
            "",
            "## 3 — Claim metrics (best book)",
            "",
            f"- Best = `{best_key}`",
            f"- total = `{_pct(best.get('total', best.get('book_total')))}` vs BTC `{_pct(btc.get('total', btc.get('book_total')))}` "
            f"(clause 1 pass={verdict.get('clause1_total_ge_btc')})",
            f"- daily PnL corr vs BTC = `{_fmt(best.get('corr_btc'))}` n={best.get('corr_btc_n')} "
            f"(clause 2 pass={verdict.get('clause2_corr_le_070')}; need ≤ 0.70)",
            f"- relative-line Sharpe = `{_fmt(best.get('rel_sharpe'))}` total `{_pct(best.get('rel_total'))}` "
            f"(STRONG need ≥ 0.50; pass={verdict.get('rel_ok')})",
            f"- MaxDD = `{_pct(best.get('maxdd'))}` vs BTC `{_pct(btc.get('maxdd'))}` "
            f"(STRONG |DD| ≤ 1.10×BTC; pass={verdict.get('maxdd_ok')})",
            "",
            "## 4 — Score-vs-vol diagnostic",
            "",
            "Mean per-date cross-sectional Spearman of score vs std_63d (what `/gauss(std)` tilts toward). Negative = low-vol tilt.",
            "",
        ]
    )
    for key, lab in BOOK_ORDER:
        b = books.get(key) or {}
        lines.append(f"- {lab}: `{_fmt(b.get('score_vol_tilt'))}` n={b.get('score_vol_tilt_n', '')}")
    lines.extend(
        [
            f"- daily DV100 btc-ex: `{_fmt(dv.get('score_vol_tilt'))}` n={dv.get('score_vol_tilt_n', '')}",
            "",
            "## 5 — Top-5 name PnL concentration (best book)",
            "",
            "| rank | id | symbol | contrib | share |",
            "|------|----|--------|---------|-------|",
        ]
    )
    for i, r in enumerate(best.get("top5") or [], start=1):
        lines.append(
            f"| {i} | {r.get('id')} | {r.get('symbol')} | {_fmt(r.get('contrib'), 4)} | {_pct(r.get('share'), 2)} |"
        )
    lines.extend(
        [
            "",
            f"Top-5 abs share of Σ|contrib| = `{_pct(best.get('top5_share'))}`.",
            "",
            "## 6 — Per-cycle honesty (total / USD Sharpe)",
            "",
            *_cycle_table(books),
            "",
            "No single cycle overrides.",
            "",
            "## 7 — Mechanical verdict",
            "",
            f"- **{verdict.get('label')}**",
            f"- Best book = `{verdict.get('best_book')}`",
            f"- Clause 1 (total ≥ BTC) = `{verdict.get('clause1_total_ge_btc')}` "
            f"(`{_pct(verdict.get('best_total'))}` vs `{_pct(verdict.get('btc_total'))}`)",
            f"- Clause 2 (corr ≤ 0.70) = `{verdict.get('clause2_corr_le_070')}` (`{_fmt(verdict.get('best_corr'))}`)",
        ]
    )
    if verdict.get("partial_which"):
        lines.append(f"- PARTIAL which: **{verdict.get('partial_which')}**")
    lines.extend(
        [
            f"- STRONG extras: rel Sharpe `{_fmt(verdict.get('best_rel_sharpe'))}` pass={verdict.get('rel_ok')}; "
            f"MaxDD pass={verdict.get('maxdd_ok')}",
            "",
            "Mechanical, no post-hoc adjustment. Nothing adopted.",
            "",
            "## Plain language",
            "",
            extra.get("plain", ""),
            "",
            "## Notes",
            "",
            "- Four books are a disclosed look; best = highest total. No parameter search.",
            "- EW basket is the no-selection line on the same mcap top-50 ex-pegged (btc-in) universe.",
            "- Frozen-spread crude book is Ladder-1 (h=14) from the 2.c cache; reference only.",
            f"- Chart: `charts/manuel2_equity.png`. Elapsed s=`{_fmt(extra.get('elapsed_sec'), 1)}`. GPU=`{extra.get('gpu_used', False)}`.",
            "",
            "COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def update_ledger_manuel2(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER MANUEL-2"
    block = [
        "",
        marker,
        "",
        "gauss(ret14)·gauss(ret28)/gauss(std63), top-50 mcap, top 5%, long-only. Backtest only. Nothing adopted. Binance-hybrid.",
        "",
        f"**{verdict.get('label')}.** Best=`{verdict.get('best_book')}` total `{extra.get('best_total')}` vs BTC `{extra.get('btc_total')}`; "
        f"corr `{extra.get('best_corr')}`. Clause1={verdict.get('clause1_total_ge_btc')} clause2={verdict.get('clause2_corr_le_070')}. "
        f"Score-vol tilt `{extra.get('tilt')}`.",
        "",
        "Mechanical, no post-hoc adjustment. Frozen products untouched.",
        "",
    ]
    new = "\n".join(block)
    if not new.startswith("\n"):
        new = "\n" + new
    if marker in text:
        pre, rest = text.split(marker, 1)
        lines = rest.splitlines()
        cut = None
        for i, ln in enumerate(lines[1:], start=1):
            if ln.startswith("## "):
                cut = i
                break
        if cut is None:
            text = pre.rstrip() + new
        else:
            text = pre.rstrip() + new + "\n".join(lines[cut:])
            if not text.endswith("\n"):
                text += "\n"
    else:
        text = text.rstrip() + "\n" + new
    path.write_text(text)
    return text


def _eq(book: dict):
    eq = book.get("equity") if isinstance(book, dict) else None
    if isinstance(eq, pd.Series) and len(eq):
        s = eq.astype(float)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).tz_convert("UTC").normalize()
        return s.sort_index()
    rets = book.get("daily_ret") if isinstance(book, dict) else None
    if rets is None:
        return None
    r = pd.Series(rets, dtype=float).fillna(0.0)
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index, utc=True)).tz_convert("UTC").normalize()
    r = r.sort_index()
    return (1.0 + r).cumprod() if len(r) else None


def plot_manuel2(best: dict, btc: dict, ew: dict, out_path: Path, *, best_label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq_b = _eq(best)
    eq_t = _eq(btc)
    eq_e = _eq(ew)
    if eq_b is None or eq_t is None:
        raise RuntimeError("plot_manuel2: missing equity")
    idx = eq_b.index.intersection(eq_t.index)
    if eq_e is not None:
        idx = idx.intersection(eq_e.index)
    eq_b, eq_t = eq_b.reindex(idx), eq_t.reindex(idx)
    rel_b = eq_b / eq_t.replace(0, np.nan)
    rel_e = None
    if eq_e is not None:
        eq_e = eq_e.reindex(idx)
        rel_e = eq_e / eq_t.replace(0, np.nan)
    roll = best.get("corr_btc_roll90")
    if isinstance(roll, pd.Series) and len(roll):
        roll = roll.copy()
        roll.index = pd.DatetimeIndex(pd.to_datetime(roll.index, utc=True)).tz_convert("UTC").normalize()
        roll = roll.reindex(idx)
    else:
        br = pd.Series(best.get("daily_ret"), dtype=float)
        bt = pd.Series(btc.get("daily_ret"), dtype=float)
        br.index = pd.DatetimeIndex(pd.to_datetime(br.index, utc=True)).tz_convert("UTC").normalize()
        bt.index = pd.DatetimeIndex(pd.to_datetime(bt.index, utc=True)).tz_convert("UTC").normalize()
        roll = br.reindex(idx).rolling(90, min_periods=30).corr(bt.reindex(idx))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]},
    )
    ax, axr, axc = axes
    lab_b = best_label
    if best.get("cagr") is not None and np.isfinite(float(best.get("cagr"))):
        lab_b += f"  CAGR={100.0 * float(best.get('cagr')):.1f}%"
    ax.plot(eq_b.index, eq_b.values, lw=1.4, color="#4C78A8", label=lab_b)
    ax.plot(eq_t.index, eq_t.values, lw=1.3, color="#F58518", label="BTC B&H")
    if eq_e is not None:
        ax.plot(eq_e.index, eq_e.values, lw=1.2, color="#54A24B", label="EW mcap-50")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("MANUEL-2 — best book vs BTC vs EW basket (information only; nothing adopted)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    axr.plot(rel_b.index, rel_b.values, lw=1.3, color="#4C78A8", label="best / BTC")
    if rel_e is not None:
        axr.plot(rel_e.index, rel_e.values, lw=1.1, color="#54A24B", label="EW / BTC")
    axr.axhline(1.0, color="0.5", lw=0.8)
    axr.set_ylabel("relative")
    axr.legend(fontsize=8)
    axr.grid(True, alpha=0.3)

    axc.plot(roll.index, roll.values, lw=1.2, color="#4C78A8", label="best vs BTC 90d corr")
    axc.axhline(0.70, color="#E45756", ls="--", lw=1.0, label="0.70")
    axc.axhline(0.0, color="0.5", lw=0.8)
    axc.set_ylabel("roll corr")
    axc.set_ylim(-0.3, 1.05)
    axc.legend(fontsize=8)
    axc.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
