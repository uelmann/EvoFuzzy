"""LONG-TIDE report, charts, ledger update. Backtest only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    LONGTIDE_CRITERION,
    LONGTIDE_PRECONDITION,
    PHASE2C_PRED_SHA256,
    PHASE2_CYCLES,
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


def _book_row(name: str, b: dict) -> str:
    fe = b.get("forced_exits") or {}
    return (
        f"| {name} | {_pct(b.get('book_total'))} | {_pct(b.get('book_cagr'))} "
        f"| {_fmt(b.get('book_sharpe'))} | {_fmt(b.get('net_sharpe_trail18m'))} "
        f"| {_fmt(b.get('rel_sharpe'))} | {_pct(b.get('rel_total'))} | {_pct(b.get('maxdd'))} "
        f"| {_pct(b.get('avg_alt_deployment'))} | {_pct(b.get('gate_on_frac'))} "
        f"| {_fmt(b.get('avg_n_names'), 2)} | {_fmt(b.get('ann_turnover'), 2)} "
        f"| {fe.get('n_events')} |"
    )


def _cycle_table(books: dict[str, dict]) -> list[str]:
    lines = [
        "| cycle | book | n | tot | CAGR | USD Sharpe | rel Sharpe | MaxDD | alt dep | #names |",
        "|-------|------|---|-----|------|------------|------------|-------|---------|--------|",
    ]
    names = [c[0] for c in PHASE2_CYCLES]
    for cyc in names:
        for bname, b in books.items():
            blob = (b.get("cycles") or {}).get(cyc) or {}
            if not blob:
                continue
            lines.append(
                f"| {cyc} | {bname} | {blob.get('n')} | {_pct(blob.get('book_total'))} "
                f"| {_pct(blob.get('book_cagr'))} | {_fmt(blob.get('book_sharpe'))} "
                f"| {_fmt(blob.get('rel_sharpe'))} | {_pct(blob.get('maxdd'))} "
                f"| {_pct(1.0 - float(blob.get('avg_w_btc') or 0.0) if blob.get('avg_w_btc') is not None else float('nan'))} "
                f"| {_fmt(blob.get('avg_n_names'), 2)} |"
            )
    return lines


def write_longtide(
    path: Path,
    *,
    precondition: str,
    verdicts: dict,
    tide: dict,
    naked: dict,
    v1: dict,
    btc: dict,
    ew: dict,
    cmc_ref: dict,
    corr: dict,
    squeeze: list,
    gate: dict,
    extra: dict,
) -> str:
    fe = tide.get("forced_exits") or {}
    vlab = "VIABLE" if verdicts.get("viable") else "NOT VIABLE"
    slab = "SUPERSEDES" if verdicts.get("supersedes") else "DOES-NOT-SUPERSEDE"
    if verdicts.get("status") == "PARALLEL-VARIANT":
        outcome = (
            "LONG-TIDE is VIABLE but does not supersede. Recorded as a **parallel long variant**. "
            "BTC-BEATER v1 stays the official long product."
        )
    elif verdicts.get("supersedes"):
        outcome = (
            "LONG-TIDE **SUPERSEDES** BTC-BEATER v1 as the official long product. "
            "v1 is demoted to record-only. SPREAD-LS (BOOK-HYBRID) is unchanged."
        )
    else:
        outcome = (
            "LONG-TIDE is **NOT VIABLE**. BTC-BEATER v1 stays the official long product. "
            "SPREAD-LS (BOOK-HYBRID) is unchanged."
        )

    sq_lines = [
        "| date | EW top-100 | LONG-TIDE |",
        "|------|------------|-----------|",
    ]
    for r in squeeze or []:
        sq_lines.append(
            f"| {r.get('date')} | {_pct(r.get('ew_basket'))} | {_pct(r.get('spread_ls'))} |"
        )

    gate_rows = [
        "| metric | value |",
        "|--------|-------|",
        f"| REGIME_BREADTH | {gate.get('REGIME_BREADTH')} |",
        f"| REGIME_OFF_HYSTERESIS | {gate.get('REGIME_OFF_HYSTERESIS')} |",
        f"| byte-identical to frozen Stage-T | {gate.get('byte_identical')} |",
        f"| % days gate ON | {_pct(tide.get('gate_on_frac'))} |",
        f"| avg alt deployment | {_pct(tide.get('avg_alt_deployment'))} |",
        f"| n ON stretches | {extra.get('n_on_stretches')} |",
        f"| mean ON length (days) | {_fmt(extra.get('mean_on_len'), 1)} |",
    ]

    lines = [
        "# BTC-BEATER LONG-TIDE — full-size long leg + frozen regime gate",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** No schedules, no live components, no retraining, "
        "no signal changes. CPU only, zero GPU. Frozen products (SPREAD-LS, COMBO, BTC-BEATER v1) "
        "untouched as products. Pricing = Binance (3.e canonical). Master only.",
        "",
        "## Precondition (mechanical, checked first)",
        "",
        f"> EXECUTE ONLY IF Phase 3.e verdict = `{LONGTIDE_PRECONDITION}`. "
        "Otherwise print `BLOCKED-BY-SUSPENSION: 3.e verdict is <verdict>` and STOP.",
        "",
        f"- 3.e verdict = **{precondition}**",
        f"- Precondition pass = **{precondition == LONGTIDE_PRECONDITION}**",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {LONGTIDE_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- Gate params byte-identical = `{gate.get('byte_identical')}` "
        f"(breadth={gate.get('REGIME_BREADTH')}, off_hyst={gate.get('REGIME_OFF_HYSTERESIS')})",
        f"- Window {tide.get('start')} → {tide.get('end')} n={tide.get('n_days')}",
        f"- Common window with v1 {extra.get('common_start')} → {extra.get('common_end')} n={extra.get('common_n')}",
        f"- GPU used = `{extra.get('gpu_used', False)}`",
        "",
        "## Mechanical verdicts",
        "",
        f"- **LONG-TIDE is {vlab}**",
        f"- **LONG-TIDE {slab} BTC-BEATER v1**",
        f"- status = `{verdicts.get('status')}`",
        f"- (a) total { _pct(verdicts.get('book_total')) } ≥ BTC { _pct(verdicts.get('btc_total')) } → {verdicts.get('a_total_ge_btc')}",
        f"- (b) rel-line Sharpe { _fmt(verdicts.get('rel_sharpe')) } > 0 → {verdicts.get('b_rel_sharpe_gt0')}",
        f"- (c) MaxDD { _pct(verdicts.get('maxdd')) } ≤ BTC MaxDD { _pct(verdicts.get('btc_maxdd')) } → {verdicts.get('c_maxdd_le_btc')}",
        f"- (d) rel-line Sharpe ≥ v1 { _fmt(verdicts.get('v1_rel_sharpe')) } + 0.15 = { _fmt(verdicts.get('need_supersede_rel')) } → {verdicts.get('d_rel_ge_v1_plus_margin')}",
        f"- (e) avg alt deployment { _pct(verdicts.get('avg_alt_deployment')) } ≥ 15% → {verdicts.get('e_alt_deployment_ge_15pct')}",
        f"- (f) no cycle rel-line Sharpe < −0.30 (worst={_fmt(verdicts.get('worst_cycle_rel'))}) → {verdicts.get('f_no_cycle_rel_below_floor')}",
        "",
        outcome,
        "",
        "Mechanical, no post-hoc adjustment.",
        "",
        "## Four-way comparison (identical window)",
        "",
        "| book | total | CAGR | USD Sharpe | trail-18m | rel Sharpe | rel total | MaxDD | alt dep | gate ON | avg #names | ann TO | forced |",
        "|------|-------|------|------------|-----------|------------|-----------|-------|---------|---------|------------|--------|--------|",
        _book_row("LONG-TIDE (spot-filter, gated, BN)", tide),
        _book_row("NAKED LONG LEG (no gate, cash idle)", naked),
        _book_row("BTC-BEATER v1 (replayed read-only)", v1),
        _book_row("BTC B&H (Binance BTCUSDT)", btc),
        _book_row("EW floored top-100 (costless CMC)", ew),
        "",
        "Unrestricted-CMC reference (not judged):",
        "",
        "| book | total | CAGR | USD Sharpe | trail-18m | rel Sharpe | rel total | MaxDD | alt dep | gate ON | avg #names | ann TO | forced |",
        "|------|-------|------|------------|-----------|------------|-----------|-------|---------|---------|------------|--------|--------|",
        _book_row("LONG-TIDE CMC unrestricted (reference)", cmc_ref),
        "",
        "## Per-cycle honesty",
        "",
        *_cycle_table(
            {
                "LONG-TIDE": tide,
                "NAKED": naked,
                "v1": v1,
                "BTC": btc,
            }
        ),
        "",
        "## Gate ribbon",
        "",
        *gate_rows,
        "",
        "## Correlations (daily PnL)",
        "",
        f"- LONG-TIDE vs BTC-BEATER v1: corr=`{_fmt((corr.get('vs_v1') or {}).get('corr'))}` n={(corr.get('vs_v1') or {}).get('n')}",
        f"- LONG-TIDE vs SPREAD-LS (BOOK-HYBRID): corr=`{_fmt((corr.get('vs_spread_ls') or {}).get('corr'))}` n={(corr.get('vs_spread_ls') or {}).get('n')}",
        "",
        "## Squeeze days (EW floored top-100 vs LONG-TIDE)",
        "",
        *sq_lines,
        "",
        "## Forced exits (death convention)",
        "",
        f"- n_events={fe.get('n_events')} n_ids={fe.get('n_ids')} weight_sum={_fmt(fe.get('weight_sum'), 4)} "
        f"cost_drag={_fmt(fe.get('cost_drag'), 6)}",
        f"- symbols={fe.get('symbols')}",
        "",
        "## Gate value (MaxDD vs naked leg)",
        "",
        f"LONG-TIDE MaxDD `{_pct(tide.get('maxdd'))}` vs NAKED MaxDD `{_pct(naked.get('maxdd'))}` "
        f"(Δ `{_pct(float(tide.get('maxdd') or 0.0) - float(naked.get('maxdd') or 0.0))}`). "
        "Positive Δ (tide less negative) is the gate's drawdown value in one number.",
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU={extra.get('gpu_used', False)}.",
        "",
        "COMBO untouched (v2.0-combo-final). SPREAD-LS BOOK-HYBRID untouched as the official long/short product. "
        "BTC-BEATER v1 replayed read-only.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def _shade_gate(ax, gate: pd.Series, ymin, ymax, color="#d9ead3"):
    if gate is None or not isinstance(gate, pd.Series) or gate.empty:
        return
    on = gate.fillna(0).astype(float) > 0.5
    if not on.any():
        return
    starts = []
    prev = False
    t0 = None
    for t, v in on.items():
        if v and not prev:
            t0 = t
        if (not v or t == on.index[-1]) and prev:
            t1 = t
            ax.axvspan(t0, t1, color=color, alpha=0.35, lw=0)
        prev = bool(v)


def plot_longtide_equity(tide: dict, naked: dict, v1: dict, out_path: Path) -> None:
    eq = tide.get("equity")
    eqb = tide.get("equity_btc")
    rel = tide.get("rel_equity")
    if eq is None or eqb is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.4), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(eq.index, eq.values, lw=1.4, label="LONG-TIDE")
    ax.plot(eqb.index, eqb.values, lw=1.2, label="BTC B&H")
    n_eq = naked.get("equity")
    if isinstance(n_eq, pd.Series) and len(n_eq):
        ax.plot(n_eq.index, n_eq.values, lw=1.1, alpha=0.9, label="NAKED LONG LEG")
    v_eq = v1.get("equity")
    if isinstance(v_eq, pd.Series) and len(v_eq):
        ax.plot(v_eq.index, v_eq.values, lw=1.1, alpha=0.85, label="BTC-BEATER v1")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("LONG-TIDE vs BTC vs naked leg vs v1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if isinstance(rel, pd.Series) and len(rel):
        ax2.plot(rel.index, rel.values, lw=1.3, label="LONG-TIDE / BTC")
    n_rel = naked.get("rel_equity")
    if isinstance(n_rel, pd.Series) and len(n_rel):
        ax2.plot(n_rel.index, n_rel.values, lw=1.1, alpha=0.9, label="NAKED / BTC")
    v_rel = v1.get("rel_equity")
    if isinstance(v_rel, pd.Series) and len(v_rel):
        ax2.plot(v_rel.index, v_rel.values, lw=1.1, alpha=0.85, label="v1 / BTC")
    ax2.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax2.set_ylabel("relative equity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gate_ribbon(tide: dict, out_path: Path) -> None:
    eq = tide.get("equity")
    gon = tide.get("gate_on")
    if not isinstance(eq, pd.Series) or eq.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.2), sharex=True, constrained_layout=True)
    ax = axes[0]
    if isinstance(gon, pd.Series):
        _shade_gate(ax, gon.reindex(eq.index).ffill().fillna(0.0), eq.min(), eq.max())
    ax.plot(eq.index, eq.values, lw=1.3, color="#4C78A8", label="LONG-TIDE")
    eqb = tide.get("equity_btc")
    if isinstance(eqb, pd.Series) and len(eqb):
        ax.plot(eqb.index, eqb.values, lw=1.1, color="#B279A2", label="BTC B&H")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("LONG-TIDE with Stage-T gate ribbon (shaded = ON)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if isinstance(gon, pd.Series) and len(gon):
        ax2.fill_between(gon.index, 0.0, gon.values.astype(float), step="post", color="#54A24B", alpha=0.7)
        ax2.set_ylim(-0.05, 1.15)
    ax2.set_ylabel("gate ON")
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def gate_stretch_stats(gate: pd.Series) -> dict:
    if not isinstance(gate, pd.Series) or gate.empty:
        return {"n_on_stretches": 0, "mean_on_len": float("nan")}
    on = (gate.fillna(0).astype(float) > 0.5).astype(int).tolist()
    lens = []
    cur = 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            lens.append(cur)
            cur = 0
    if cur:
        lens.append(cur)
    return {
        "n_on_stretches": int(len(lens)),
        "mean_on_len": float(np.mean(lens)) if lens else float("nan"),
        "max_on_len": int(max(lens)) if lens else 0,
        "n_on_days": int(sum(on)),
    }


def update_ledger_longtide(
    path: Path,
    *,
    verdicts: dict,
    tide: dict,
    extra: dict,
) -> str:
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER LONG-TIDE"
    block = [
        "",
        marker,
        "",
        "Full-size long leg + frozen Stage-T gate, BTC parking. Backtest only. "
        "Binance-priced (3.e canonical). 2.c spread cache reused. No shorts, no funding. "
        "SPREAD-LS BOOK-HYBRID remains the official long/short product. COMBO untouched.",
        "",
    ]
    if verdicts.get("supersedes"):
        block += [
            "**OFFICIAL long product = LONG-TIDE.** SUPERSEDES BTC-BEATER v1 (v1 demoted to record-only).",
            "",
        ]
    elif verdicts.get("viable"):
        block += [
            "**LONG-TIDE = parallel long variant.** VIABLE but does not supersede. "
            "BTC-BEATER v1 stays the official long product.",
            "",
        ]
    else:
        block += [
            "**LONG-TIDE is NOT VIABLE.** BTC-BEATER v1 stays the official long product.",
            "",
        ]
    block += [
        f"LONG-TIDE rel-line Sharpe `{_fmt(verdicts.get('rel_sharpe'))}` / USD Sharpe "
        f"`{_fmt(tide.get('book_sharpe'))}` / trail `{_fmt(tide.get('net_sharpe_trail18m'))}` / "
        f"total `{_pct(tide.get('book_total'))}` / MaxDD `{_pct(tide.get('maxdd'))}` / "
        f"alt deployment `{_pct(verdicts.get('avg_alt_deployment'))}` / "
        f"gate ON `{_pct(tide.get('gate_on_frac'))}`. "
        f"status=`{verdicts.get('status')}`. "
        f"Window {tide.get('start')}→{tide.get('end')} n={tide.get('n_days')}.",
        "",
        "Mechanical, no post-hoc adjustment.",
        "",
    ]
    new = "\n".join(block)
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
        text = text.rstrip() + new
    path.write_text(text)
    return text
