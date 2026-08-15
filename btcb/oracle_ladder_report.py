"""ORACLE LADDER report and CAGR-vs-RankIC curve. Analysis only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    ORACLE_LADDER_CRITERION,
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
        v = float(x)
        if abs(v) >= 100.0:
            return f"{v:.3e}"
        return f"{100.0 * v:.{nd}f}%"
    except Exception:
        return str(x)


def _sci_or_pct(x, nd=1):
    return _pct(x, nd)


def _row(name: str, b: dict, *, rng: bool = False) -> str:
    if rng:
        return (
            f"| {name} | {_fmt(b.get('rankic'), 4)} "
            f"[{_fmt(b.get('rankic_lo'), 4)}, {_fmt(b.get('rankic_hi'), 4)}] "
            f"| {_sci_or_pct(b.get('total'))} "
            f"[{_sci_or_pct(b.get('total_lo'))}, {_sci_or_pct(b.get('total_hi'))}] "
            f"| {_pct(b.get('cagr'))} "
            f"[{_pct(b.get('cagr_lo'))}, {_pct(b.get('cagr_hi'))}] "
            f"| {_pct(b.get('maxdd'))} "
            f"[{_pct(b.get('maxdd_lo'))}, {_pct(b.get('maxdd_hi'))}] |"
        )
    return (
        f"| {name} | {_fmt(b.get('rankic'), 4)} | {_sci_or_pct(b.get('total'))} "
        f"| {_pct(b.get('cagr'))} | {_pct(b.get('maxdd'))} | {_fmt(b.get('sharpe'))} "
        f"| {_fmt(b.get('avg_n_names'), 1)} | {b.get('n_formations')} | {b.get('forced_n')} |"
    )


def _cycle_block(name: str, b: dict) -> list[str]:
    lines = []
    for cyc, _a, _b in PHASE2_CYCLES:
        blob = (b.get("cycles") or {}).get(cyc) or {}
        if not blob:
            continue
        lines.append(
            f"| {cyc} | {name} | {blob.get('n')} | {_sci_or_pct(blob.get('total'))} "
            f"| {_pct(blob.get('cagr'))} | {_pct(blob.get('maxdd'))} | {_fmt(blob.get('sharpe'))} |"
        )
    return lines


def write_oracle_ladder(
    path: Path,
    *,
    oracle_gross: dict,
    oracle_net: dict,
    oracle7_gross: dict,
    oracle7_net: dict,
    ladder: dict,
    model: dict,
    naive: dict,
    random: dict,
    verdict: dict,
    extra: dict,
) -> str:
    cyc_lines = [
        "| cycle | book | n | total | CAGR | MaxDD | Sharpe |",
        "|-------|------|---|-------|------|-------|--------|",
    ]
    cyc_lines += _cycle_block("ORACLE NET h=14", oracle_net)
    cyc_lines += _cycle_block("OUR MODEL", model)
    cyc_lines += _cycle_block("NAIVE 90d", naive)

    lad_lines = [
        "| target IC | realized RankIC mean [range] | total mean [range] | CAGR mean [range] | MaxDD mean [range] |",
        "|-----------|------------------------------|--------------------|-------------------|--------------------|",
    ]
    for tgt in sorted(ladder.keys(), reverse=True):
        lad_lines.append(_row(f"{tgt:.2f}", ladder[tgt], rng=True))

    vlab = verdict.get("label")
    if verdict.get("on_curve"):
        conclusion = (
            "the binding constraint is **INFORMATION**, and improvement means new data, not new architecture."
        )
    elif verdict.get("below_curve"):
        conclusion = (
            "TRANSLATION slack exists in the signal→book layer, quantified as the CAGR gap "
            f"`{_pct(verdict.get('model_cagr'))}` vs curve `{_pct(verdict.get('curve_cagr'))}`."
        )
    elif verdict.get("above_curve"):
        conclusion = (
            "the model is **above** the interpolation band; there is no evidence of translation slack. "
            "The binding constraint is still treated as INFORMATION."
        )
    else:
        conclusion = "efficiency band could not be evaluated."

    ic16 = (ladder.get(0.16) or ladder.get("0.16") or {})
    ceiling = oracle_net
    y = verdict.get("capture_of_oracle_cagr")
    ypct = f"{100.0 * y:.1f}%" if y is not None and np.isfinite(float(y or float("nan"))) else "nan"
    word = verdict.get("consistent_word") or "below"
    plain = (
        f"the ceiling is {_sci_or_pct(ceiling.get('total'))} total / {_pct(ceiling.get('cagr'))} CAGR "
        f"(NET h=14; GROSS {_sci_or_pct(oracle_gross.get('total'))} / {_pct(oracle_gross.get('cagr'))}); "
        f"at IC 1.0→0.16 the curve passes through oracle CAGR {_pct(oracle_net.get('cagr'))} → "
        f"ladder-0.16 CAGR {_pct(ic16.get('cagr'))} (realized RankIC {_fmt(ic16.get('rankic'), 4)}); "
        f"our model captures {ypct} of the oracle CAGR, which is {word} its information content."
    )

    lines = [
        "# BTC-BEATER ORACLE LADDER — perfect-foresight ceiling and IC curve",
        "",
        "**ANALYSIS ONLY.** Nothing adopted. No retraining, no product changes. "
        "CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.",
        "",
        "Construction is **14-day full rebalance** (not the production overlapping-tranche book). "
        "Every ladder/reference point uses the same construction.",
        "",
        "## Pre-registered verdict (verbatim, before results)",
        "",
        f"> {ORACLE_LADDER_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- Window {oracle_net.get('start')} → {oracle_net.get('end')} n={oracle_net.get('n_days')} "
        f"formations={oracle_net.get('n_formations')}",
        f"- GPU used = `{extra.get('gpu_used', False)}`",
        "",
        "## Mechanical verdict",
        "",
        f"- **MODEL EFFICIENCY = {vlab}**",
        f"- model RankIC `{_fmt(verdict.get('model_rankic'), 4)}` CAGR `{_pct(verdict.get('model_cagr'))}`",
        f"- ladder interpolation at that IC `{_pct(verdict.get('curve_cagr'))}` "
        f"(band `{_pct(verdict.get('need_lo'))}` – `{_pct(verdict.get('need_hi'))}`)",
        f"- ratio vs curve `{_fmt(verdict.get('cagr_ratio_vs_curve'))}`  log-gap `{_fmt(verdict.get('log_gap'))}`  "
        f"(need |log-gap| ≤ ln(1.20) = 0.182)",
        f"- capture of oracle CAGR `{ypct}`",
        f"- {conclusion}",
        "",
        "Mechanical, no post-hoc adjustment.",
        "",
        "## Plain language",
        "",
        plain,
        "",
        "## 1 — Oracle (perfect foresight)",
        "",
        "| book | RankIC | total | CAGR | MaxDD | Sharpe | avg #names | formations | forced |",
        "|------|--------|-------|------|-------|--------|------------|------------|--------|",
        _row("ORACLE GROSS h=14", oracle_gross),
        _row("ORACLE NET h=14 (10 bps/side)", oracle_net),
        _row("ORACLE GROSS h=7 (secondary)", oracle7_gross),
        _row("ORACLE NET h=7 (secondary)", oracle7_net),
        "",
        "The h=14 NET row is the ceiling used on the curve.",
        "",
        "## 2 — Degraded-oracle ladder (h=14, costs on, 5 seeds)",
        "",
        *lad_lines,
        "",
        "Realized RankIC is the mean per-date Spearman(score, future excess) after per-date noise calibration.",
        "",
        "## 3 — Real reference points (same construction, same window, costs on)",
        "",
        "| book | RankIC | total | CAGR | MaxDD | Sharpe | avg #names | formations | forced |",
        "|------|--------|-------|------|-------|--------|------------|------------|--------|",
        _row("OUR MODEL (frozen 2.c spread)", model),
        _row("NAIVE 90d excess", naive),
        _row("RANDOM (IC≈0, 5-seed mean)", {**random, "avg_n_names": random.get("avg_n_names"), "n_formations": random.get("n_formations"), "forced_n": random.get("forced_n")}),
        "",
        f"RANDOM RankIC mean `{_fmt(random.get('rankic'), 4)}` "
        f"[{_fmt(random.get('rankic_lo'), 4)}, {_fmt(random.get('rankic_hi'), 4)}]; "
        f"CAGR `{_pct(random.get('cagr'))}`.",
        "",
        "## Per-cycle (NET)",
        "",
        *cyc_lines,
        "",
        "## Notes",
        "",
        "- OUR MODEL here is a naked 14d-full-rebalance long book on the frozen spread. "
        "It is **not** LONG-TIDE and **not** SPREAD-LS (those are overlapping-tranche products).",
        "- Nothing is adopted. This is a map of information vs translation.",
        "",
        f"Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}. GPU={extra.get('gpu_used', False)}.",
        "",
        "COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_curve(
    oracle_net: dict,
    ladder: dict,
    model: dict,
    naive: dict,
    random: dict,
    verdict: dict,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)

    xs, ys, ylo, yhi = [], [], [], []
    for tgt in sorted(ladder.keys()):
        b = ladder[tgt]
        xs.append(float(b.get("rankic") or np.nan))
        ys.append(float(b.get("cagr") or np.nan))
        ylo.append(float(b.get("cagr_lo") or np.nan))
        yhi.append(float(b.get("cagr_hi") or np.nan))
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ylo = np.asarray(ylo, dtype=float)
    yhi = np.asarray(yhi, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    if ok.any():
        yerr = np.vstack([np.maximum(ys[ok] - ylo[ok], 0.0), np.maximum(yhi[ok] - ys[ok], 0.0)])
        ax.errorbar(
            xs[ok],
            ys[ok],
            yerr=yerr,
            fmt="o",
            color="#4C78A8",
            ecolor="#4C78A8",
            capsize=3,
            label="ladder (mean ± range)",
        )
        order = np.argsort(xs[ok])
        ax.plot(xs[ok][order], ys[ok][order], color="#4C78A8", lw=1.1, alpha=0.8)

    def _pt(ic, cagr, marker, color, label, ms=9):
        if ic is None or cagr is None:
            return
        ic, cagr = float(ic), float(cagr)
        if not (np.isfinite(ic) and np.isfinite(cagr)):
            return
        if cagr <= 0:
            ax.scatter([ic], [max(cagr, 1e-4)], marker=marker, color=color, s=ms ** 2, label=label + " (≤0, pinned)", zorder=5)
            return
        ax.scatter([ic], [cagr], marker=marker, color=color, s=ms ** 2, label=label, zorder=5)

    _pt(oracle_net.get("rankic"), oracle_net.get("cagr"), "*", "#B279A2", "ORACLE NET", 12)
    _pt(model.get("rankic"), model.get("cagr"), "D", "#E45756", "OUR MODEL", 8)
    _pt(naive.get("rankic"), naive.get("cagr"), "s", "#F58518", "NAIVE 90d", 8)
    _pt(random.get("rankic"), random.get("cagr"), "x", "#54A24B", "RANDOM", 8)

    cx = verdict.get("model_rankic")
    cy = verdict.get("curve_cagr")
    if cx is not None and cy is not None and np.isfinite(float(cx)) and np.isfinite(float(cy)) and float(cy) > 0:
        ax.scatter([float(cx)], [float(cy)], marker="_", s=120, color="0.3", label="interp at model IC", zorder=4)
        lo, hi = verdict.get("need_lo"), verdict.get("need_hi")
        if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
            ax.vlines(float(cx), max(float(lo), 1e-4), float(hi), color="0.5", lw=4, alpha=0.25, label="±20% log band")

    ax.set_xlabel("realized mean per-date RankIC")
    ax.set_ylabel("CAGR")
    ax.set_yscale("log")
    ax.set_title("ORACLE LADDER — log CAGR vs RankIC")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def update_ledger_oracle(path: Path, *, verdict: dict, oracle_net: dict, model: dict) -> str:
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER ORACLE LADDER"
    block = [
        "",
        marker,
        "",
        "Perfect-foresight ceiling and IC-degraded oracle ladder. Analysis only. Nothing adopted. "
        "14d full-rebalance long construction (not the production tranche books). Binance-priced.",
        "",
        f"**MODEL EFFICIENCY = {verdict.get('label')}.** Binding constraint: "
        f"**{verdict.get('binding_constraint')}**. "
        f"Oracle NET h=14 total `{_sci_or_pct(oracle_net.get('total'))}` / CAGR `{_pct(oracle_net.get('cagr'))}` / "
        f"MaxDD `{_pct(oracle_net.get('maxdd'))}`. "
        f"OUR MODEL RankIC `{_fmt(model.get('rankic'), 4)}` CAGR `{_pct(model.get('cagr'))}` "
        f"({_fmt(100.0 * float(verdict.get('capture_of_oracle_cagr') or float('nan')), 1)}% of oracle CAGR) "
        f"vs curve `{_pct(verdict.get('curve_cagr'))}`.",
        "",
        "Mechanical, no post-hoc adjustment. Frozen products untouched.",
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
