"""ORACLE LADDER 2 report: overlap / tail-IC / variants. Analysis only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    ORACLE_LADDER2_CRITERION,
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
        pct = 100.0 * v
        if abs(pct) >= 10000.0:
            return f"{pct:.3e}%"
        return f"{pct:.{nd}f}%"
    except Exception:
        return str(x)


def _sci_or_pct(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        v = float(x)
        if abs(v) >= 10.0:
            return f"{v:.3e}"
        return f"{100.0 * v:.{nd}f}%"
    except Exception:
        return str(x)


def _pp(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{100.0 * float(x):.{nd}f}pp"
    except Exception:
        return str(x)


def _rng(b, key, nd=4):
    if b.get(f"{key}_lo") is None:
        return _fmt(b.get(key), nd)
    return f"{_fmt(b.get(key), nd)} [{_fmt(b.get(key + '_lo'), nd)}, {_fmt(b.get(key + '_hi'), nd)}]"


def _book_row(name, b):
    return (
        f"| {name} | {_sci_or_pct(b.get('total'))} | {_pct(b.get('cagr'))} | {_pct(b.get('maxdd'))} "
        f"| {_fmt(b.get('sharpe'))} | {_fmt(b.get('ann_turnover'), 2)} "
        f"| {_fmt(b.get('avg_n_names'), 1)} | {b.get('n_formations', b.get('n_days', ''))} |"
    )


def _cycle_rows(name, b, cagr_key="cagr", tot_key="total", mdd_key="maxdd"):
    lines = []
    for cyc, _a, _bb in PHASE2_CYCLES:
        blob = (b.get("cycles") or {}).get(cyc) or {}
        if not blob:
            continue
        tot = blob.get(tot_key, blob.get("book_total", blob.get("total")))
        cagr = blob.get(cagr_key, blob.get("book_cagr", blob.get("cagr")))
        mdd = blob.get(mdd_key, blob.get("maxdd"))
        lines.append(
            f"| {cyc} | {name} | {blob.get('n')} | {_sci_or_pct(tot)} | {_pct(cagr)} | {_pct(mdd)} |"
        )
    return lines


def write_ladder2(
    path: Path,
    *,
    diags: dict,
    books: dict,
    verdict: dict,
    extra: dict,
) -> str:
    model = diags["model"]
    lad = diags["ladder0116"]
    lad16 = diags["ladder016"]
    ora = diags["oracle"]
    base = books["base"]
    v1, v2, v3 = books["v1"], books["v2"], books["v3"]
    lad_crude = books["ladder_crude"]
    lad_prod = books["ladder_prod"]

    ov_lines = [
        "| signal | RankIC | top-decile overlap | tail-IC top half | tail-IC bottom half | bottom−top | monster top-3 |",
        "|--------|--------|--------------------|------------------|---------------------|------------|---------------|",
        f"| OUR SPREAD | {_fmt(model.get('rankic'), 4)} | {_fmt(model.get('overlap'), 4)} | {_fmt(model.get('tail_ic_top'), 4)} | {_fmt(model.get('tail_ic_bot'), 4)} | {_fmt(model.get('bottom_minus_top'), 4)} | {_fmt(model.get('monster'), 4)} |",
        f"| ladder-0.116 | {_rng(lad, 'rankic')} | {_rng(lad, 'overlap')} | {_rng(lad, 'tail_ic_top')} | {_rng(lad, 'tail_ic_bot')} | {_rng(lad, 'bottom_minus_top')} | {_rng(lad, 'monster')} |",
        f"| ladder-0.16 | {_rng(lad16, 'rankic')} | {_rng(lad16, 'overlap')} | {_rng(lad16, 'tail_ic_top')} | {_rng(lad16, 'tail_ic_bot')} | {_rng(lad16, 'bottom_minus_top')} | {_rng(lad16, 'monster')} |",
        f"| ORACLE | {_fmt(ora.get('rankic'), 4)} | {_fmt(ora.get('overlap'), 4)} | {_fmt(ora.get('tail_ic_top'), 4)} | {_fmt(ora.get('tail_ic_bot'), 4)} | {_fmt(ora.get('bottom_minus_top'), 4)} | {_fmt(ora.get('monster'), 4)} |",
    ]
    cyc_ov = [
        "| cycle | OUR overlap | ladder-0.116 overlap | ladder-0.16 overlap | ORACLE overlap |",
        "|-------|-------------|----------------------|---------------------|----------------|",
    ]
    for name, _a, _b in PHASE2_CYCLES:
        def _cm(blob):
            return _fmt(((blob.get("overlap_cycles") or {}).get(name) or {}).get("mean"), 4)
        cyc_ov.append(
            f"| {name} | {_cm(model)} | {_cm(lad)} | {_cm(lad16)} | {_cm(ora)} |"
        )

    pri = verdict.get("priority")
    if pri == "TRANSLATION":
        conclusion = (
            "the best translation variant clears +10pp CAGR at comparable MaxDD — "
            "**translation work is the next priority** (not adopted here)."
        )
    else:
        conclusion = (
            "no V1–V3 variant clears +10pp CAGR at comparable MaxDD — "
            "**the right-tail information hunt (catalysts/attention data) is the next priority**."
        )
    gap = verdict.get("gap_pp")
    plain = (
        f"of the {_pp(gap)} gap, ~{_pp(verdict.get('tail_pp'))} is tail information, "
        f"~{_pp(verdict.get('construction_pp'))} is construction, "
        f"~{_pp(verdict.get('unexplained_pp'))} unexplained."
    )

    lines = [
        "# BTC-BEATER ORACLE LADDER 2 — tail-blindness vs translation slack",
        "",
        "**ANALYSIS ONLY.** Nothing adopted. No retraining, no product changes. "
        "CPU only, zero GPU. Frozen products untouched. Pricing = Binance (3.e canonical). Master only.",
        "",
        "Splits the Ladder-1 BELOW-CURVE gap. White-noise oracles are tail-aware; our model is bottom-heavy.",
        "",
        "## Pre-registered reading (verbatim, before results)",
        "",
        f"> {ORACLE_LADDER2_CRITERION}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- Window {base.get('start')} → {base.get('end')} n={base.get('n_days')} formations={base.get('n_formations')}",
        f"- GPU used = `{extra.get('gpu_used', False)}`",
        "",
        "## Mechanical verdict",
        "",
        f"- **PRIORITY = {pri}**",
        f"- gap (ladder-0.116 crude − model crude) `{_pp(gap)}`",
        f"- TAIL-INFORMATION `{_pp(verdict.get('tail_pp'))}` "
        f"(overlap-implied CAGR `{_pct(verdict.get('cagr_at_model_overlap'))}` vs ladder `{_pct(verdict.get('ladder_cagr'))}`)",
        f"- CONSTRUCTION `{_pp(verdict.get('construction_pp'))}` "
        f"(best variant `{verdict.get('variant_best')}` Δ `{_pp(verdict.get('variant_best_delta_pp'))}` "
        f"+ production Δ `{_pp(verdict.get('prod_delta_pp'))}`)",
        f"- UNEXPLAINED `{_pp(verdict.get('unexplained_pp'))}`",
        f"- qualified variant (comparable MaxDD) `{verdict.get('qualified_best')}` "
        f"Δ `{_pp(verdict.get('qualified_best_delta_pp'))}` (need ≥ +10pp)",
        f"- {conclusion}",
        "",
        "Mechanical, no post-hoc adjustment. Nothing adopted.",
        "",
        "## Plain language",
        "",
        plain,
        "",
        "## 1 — Tail diagnostics",
        "",
        *ov_lines,
        "",
        "Overlap = fraction of the signal's top-decile picks that land in the realized top decile. "
        "Tail-IC = Spearman(score, next-14d excess) within the top / bottom half of the signal ranking. "
        "Monster = fraction of realized top-3 movers held in the signal's top-decile book.",
        "",
        *cyc_ov,
        "",
        "## 2 — Construction slack",
        "",
        "| book | total | CAGR | MaxDD | Sharpe | ann TO | avg #names | n |",
        "|------|-------|------|-------|--------|--------|------------|---|",
        _book_row("OUR MODEL crude 14d (base)", base),
        _book_row("V1 score-weighted", v1),
        _book_row("V2 concentrated top-5", v2),
        _book_row("V3 tail-threshold p95", v3),
        _book_row("ladder-0.116 crude 14d", lad_crude),
        _book_row("ladder-0.116 PRODUCTION (no gate)", lad_prod),
        "",
        f"Production = h=14 tranches, k_enter=10 / k_stay=20, n_hold=10, cap 10%, anti-blowoff, BTC park. "
        f"Not LONG-TIDE (no Stage-T gate). Production Δ on ladder-0.116 = `{_pp(verdict.get('prod_delta_pp'))}`.",
        "",
        "## Per-cycle (NET)",
        "",
        "| cycle | book | n | total | CAGR | MaxDD |",
        "|-------|------|---|-------|------|-------|",
        *_cycle_rows("BASE", base),
        *_cycle_rows("V1", v1),
        *_cycle_rows("V2", v2),
        *_cycle_rows("V3", v3),
        *_cycle_rows("LADDER crude", lad_crude),
        *_cycle_rows("LADDER prod", lad_prod, cagr_key="book_cagr", tot_key="book_total"),
        "",
        "## Notes",
        "",
        "- Crude 14d is the Ladder-1 construction (idle cash). V1–V3 change only the weight rule.",
        "- The brief's 33.6% figure is not a gate; the mechanical base is this run's crude book.",
        "- Ladder diagnostics for 0.116 / 0.16 are 5-seed mean [range].",
        "- Nothing is adopted. Any adoption needs a fresh pre-registered phase.",
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


def plot_overlap(diags: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["OUR SPREAD", "ladder-0.116", "ladder-0.16", "ORACLE"]
    keys = ["model", "ladder0116", "ladder016", "oracle"]
    xs = np.arange(len(labels))
    ys, ylo, yhi = [], [], []
    for k in keys:
        b = diags[k]
        ys.append(float(b.get("overlap") or np.nan))
        ylo.append(float(b.get("overlap_lo") if b.get("overlap_lo") is not None else b.get("overlap") or np.nan))
        yhi.append(float(b.get("overlap_hi") if b.get("overlap_hi") is not None else b.get("overlap") or np.nan))
    ys, ylo, yhi = np.asarray(ys), np.asarray(ylo), np.asarray(yhi)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    colors = ["#E45756", "#4C78A8", "#72B7B2", "#B279A2"]
    ax.bar(xs, ys, color=colors, width=0.65)
    yerr = np.vstack([np.maximum(ys - ylo, 0.0), np.maximum(yhi - ys, 0.0)])
    ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="0.2", capsize=4, lw=1)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("mean top-decile overlap")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ORACLE LADDER 2 — top-decile overlap")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_variants(books: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    spec = (
        ("base", "crude 14d base", "#E45756"),
        ("v1", "V1 score-weighted", "#4C78A8"),
        ("v2", "V2 top-5", "#F58518"),
        ("v3", "V3 p95 threshold", "#54A24B"),
    )
    for key, lab, col in spec:
        b = books.get(key) or {}
        eq = b.get("equity")
        if eq is None:
            rets = b.get("daily_ret")
            if rets is None:
                continue
            eq = (1.0 + pd.Series(rets).astype(float).fillna(0.0)).cumprod()
        eq = pd.Series(eq).astype(float)
        eq.index = pd.to_datetime(eq.index, utc=True)
        ax.plot(eq.index, eq.to_numpy(), color=col, lw=1.3, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("ORACLE LADDER 2 — V1–V3 vs crude 14d base")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def update_ledger_ladder2(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER ORACLE LADDER 2"
    block = [
        "",
        marker,
        "",
        "BELOW-CURVE gap split: tail-blindness vs translation. Analysis only. Nothing adopted. Binance-priced.",
        "",
        f"**PRIORITY = {verdict.get('priority')}.** "
        f"Gap `{_pp(verdict.get('gap_pp'))}` = tail `{_pp(verdict.get('tail_pp'))}` "
        f"+ construction `{_pp(verdict.get('construction_pp'))}` "
        f"+ unexplained `{_pp(verdict.get('unexplained_pp'))}`. "
        f"Best variant `{verdict.get('variant_best')}` Δ `{_pp(verdict.get('variant_best_delta_pp'))}`.",
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
