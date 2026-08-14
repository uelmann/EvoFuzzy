"""Symmetry audit report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from symmetry.constants import ALT_SEASON_END, ALT_SEASON_START, CLASSIFICATION_CRITERION, N_BUCKETS


def _fmt(x, nd=4):
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


def plot_buckets(curves: dict, out_path: Path) -> None:
    """curves[(h, uni, window)] = bucket_curve_for_window output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unis = ["top20", "top40", "top120"]
    hs = [7, 10]
    windows = ["full", "trail18m"]
    fig, axes = plt.subplots(len(hs) * len(unis), len(windows), figsize=(11, 16), constrained_layout=True)
    for i, h in enumerate(hs):
        for j, uni in enumerate(unis):
            row = i * len(unis) + j
            for k, w in enumerate(windows):
                ax = axes[row][k]
                blob = curves.get((h, uni, w)) or {}
                bucks = blob.get("buckets") or []
                if not bucks:
                    ax.set_title(f"h={h} {uni} {w} (empty)")
                    continue
                xs = [b["bucket"] for b in bucks]
                ys = [b["mean_y"] for b in bucks]
                ax.bar(xs, ys, color="#4C78A8", width=0.8)
                cs = blob.get("cs_mean")
                if cs is not None and np.isfinite(cs):
                    ax.axhline(cs, color="#E45756", ls="--", lw=1.2, label="CS mean")
                ax.axhline(0.0, color="black", lw=0.6)
                ax.set_xticks(xs)
                ax.set_title(f"h={h} {uni} {w}  ρ={_fmt(blob.get('spearman'), 2)}")
                ax.grid(True, axis="y", alpha=0.3)
                if k == 0:
                    ax.set_ylabel("mean residual")
                if row == 0 and k == 1:
                    ax.legend(fontsize=7, loc="best")
    fig.suptitle("A0 score buckets — mean h-day residual (red = cross-sectional mean)", fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_tide(tides: dict, out_path: Path) -> None:
    """tides[(h, uni)] = tide_tables output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unis = ["top20", "top40", "top120"]
    fig, axes = plt.subplots(len(unis), 2, figsize=(12, 10), constrained_layout=True)
    t0 = pd.Timestamp(ALT_SEASON_START, tz="UTC")
    t1 = pd.Timestamp(ALT_SEASON_END, tz="UTC")
    for j, uni in enumerate(unis):
        axp = axes[j][0]
        axt = axes[j][1]
        for h, color in ((7, "#4C78A8"), (10, "#F58518")):
            blob = tides.get((h, uni)) or {}
            roll = blob.get("roll90")
            cs = blob.get("cs_mean")
            top = blob.get("top_mean")
            if isinstance(roll, pd.Series) and len(roll):
                axp.plot(roll.index, roll.values, color=color, lw=1.3, label=f"h={h}")
            if isinstance(cs, pd.Series) and len(cs):
                axt.plot(cs.index, cs.values, color=color, lw=0.9, alpha=0.85, label=f"h={h} tide")
            if isinstance(top, pd.Series) and len(top):
                axt.plot(top.index, top.values, color=color, lw=0.8, ls="--", alpha=0.7, label=f"h={h} top")
        for ax in (axp, axt):
            ax.axvspan(t0, t1, color="#54A24B", alpha=0.12, zorder=0)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axp.axhline(0.5, color="black", lw=0.6, ls=":")
        axp.set_ylim(0.0, 1.0)
        axp.set_ylabel("% days top>0 (90d)")
        axp.set_title(f"{uni} — rolling 90d share of days top-bucket residual > 0")
        axp.legend(fontsize=7)
        axt.axhline(0.0, color="black", lw=0.6)
        axt.set_ylabel("residual")
        axt.set_title(f"{uni} — CS mean (solid) vs top-bucket mean (dashed)")
        axt.legend(fontsize=6, ncol=2)
    fig.suptitle("Alt tide and top-bucket positivity (green band = 2023–24)", fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _spread_row(h, uni, blob) -> str:
    t = blob["top"]["full"]
    b = blob["bottom"]["full"]
    tt = blob["top"]["trail18m"]
    bt = blob["bottom"]["trail18m"]
    return (
        f"| h={h} {uni} | {_fmt(t['mean'])} | {_fmt(t['nw_t'], 2)} | {_fmt(b['mean'])} | {_fmt(b['nw_t'], 2)} "
        f"| {_fmt(blob['ratio']['full'], 3)} | {blob['pass']} | {_fmt(tt['mean'])} | {_fmt(tt['nw_t'], 2)} "
        f"| {_fmt(bt['mean'])} | {_fmt(bt['nw_t'], 2)} | {_fmt(blob['ratio']['trail18m'], 3)} |"
    )


def _alpha_row(book, bench, blob) -> str:
    f = blob.get("full") or {}
    t = blob.get("trail18m") or {}
    return (
        f"| {book} vs {bench} | {_fmt(f.get('alpha_ann'), 3)} | {_fmt(f.get('beta'), 3)} "
        f"| {_fmt(f.get('nw_t_alpha'), 2)} | {f.get('n', '')} | {_fmt(t.get('alpha_ann'), 3)} "
        f"| {_fmt(t.get('beta'), 3)} | {_fmt(t.get('nw_t_alpha'), 2)} | {t.get('n', '')} |"
    )


def write_report(
    path: Path,
    *,
    frozen_hash: str,
    pred_hashes: dict,
    classification: dict,
    cells: dict,
    curves: dict,
    tides: dict,
    lo_alpha: dict,
    picks: dict,
    extra: dict,
) -> str:
    years = list(range(2022, 2027))
    lines = [
        "# Symmetry audit — does the model predict outperformance?",
        "",
        "**ANALYSIS ONLY.** Frozen A0 predictions and residual labels reused byte-identical. "
        "No retraining, no portfolio changes, no live components. CPU only. Zero GPU.",
        "",
        f"**Frozen A0 SHA256:** `{frozen_hash}`",
        f"**Prediction SHA256:** h7=`{pred_hashes.get('h7')}` h10=`{pred_hashes.get('h10')}`",
        "**Reference book:** COMBO v2.0-combo-final. **UNCHANGED.** Ledger and system card untouched.",
        "",
        "## Pre-registered classification",
        "",
        f"> {CLASSIFICATION_CRITERION}",
        "",
        "Mechanical reading: a (h, universe) cell passes if full-period mean(TOP spread) > 0 "
        "with NW-t ≥ 2.0 (lag = h) and symmetry ratio ≥ 0.4. The engine is SYMMETRIC if "
        "n_pass(h=7) ≥ 2 or n_pass(h=10) ≥ 2; otherwise LONG-SIDE GAP.",
        "",
        "## Mechanical label",
        "",
        f"- **{classification.get('label')}** — "
        f"n_pass(h=7)={classification.get('n_pass_h7')} / 3, "
        f"n_pass(h=10)={classification.get('n_pass_h10')} / 3 "
        f"(need ≥ {classification.get('need_universes')} universes at either horizon).",
        "",
    ]
    for r in classification.get("rows") or []:
        lines.append(
            f"- h={r['horizon']} {r['universe']}: {'PASS' if r['pass'] else 'FAIL'} "
            f"(TOP={_fmt(r.get('top_mean'))}, t={_fmt(r.get('top_nw_t'), 2)}, "
            f"BOTTOM={_fmt(r.get('bot_mean'))}, t={_fmt(r.get('bot_nw_t'), 2)}, "
            f"ratio={_fmt(r.get('ratio'), 3)})"
        )
    lines += [
        "",
        "## 1 — Bucket curves",
        "",
        "Per date, names are bucketed by frozen A0 score (quintiles on top-20/top-40, "
        "deciles on pit-120). Values are mean realized h-day residual (`y_h{h}`). "
        "Spearman ρ is bucket-rank vs window-average bucket mean. Hit rate = fraction of names with residual > 0.",
        "",
    ]
    for h in (7, 10):
        for uni in ("top20", "top40", "top120"):
            nb = N_BUCKETS[uni]
            lines.append(f"### h={h} {uni} ({nb} buckets)")
            lines.append("")
            lines.append("| window | " + " | ".join(f"B{i}" for i in range(1, nb + 1)) + " | CS mean | Spearman |")
            lines.append("|--------|" + "|".join(["------"] * nb) + "|---------|----------|")
            for w in ["full", "trail18m"] + [f"y{y}" for y in years]:
                blob = curves.get((h, uni, w)) or {}
                bucks = {b["bucket"]: b["mean_y"] for b in (blob.get("buckets") or [])}
                cells_s = " | ".join(_fmt(bucks.get(i)) for i in range(1, nb + 1))
                lines.append(
                    f"| {w} | {cells_s} | {_fmt(blob.get('cs_mean'))} | {_fmt(blob.get('spearman'), 3)} |"
                )
            lines.append("")
            lines.append("| window | " + " | ".join(f"hit{i}" for i in range(1, nb + 1)) + " |")
            lines.append("|--------|" + "|".join(["------"] * nb) + "|")
            for w in ["full", "trail18m"]:
                blob = curves.get((h, uni, w)) or {}
                bucks = {b["bucket"]: b["hit_rate"] for b in (blob.get("buckets") or [])}
                cells_s = " | ".join(_fmt(bucks.get(i), 3) for i in range(1, nb + 1))
                lines.append(f"| {w} | {cells_s} |")
            lines.append("")
    lines += [
        "## 2 — Tail spreads",
        "",
        "TOP = mean(top bucket) − mean(middle). BOTTOM = mean(middle) − mean(bottom). "
        "Middle = Q3 (quintiles) or (D5+D6)/2 (deciles). NW t uses Bartlett HAC with lag = h.",
        "",
        "| cell | TOP full | t | BOTTOM full | t | ratio | pass | TOP 18m | t | BOTTOM 18m | t | ratio 18m |",
        "|------|----------|---|-------------|---|-------|------|---------|---|------------|---|-----------|",
    ]
    for h in (7, 10):
        for uni in ("top20", "top40", "top120"):
            blob = cells.get((h, uni))
            if blob:
                lines.append(_spread_row(h, uni, blob))
    lines += ["", "### Per calendar year (TOP / BOTTOM mean and t)", ""]
    for h in (7, 10):
        for uni in ("top20", "top40", "top120"):
            blob = cells.get((h, uni))
            if not blob:
                continue
            lines.append(f"**h={h} {uni}**")
            lines.append("")
            lines.append("| year | TOP | t | BOTTOM | t | ratio |")
            lines.append("|------|-----|---|--------|---|-------|")
            for y in years:
                key = f"y{y}"
                t = blob["top"].get(key) or {}
                b = blob["bottom"].get(key) or {}
                lines.append(
                    f"| {y} | {_fmt(t.get('mean'))} | {_fmt(t.get('nw_t'), 2)} "
                    f"| {_fmt(b.get('mean'))} | {_fmt(b.get('nw_t'), 2)} "
                    f"| {_fmt(blob['ratio'].get(key), 3)} |"
                )
            lines.append("")
    lines += [
        "## 3 — Raw-material tide",
        "",
        "| cell | % days top>0 full | % days top>0 18m |",
        "|------|-------------------|------------------|",
    ]
    for h in (7, 10):
        for uni in ("top20", "top40", "top120"):
            t = tides.get((h, uni)) or {}
            lines.append(
                f"| h={h} {uni} | {_pct(t.get('pct_top_pos_full'))} | {_pct(t.get('pct_top_pos_trail18m'))} |"
            )
    lines += ["", "### Per year: CS mean, top-bucket mean, bottom-bucket mean", ""]
    for h in (7, 10):
        for uni in ("top20", "top40", "top120"):
            t = tides.get((h, uni)) or {}
            by = t.get("by_year") or {}
            if not by:
                continue
            lines.append(f"**h={h} {uni}**")
            lines.append("")
            lines.append("| year | CS mean | top mean | bottom mean | % top>0 |")
            lines.append("|------|---------|----------|-------------|---------|")
            for y in years:
                r = by.get(y) or {}
                lines.append(
                    f"| {y} | {_fmt(r.get('cs_mean'))} | {_fmt(r.get('top_mean'))} "
                    f"| {_fmt(r.get('bot_mean'))} | {_pct(r.get('pct_top_pos'))} |"
                )
            lines.append("")
    lines += [
        "## 4 — Fair-benchmark re-test of the long-only books",
        "",
        "OLS of COMBO-LO daily net on costless EW PIT baskets (the fair relative-long benchmark). "
        "BTC B&H is kept from the previous report as a reference only. HAC lag = 10.",
        "",
        "| book vs bench | α_ann full | β full | NW-t α | n | α_ann 18m | β 18m | NW-t 18m | n 18m |",
        "|---------------|------------|--------|--------|---|-----------|-------|----------|-------|",
    ]
    for book, benches in lo_alpha.items():
        for bench, blob in benches.items():
            lines.append(_alpha_row(book, bench, blob))
    lines += [
        "",
        "### Long-pick quality (reference book longs vs same-date CS mean residual)",
        "",
        "Direct measure: does the frozen book hold the better alts that day?",
        "",
        "| sleeve | mean residual of longs | CS mean | excess | % days excess>0 | n days |",
        "|--------|------------------------|---------|--------|-----------------|--------|",
    ]
    for name, blob in picks.items():
        lines.append(
            f"| {name} | {_fmt(blob.get('mean_long'))} | {_fmt(blob.get('mean_cs'))} "
            f"| {_fmt(blob.get('mean_excess'))} | {_pct(blob.get('pct_excess_pos'))} "
            f"| {blob.get('n_days', '')} |"
        )
    lines += [
        "",
        "## Plain-language conclusion",
        "",
        extra.get("conclusion", ""),
        "",
        "## Reference book is unchanged",
        "",
        "This audit attaches a diagnostic label only. It does not change COMBO, the sleeves, "
        "the numbers ledger, or the system card.",
        "",
        f"Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. "
        f"Scheduled jobs created: false.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def print_stdout(classification: dict, cells: dict, lo_alpha: dict, picks: dict, extra: dict) -> None:
    print(f"SYMMETRY LABEL: {classification.get('label')}", flush=True)
    for h in (7, 10):
        for uni in ("top20", "top40", "top120"):
            blob = cells.get((h, uni)) or {}
            t = (blob.get("top") or {}).get("full") or {}
            b = (blob.get("bottom") or {}).get("full") or {}
            print(
                f"TOP/BOTTOM h={h} {uni}: TOP={_fmt(t.get('mean'))} (t={_fmt(t.get('nw_t'), 2)}) "
                f"BOTTOM={_fmt(b.get('mean'))} (t={_fmt(b.get('nw_t'), 2)}) "
                f"ratio={_fmt((blob.get('ratio') or {}).get('full'), 3)} "
                f"{'PASS' if blob.get('pass') else 'FAIL'}",
                flush=True,
            )
    bits = []
    for book, benches in lo_alpha.items():
        for bench, blob in benches.items():
            f = blob.get("full") or {}
            bits.append(f"{book} vs {bench}: α={_fmt(f.get('alpha_ann'), 3)} t={_fmt(f.get('nw_t_alpha'), 2)}")
    print("EW-basket alpha: " + "; ".join(bits), flush=True)
    pbits = []
    for name, blob in picks.items():
        pbits.append(
            f"{name} excess={_fmt(blob.get('mean_excess'))} "
            f"(long={_fmt(blob.get('mean_long'))} vs CS={_fmt(blob.get('mean_cs'))})"
        )
    print("Long-pick quality: " + "; ".join(pbits), flush=True)
    print("Reference book UNCHANGED (v2.0-combo-final).", flush=True)
    if extra.get("conclusion"):
        print(extra["conclusion"], flush=True)
