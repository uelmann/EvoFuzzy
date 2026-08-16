"""Phase 8 MODEL-ZOO report, charts, ledger. Analysis only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    PHASE2_CYCLES,
    PHASE2C_PRED_SHA256,
    PHASE4V2_PI_SCOPE,
    PHASE8_CRITERION,
    PHASE8_DATE_SUBSAMPLE,
    PHASE8_FIREWALL,
    PHASE8_NULL_REGISTRATION,
    PHASE8_TABPFN_CAVEAT,
)

SIGNAL_ORDER = (
    ("frozen_spread", "frozen spread"),
    ("cs_attn", "CS-ATTN-DAILY"),
    ("tabpfn", "TabPFN v2"),
    ("ridge", "RIDGE ON RANKS"),
)
SIGNAL_COLORS = {
    "frozen_spread": "#4C78A8",
    "cs_attn": "#E45756",
    "tabpfn": "#F58518",
    "ridge": "#54A24B",
}


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


def _delta(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        v = float(x)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{nd}f}"
    except Exception:
        return str(x)


def _null_table(cells: list, real_key: str) -> list[str]:
    lines = [
        "| fold | n | null mean | centre | 2·SE | bias_ok | p95 | real | exceeds p95 |",
        "|------|---|-----------|--------|------|---------|-----|------|-------------|",
    ]
    for c in cells or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'))} | {_fmt(c.get('center'))} "
            f"| {_fmt(c.get('bias_lim'))} | {c.get('bias_ok')} | {_fmt(c.get('p95'))} "
            f"| {_fmt(c.get(real_key))} | {c.get('exceeds_p95')} |"
        )
    return lines


def _null_summary(title: str, null: dict, metric: str, real_key: str) -> list[str]:
    blob = (null or {}).get(metric) or {}
    lines = [
        f"**{title}** verdict=`{blob.get('verdict')}` bias_pass={blob.get('bias_pass')} "
        f"skill_pass={blob.get('skill_pass')} exceed={blob.get('n_exceed')}/{blob.get('n_folds')} "
        f"violations={blob.get('n_violate')} Stouffer z=`{_fmt(blob.get('stouffer_z'), 3)}`.",
        "",
        *_null_table((null or {}).get(f"{metric}_cells"), real_key),
        "",
    ]
    return lines


def _grid_row(name: str, m: dict) -> str:
    m = m or {}
    return (
        f"| {name} | {_fmt(m.get('tail_ic_top'))} | {_fmt(m.get('tail_ic_top_nw_t'), 2)} "
        f"| {_fmt(m.get('tail_ic_bot'))} | {_fmt(m.get('overlap'))} | {_fmt(m.get('monster'))} "
        f"| {_fmt(m.get('rankic'))} | {m.get('n_dates', '')} |"
    )


def _trail_row(name: str, m: dict) -> str:
    m = m or {}
    return (
        f"| {name} | {_fmt(m.get('tail_ic_top_trail'))} | {_fmt(m.get('tail_ic_top_trail_nw_t'), 2)} "
        f"| {_fmt(m.get('tail_ic_bot_trail'))} | {_fmt(m.get('overlap_trail'))} "
        f"| {_fmt(m.get('monster_trail'))} | {_fmt(m.get('rankic_trail'))} |"
    )


def _cycle_rows(grid: dict, field: str) -> list[str]:
    lines = []
    header = "| cycle | " + " | ".join(lab for _, lab in SIGNAL_ORDER) + " |"
    sep = "|-------|" + "|".join(["------"] * len(SIGNAL_ORDER)) + "|"
    lines.extend([header, sep])
    for cyc, *_ in PHASE2_CYCLES:
        cells = []
        for key, _lab in SIGNAL_ORDER:
            blob = ((grid.get(key) or {}).get(field) or {}).get(cyc) or {}
            cells.append(_fmt(blob.get("mean")))
        lines.append(f"| {cyc} | " + " | ".join(cells) + " |")
    return lines


def _book_row(name: str, b: dict) -> str:
    if not b:
        return f"| {name} | nan | nan | nan | nan |  |"
    return (
        f"| {name} | {_pct(b.get('total'))} "
        f"| {_pct(b.get('cagr'))} | {_pct(b.get('maxdd'))} | {_fmt(b.get('sharpe'), 3)} "
        f"| {b.get('n_formations', b.get('n_days', ''))} |"
    )


def _book_equity(book: dict):
    if not isinstance(book, dict):
        return None
    eq = book.get("equity")
    if isinstance(eq, pd.Series) and len(eq):
        s = eq.astype(float)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).tz_convert("UTC").normalize()
        return s.sort_index()
    rets = book.get("daily_ret")
    if rets is None:
        return None
    r = pd.Series(rets, dtype=float).fillna(0.0)
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index, utc=True)).tz_convert("UTC").normalize()
    r = r.sort_index()
    if r.empty:
        return None
    return (1.0 + r).cumprod()


def write_phase8(
    path: Path,
    *,
    grid: dict,
    grid_full: dict | None,
    books: dict,
    null: dict | None,
    corr: dict,
    verdict: dict,
    attn: dict | None,
    configs: dict,
    extra: dict,
) -> str:
    v_arms = (verdict or {}).get("arms") or {}
    lines = [
        "# BTC-BEATER Phase 8 — MODEL-ZOO",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. "
        "Frozen products untouched. Pricing = Binance-hybrid (3.e canonical). Master only. "
        "GPU only for Arm B if TabPFN required it.",
        "",
        "Independent of Phases 7.c / 7.d. One config per arm. Zero architecture search.",
        "",
        "## Firewall (verbatim, before results)",
        "",
        f"> {PHASE8_FIREWALL}",
        "",
        "## PI data-perimeter (verbatim)",
        "",
        f"> {PHASE4V2_PI_SCOPE}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE8_CRITERION}",
        "",
        "## TabPFN caveat (verbatim, before results)",
        "",
        f"> {PHASE8_TABPFN_CAVEAT}",
        "",
        "## Date subsample (verbatim, before results)",
        "",
        f"> {PHASE8_DATE_SUBSAMPLE}",
        "",
        "## Vol-matched null (verbatim, before results)",
        "",
        f"> {PHASE8_NULL_REGISTRATION}",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- Window {extra.get('start')} → {extra.get('end')} n_dates={extra.get('n_eval_dates')}",
        f"- Judgment date set = `{extra.get('judgment_set')}` n={extra.get('n_judgment_dates')}",
        f"- GPU used = `{extra.get('gpu_used', False)}` type=`{extra.get('gpu_type')}` "
        f"estimated USD=`{_fmt(extra.get('gpu_usd_est'), 2)}` cap=$20",
        f"- TabPFN subsample flag = `{extra.get('tabpfn_subsample_flag')}`",
        f"- Best arm by RankIC = `{verdict.get('best_arm')}`",
        f"- Elapsed sec = `{_fmt(extra.get('elapsed_sec'), 1)}`",
        "",
        "## Per-arm config dumps (frozen, one each)",
        "",
        "```json",
        json.dumps(configs or {}, indent=2, default=str),
        "```",
        "",
        "## Wall-times and budget flags",
        "",
        f"- Arm A CS-ATTN elapsed_sec = `{_fmt(extra.get('cs_attn_sec'), 1)}` (CPU)",
        f"- Arm B TabPFN elapsed_sec = `{_fmt(extra.get('tabpfn_sec'), 1)}` "
        f"pred_sec_total=`{_fmt(extra.get('tabpfn_pred_sec'), 1)}` "
        f"pred_sec_per_date=`{_fmt(extra.get('tabpfn_pred_per_date'), 3)}` "
        f"status=`{extra.get('tabpfn_status')}`",
        f"- Arm C ridge elapsed_sec = `{_fmt(extra.get('ridge_sec'), 1)}` (CPU)",
        f"- GPU estimate USD = `{_fmt(extra.get('gpu_usd_est'), 2)}` / $20 cap; flag=`{extra.get('budget_flag')}`",
        "",
        "## Mechanical verdicts",
        "",
        f"- Arm A CS-ATTN-DAILY: **{(v_arms.get('cs_attn') or {}).get('verdict')}** "
        f"(Δtail-IC `{_delta((v_arms.get('cs_attn') or {}).get('delta_tail_ic'))}`, "
        f"Δoverlap `{_delta((v_arms.get('cs_attn') or {}).get('delta_overlap'))}`, "
        f"ΔRankIC `{_delta((v_arms.get('cs_attn') or {}).get('delta_rankic'))}`)",
        f"- Arm B TabPFN v2: **{(v_arms.get('tabpfn') or {}).get('verdict')}** "
        f"(Δtail-IC `{_delta((v_arms.get('tabpfn') or {}).get('delta_tail_ic'))}`, "
        f"Δoverlap `{_delta((v_arms.get('tabpfn') or {}).get('delta_overlap'))}`, "
        f"ΔRankIC `{_delta((v_arms.get('tabpfn') or {}).get('delta_rankic'))}`)",
        f"- Arm C RIDGE ON RANKS: **{(v_arms.get('ridge') or {}).get('verdict')}** "
        f"(Δtail-IC `{_delta((v_arms.get('ridge') or {}).get('delta_tail_ic'))}`, "
        f"Δoverlap `{_delta((v_arms.get('ridge') or {}).get('delta_overlap'))}`, "
        f"ΔRankIC `{_delta((v_arms.get('ridge') or {}).get('delta_rankic'))}`)",
        f"- {verdict.get('linear_ceiling_text') or 'LINEAR-CEILING: n/a'}",
        f"- ORTHOGONAL SIGNAL: "
        + (
            ", ".join(
                f"{o.get('arm')} corr={_fmt(o.get('corr'))} RankIC={_fmt(o.get('rankic'))}"
                for o in (verdict.get("orthogonal") or [])
            )
            or "none"
        ),
        "",
        extra.get("plain") or "",
        "",
        "## 1 — Judgment grid (judgment date set)",
        "",
        "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | n |",
        "|--------|-------------|------|-------------|---------|---------|--------|---|",
        *[_grid_row(lab, grid.get(key) or {}) for key, lab in SIGNAL_ORDER],
        "",
        "Trailing 18m:",
        "",
        "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC |",
        "|--------|-------------|------|-------------|---------|---------|--------|",
        *[_trail_row(lab, grid.get(key) or {}) for key, lab in SIGNAL_ORDER],
        "",
        "Per-cycle RankIC:",
        "",
        *_cycle_rows(grid, "rankic_cycles"),
        "",
        "Per-cycle tail-IC(top-half):",
        "",
        *_cycle_rows(grid, "tail_ic_top_cycles"),
        "",
    ]
    if grid_full and extra.get("judgment_set") == "1-in-3":
        lines += [
            "### Informational full-OOS grid (Arms A/C + frozen; not the verdict table)",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | n |",
            "|--------|-------------|------|-------------|---------|---------|--------|---|",
            *[_grid_row(lab, grid_full.get(key) or {}) for key, lab in SIGNAL_ORDER],
            "",
        ]
    disp = extra.get("seed_dispersion") or {}
    lines += [
        "### Arm A seed dispersion (unbagged spread RankIC)",
        "",
        f"- seeds {{42,43,44}}: `{json.dumps(disp, default=str)}`",
        "",
        "## 2 — Vol-matched null (best arm only)",
        "",
        f"Best arm = `{verdict.get('best_arm')}`. Completions: CS-ATTN null = seed 42 cold-start; "
        "ridge/TabPFN = full frozen procedure. Folds {5,15,21,24} × 15.",
        "",
    ]
    if null:
        lines += _null_summary("tail-IC(top-half) [LIVE gate]", null, "tail_ic_top", "real_tail_ic_top")
        lines += _null_summary("overlap", null, "overlap", "real_overlap")
        lines += _null_summary("monster top-3", null, "monster", "real_monster")
        lines += _null_summary("whole-list RankIC [LEAD gate]", null, "rankic", "real_rankic")
    else:
        lines += ["Null not run.", ""]
    mat = (corr or {}).get("matrix") or {}
    names = (corr or {}).get("names") or [k for k, _ in SIGNAL_ORDER]
    lines += [
        "## 3 — Signal correlation (mean per-date Spearman)",
        "",
        f"n_dates=`{(corr or {}).get('n_dates')}`",
        "",
        "|  | " + " | ".join(names) + " |",
        "|--|" + "|".join(["---"] * len(names)) + "|",
    ]
    for a in names:
        row = [a] + [_fmt((mat.get(a) or {}).get(b)) for b in names]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 4 — Arm A attention diagnostics",
        "",
        f"- mean attention entropy (normalized) = `{_fmt((attn or {}).get('mean_entropy'))}`",
        f"- mean self-weight of highest-scored coin = `{_fmt((attn or {}).get('mean_self_weight'))}`",
        f"- collapse_to_self (≥0.50 self-weight) = `{(attn or {}).get('collapse_to_self')}`",
        f"- n diagnostic dates = `{(attn or {}).get('n_dates')}`",
        "",
        "Top-5 attended peers (highest-scored coin, 10 linspace OOS dates, fold model seed 42):",
        "",
        "```json",
        json.dumps((attn or {}).get("top5_peers") or [], indent=2, default=str),
        "```",
        "",
        "## 5 — Crude 14d books (information only; nothing adopted)",
        "",
        "| signal | total | CAGR | MaxDD | Sharpe | n_form |",
        "|--------|-------|------|-------|--------|--------|",
        *[_book_row(lab, (books or {}).get(key) or {}) for key, lab in SIGNAL_ORDER],
        "",
        "- RankIC by arm with null band: `charts/btcb_phase8_rankic.png`",
        "- Signal correlation heatmap: `charts/btcb_phase8_corr.png`",
        "- Crude equity curves: `charts/btcb_phase8_equity.png`",
        "",
        "Mechanical, no post-hoc adjustment. Frozen products untouched. Nothing adopted.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text)
    return text


def plot_rankic(grid: dict, null: dict | None, best_arm: str | None, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [lab for _, lab in SIGNAL_ORDER]
    keys = [k for k, _ in SIGNAL_ORDER]
    ys = [float((grid.get(k) or {}).get("rankic") or np.nan) for k in keys]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    colors = [SIGNAL_COLORS[k] for k in keys]
    xs = np.arange(len(labels))
    ax.bar(xs, ys, color=colors, width=0.72)
    cells = (null or {}).get("rankic_cells") or []
    p95s = [c.get("p95") for c in cells if c.get("p95") is not None]
    if p95s and best_arm in keys:
        p95 = float(np.mean([float(x) for x in p95s if np.isfinite(float(x))]))
        i = keys.index(best_arm)
        ax.axhspan(p95, p95, xmin=0, xmax=1, color="0.2", lw=0)  # no-op keep
        ax.plot([xs[i] - 0.4, xs[i] + 0.4], [p95, p95], color="k", lw=2.2, zorder=5, label="best-arm null mean p95")
        lo = [c.get("mean") for c in cells if c.get("mean") is not None]
        if lo:
            mu = float(np.mean([float(x) for x in lo if np.isfinite(float(x))]))
            ax.plot([xs[i] - 0.4, xs[i] + 0.4], [mu, mu], color="0.35", lw=1.6, ls="--", zorder=5, label="best-arm null mean")
    ax.set_xticks(xs, labels, rotation=12, ha="right")
    ax.set_ylabel("whole-list RankIC")
    ax.set_title("Phase 8 MODEL-ZOO — RankIC by arm (black = vol-matched null p95 on best arm)")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_corr(corr: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = (corr or {}).get("names") or []
    mat = (corr or {}).get("matrix") or {}
    n = len(names)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = (mat.get(a) or {}).get(b)
            if v is not None:
                M[i, j] = float(v)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), names, rotation=18, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Phase 8 — mean per-date Spearman of signals")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_equity(books: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax, ax2 = axes
    for key, lab in SIGNAL_ORDER:
        eq = _book_equity((books or {}).get(key) or {})
        if eq is None or eq.empty:
            continue
        c = SIGNAL_COLORS.get(key, "0.3")
        ax.plot(eq.index, np.log(eq.clip(lower=1e-12)), color=c, lw=1.4, label=lab)
        dd = eq / eq.cummax() - 1.0
        ax2.plot(dd.index, dd, color=c, lw=1.1)
    ax.set_ylabel("log equity")
    ax.set_title("Phase 8 — crude 14d books (information only)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def update_ledger_phase8(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER PHASE 8 MODEL-ZOO"
    arms = (verdict or {}).get("arms") or {}
    orth = verdict.get("orthogonal") or []
    orth_s = (
        ", ".join(f"{o.get('arm')} corr={o.get('corr')} RankIC={o.get('rankic')}" for o in orth) or "none"
    )
    block = [
        "",
        marker,
        "",
        "Three non-GBM classes on the daily CS (CS-ATTN / TabPFN v2 / ridge-on-ranks). "
        "Backtest/analysis only. Nothing adopted. Binance-hybrid priced. Independent of 7.c/7.d.",
        "",
        f"**A CS-ATTN `{(arms.get('cs_attn') or {}).get('verdict')}`.** "
        f"**B TabPFN `{(arms.get('tabpfn') or {}).get('verdict')}`.** "
        f"**C RIDGE `{(arms.get('ridge') or {}).get('verdict')}`.**",
        "",
        f"{verdict.get('linear_ceiling_text') or 'LINEAR-CEILING n/a'}",
        "",
        f"ORTHOGONAL SIGNAL: {orth_s}. Best arm by RankIC=`{verdict.get('best_arm')}`. "
        f"Judgment set=`{extra.get('judgment_set')}`. GPU est USD=`{extra.get('gpu_usd_est')}`.",
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
        text = text.rstrip() + "\n" + new
    path.write_text(text)
    return text
