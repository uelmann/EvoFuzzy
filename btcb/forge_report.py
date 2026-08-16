"""PROJECT FORGE F0 report, charts, ledger. Analysis only."""

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
    FORGE_CRITERION,
    FORGE_FITNESS,
    FORGE_GENS,
    FORGE_JUDGE_END,
    FORGE_JUDGE_START,
    FORGE_MINE_END,
    FORGE_MINE_START,
    FORGE_NONCONTAMINATION,
    FORGE_NULL_GENS,
    FORGE_POP,
    FORGE_SELECT_END,
    FORGE_SELECT_START,
    PHASE2C_PRED_SHA256,
)


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


def jsonable(x, drop=None):
    drop = drop or {
        "daily_ret",
        "equity",
        "btc_ret",
        "equity_btc",
        "rel_equity",
        "w_btc",
        "n_names",
        "turnover",
        "corr_btc_roll90",
        "headline",
    }
    if isinstance(x, dict):
        return {str(k): jsonable(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [jsonable(v, drop) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


def _cycle_table(book: dict) -> list[str]:
    lines = [
        "| cycle | n | book total | BTC total | rel Sharpe | MaxDD | BTC MaxDD |",
        "|-------|---|------------|-----------|------------|-------|-----------|",
    ]
    cycles = (book or {}).get("cycles") or {}
    for name, row in cycles.items():
        lines.append(
            f"| {name} | {row.get('n')} | {_fmt(row.get('book_total'))} | {_fmt(row.get('btc_total'))} "
            f"| {_fmt(row.get('rel_sharpe'))} | {_fmt(row.get('maxdd'))} | {_fmt(row.get('btc_maxdd'))} |"
        )
    if len(lines) == 2:
        lines.append("| — |  |  |  |  |  |  |")
    return lines


def write_forge_f0(
    path: Path,
    *,
    addendum: str,
    extra: dict,
    history: list | None = None,
    decay: list | None = None,
    champions: list | None = None,
    census: dict | None = None,
    null: dict | None = None,
    judge: dict | None = None,
    bench: dict | None = None,
    verdict: dict | None = None,
    stage: str = "pre_judge",
) -> str:
    extra = extra or {}
    history = history or []
    decay = decay or []
    champions = champions or []
    census = census or {}
    null = null or {}
    judge = judge or {}
    bench = bench or {}
    lines = [
        "# PROJECT FORGE — Phase F0",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Evolutionary strategy miner. Compounding fitness. Nested windows.",
        "Master only. CPU only. Zero GPU. Frozen products untouched. Nothing adopted.",
        "",
        f"**Stage:** `{stage}`.",
        "",
        "## Non-contamination clause (verbatim)",
        "",
        f"> {FORGE_NONCONTAMINATION}",
        "",
        "## Search space and fitness (frozen)",
        "",
        "See `reports/forge_f0_addendum.md` for the full freeze, including search-space completions.",
        "",
        f"> {FORGE_FITNESS}",
        "",
        "## Nested windows",
        "",
        f"- MINE: {FORGE_MINE_START} → {FORGE_MINE_END} (pop {FORGE_POP}, gens {FORGE_GENS})",
        f"- SELECT: {FORGE_SELECT_START} → {FORGE_SELECT_END} (untouched by evolution)",
        f"- JUDGE: {FORGE_JUDGE_START} → {FORGE_JUDGE_END} (touched once, after champions committed)",
        "",
        "## Pre-registered judgment (verbatim, before results)",
        "",
        f"> {FORGE_CRITERION}",
        "",
        "## Death-in-position",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Run hygiene",
        "",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert)",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- gpu_used = `{extra.get('gpu_used')}`",
        f"- elapsed_sec = `{extra.get('elapsed_sec')}`",
        f"- budget_flag = `{extra.get('budget_flag')}`",
        f"- null_budget_flag = `{extra.get('null_budget_flag')}`",
        f"- n_jobs = `{extra.get('n_jobs')}`",
        f"- cube mean DV100 / MCAP50 = `{_fmt(extra.get('mean_dv'), 1)}` / `{_fmt(extra.get('mean_mcap'), 1)}`",
        f"- code/freeze sha = `{extra.get('code_sha')}`",
        f"- champions commit sha (pre-JUDGE freeze) = `{extra.get('champions_sha')}`",
        "",
        "## Evolution (MINE)",
        "",
        "| gen | best fitness | median finite | n finite | best nodes |",
        "|-----|--------------|---------------|----------|------------|",
    ]
    for rec in history:
        lines.append(
            f"| {rec.get('generation')} | {_fmt(rec.get('best'))} | {_fmt(rec.get('median'))} "
            f"| {rec.get('n_finite')} | {rec.get('best_nodes')} |"
        )
    if not history:
        lines.append("| — |  |  |  |  |")
    lines += [
        "",
        f"Early stop / last gen: `{extra.get('last_generation')}`. Best MINE formula: `{extra.get('best_mine_expr')}`.",
        "",
        "## MINE → SELECT decay (top-20 by MINE fitness)",
        "",
        "| rank | nodes | MINE fitness | SELECT fitness | formula |",
        "|------|-------|--------------|----------------|---------|",
    ]
    for row in decay:
        expr = str(row.get("expr") or "").replace("|", "/")
        lines.append(
            f"| {row.get('rank_mine')} | {row.get('nodes')} | {_fmt(row.get('mine_fitness'))} "
            f"| {_fmt(row.get('select_fitness'))} | `{expr}` |"
        )
    if not decay:
        lines.append("| — |  |  |  |  |")
    lines += [
        "",
        "## Champions (formulas verbatim; frozen before JUDGE)",
        "",
    ]
    if stage == "pre_judge":
        lines += [
            "**JUDGE has not been touched.** The five formulas below are the freeze set.",
            "",
        ]
    for ch in champions:
        lines += [
            f"### Champion {ch.get('rank')}",
            "",
            f"`{ch.get('expr')}`",
            "",
            f"- nodes = `{ch.get('nodes')}`",
            f"- MINE fitness = `{_fmt(ch.get('mine_fitness'))}`",
            f"- SELECT fitness = `{_fmt(ch.get('select_fitness'))}`",
            f"- SELECT 12-book median / spread = `{_fmt(ch.get('select_median'))}` / `{_fmt(ch.get('select_spread'))}`",
            f"- max pairwise headline corr (SELECT) = `{_fmt(ch.get('max_pairwise_corr'))}`",
            "",
        ]
    if not champions:
        lines += ["No champions (empty finite population).", ""]
    lines += [
        "## Null-mining floor",
        "",
        f"- null gens = `{null.get('n_gen', FORGE_NULL_GENS)}`",
        f"- best null-mined SELECT fitness = `{_fmt(null.get('best_select_fitness'))}`",
        f"- best null expr = `{null.get('best_expr')}`",
        "",
        "Champions' SELECT fitness vs null floor:",
        "",
    ]
    nf = null.get("best_select_fitness")
    for ch in champions:
        sf = ch.get("select_fitness")
        delta = None
        try:
            if sf is not None and nf is not None and np.isfinite(float(sf)) and np.isfinite(float(nf)):
                delta = float(sf) - float(nf)
        except Exception:
            delta = None
        lines.append(
            f"- C{ch.get('rank')} SELECT `{_fmt(sf)}` − null `{_fmt(nf)}` = `{_fmt(delta)}`"
        )
    lines += [
        "",
        "## Ingredient census (top-50 by MINE fitness)",
        "",
        f"- `{census.get('one_liner', '')}`",
        "",
        "| primitive | count |",
        "|-----------|-------|",
    ]
    for name, c in (census.get("primitives") or [])[:20]:
        lines.append(f"| `{name}` | {c} |")
    lines += [
        "",
        "| operator | count |",
        "|----------|-------|",
    ]
    for name, c in (census.get("operators") or [])[:20]:
        lines.append(f"| `{name}` | {c} |")
    lines += ["", "## JUDGE"]
    if stage != "judge" or not verdict:
        lines += [
            "",
            "Not evaluated in this document. Formulas above are frozen; JUDGE is a later one-shot job.",
            "",
        ]
    else:
        lines += [
            "",
            f"**Verdict: {verdict.get('label')}** (n_pass={verdict.get('n_pass')} / {verdict.get('n')}).",
            "",
            "Headline book = k=5, daily, floored PIT top-100 DV. Mechanical, no post-hoc adjustment.",
            "",
            "| champion | formula | total | BTC | rel Sharpe | MaxDD | BTC MaxDD | a | b | c | pass |",
            "|----------|---------|-------|-----|------------|-------|-----------|---|---|---|------|",
        ]
        for row in judge.get("champions") or []:
            expr = str(row.get("expr") or "").replace("|", "/")
            p = row.get("pass_bits") or {}
            lines.append(
                f"| C{row.get('rank')} | `{expr}` | {_fmt(row.get('book_total'))} | {_fmt(row.get('btc_total'))} "
                f"| {_fmt(row.get('rel_sharpe'))} | {_fmt(row.get('maxdd'))} | {_fmt(row.get('btc_maxdd'))} "
                f"| {p.get('a_total_ge_btc')} | {p.get('b_rel_sharpe_gt0')} | {p.get('c_maxdd_le')} "
                f"| {p.get('pass')} |"
            )
        lines += ["", "### Per-champion honesty (PHASE2 cycles on JUDGE window)", ""]
        for row in judge.get("champions") or []:
            lines += [f"#### C{row.get('rank')} `{row.get('expr')}`", ""]
            lines += _cycle_table(row.get("book") or {})
            lines.append("")
        lines += [
            "### Benchmarks (JUDGE)",
            "",
            "| book | total | CAGR | Sharpe | MaxDD | rel Sharpe | corr BTC |",
            "|------|-------|------|--------|-------|------------|----------|",
        ]
        for name, bk in (bench or {}).items():
            if not isinstance(bk, dict):
                continue
            lines.append(
                f"| {name} | {_fmt(bk.get('total', bk.get('book_total')))} | {_fmt(bk.get('cagr', bk.get('book_cagr')))} "
                f"| {_fmt(bk.get('sharpe', bk.get('book_sharpe')))} | {_fmt(bk.get('maxdd'))} "
                f"| {_fmt(bk.get('rel_sharpe'))} | {_fmt(bk.get('corr_btc'))} |"
            )
        lines += ["", "If FORGE-DEAD: GP-mined formulaic strategies do not survive nesting on this data.", ""]
    lines += [
        "Mechanical, no post-hoc adjustment. Frozen products untouched. Nothing adopted.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text)
    return text


def plot_fitness_gen(history: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("FORGE F0 fitness by generation (empty)")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    gens = [r.get("generation") for r in history]
    best = [r.get("best") for r in history]
    med = [r.get("median") for r in history]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(gens, best, label="best", color="#1f77b4")
    ax.plot(gens, med, label="median finite", color="#ff7f0e")
    ax.set_xlabel("generation")
    ax.set_ylabel("MINE fitness")
    ax.set_title("FORGE F0 — fitness by generation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mine_vs_select(decay: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    for r in decay or []:
        a, b = r.get("mine_fitness"), r.get("select_fitness")
        if a is None or b is None:
            continue
        if np.isfinite(float(a)) and np.isfinite(float(b)):
            xs.append(float(a))
            ys.append(float(b))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if xs:
        ax.scatter(xs, ys, c="#1f77b4", s=28)
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], ls="--", c="gray", lw=1)
    ax.set_xlabel("MINE fitness")
    ax.set_ylabel("SELECT fitness")
    ax.set_title("FORGE F0 — MINE vs SELECT (top-20)")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_judge_equity(champs: list[dict], btc: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    def _eq(book):
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
        return (1.0 + r).cumprod()

    btc_eq = _eq(btc)
    ax0, ax1 = axes
    if btc_eq is not None:
        ax0.plot(btc_eq.index, np.log(btc_eq.replace(0, np.nan)), label="BTC B&H", color="black", lw=1.4)
        ax1.axhline(1.0, color="black", lw=0.8)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, ch in enumerate(champs or []):
        book = ch.get("book") or ch
        eq = _eq(book)
        if eq is None:
            continue
        lab = f"C{ch.get('rank', i + 1)}"
        ax0.plot(eq.index, np.log(eq.replace(0, np.nan)), label=lab, color=colors[i % len(colors)], lw=1.2)
        if btc_eq is not None:
            rel = eq.reindex(btc_eq.index).ffill() / btc_eq.replace(0, np.nan)
            ax1.plot(rel.index, rel, label=lab, color=colors[i % len(colors)], lw=1.2)
    ax0.set_ylabel("log equity")
    ax0.set_title("FORGE F0 JUDGE — log equity vs BTC")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax1.set_ylabel("book / BTC")
    ax1.set_title("Relative line")
    ax1.grid(True, alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def update_ledger_forge(path: Path, *, verdict: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER FORGE F0"
    champs = extra.get("champion_one_liners") or ""
    block = [
        "",
        marker,
        "",
        "Evolutionary strategy miner, compounding 12-book fitness, nested MINE/SELECT/JUDGE. Backtest only. Nothing adopted. Binance-hybrid.",
        "",
        f"**{verdict.get('label')}.** n_pass=`{verdict.get('n_pass')}` / `{verdict.get('n')}`. "
        f"Null SELECT floor `{extra.get('null_floor')}`. Census `{extra.get('census_line')}`.",
        "",
        champs,
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
    Path(path).write_text(text)
    return text
