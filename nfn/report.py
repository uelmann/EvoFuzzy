"""Phase 7 NFN report + charts. Analysis only. Nothing adopted."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btcb.constants import DEATH_CONVENTION, PHASE2C_PRED_SHA256
from nfn.constants import (
    FIREWALL,
    FIREWALL_PHASE7,
    PHASE7_CEILING,
    PHASE7_CRITERION,
    PHASE7_NULL_REGISTRATION,
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
        "| fold | n | null mean | 2·SE | bias_ok | p95 | real | exceeds p95 |",
        "|------|---|-----------|------|---------|-----|------|-------------|",
    ]
    for c in cells or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'))} "
            f"| {_fmt(c.get('bias_lim'))} | {c.get('bias_ok')} | {_fmt(c.get('p95'))} "
            f"| {_fmt(c.get(real_key))} | {c.get('exceeds_p95')} |"
        )
    return lines


def _metric_row(name: str, met: dict) -> str:
    return (
        f"| {name} | {_fmt(met.get('tail_ic_top'))} | {_fmt(met.get('tail_ic_top_nw_t'), 2)} "
        f"| {_fmt(met.get('tail_ic_bot'))} | {_fmt(met.get('overlap'))} "
        f"| {_fmt(met.get('monster'))} | {_fmt(met.get('rankic'))} "
        f"| {_fmt(met.get('tail_ic_top_trail'))} | {met.get('n_dates')} |"
    )


def write_phase7(
    path: Path,
    *,
    grid: dict,
    books: dict,
    null: dict,
    hygiene: list[dict],
    seed_metrics: dict,
    verdict: dict,
    rules: list[dict],
    movement: list[dict],
    extra: dict,
) -> str:
    n_under = int(sum(1 for h in hygiene if h.get("undertrained")))
    lines = [
        "# BTC-BEATER Phase 7 — NEURO-FUZZY NET v0",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched. Master only.",
        "",
        "## Firewall (verbatim)",
        "",
        f"> {FIREWALL}",
        "",
        f"> {FIREWALL_PHASE7}",
        "",
        f"Firewall grep: passed=`{extra.get('firewall_passed')}` warm-start path=`{extra.get('warmstart_path')}` "
        f"viable=`{extra.get('warmstart_viable')}` n_rules_copied=`{extra.get('warmstart_rules')}`.",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE7_CRITERION}",
        "",
        "## Config dump (one architecture; zero search)",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` readonly_ok=`{extra.get('cmc_readonly_ok')}`",
        f"- n_params = `{extra.get('n_params')}` gpu=`{extra.get('gpu_used')}`",
        f"- seeds = `{extra.get('seeds')}` folds = `{extra.get('n_folds')}`",
        f"- window `{extra.get('start')}` → `{extra.get('end')}`",
        f"- {PHASE7_NULL_REGISTRATION}",
        "",
        "## 1 — Training hygiene",
        "",
        "| seed | fold | best_epoch | holdout tail-IC | UNDERTRAINED | n_train_dates | elapsed_s |",
        "|------|------|------------|-----------------|--------------|---------------|-----------|",
    ]
    for h in hygiene:
        lines.append(
            f"| {h.get('seed')} | {h.get('fold_id')} | {h.get('best_epoch')} "
            f"| {_fmt(h.get('best_holdout_tail_ic'))} | {h.get('undertrained')} "
            f"| {h.get('n_train_dates')} | {_fmt(h.get('elapsed'), 1)} |"
        )
    lines.extend(
        [
            "",
            f"**UNDERTRAINED count** = `{n_under}` (best_epoch < 10).",
            "",
            "## 2 — Judgment grid vs frozen spread",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | trail-18m tail-IC | n |",
            "|--------|-------------|------|-------------|---------|---------|--------|-------------------|---|",
            _metric_row("frozen spread", grid.get("frozen_spread") or {}),
        ]
    )
    for s, met in sorted(seed_metrics.items()):
        lines.append(_metric_row(f"NFN seed {s}", met or {}))
    lines.append(_metric_row("NFN ensemble", grid.get("nfn_ensemble") or {}))
    lines.extend(
        [
            "",
            f"Seed dispersion tail-IC(top) = `{_fmt(verdict.get('seed_dispersion_tail_ic'))}` "
            f"(need ≤ 0.010). Δtail-IC = `{_delta(verdict.get('delta_tail_ic'))}` "
            f"Δoverlap = `{_delta(verdict.get('delta_overlap'))}`.",
            "",
            "## 3 — Vol-matched null",
            "",
            f"Design `{null.get('null_design')}` judged=`{null.get('judged')}` "
            f"passed=`{null.get('passed')}` tail-IC verdict=`{(null.get('tail_ic_top') or {}).get('verdict')}`.",
            "",
            "### tail-IC(top)",
            "",
        ]
    )
    lines.extend(_null_table(null.get("tail_ic_top_cells") or [], "real_tail_ic_top"))
    lines.extend(["", "### overlap", ""])
    lines.extend(_null_table(null.get("overlap_cells") or [], "real_overlap"))
    lines.extend(
        [
            "",
            "## 4 — Crude-14d books (information only)",
            "",
            "| book | total | CAGR | Sharpe | MaxDD | RankIC | n_form | forced |",
            "|------|-------|------|--------|-------|--------|--------|--------|",
        ]
    )
    for name, packed in (books or {}).items():
        lines.append(
            f"| {name} | {_pct(packed.get('total'))} | {_pct(packed.get('cagr'))} "
            f"| {_fmt(packed.get('sharpe'), 3)} | {_pct(packed.get('maxdd'))} "
            f"| {_fmt(packed.get('rankic'))} | {packed.get('n_formations')} "
            f"| {packed.get('n_forced', packed.get('forced_n'))} |"
        )
    lines.extend(["", "## 5 — Interpretability", "", "### Top rules (last-fold seed 42, by |w|·L1(e))", ""])
    for rec in rules[:8]:
        lines.append(f"- **r_{rec.get('rule')}** w=`{_fmt(rec.get('weight'), 3)}` L1(e)=`{_fmt(rec.get('l1_e'), 3)}`: `{rec.get('formula')}`")
    lines.extend(["", "### Membership movement from init (top-20 features)", "", "| feature | movement | c | s | c_init | s_init |", "|---------|----------|---|---|--------|--------|"])
    for rec in movement[:20]:
        lines.append(
            f"| {rec.get('feature')} | {_fmt(rec.get('movement'))} | {rec.get('c')} | {rec.get('s')} "
            f"| {rec.get('c_init')} | {rec.get('s_init')} |"
        )
    lines.extend(
        [
            "",
            "## 6 — Mechanical verdicts",
            "",
            f"- **{verdict.get('label')}**",
            f"- clause (a) deltas: `{verdict.get('clause_a')}` ΔIC `{_delta(verdict.get('delta_tail_ic'))}` Δov `{_delta(verdict.get('delta_overlap'))}`",
            f"- clause (b) dispersion: `{verdict.get('clause_b')}` `{_fmt(verdict.get('seed_dispersion_tail_ic'))}`",
            f"- clause (c) vol-matched null: `{verdict.get('clause_c')}` `{verdict.get('null_verdict')}`",
            f"- failed clauses: `{verdict.get('failed_clauses')}`",
            "",
        ]
    )
    if verdict.get("ceiling"):
        lines.extend([f"- Ledger clause: **{verdict.get('ceiling')}**", ""])
    lines.extend(
        [
            "Mechanical, no post-hoc adjustment. Nothing adopted.",
            "",
            "## Plain language",
            "",
            extra.get("plain", ""),
            "",
            "## Notes",
            "",
            f"- Elapsed s=`{_fmt(extra.get('elapsed_sec'), 1)}`. GPU=`{extra.get('gpu_used', False)}`.",
            "- Frozen spread is the 2.c cache (not retrained).",
            "- Crude 14d CAGR/MaxDD is an information check.",
            "",
            "COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_tail_ic_bars(grid: dict, null: dict, path: Path) -> None:
    labels = ["frozen spread", "NFN ensemble"]
    vals = [
        (grid.get("frozen_spread") or {}).get("tail_ic_top"),
        (grid.get("nfn_ensemble") or {}).get("tail_ic_top"),
    ]
    cells = (null or {}).get("tail_ic_top_cells") or []
    p95 = np.nanmean([c.get("p95") for c in cells if c.get("p95") is not None]) if cells else float("nan")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x, [float(v) if v is not None and np.isfinite(v) else 0.0 for v in vals], color=["#888", "#2a6"])
    if np.isfinite(p95):
        ax.axhline(p95, color="crimson", ls="--", lw=1.2, label=f"null p95 mean {p95:.3f}")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("tail-IC (top-half)")
    ax.set_title("Phase 7 NFN vs frozen spread")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_membership(movement: list[dict], path: Path) -> None:
    xs = np.linspace(-3, 3, 200)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()
    for ax, rec in zip(axes, movement[:4]):
        name = rec.get("feature")
        c, s = rec.get("c") or [0, 0, 0], rec.get("s") or [1, 1, 1]
        c0, s0 = rec.get("c_init") or [-0.67, 0, 0.67], rec.get("s_init") or [1, 1, 1]
        for k in range(3):
            mu = 1.0 / (1.0 + np.exp(-(xs - float(c[k])) / max(float(s[k]), 0.2)))
            mu0 = 1.0 / (1.0 + np.exp(-(xs - float(c0[k])) / max(float(s0[k]), 0.2)))
            ax.plot(xs, mu0, color="#aaa", lw=1)
            ax.plot(xs, mu, lw=1.5)
        ax.set_title(str(name))
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Membership before (grey) / after (color)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_film_ribbon(hygiene: list[dict], path: Path) -> None:
    # last fold of seed 42 if present
    rec = None
    for h in hygiene:
        if int(h.get("seed", -1)) == 42:
            rec = h
    if rec is None and hygiene:
        rec = hygiene[-1]
    film = (rec or {}).get("film") or {}
    g = film.get("gamma_mean") or []
    b = film.get("beta_mean") or []
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if g:
        ax.plot(g, label="mean γ", color="#1f77b4")
        ax.fill_between(np.arange(len(g)), np.asarray(g) - 0.05, np.asarray(g) + 0.05, color="#1f77b4", alpha=0.15)
    if b:
        ax.plot(b, label="mean β", color="#ff7f0e")
    ax.axhline(1.0, color="#1f77b4", ls=":", lw=0.8)
    ax.axhline(0.0, color="#ff7f0e", ls=":", lw=0.8)
    ax.set_title("FiLM gate ribbon (val dates, last recorded fold)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def update_ledger(path: Path, verdict: dict, extra: dict) -> None:
    if not path.exists():
        return
    text = path.read_text()
    block = (
        "\n## Phase 7 NFN (record only; nothing adopted)\n\n"
        f"- **{verdict.get('label')}** clauses a/b/c = "
        f"{verdict.get('clause_a')}/{verdict.get('clause_b')}/{verdict.get('clause_c')}\n"
        f"- Δtail-IC `{verdict.get('delta_tail_ic')}` Δoverlap `{verdict.get('delta_overlap')}` "
        f"dispersion `{verdict.get('seed_dispersion_tail_ic')}`\n"
        f"- UNDERTRAINED count `{extra.get('n_undertrained')}` warm-start `{extra.get('warmstart_path')}`\n"
    )
    if verdict.get("ceiling"):
        block += f"- Ceiling: {verdict.get('ceiling')}\n"
    if "Phase 7 NFN" not in text:
        path.write_text(text.rstrip() + "\n" + block)
