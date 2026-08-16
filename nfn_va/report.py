"""Phase 7.d Variant A report + charts. Analysis only. Nothing adopted."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btcb.constants import DEATH_CONVENTION, PHASE2C_PRED_SHA256
from nfn_va.constants import (
    FIREWALL,
    FIREWALL_PHASE7,
    PHASE7D_CRITERION,
    PHASE7_ARCH,
    PHASE7_NULL_REGISTRATION,
    VARIANT_A_N_PARAMS,
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


def write_phase7d(
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
    mag: dict,
    deciles: dict,
    yearly: list,
    extra: dict,
) -> str:
    n_under = int(sum(1 for h in hygiene if h.get("undertrained")))
    mag_rec = (verdict or {}).get("magnitude_gain_rec") or mag or {}
    lines = [
        "# BTC-BEATER Phase 7.d — NFN VARIANT A",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched.",
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
        f"> {PHASE7D_CRITERION}",
        "",
        "## Config diff vs Phase 7 (must show label/loss/head only)",
        "",
        f"- Frozen architecture block = `{PHASE7_ARCH}`",
        f"- n_params Variant A = `{extra.get('n_params')}` (expected `{VARIANT_A_N_PARAMS}` = 5488 − 25)",
        f"- config-equality passed = `{extra.get('config_equal')}`",
        f"- Allowed diffs: single scalar head (no twin, no isotonic); magnitude labels (rank01 + winsor 1/99); "
        f"L = 1.0 ListNet(softmax scores/τ vs softmax winsor/τ) + 0.5 Huber(δ=1) on z-winsor + L1(e); "
        f"7.c craft (trail-3 SWA, ES floor 10 / patience 8 / cap 40, 5-init bag, fold warm-start, AdamW 1e-4 cosine→1e-5, inner 120 dates purged).",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` readonly_ok=`{extra.get('cmc_readonly_ok')}`",
        f"- seeds = `{extra.get('seeds')}` folds = `{extra.get('n_folds')}` gpu=`{extra.get('gpu_used')}`",
        f"- window `{extra.get('start')}` → `{extra.get('end')}`",
        f"- {PHASE7_NULL_REGISTRATION}",
        "",
        "## 1 — Training hygiene",
        "",
        "| seed | fold | selected epoch | window | 5-init spread | UNDERTRAINED | n_train_dates | elapsed_s |",
        "|------|------|----------------|--------|---------------|--------------|---------------|-----------|",
    ]
    for h in hygiene:
        win = h.get("selected_epoch_window")
        lines.append(
            f"| {h.get('seed')} | {h.get('fold_id')} | {h.get('selected_epoch')} "
            f"| {win} | {_fmt(h.get('init_spread_trail_ic'))} | {h.get('undertrained')} "
            f"| {h.get('n_train_dates')} | {_fmt(h.get('elapsed'), 1)} |"
        )
    lines.extend(
        [
            "",
            f"**UNDERTRAINED count** = `{n_under}` (selected-epoch window centre < 10).",
            "",
            "## 2 — Judgment grid vs frozen spread and NFN v0",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | trail-18m tail-IC | n |",
            "|--------|-------------|------|-------------|---------|---------|--------|-------------------|---|",
            _metric_row("frozen spread", grid.get("frozen_spread") or {}),
        ]
    )
    if grid.get("nfn_v0"):
        lines.append(_metric_row("NFN v0 (read-only)", grid.get("nfn_v0") or {}))
    for s, met in sorted(seed_metrics.items()):
        lines.append(_metric_row(f"Variant A seed {s}", met or {}))
    lines.append(_metric_row("Variant A ensemble", grid.get("variant_a_ensemble") or {}))
    nfn0 = grid.get("nfn_v0") or {}
    ens = grid.get("variant_a_ensemble") or {}
    lines.extend(
        [
            "",
            f"Seed dispersion tail-IC(top) = `{_fmt(verdict.get('seed_dispersion_tail_ic'))}` (need ≤ 0.010). "
            f"Δ vs frozen: tail-IC `{_delta(verdict.get('delta_tail_ic'))}` overlap `{_delta(verdict.get('delta_overlap'))}`. "
            f"Δ vs NFN v0: tail-IC `{_delta(_fsub(ens.get('tail_ic_top'), nfn0.get('tail_ic_top')))}` "
            f"overlap `{_delta(_fsub(ens.get('overlap'), nfn0.get('overlap')))}`.",
            "",
            "### Per-cycle overlap / tail-IC(top)",
            "",
        ]
    )
    cyc_ov = (ens.get("overlap_cycles") or {}) if isinstance(ens.get("overlap_cycles"), dict) else {}
    cyc_ic = (ens.get("tail_ic_top_cycles") or {}) if isinstance(ens.get("tail_ic_top_cycles"), dict) else {}
    base_ov = ((grid.get("frozen_spread") or {}).get("overlap_cycles") or {}) if isinstance((grid.get("frozen_spread") or {}).get("overlap_cycles"), dict) else {}
    base_ic = ((grid.get("frozen_spread") or {}).get("tail_ic_top_cycles") or {}) if isinstance((grid.get("frozen_spread") or {}).get("tail_ic_top_cycles"), dict) else {}
    keys = sorted(set(cyc_ov) | set(cyc_ic) | set(base_ov) | set(base_ic))
    if keys:
        lines.extend(["| cycle | frozen overlap | VA overlap | frozen tail-IC | VA tail-IC |", "|-------|----------------|------------|----------------|------------|"])
        for k in keys:
            lines.append(
                f"| {k} | {_fmt(base_ov.get(k))} | {_fmt(cyc_ov.get(k))} | {_fmt(base_ic.get(k))} | {_fmt(cyc_ic.get(k))} |"
            )
        lines.append("")
    lines.extend(
        [
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
            "## 4 — Magnitude diagnostics",
            "",
            f"- Top-10 mean realized excess: Variant A `{_fmt(mag_rec.get('va'))}` vs frozen `{_fmt(mag_rec.get('frozen'))}` "
            f"relative `{_fmt(mag_rec.get('rel'))}` need ≥ `{mag_rec.get('need')}`. "
            f"**MAGNITUDE-GAIN = {'yes' if mag_rec.get('yes') else 'no'}**.",
            "",
            "| decile (1=low score) | Variant A mean excess | frozen mean excess |",
            "|----------------------|-----------------------|--------------------|",
        ]
    )
    va_d = {int(r["decile"]): r for r in (deciles.get("variant_a") or [])}
    fr_d = {int(r["decile"]): r for r in (deciles.get("frozen") or [])}
    for d in range(1, 11):
        lines.append(
            f"| {d} | {_fmt((va_d.get(d) or {}).get('mean_excess'))} | {_fmt((fr_d.get(d) or {}).get('mean_excess'))} |"
        )
    lines.extend(["", "### Score CS distribution by year", "", "| year | n_dates | mean | std | p10 | p90 |", "|------|---------|------|-----|-----|-----|"])
    for rec in yearly or []:
        lines.append(
            f"| {rec.get('year')} | {rec.get('n_dates')} | {_fmt(rec.get('mean'))} "
            f"| {_fmt(rec.get('std'))} | {_fmt(rec.get('p10'))} | {_fmt(rec.get('p90'))} |"
        )
    lines.extend(
        [
            "",
            "## 5 — Crude-14d books (information only)",
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
    lines.extend(["", "## 6 — Interpretability", "", "### Top rules (last-fold seed 42, by |w|·L1(e))", ""])
    for rec in rules[:8]:
        lines.append(
            f"- **r_{rec.get('rule')}** w=`{_fmt(rec.get('weight'), 3)}` L1(e)=`{_fmt(rec.get('l1_e'), 3)}`: `{rec.get('formula')}`"
        )
    lines.extend(
        [
            "",
            "### Membership movement from init (top-20 features)",
            "",
            "| feature | movement | c | s | c_init | s_init |",
            "|---------|----------|---|---|--------|--------|",
        ]
    )
    for rec in movement[:20]:
        lines.append(
            f"| {rec.get('feature')} | {_fmt(rec.get('movement'))} | {rec.get('c')} | {rec.get('s')} "
            f"| {rec.get('c_init')} | {rec.get('s_init')} |"
        )
    failed = verdict.get("failed_clauses") or []
    lines.extend(
        [
            "",
            "## 7 — Mechanical verdicts",
            "",
            f"- **{verdict.get('label')}**",
            f"- clause (a) deltas: `{verdict.get('clause_a')}` ΔIC `{_delta(verdict.get('delta_tail_ic'))}` Δov `{_delta(verdict.get('delta_overlap'))}`",
            f"- clause (b) dispersion: `{verdict.get('clause_b')}` `{_fmt(verdict.get('seed_dispersion_tail_ic'))}`",
            f"- clause (c) vol-matched null: `{verdict.get('clause_c')}` `{verdict.get('null_verdict')}`",
            f"- failed clauses: `{failed if failed else 'none'}`",
            f"- **MAGNITUDE-GAIN = {'yes' if verdict.get('magnitude_gain') else 'no'}** "
            f"(VA `{_fmt(mag_rec.get('va'))}` vs frozen `{_fmt(mag_rec.get('frozen'))}`)",
            "",
            "Mechanical, no post-hoc adjustment. Nothing adopted.",
            "",
            "## Plain language",
            "",
            extra.get("plain", ""),
            "",
            "## Notes",
            "",
            f"- Elapsed s=`{_fmt(extra.get('elapsed_sec'), 1)}`. GPU=`{extra.get('gpu_used', False)}`.",
            "- Frozen spread is the 2.c cache (not retrained). NFN v0 is read-only.",
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


def _fsub(a, b):
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return float("nan")


def plot_tail_ic_bars(grid: dict, null: dict, path: Path) -> None:
    labels = ["frozen spread", "NFN v0", "Variant A"]
    vals = [
        (grid.get("frozen_spread") or {}).get("tail_ic_top"),
        (grid.get("nfn_v0") or {}).get("tail_ic_top"),
        (grid.get("variant_a_ensemble") or {}).get("tail_ic_top"),
    ]
    cells = (null or {}).get("tail_ic_top_cells") or []
    p95s = [c.get("p95") for c in cells if c.get("p95") is not None]
    p95 = float(np.nanmean(p95s)) if p95s else float("nan")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#888", "#E45756"]
    ax.bar(x, [float(v) if v is not None and np.isfinite(v) else 0.0 for v in vals], color=colors)
    if np.isfinite(p95):
        ax.plot([2 - 0.4, 2 + 0.4], [p95, p95], color="black", lw=2.5, label=f"null p95 mean {p95:.3f}")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("tail-IC (top-half)")
    ax.set_title("Phase 7.d Variant A — tail-IC(top-half); black tick = vol-matched null mean p95")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_decile_curve(deciles: dict, path: Path) -> None:
    va = {int(r["decile"]): r.get("mean_excess") for r in (deciles.get("variant_a") or [])}
    fr = {int(r["decile"]): r.get("mean_excess") for r in (deciles.get("frozen") or [])}
    xs = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(xs, [fr.get(d) for d in xs], marker="o", label="frozen spread", color="#4C78A8")
    ax.plot(xs, [va.get(d) for d in xs], marker="s", label="Variant A", color="#E45756")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xlabel("score decile (1 = lowest)")
    ax.set_ylabel("mean 14d log excess vs BTC")
    ax.set_title("Phase 7.d — decile-mean-return curve")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def update_ledger(path: Path, verdict: dict, extra: dict) -> None:
    if not path.exists():
        return
    text = path.read_text()
    mag = (verdict or {}).get("magnitude_gain_rec") or {}
    block = (
        "\n## BTC-BEATER Phase 7.d NFN VARIANT A\n\n"
        "Magnitude labels + ranking loss, single head. Backtest only. Nothing adopted.\n\n"
        f"**{verdict.get('label')}.** Failed clauses=`{verdict.get('failed_clauses') or 'none'}`. "
        f"MAGNITUDE-GAIN={'yes' if verdict.get('magnitude_gain') else 'no'} "
        f"(VA `{mag.get('va')}` vs frozen `{mag.get('frozen')}`).\n\n"
        f"Δtail-IC `{verdict.get('delta_tail_ic')}` Δoverlap `{verdict.get('delta_overlap')}` "
        f"dispersion `{verdict.get('seed_dispersion_tail_ic')}`. "
        f"UNDERTRAINED `{extra.get('n_undertrained')}`.\n\n"
        "Mechanical, no post-hoc adjustment. Frozen products untouched.\n"
    )
    if "Phase 7.d NFN VARIANT A" not in text:
        path.write_text(text.rstrip() + "\n" + block)
