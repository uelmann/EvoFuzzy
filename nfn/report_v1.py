"""Phase 7.c NFN v1 report + charts. Analysis only. Nothing adopted."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btcb.constants import DEATH_CONVENTION, PHASE2C_PRED_SHA256
from nfn.constants import FIREWALL, FIREWALL_PHASE7
from nfn.constants_v1 import PHASE7C_CRITERION, PHASE7C_NULL_REGISTRATION
from nfn.report import _delta, _fmt, _metric_row, _null_table, _pct, plot_film_ribbon, plot_membership


def aggregate_seed_fold(hygiene: list[dict]) -> list[dict]:
    from collections import defaultdict

    g = defaultdict(list)
    for h in hygiene or []:
        if h.get("status") != "ok":
            continue
        g[(int(h.get("seed", -1)), int(h.get("fold_id", -1)))].append(h)
    rows = []
    for (seed, fid), hs in sorted(g.items()):
        ics = []
        for h in hs:
            v = h.get("best_holdout_tail_ic")
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                ics.append(fv)
        epochs = []
        spans = []
        for h in hs:
            try:
                epochs.append(float(h.get("selected_epoch", h.get("best_epoch"))))
            except (TypeError, ValueError):
                pass
            try:
                spans.append(float(h.get("swa_span") or 0))
            except (TypeError, ValueError):
                pass
        bag0 = next((h for h in hs if int(h.get("bag", 0)) == 0), hs[0])
        rows.append(
            {
                "seed": seed,
                "fold_id": fid,
                "selected_epoch": float(np.mean(epochs)) if epochs else float("nan"),
                "selected_epoch_window": bag0.get("selected_epoch_window"),
                "swa_span": float(np.mean(spans)) if spans else 0.0,
                "bag_spread": (max(ics) - min(ics)) if len(ics) >= 2 else 0.0,
                "n_holdout_dates": bag0.get("n_holdout_dates"),
                "elapsed": float(sum(float(h.get("elapsed") or 0) for h in hs)),
                "n_bags": int(len(hs)),
                "n_cache_hits": int(sum(1 for h in hs if h.get("cache_hit"))),
            }
        )
    return rows


def write_phase7c(
    path: Path,
    *,
    grid: dict,
    books: dict,
    null: dict,
    hygiene: list[dict],
    hygiene_seed_fold: list[dict],
    seed_metrics: dict,
    verdict: dict,
    rules: list[dict],
    movement: list[dict],
    extra: dict,
    cold_control: list[dict],
    arch: dict,
) -> str:
    lines = [
        "# BTC-BEATER Phase 7.c — NFN v1 (corrected training craft)",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched. Master only.",
        "",
        "## Firewall (verbatim)",
        "",
        f"> {FIREWALL}",
        "",
        f"> {FIREWALL_PHASE7}",
        "",
        f"Firewall grep: passed=`{extra.get('firewall_passed')}` architecture_assert=`{arch.get('passed')}` "
        f"fold-0 init=`random` (no VIABLE Phase-6 bank).",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Pre-registered criteria (verbatim, before results)",
        "",
        f"> {PHASE7C_CRITERION}",
        "",
        "## Config diff vs Phase 7 (six craft items only)",
        "",
        "| item | Phase 7 v0 | Phase 7.c v1 |",
        "|------|------------|--------------|",
        "| 1 model selection | argmax single-epoch holdout tail-IC | 3-epoch trailing mean; SWA of best 3 epochs |",
        "| 1 ES | floor 8, patience 6, cap 30 | floor 10, patience 8, cap 40 |",
        "| 2 intra-fold bagging | 1 init / fold | 5 independent inits; mean calibrated spread |",
        "| 3 warm-start | Phase-6 RULE-FORGE (absent → random every fold) | previous fold SWA weights (fold 0 random); cold control folds {9,18,27} |",
        "| 4 optimizer | AdamW 3e-4, wd 0, clip 5.0 | AdamW 1e-4 cosine→1e-5, wd 1e-4, clip 1.0 |",
        "| 5 inner holdout | last 90 calendar days | last 120 dates, purged by h+3 |",
        "| 6 seeds | {42,43,44} one run each | {42,43,44} at bagging-ensemble level (3×5=15 inits/fold) |",
        "",
        "Architecture (memberships, 198 primitives, 24 log-space rules, FiLM(8), twin heads, loss) unchanged. "
        f"n_params = `{extra.get('n_params')}` (expected 5488). gpu=`{extra.get('gpu_used')}`.",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` readonly_ok=`{extra.get('cmc_readonly_ok')}`",
        f"- seeds = `{extra.get('seeds')}` bags = `{extra.get('n_bag')}` folds = `{extra.get('n_folds')}`",
        f"- window `{extra.get('start')}` → `{extra.get('end')}`",
        f"- {PHASE7C_NULL_REGISTRATION}",
        "",
        "## 1 — Hygiene table v2 (per seed × fold, bags aggregated)",
        "",
        "| seed | fold | sel. epoch (mean) | SWA window | SWA span | 5-init spread | n_holdout | elapsed_s | cache |",
        "|------|------|-------------------|------------|----------|---------------|-----------|-----------|-------|",
    ]
    for h in hygiene_seed_fold:
        win = h.get("selected_epoch_window") or []
        win_s = ",".join(str(x) for x in win) if isinstance(win, list) else str(win)
        lines.append(
            f"| {h.get('seed')} | {h.get('fold_id')} | {_fmt(h.get('selected_epoch'), 1)} "
            f"| {win_s} | {h.get('swa_span')} | {_fmt(h.get('bag_spread'))} "
            f"| {h.get('n_holdout_dates')} | {_fmt(h.get('elapsed'), 1)} | {h.get('n_cache_hits')}/{h.get('n_bags')} |"
        )
    lines.extend(
        [
            "",
            f"Mean selected-epoch window (seed×fold, bags averaged) = `{_fmt(verdict.get('mean_selected_epoch'), 2)}` "
            f"(CRAFT bar ≥ 8). v0 mean best_epoch = `{_fmt(extra.get('v0_mean_best_epoch'), 2)}`.",
            f"Seed dispersion v0 → v1: `{_fmt(extra.get('v0_dispersion'))}` → `{_fmt(verdict.get('seed_dispersion_tail_ic'))}`.",
            "",
            "### Warm-start vs cold-start control (seed 42, folds {9,18,27})",
            "",
            "| fold | bag | warm sel.epoch | cold sel.epoch | Δ holdout trail3 (warm−cold) | Δ holdout IC |",
            "|------|-----|----------------|----------------|------------------------------|--------------|",
        ]
    )
    for rec in cold_control or []:
        lines.append(
            f"| {rec.get('fold_id')} | {rec.get('bag')} | {rec.get('warm_epoch')} | {rec.get('cold_epoch')} "
            f"| {_delta(rec.get('delta_trail3'))} | {_delta(rec.get('delta_ic'))} |"
        )
    if not cold_control:
        lines.append("| — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            f"Mean Δ trail3 (warm−cold) = `{_delta(extra.get('cold_mean_delta_trail3'))}`.",
            "",
            "## 2 — Judgment grid vs frozen spread",
            "",
            "| signal | tail-IC top | NW-t | tail-IC bot | overlap | monster | RankIC | trail-18m tail-IC | n |",
            "|--------|-------------|------|-------------|---------|---------|--------|-------------------|---|",
            _metric_row("frozen spread", grid.get("frozen_spread") or {}),
        ]
    )
    for s, met in sorted(seed_metrics.items()):
        lines.append(_metric_row(f"NFN-v1 seed {s}", met or {}))
    lines.append(_metric_row("NFN-v1 ensemble", grid.get("nfn_ensemble") or {}))
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
    lines.extend(["", "## 5 — Interpretability", "", "### Top rules (last-fold seed 42 bag 0, by |w|·L1(e))", ""])
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
            f"- **CRAFT-CONFIRMED** = `{verdict.get('craft_confirmed')}` "
            f"(disp_ok=`{verdict.get('disp_ok')}` epoch_ok=`{verdict.get('epoch_ok')}` "
            f"mean selected epoch=`{_fmt(verdict.get('mean_selected_epoch'), 2)}`)",
            "",
        ]
    )
    if verdict.get("craft_ledger"):
        lines.extend([f"- Ledger clause: **{verdict.get('craft_ledger')}**", ""])
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
            "- Architecture asserted equal to Phase 7 except the six craft items.",
            "",
            "COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_tail_ic_null(grid: dict, null: dict, path: Path) -> None:
    labels = ["frozen spread", "NFN-v1 ensemble"]
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
    ax.set_title("Phase 7.c NFN v1 vs frozen spread")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_dispersion(v0_ics: list[float], v1_ics: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([0] * len(v0_ics), v0_ics, color="#c44", s=60, label="v0 seeds")
    ax.scatter([1] * len(v1_ics), v1_ics, color="#2a6", s=60, label="v1 seeds")
    if v0_ics:
        ax.plot([0, 0], [min(v0_ics), max(v0_ics)], color="#c44", lw=2)
    if v1_ics:
        ax.plot([1, 1], [min(v1_ics), max(v1_ics)], color="#2a6", lw=2)
    ax.axhline(0.0, color="#aaa", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Phase 7 v0", "Phase 7.c v1"])
    ax.set_ylabel("seed tail-IC (top)")
    d0 = (max(v0_ics) - min(v0_ics)) if len(v0_ics) >= 2 else float("nan")
    d1 = (max(v1_ics) - min(v1_ics)) if len(v1_ics) >= 2 else float("nan")
    ax.set_title(f"Seed dispersion {d0:.4f} → {d1:.4f} (bar ≤ 0.010)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_epoch_hist(v0_epochs: list[float], v1_epochs: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.arange(0.5, 41.5, 1.0)
    if v0_epochs:
        ax.hist(v0_epochs, bins=bins, alpha=0.55, color="#c44", label=f"v0 best_epoch n={len(v0_epochs)}")
    if v1_epochs:
        ax.hist(v1_epochs, bins=bins, alpha=0.55, color="#2a6", label=f"v1 selected_epoch n={len(v1_epochs)}")
    ax.axvline(8.0, color="#333", ls="--", lw=1.0, label="CRAFT bar 8")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fold × seed (v0) / fold × seed × bag (v1)")
    ax.set_title("Selected-epoch histogram: Phase 7 v0 vs 7.c v1")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def update_ledger_v1(path: Path, verdict: dict, extra: dict) -> None:
    if not path.exists():
        return
    text = path.read_text()
    block = (
        "\n## Phase 7.c NFN v1 (record only; nothing adopted)\n\n"
        f"- **{verdict.get('label')}** clauses a/b/c = "
        f"{verdict.get('clause_a')}/{verdict.get('clause_b')}/{verdict.get('clause_c')}\n"
        f"- CRAFT-CONFIRMED `{verdict.get('craft_confirmed')}` mean selected epoch "
        f"`{verdict.get('mean_selected_epoch')}` dispersion v0→v1 "
        f"`{extra.get('v0_dispersion')}` → `{verdict.get('seed_dispersion_tail_ic')}`\n"
        f"- Δtail-IC `{verdict.get('delta_tail_ic')}` Δoverlap `{verdict.get('delta_overlap')}`\n"
    )
    if verdict.get("craft_ledger"):
        block += f"- Craft ledger: {verdict.get('craft_ledger')}\n"
    if "Phase 7.c NFN v1" not in text:
        path.write_text(text.rstrip() + "\n" + block)
