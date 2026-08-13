"""Phase E.1b report writer."""

from __future__ import annotations

from pathlib import Path

from phase_e1.nullgate import E1B_GATE
from phase_e1.report import CONFIRM_CRITERION, _fmt, write_report


def write_e1b_report(path: Path, **kw) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase E.1b — empirical-null GRU label-shuffle gate\n")
    lines.append("**Verification only.** No GRU retuning. Backtest/analysis only.\n")
    lines.append(f"- Frozen A0 hash: `{kw.get('frozen_hash')}`")
    lines.append("- Addendum (written before the null was observed): `reports/phaseE1_addendum.md`")
    lines.append("- Original E.1 `|IC|<0.005` FAIL is preserved; this gate replaces clause (i) only if GREEN.\n")
    lines.append("## Pre-registered gate (verbatim, before results)\n")
    lines.append(f"> {E1B_GATE}\n")
    lines.append(f"- Budget: `{kw.get('budget')}`")
    lines.append(f"- Folds used: `{kw.get('folds_used')}`")
    lines.append(f"- Horizons: `{kw.get('horizons')}`")
    lines.append(f"- Shuffle seeds: `{kw.get('shuffle_seeds')}`")
    lines.append(f"- Dropped to 2 folds: `{kw.get('dropped_folds')}`\n")
    lines.append(f"**Mechanical E.1b verdict: {kw.get('e1b_verdict')}**\n")
    lines.append(f"Details: `{kw.get('e1b_details')}`\n")

    lines.append("## Null histograms (pit-120 primary)\n")
    lines.append("| h | fold | n | mean | SD | 95th pct | real 3-seed IC |")
    lines.append("|---|------|---|------|----|----------|----------------|")
    for r in kw.get("null_table") or []:
        if r.get("universe") != "pit120":
            continue
        lines.append(
            f"| {r.get('horizon')} | {r.get('fold_id')} | {r.get('n')} | {_fmt(r.get('mean'))} | "
            f"{_fmt(r.get('sd'))} | {_fmt(r.get('p95'))} | {_fmt(r.get('real_ic'))} |"
        )
    lines.append("\nInformational top-20:\n")
    lines.append("| h | fold | n | mean | SD | 95th pct | real 3-seed IC |")
    lines.append("|---|------|---|------|----|----------|----------------|")
    for r in kw.get("null_table") or []:
        if r.get("universe") != "top20":
            continue
        lines.append(
            f"| {r.get('horizon')} | {r.get('fold_id')} | {r.get('n')} | {_fmt(r.get('mean'))} | "
            f"{_fmt(r.get('sd'))} | {_fmt(r.get('p95'))} | {_fmt(r.get('real_ic'))} |"
        )

    lines.append("\n## Bias test (primary pit-120)\n")
    lines.append("| h | fold | mean | SD | SE=SD/√R | 2·SE | \\|mean\\| | pass |")
    lines.append("|---|------|------|----|----------|------|---------|------|")
    for r in kw.get("null_table") or []:
        if r.get("universe") != "pit120":
            continue
        lines.append(
            f"| {r.get('horizon')} | {r.get('fold_id')} | {_fmt(r.get('mean'))} | {_fmt(r.get('sd'))} | "
            f"{_fmt(r.get('se'))} | {_fmt(r.get('bias_lim'))} | {_fmt(abs(r.get('mean') or float('nan')))} | "
            f"{'PASS' if r.get('bias_ok') else 'FAIL'} |"
        )

    lines.append("\n## Skill test (real 3-seed vs null 95th, pit-120)\n")
    lines.append("| h | fold | real IC | null 95th | exceeds |")
    lines.append("|---|------|---------|-----------|---------|")
    for r in kw.get("null_table") or []:
        if r.get("universe") != "pit120":
            continue
        ex = _finite(r.get("real_ic")) and _finite(r.get("p95")) and float(r["real_ic"]) > float(r["p95"])
        lines.append(
            f"| {r.get('horizon')} | {r.get('fold_id')} | {_fmt(r.get('real_ic'))} | {_fmt(r.get('p95'))} | {ex} |"
        )
    lines.append(f"\nSkill-by-horizon: `{kw.get('skill_by_h')}`\n")

    lines.append("## A0 LightGBM empirical null (informational, no verdict)\n")
    lines.append("| h | fold | n | mean | SD | 95th pct | real A0 IC | exceeds 95th |")
    lines.append("|---|------|---|------|----|----------|------------|--------------|")
    for r in kw.get("a0_table") or []:
        if r.get("universe") != "pit120":
            continue
        ex = _finite(r.get("real_ic")) and _finite(r.get("p95")) and float(r["real_ic"]) > float(r["p95"])
        lines.append(
            f"| {r.get('horizon')} | {r.get('fold_id')} | {r.get('n')} | {_fmt(r.get('mean'))} | "
            f"{_fmt(r.get('sd'))} | {_fmt(r.get('p95'))} | {_fmt(r.get('real_ic'))} | {ex} |"
        )

    if kw.get("e1b_verdict") == "GREEN" and kw.get("resume"):
        lines.append("\n## Resumed Phase E.1 §2–§4\n")
        lines.append("Clause (i) replaced by the E.1b empirical-null gate (GREEN). Clauses (ii)–(iv) unchanged.\n")
        lines.append("## Pre-registered confirmation criterion (verbatim)\n")
        lines.append(f"> {CONFIRM_CRITERION}\n")
        res = kw["resume"]
        lines.append(f"**Mechanical confirmation verdict: {res.get('verdict')}**\n")
        lines.append(f"Details: `{res.get('details')}`\n")
        tmp = path.parent / "_e1_resume_tmp.md"
        write_report(
            tmp,
            frozen_hash=kw.get("frozen_hash"),
            verdict=res.get("verdict"),
            verdict_details=res.get("details"),
            gates=kw.get("gates") or [{"name": "e1b_empirical_null", "passed": True}],
            gates_ok=True,
            budget=res.get("extra_gpu", {}).get("projection") if isinstance(res.get("extra_gpu"), dict) else None,
            horizons_trained=res.get("details", {}).get("horizons_trained") if isinstance(res.get("details"), dict) else None,
            ensemble_keep_lines=res.get("keep_lines"),
            ensemble_table=res.get("ensemble_table"),
            seed_dist=res.get("seed_dist"),
            nw_table=res.get("nw_table"),
            corr_table=res.get("corr_table"),
            year_table=res.get("year_table"),
            acf_table=res.get("acf_table"),
            port_table=res.get("port_table"),
        )
        extra = tmp.read_text().split("## 2. Seed robustness", 1)
        if len(extra) == 2:
            lines.append("## 2. Seed robustness" + extra[1])
        else:
            lines.append(tmp.read_text())
        tmp.unlink(missing_ok=True)
    else:
        lines.append("\n§2–§4 not resumed.\n")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def _finite(x) -> bool:
    try:
        return x is not None and float(x) == float(x)
    except Exception:
        return False
