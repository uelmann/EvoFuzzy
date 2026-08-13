"""Phase E report + stdout summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phase_e.evalutil import SIG_CRITERION, S_VIABLE_CRITERION


def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_phaseE_report(
    path: Path,
    frozen_hash: str,
    sig_ablation: dict,
    sig_keep: dict,
    sig_imp: dict,
    seq_s: dict,
    seq_blend: dict,
    seq_criteria: dict,
    score_corr: dict,
    seed_var: dict,
    gru_info: dict,
    budget: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase E — path signatures + tiny sequence model\n")
    lines.append("**Run order:** Phase E runs BEFORE the queued D.2 universe test, at PI direction. "
                 "Every result is reported on both evaluation slices (top-20 PIT and pit-120) with "
                 "separate mechanical verdicts.\n")
    lines.append(f"- Frozen A0 hash: `{frozen_hash}`")
    lines.append("- Scope: backtest/analysis only; no schedules or live components.")
    lines.append("- Frozen A0 config untouched; arms only ADD features (E1) or ADD a separate model (S).\n")

    lines.append("## Arm 1 — Path signatures (CPU)\n")
    lines.append("### Pre-registered criterion (verbatim)\n")
    lines.append(f"> {SIG_CRITERION}\n")
    lines.append("### Mechanical verdicts\n")
    for uni, blob in (sig_keep.get("universes") or {}).items():
        lines.append(f"- **SIG {uni}: {blob.get('verdict')}** reasons={blob.get('keep_reasons')} details=`{blob.get('details')}`")
    lines.append("")

    lines.append("### Ablation tables (A = frozen A0, E1 = A0 + signatures)\n")
    lines.append("| h | universe | window | A IC | E1 IC | ΔIC | NW-t Δ | n_days |")
    lines.append("|---|----------|--------|------|-------|-----|--------|--------|")
    for h, blob in sorted((sig_ablation or {}).items(), key=lambda kv: int(kv[0])):
        paired = blob.get("paired_nw") or {}
        for t in blob.get("tables") or []:
            nw = ""
            if t["window"] in ("full", "trail18m"):
                p = (paired.get(t["universe"]) or {}).get(t["window"]) or {}
                nw = _fmt(p.get("nw_tstat"), 2)
            lines.append(
                f"| {t['horizon']} | {t['universe']} | {t['window']} | {_fmt(t.get('A_ic'))} | "
                f"{_fmt(t.get('E1_ic', t.get('B_ic')))} | {_fmt(t.get('delta_ic'))} | {nw} | {t.get('n_days')} |"
            )

    lines.append("\n### Δ median-τ net Sharpe (tranche, funding on, top-20, paired days)\n")
    lines.append("| h | A Sharpe | E1 Sharpe | Δ | n_days |")
    lines.append("|---|----------|-----------|---|--------|")
    for h, blob in sorted((sig_ablation or {}).items(), key=lambda kv: int(kv[0])):
        s = blob.get("sharpe_delta") or {}
        lines.append(
            f"| {h} | {_fmt(s.get('A_sharpe'), 3)} | {_fmt(s.get('E1_sharpe', s.get('B_sharpe')), 3)} | "
            f"{_fmt(s.get('delta_sharpe'), 3)} | {s.get('n_days')} |"
        )

    lines.append("\n### Signature LightGBM gain importances\n")
    lines.append("| h | feature | mean_gain | median_gain |")
    lines.append("|---|---------|-----------|-------------|")
    for h, imp in sorted((sig_imp or {}).items(), key=lambda kv: int(kv[0])):
        for feat, st in sorted((imp or {}).items(), key=lambda kv: -kv[1].get("mean_gain", 0)):
            lines.append(f"| {h} | {feat} | {_fmt(st.get('mean_gain'), 2)} | {_fmt(st.get('median_gain'), 2)} |")

    lines.append("\n## Arm 2 — Tiny GRU sequence model (GPU, budget-guarded)\n")
    lines.append("### Pre-registered criteria (verbatim)\n")
    lines.append(f"> {S_VIABLE_CRITERION}\n")
    lines.append(f"- GRU param count: **{gru_info.get('n_params')}**")
    lines.append(f"- Device: `{gru_info.get('device')}`; max_epochs={gru_info.get('max_epochs')}; seeds={gru_info.get('seeds')}")
    lines.append(f"- Budget: `{budget}`\n")

    lines.append("### Mechanical verdicts\n")
    for uni, blob in (seq_criteria.get("universes") or {}).items():
        lines.append(
            f"- **S {uni}: {blob.get('S_viable')}** reasons={blob.get('S_reasons')} `{blob.get('S_details')}`"
        )
        lines.append(
            f"- **BLEND {uni}: {blob.get('BLEND_verdict')}** reasons={blob.get('BLEND_reasons')} `{blob.get('BLEND_details')}`"
        )
    lines.append("")

    lines.append("### S standalone vs A0\n")
    lines.append("| h | universe | window | A IC | S IC | ΔIC | n_days |")
    lines.append("|---|----------|--------|------|------|-----|--------|")
    for h, blob in sorted((seq_s or {}).items(), key=lambda kv: int(kv[0])):
        for t in blob.get("tables") or []:
            lines.append(
                f"| {t['horizon']} | {t['universe']} | {t['window']} | {_fmt(t.get('A_ic'))} | "
                f"{_fmt(t.get('S_ic', t.get('B_ic')))} | {_fmt(t.get('delta_ic'))} | {t.get('n_days')} |"
            )

    lines.append("\n### BLEND (50/50 per-date z A0+S) vs A0\n")
    lines.append("| h | universe | window | A IC | BLEND IC | ΔIC | NW-t Δ | n_days |")
    lines.append("|---|----------|--------|------|----------|-----|--------|--------|")
    for h, blob in sorted((seq_blend or {}).items(), key=lambda kv: int(kv[0])):
        paired = blob.get("paired_nw") or {}
        for t in blob.get("tables") or []:
            nw = ""
            if t["window"] in ("full", "trail18m"):
                p = (paired.get(t["universe"]) or {}).get(t["window"]) or {}
                nw = _fmt(p.get("nw_tstat"), 2)
            lines.append(
                f"| {t['horizon']} | {t['universe']} | {t['window']} | {_fmt(t.get('A_ic'))} | "
                f"{_fmt(t.get('BLEND_ic', t.get('B_ic')))} | {_fmt(t.get('delta_ic'))} | {nw} | {t.get('n_days')} |"
            )

    lines.append("\n### Δ median-τ net Sharpe BLEND vs A0 (top-20, funding on, paired days)\n")
    lines.append("| h | A Sharpe | BLEND Sharpe | Δ | n_days |")
    lines.append("|---|----------|--------------|---|--------|")
    for h, blob in sorted((seq_blend or {}).items(), key=lambda kv: int(kv[0])):
        s = blob.get("sharpe_delta") or {}
        lines.append(
            f"| {h} | {_fmt(s.get('A_sharpe'), 3)} | {_fmt(s.get('BLEND_sharpe', s.get('B_sharpe')), 3)} | "
            f"{_fmt(s.get('delta_sharpe'), 3)} | {s.get('n_days')} |"
        )

    lines.append("\n### Per-seed S RankIC (variance)\n")
    lines.append("| h | seed | universe | window | RankIC |")
    lines.append("|---|------|----------|--------|--------|")
    for row in seed_var.get("rows") or []:
        lines.append(
            f"| {row.get('horizon')} | {row.get('seed')} | {row.get('universe')} | "
            f"{row.get('window')} | {_fmt(row.get('mean_ic'))} |"
        )
    lines.append(f"\nSpread summary: `{seed_var.get('spread')}`\n")

    lines.append("### A0 ↔ S daily score Spearman\n")
    for h, blob in sorted((score_corr or {}).items(), key=lambda kv: str(kv[0])):
        lines.append(f"- h={h}: mean={_fmt(blob.get('mean_spearman'))} median={_fmt(blob.get('median_spearman'))} n_days={blob.get('n_days')}")
    lines.append("")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def print_stdout_summary(
    sig_keep: dict,
    seq_criteria: dict,
    sig_ablation: dict,
    seq_s: dict,
    seq_blend: dict,
    gru_info: dict,
    seed_var: dict,
) -> None:
    print("\n========== PHASE E SUMMARY ==========", flush=True)
    print("RUN ORDER: Phase E before queued D.2 universe test; both slices reported.", flush=True)
    print(f"GRU n_params={gru_info.get('n_params')} max_epochs={gru_info.get('max_epochs')} device={gru_info.get('device')}", flush=True)
    print("arm | universe | window | h=7 Δ | h=10 Δ | verdict", flush=True)
    for uni in ("top20", "pit120"):
        sv = (sig_keep.get("universes") or {}).get(uni) or {}
        d7 = ((sig_ablation.get(7) or {}).get("by_universe") or {}).get(uni) or {}
        d10 = ((sig_ablation.get(10) or {}).get("by_universe") or {}).get(uni) or {}
        print(
            f"SIG | {uni} | trail18m | {_fmt(d7.get('delta_trail18m'))} | {_fmt(d10.get('delta_trail18m'))} | SIG_{uni}={sv.get('verdict')}",
            flush=True,
        )
        print(
            f"SIG | {uni} | full     | {_fmt(d7.get('delta_full'))} | {_fmt(d10.get('delta_full'))} |",
            flush=True,
        )
        bv = (seq_criteria.get("universes") or {}).get(uni) or {}
        s7 = ((seq_s.get(7) or {}).get("by_universe") or {}).get(uni) or {}
        s10 = ((seq_s.get(10) or {}).get("by_universe") or {}).get(uni) or {}
        b7 = ((seq_blend.get(7) or {}).get("by_universe") or {}).get(uni) or {}
        b10 = ((seq_blend.get(10) or {}).get("by_universe") or {}).get(uni) or {}
        print(
            f"S   | {uni} | trail18m | {_fmt(s7.get('delta_trail18m'))} | {_fmt(s10.get('delta_trail18m'))} | S_{uni}={bv.get('S_viable')}",
            flush=True,
        )
        print(
            f"BLN | {uni} | trail18m | {_fmt(b7.get('delta_trail18m'))} | {_fmt(b10.get('delta_trail18m'))} | BLEND_{uni}={bv.get('BLEND_verdict')}",
            flush=True,
        )
        print(
            f"BLN | {uni} | full     | {_fmt(b7.get('delta_full'))} | {_fmt(b10.get('delta_full'))} |",
            flush=True,
        )
    print(f"PER-SEED SPREAD: {seed_var.get('spread')}", flush=True)
    print("=====================================\n", flush=True)
