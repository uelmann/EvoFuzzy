"""Write FuzzyX report. Mechanical verdict only."""

from __future__ import annotations

import json
from pathlib import Path


def write_report(
    path: Path,
    *,
    mode: str,
    gates: list[dict],
    bias_folds: list[dict],
    book: dict,
    a0_delta: dict | None,
    verdict: dict,
    folds: list[dict],
    rules: list[str],
    n_params: int,
    notes: list[str],
    title: str = "FuzzyX-v1b report",
    addendum: str = "reports/fuzzyx_addendum_v1b.md",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leak = "PASS" if verdict.get("leak_ok") else "FAIL"
    bias = "PASS" if verdict.get("bias_ok") else "FAIL"
    skill = "PASS" if verdict.get("skill_ok") else "FAIL"
    lines = [
        f"# {title}",
        "",
        "**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. "
        f"Does not replace COMBO / A0. Addendum: `{addendum}`.",
        "",
        f"**Mode:** `{mode}`",
        f"**Verdict:** **{verdict.get('verdict')}**",
        f"**Params:** {n_params}",
        "",
        "## Keep rule (verbatim)",
        "",
        f"> See `{addendum}`. VIABLE only if leakage, shuffle-bias on mean weekly "
        "net PnL (per-fold weights), full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs "
        "A0 Sleeve A when A0 preds exist. LOCAL-RESTRICTED cannot be official VIABLE.",
        "",
        "## Gates",
        "",
        f"| clause | result |",
        f"|---|---|",
        f"| (i) leakage | {leak} |",
        f"| (ii) shuffle-bias | {bias} |",
        f"| (iii) Sharpe ≥ 0 | {skill} ({verdict.get('net_sharpe_weekly'):.3f} weekly) |",
        f"| (iv) vs A0 | {'SKIP' if verdict.get('vs_a0_skip') else ('PASS' if verdict.get('vs_a0_ok') else 'FAIL')} |",
        "",
    ]
    for g in gates:
        lines.append(f"- `{g.get('name')}`: **{'PASS' if g.get('passed') else 'FAIL'}** `{g}`")
    lines += ["", "## Shuffle-bias folds", ""]
    for b in bias_folds:
        lines.append(f"- `{b}`")
    lines += ["", "## Book", "", f"```json\n{json.dumps(book, indent=2, default=str)}\n```", ""]
    if a0_delta:
        lines += ["## vs A0", "", f"```json\n{json.dumps(a0_delta, indent=2, default=str)}\n```", ""]
    lines += ["## Folds", ""]
    for f in folds:
        lines.append(
            f"- fold {f.get('fold_id')} {f.get('val_start')}→{f.get('val_end')} "
            f"status={f.get('status')} best_epoch={f.get('best_epoch')} "
            f"hold_loss={f.get('best_val')} n_reb={f.get('n_reb')}"
        )
    lines += ["", "## Sample rules (eval argmax literals)", ""]
    for r in rules[:16]:
        lines.append(f"- `{r}`")
    if notes:
        lines += ["", "## Notes", ""]
        for n in notes:
            lines.append(f"- {n}")
    path.write_text("\n".join(lines) + "\n")
