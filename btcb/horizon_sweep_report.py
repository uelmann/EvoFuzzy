"""Horizon-sweep report and charts for SPREAD-LS."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    HORIZON_SWEEP_CRITERION,
    HORIZON_SWEEP_HS,
    PHASE2C_NULL_GATE,
    PHASE3_FUNDING_CAVEAT,
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
        return f"{100.0 * float(x):.{nd}f}%"
    except Exception:
        return str(x)


def _null_table(null: dict) -> list[str]:
    ric = null.get("rankic") or {}
    lines = [
        f"Bias pass={ric.get('bias_pass')}; skill {ric.get('verdict')}; "
        f"{ric.get('n_exceed')}/6 exceed p95; Stouffer z={_fmt(ric.get('stouffer_z'))}. "
        f"passed={null.get('passed')}. Failure = PARKED, no override, no retest with different folds.",
        "",
        "| fold | n | null mean | SD | 95th | real | bias_ok | exceeds_p95 |",
        "|------|---|-----------|----|------|------|---------|-------------|",
    ]
    for c in null.get("rankic_cells") or []:
        lines.append(
            f"| {c.get('fold_id')} | {c.get('n')} | {_fmt(c.get('mean'), 4)} | {_fmt(c.get('sd'), 4)} "
            f"| {_fmt(c.get('p95'), 4)} | {_fmt(c.get('real_rankic'), 4)} | {c.get('bias_ok')} "
            f"| {c.get('exceeds_p95')} |"
        )
    return lines


def write_horizon_sweep(
    path: Path,
    *,
    choice: dict,
    books: dict,
    nulls: dict,
    extra: dict,
) -> str:
    ch = choice or {}
    lines = [
        "# BTC-BEATER SPREAD-LS — horizon sweep (h=3 / 7 / 14 / 30)",
        "",
        "**BACKTEST ONLY.** Twin heads trained at h=3 and h=7 only. h=14/h=30 caches reused "
        "byte-identical. Production U = floored PIT top-100 dollar-volume. β-matched. "
        "FUNDING=OFF (3.b has not run). CPU only, zero GPU. COMBO untouched.",
        "",
        "## Pre-registered reading (verbatim, before results)",
        "",
        f"> {HORIZON_SWEEP_CRITERION}",
        "",
        "## Repowered skill null (verbatim)",
        "",
        f"> {PHASE2C_NULL_GATE}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Funding caveat (verbatim)",
        "",
        f"> {PHASE3_FUNDING_CAVEAT}",
        "",
        "## Mechanical choice",
        "",
        f"- **Chosen production horizon = h={ch.get('chosen_h')}** "
        f"(fallback={ch.get('fallback')}; incumbent=h={ch.get('incumbent')})",
        f"- incumbent full={_fmt(ch.get('inc_full'))} so challengers need ≥ {_fmt(ch.get('need_full'))}; "
        f"incumbent trail-18m={_fmt(ch.get('inc_trail'))} so challengers need ≥ {_fmt(ch.get('need_trail'))}",
        f"- qualifiers={ch.get('qualifiers')}",
        "",
    ]
    det = ch.get("details") or {}
    lines += [
        "| h | judged | null passed | full | trail | full_ok | trail_ok | qualifies |",
        "|---|--------|-------------|------|-------|---------|----------|-----------|",
    ]
    for h in HORIZON_SWEEP_HS:
        d = det.get(h) or det.get(str(h)) or {}
        lines.append(
            f"| {h} | {d.get('judged')} | {d.get('null_passed')} | {_fmt(d.get('full'))} "
            f"| {_fmt(d.get('trail'))} | {d.get('full_ok')} | {d.get('trail_ok')} | {d.get('qualifies')} |"
        )
    lines += ["", "## Per-trade economics (slot-level round-trips, β-matched, funding-off)", "", 
              "| h | avg hold (d) | RT / year | gross edge (bps) | cost / RT (bps) | net edge (bps) | ann cost drag | n RT |",
              "|---|--------------|-----------|------------------|-----------------|----------------|---------------|------|"]
    for h in HORIZON_SWEEP_HS:
        b = books.get(h) or books.get(str(h)) or {}
        e = b.get("econ") or {}
        lines.append(
            f"| {h} | {_fmt(e.get('avg_hold_days'), 1)} | {_fmt(e.get('round_trips_per_year'), 1)} "
            f"| {_fmt(e.get('avg_gross_bps'), 1)} | {_fmt(e.get('avg_cost_bps'), 1)} "
            f"| {_fmt(e.get('avg_net_bps'), 1)} | {_fmt(e.get('ann_cost_drag'), 4)} "
            f"| {e.get('n_round_trips')} |"
        )
    lines += [
        "",
        "Round-trip = one name entering and later leaving a single tranche slot. "
        "Gross edge is the signed simple return of that name over the hold, in bps. "
        "Cost / RT = 2 × one-way (20 bps long, 16 bps short). Open trades at the end are excluded. "
        "Ann cost drag = mean(daily book cost) × 365 (NAV return units).",
        "",
        "## Books (β-matched, top-100 DV, funding-off)",
        "",
        "| h | net Sharpe | trail-18m | MaxDD | total | #long | #short | shortable | % inc. short | ann TO | squeeze mean | β vs BTC | RankIC |",
        "|---|------------|-----------|-------|-------|-------|--------|-----------|--------------|--------|--------------|----------|--------|",
    ]
    for h in HORIZON_SWEEP_HS:
        b = books.get(h) or books.get(str(h)) or {}
        lines.append(
            f"| {h} | {_fmt(b.get('net_sharpe'))} | {_fmt(b.get('net_sharpe_trail18m'))} "
            f"| {_pct(b.get('maxdd'))} | {_pct(b.get('book_total'))} "
            f"| {_fmt(b.get('avg_n_long'), 2)} | {_fmt(b.get('avg_n_short'), 2)} "
            f"| {_fmt(b.get('avg_shortable'), 1)} | {_pct(b.get('pct_incomplete_short'))} "
            f"| {_fmt(b.get('ann_turnover'), 2)} | {_fmt(b.get('squeeze_mean'))} "
            f"| {_fmt(b.get('realized_beta_full'))} | {_fmt(b.get('rankic'), 4)} |"
        )
    lines += ["", "## Per-cycle net Sharpe", "",
              "| h | cycle | n | net Sharpe | MaxDD | #long | #short |",
              "|---|-------|---|------------|-------|-------|--------|"]
    for h in HORIZON_SWEEP_HS:
        b = books.get(h) or books.get(str(h)) or {}
        for cyc, c in (b.get("cycles") or {}).items():
            lines.append(
                f"| {h} | {cyc} | {c.get('n')} | {_fmt(c.get('net_sharpe'))} | {_pct(c.get('maxdd'))} "
                f"| {_fmt(c.get('avg_n_long'), 2)} | {_fmt(c.get('avg_n_short'), 2)} |"
            )

    for h in (3, 7):
        ng = nulls.get(h) or nulls.get(str(h)) or {}
        lines += ["", f"## §2 null — spread per-date RankIC (h={h}, judged signal)", ""]
        if ng.get("skipped"):
            lines.append(f"Not run: {ng.get('reason')}")
        else:
            lines += _null_table(ng)

    ng14 = nulls.get(14) or nulls.get("14") or {}
    lines += [
        "",
        "## §2 null — h=14 (reused from Phase 2.c, not re-run)",
        "",
        f"passed={ng14.get('passed')}; verdict={ (ng14.get('rankic') or {}).get('verdict') }; "
        f"{(ng14.get('rankic') or {}).get('n_exceed')}/6; "
        f"Stouffer z={_fmt((ng14.get('rankic') or {}).get('stouffer_z'))}. "
        f"{ng14.get('reason') or ''}",
        "",
        "## h=30 null",
        "",
        f"{(nulls.get(30) or nulls.get('30') or {}).get('reason')}",
        "",
        "## Cache / reuse",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` n_files={extra.get('pred_n_files')} "
        f"(expected `{extra.get('pred_sha256_expected')}`)",
        f"- new-head pred dir sha256 = `{extra.get('new_pred_sha256')}` n_files={extra.get('new_pred_n_files')}",
        f"- BTC in book hits (all runs) = {extra.get('btc_hits_total')}",
        f"- GPU={extra.get('gpu_used', False)}. Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}.",
        "",
        "Charts: `charts/btcb_horizon_equity.png`, `charts/btcb_horizon_rankic.png`.",
        "",
        "COMBO untouched (v2.0-combo-final).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_horizon_equity(books: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    colors = {3: "#4C78A8", 7: "#F58518", 14: "#54A24B", 30: "#E45756"}
    for h in HORIZON_SWEEP_HS:
        b = books.get(h) or books.get(str(h)) or {}
        eq = b.get("equity")
        if eq is None or len(eq) == 0:
            continue
        sh = b.get("net_sharpe")
        tr = b.get("net_sharpe_trail18m")
        lab = f"h={h}"
        if sh is not None and np.isfinite(float(sh)):
            lab += f"  Sh={float(sh):.2f}"
        if tr is not None and np.isfinite(float(tr)):
            lab += f"  trail={float(tr):.2f}"
        ax.plot(eq.index, eq.values, lw=1.4, color=colors[h], label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("equity (log)")
    ax.set_title("SPREAD-LS horizon sweep — β-matched, top-100 DV, funding-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_horizon_rankic(series: dict, means: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), constrained_layout=True, gridspec_kw={"height_ratios": [1.2, 2.2]})
    colors = {3: "#4C78A8", 7: "#F58518", 14: "#54A24B", 30: "#E45756"}
    axb = axes[0]
    hs = [h for h in HORIZON_SWEEP_HS if h in means or str(h) in means]
    vals = [float(means.get(h, means.get(str(h), float("nan")))) for h in hs]
    axb.bar([str(h) for h in hs], vals, color=[colors[h] for h in hs])
    axb.axhline(0.0, color="0.5", lw=0.8)
    axb.set_ylabel("mean RankIC")
    axb.set_title("Spread per-date RankIC by horizon (last-fold-wins, PIT top-100)")
    axb.grid(True, axis="y", alpha=0.3)
    ax = axes[1]
    for h in HORIZON_SWEEP_HS:
        s = series.get(h, series.get(str(h)))
        if not isinstance(s, pd.Series) or s.empty:
            continue
        ax.plot(s.index, s.values, lw=0.6, alpha=0.25, color=colors[h])
        rs = s.rolling(30, min_periods=10).mean()
        mu = means.get(h, means.get(str(h)))
        lab = f"h={h} 30d"
        if mu is not None and np.isfinite(float(mu)):
            lab += f"  mean={float(mu):.3f}"
        ax.plot(rs.index, rs.values, lw=1.4, color=colors[h], label=lab)
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("RankIC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
