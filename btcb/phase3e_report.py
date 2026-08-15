"""Phase 3.e forensics report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import DEATH_CONVENTION, PHASE2C_PRED_SHA256, PHASE3E_OUTCOME


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


def write_phase3e(
    path: Path,
    *,
    verdict: dict,
    fr: dict,
    side: dict,
    tier: dict,
    conc: dict,
    wf: dict,
    rankic: dict,
    funding: dict,
    never: dict,
    extra: dict,
) -> str:
    vlab = verdict.get("label")
    if verdict.get("confirmed"):
        resume = (
            "Official SPREAD-LS record **RESUMES as BOOK-HYBRID (funding-on)**. "
            "Binance-priced numbers are canonical for all future phases. "
            "The 3.c suspension footnote is replaced by this forensic decomposition."
        )
    else:
        resume = (
            "Official SPREAD-LS record **stays SUSPENDED**. "
            "The next phase MUST re-derive the book on Binance-only pricing before anything else."
        )

    yr_lines = [
        "| year | n | Sharpe ON | Sharpe OFF | Sharpe CMC | ΔSh funding | ΔSh repricing | funding PnL | repricing PnL |",
        "|------|---|-----------|------------|------------|-------------|---------------|-------------|---------------|",
    ]
    for y, r in sorted((fr.get("by_year") or {}).items()):
        yr_lines.append(
            f"| {y} | {r.get('n')} | {_fmt(r.get('sharpe_bn_on'))} | {_fmt(r.get('sharpe_bn_off'))} "
            f"| {_fmt(r.get('sharpe_cmc'))} | {_fmt(r.get('d_sharpe_funding'))} "
            f"| {_fmt(r.get('d_sharpe_repricing'))} | {_fmt(r.get('funding_pnl'), 4)} "
            f"| {_fmt(r.get('repricing_pnl'), 4)} |"
        )

    top_lines = [
        "| date | id | symbol | side | w | r_cmc | r_bn | Δr | w·Δr | class | rank |",
        "|------|----|--------|------|---|-------|------|----|------|-------|------|",
    ]
    id_to_sym = extra.get("id_to_sym") or {}
    for r in conc.get("top_rows") or []:
        sid = str(id_to_sym.get(str(r.get("id")), id_to_sym.get(r.get("id"), "")))
        top_lines.append(
            f"| {r.get('date')} | {r.get('id')} | {sid} | {r.get('side')} | {_fmt(r.get('w'), 4)} "
            f"| {_fmt(r.get('r_cmc'), 4)} | {_fmt(r.get('r_bn'), 4)} | {_fmt(r.get('d_r'), 4)} "
            f"| {_fmt(r.get('contrib_diff'), 5)} | {r.get('stale_class')} | {r.get('rank')} |"
        )

    cls = conc.get("by_class") or {}
    buck = rankic.get("buckets") or []
    buck_lines = [
        "| Q | n | mean excess BN | mean excess CMC |",
        "|---|---|----------------|-----------------|",
    ]
    for b in buck:
        buck_lines.append(
            f"| {b.get('quintile')} | {b.get('n')} | {_fmt(b.get('mean_ex_bn'), 4)} | {_fmt(b.get('mean_ex_cmc'), 4)} |"
        )

    fund_y = [
        "| year | n name-days | funding PnL | mean rate |",
        "|------|-------------|-------------|-----------|",
    ]
    for y, r in sorted((funding.get("by_year") or {}).items()):
        fund_y.append(
            f"| {y} | {r.get('n_name_days')} | {_fmt(r.get('funding_pnl'), 4)} | {_fmt(r.get('mean_rate'), 6)} |"
        )

    lines = [
        "# BTC-BEATER Phase 3.e — pricing-gap forensics",
        "",
        "**ANALYSIS ONLY.** Same 3.c positions. No signal or book changes. CPU only, zero GPU. "
        "COMBO untouched. 3.c artifacts reused byte-identical.",
        "",
        "## Pre-registered outcomes (verbatim, frozen before results)",
        "",
        f"> {PHASE3E_OUTCOME}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Mechanical verdict",
        "",
        f"- **{vlab}**",
        f"- RankIC Binance full `{_fmt(verdict.get('rankic_bn_full'), 4)}` vs same-names CMC "
        f"`{_fmt(verdict.get('rankic_cmc_full'), 4)}` (Δ `{_fmt(verdict.get('d_full'), 4)}`; "
        f"need ≥ −{_fmt(verdict.get('need_tol'), 2)}; pass={verdict.get('pass_full')})",
        f"- RankIC Binance trail-18m `{_fmt(verdict.get('rankic_bn_trail'), 4)}` vs CMC "
        f"`{_fmt(verdict.get('rankic_cmc_trail'), 4)}` (Δ `{_fmt(verdict.get('d_trail'), 4)}`; "
        f"pass={verdict.get('pass_trail')})",
        f"- {resume}",
        "",
        "Mechanical, no post-hoc adjustment.",
        "",
        "## Identity",
        "",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` (expected `{PHASE2C_PRED_SHA256}`)",
        f"- Position-log sha256 = `{extra.get('position_sha256')}`",
        f"- CMC panel sha256 = `{extra.get('cmc_panel_sha256')}` (read-only assert {extra.get('cmc_readonly_ok')})",
        f"- BOOK-BINANCE-ONLY Sharpe ON `{_fmt(fr.get('sharpe_bn_on'))}` / OFF `{_fmt(fr.get('sharpe_bn_off'))}` "
        f"/ same-days CMC `{_fmt(fr.get('sharpe_cmc_sub'))}`",
        f"- GPU used = `{extra.get('gpu_used')}`",
        "",
        "## (a) Funding vs repricing",
        "",
        f"ΔSharpe(funding) = `{_fmt(fr.get('d_sharpe_funding'))}` "
        f"(ON `{_fmt(fr.get('sharpe_bn_on'))}` − OFF `{_fmt(fr.get('sharpe_bn_off'))}`).",
        f"ΔSharpe(repricing) = `{_fmt(fr.get('d_sharpe_repricing'))}` "
        f"(OFF `{_fmt(fr.get('sharpe_bn_off'))}` − CMC `{_fmt(fr.get('sharpe_cmc_sub'))}`).",
        "",
        *yr_lines,
        "",
        "## (b) By side (repricing PnL = Σ w·Δr on replayable name-days)",
        "",
        "| side | name-days | PnL diff sum | share of gap |",
        "|------|-----------|--------------|--------------|",
        f"| long | {(side.get('long') or {}).get('n_name_days')} | {_fmt((side.get('long') or {}).get('pnl_diff_sum'), 4)} | {_pct((side.get('long') or {}).get('share_of_gap'))} |",
        f"| short | {(side.get('short') or {}).get('n_name_days')} | {_fmt((side.get('short') or {}).get('pnl_diff_sum'), 4)} | {_pct((side.get('short') or {}).get('share_of_gap'))} |",
        "",
        "## (c) By PIT rank tier",
        "",
        "| tier | name-days | PnL diff sum | share of gap |",
        "|------|-----------|--------------|--------------|",
    ]
    for name in ("1-30", "31-60", "61-100"):
        r = (tier or {}).get(name) or {}
        lines.append(
            f"| {name} | {r.get('n_name_days')} | {_fmt(r.get('pnl_diff_sum'), 4)} | {_pct(r.get('share_of_gap'))} |"
        )
    lines += [
        "",
        "## (d) Concentration (top-30 name-days by |w·Δr|)",
        "",
        f"Top-30 share of signed gap = `{_pct(conc.get('top_share_of_gap'))}`; "
        f"share of |contrib| = `{_pct(conc.get('top_share_of_abs'))}`.",
        "",
        *top_lines,
        "",
        "## Stale-price classification",
        "",
        f"STALE share of signed repricing gap = `{_pct(conc.get('stale_share_of_gap'))}`.",
        "",
        "| class | n | PnL diff sum | share of gap | share of |contrib| |",
        "|-------|---|--------------|--------------|------------------|",
        f"| STALE | {(cls.get('STALE') or {}).get('n')} | {_fmt((cls.get('STALE') or {}).get('pnl_diff_sum'), 4)} | {_pct((cls.get('STALE') or {}).get('share_of_gap'))} | {_pct((cls.get('STALE') or {}).get('share_of_abs'))} |",
        f"| LEVEL-DIFF | {(cls.get('LEVEL-DIFF') or {}).get('n')} | {_fmt((cls.get('LEVEL-DIFF') or {}).get('pnl_diff_sum'), 4)} | {_pct((cls.get('LEVEL-DIFF') or {}).get('share_of_gap'))} | {_pct((cls.get('LEVEL-DIFF') or {}).get('share_of_abs'))} |",
        f"| OTHER | {(cls.get('OTHER') or {}).get('n')} | {_fmt((cls.get('OTHER') or {}).get('pnl_diff_sum'), 4)} | {_pct((cls.get('OTHER') or {}).get('share_of_gap'))} | {_pct((cls.get('OTHER') or {}).get('share_of_abs'))} |",
        "",
        "## Gap waterfall (PnL: BN_on − CMC_sub)",
        "",
        f"- funding `{_fmt(wf.get('funding_pnl'), 4)}` (`{_pct(wf.get('pct_funding'))}`)",
        f"- stale repricing `{_fmt(wf.get('stale_pnl'), 4)}` (`{_pct(wf.get('pct_stale'))}`)",
        f"- diffuse repricing `{_fmt(wf.get('diffuse_pnl'), 4)}` (`{_pct(wf.get('pct_diffuse'))}`)",
        f"- total `{_fmt(wf.get('total_pnl_gap'), 4)}`",
        "",
        "## RankIC of frozen spread vs h=14 excess (replayable names, same-names CMC)",
        "",
        "| window | RankIC BN | RankIC CMC | Δ | n dates |",
        "|--------|-----------|------------|---|---------|",
        f"| full | {_fmt((rankic.get('binance') or {}).get('full'), 4)} | {_fmt((rankic.get('cmc_same_names') or {}).get('full'), 4)} | {_fmt(verdict.get('d_full'), 4)} | {(rankic.get('binance') or {}).get('n_full')} |",
        f"| trail-18m | {_fmt((rankic.get('binance') or {}).get('trail18m'), 4)} | {_fmt((rankic.get('cmc_same_names') or {}).get('trail18m'), 4)} | {_fmt(verdict.get('d_trail'), 4)} | {(rankic.get('binance') or {}).get('n_trail')} |",
        "",
        "Quintile bucket curve (mean h=14 excess):",
        "",
        *buck_lines,
        "",
        "## Structural funding",
        "",
        f"Held-short funding `{_fmt(funding.get('held_bps_day'), 2)}` bps/day vs shortable-universe "
        f"`{_fmt(funding.get('universe_bps_day'), 2)}` (Δ `{_fmt(funding.get('delta_bps_day'), 2)}`). "
        f"Total funding PnL `{_fmt(funding.get('funding_pnl_total'), 4)}`.",
        "",
        *fund_y,
        "",
        "## Never-listed longs (CMC book contribution)",
        "",
        f"{never.get('n_names')} names, {never.get('n_name_days')} name-days, CMC PnL "
        f"`{_fmt(never.get('pnl_cmc'), 4)}` = `{_pct(never.get('share'))}` of CMC-book name-day gross.",
        "",
        "Charts: `charts/btcb_phase3e_gap_waterfall.png`, `charts/btcb_phase3e_rankic.png`.",
        "",
        "## Ledger",
        "",
        resume,
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def plot_gap_waterfall(wf: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["funding", "stale", "diffuse"]
    vals = [float(wf.get("funding_pnl") or 0), float(wf.get("stale_pnl") or 0), float(wf.get("diffuse_pnl") or 0)]
    colors = ["#F58518", "#E45756", "#4C78A8"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.7)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PnL gap (BN − CMC)")
    ax.set_title(
        f"3.e gap waterfall  funding={_pct(wf.get('pct_funding'))}  "
        f"stale={_pct(wf.get('pct_stale'))}  diffuse={_pct(wf.get('pct_diffuse'))}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rankic_bars(rankic: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bn = rankic.get("binance") or {}
    cmc = rankic.get("cmc_same_names") or {}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    x = np.arange(2)
    w = 0.35
    axes[0].bar(x - w / 2, [bn.get("full") or 0, bn.get("trail18m") or 0], width=w, color="#4C78A8", label="Binance")
    axes[0].bar(x + w / 2, [cmc.get("full") or 0, cmc.get("trail18m") or 0], width=w, color="#54A24B", label="CMC same-names")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["full", "trail-18m"])
    axes[0].set_ylabel("mean RankIC")
    axes[0].set_title("Spread RankIC: Binance vs CMC excess h=14")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    buckets = rankic.get("buckets") or []
    if buckets:
        qs = [b.get("quintile") for b in buckets]
        axes[1].plot(qs, [b.get("mean_ex_bn") for b in buckets], marker="o", color="#4C78A8", label="BN excess")
        axes[1].plot(qs, [b.get("mean_ex_cmc") for b in buckets], marker="o", color="#54A24B", label="CMC excess")
        axes[1].axhline(0.0, color="0.5", lw=0.8)
        axes[1].set_xlabel("spread quintile (1=bottom)")
        axes[1].set_ylabel("mean h=14 excess")
        axes[1].set_title("Quintile bucket curve")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def update_ledger_3e(path: Path, *, confirmed: bool, verdict: dict, extra: dict, hybrid: dict, cmc: dict) -> str:
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER SPREAD-LS (Phase 3.c Binance replay)"
    if confirmed:
        block = [
            "",
            marker,
            "",
            "Production book config: β-matched, h=14, floored PIT top-100 dollar-volume. "
            "Positions from the 2.c spread cache (signals not recomputed). "
            f"COMBO overlap corr remains {extra.get('combo_corr', 0.157)} for allocation. "
            "MASTER combination book is out of scope (PI).",
            "",
            "**OFFICIAL SPREAD-LS = BOOK-HYBRID (funding-on).** Resumed by Phase 3.e **SIGNAL-CONFIRMED**. "
            "Binance-priced numbers are canonical for all future phases.",
            "",
            f"BOOK-HYBRID Sharpe `{_fmt(hybrid.get('net_sharpe') or extra.get('hybrid_sharpe'))}` / trail "
            f"`{_fmt(hybrid.get('net_sharpe_trail18m') or extra.get('hybrid_trail'))}`. "
            f"RankIC BN `{_fmt(verdict.get('rankic_bn_full'), 4)}` vs same-names CMC "
            f"`{_fmt(verdict.get('rankic_cmc_full'), 4)}` (Δ `{_fmt(verdict.get('d_full'), 4)}`).",
            "",
            f"Footnote: funding-off CMC BOOK-CMC Sharpe `{_fmt(cmc.get('net_sharpe') or extra.get('cmc_sharpe'))}` "
            "is **deprecated** as of Phase 3.e (signal confirmed on Binance returns; gap is execution/pricing-level).",
            "",
        ]
    else:
        block = [
            "",
            marker,
            "",
            "Production book config: β-matched, h=14, floored PIT top-100 dollar-volume. "
            "Positions from the 2.c spread cache (signals not recomputed). "
            f"COMBO overlap corr remains {extra.get('combo_corr', 0.157)} for allocation. "
            "MASTER combination book is out of scope (PI).",
            "",
            "**OFFICIAL SPREAD-LS RECORD SUSPENDED.** Phase 3.e **SIGNAL-PARTLY-ARTIFACT**. "
            "The next phase MUST re-derive the book on Binance-only pricing before anything else.",
            "",
            f"RankIC BN `{_fmt(verdict.get('rankic_bn_full'), 4)}` vs same-names CMC "
            f"`{_fmt(verdict.get('rankic_cmc_full'), 4)}` (Δ `{_fmt(verdict.get('d_full'), 4)}`; "
            f"trail Δ `{_fmt(verdict.get('d_trail'), 4)}`).",
            "",
            f"BOOK-CMC (funding-off, reference) Sharpe `{_fmt(cmc.get('net_sharpe') or extra.get('cmc_sharpe'))}`. "
            f"BOOK-HYBRID (unofficial) Sharpe `{_fmt(hybrid.get('net_sharpe') or extra.get('hybrid_sharpe'))}`.",
            "",
        ]
    start = text.find(marker)
    if start < 0:
        new = text.rstrip() + "\n" + "\n".join(block)
    else:
        rest = text[start + len(marker) :]
        nxt = rest.find("\n## ")
        if nxt >= 0:
            end = start + len(marker) + nxt
            new = text[:start].rstrip() + "\n" + "\n".join(block) + rest[nxt:]
        else:
            new = text[:start].rstrip() + "\n" + "\n".join(block)
    path.write_text(new)
    return "\n".join(block)
