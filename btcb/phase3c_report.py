"""Phase 3.c Binance-replay report and charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcb.constants import (
    DEATH_CONVENTION,
    PHASE2C_PRED_SHA256,
    PHASE3C_BETA_MATCH_DESIGNATION,
    PHASE3C_HOUSE_RULE,
    PHASE3C_MASTER_NOTE,
    PHASE3C_VALIDATION,
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


def _cycle_cell(book: dict, name: str) -> str:
    c = (book.get("cycles") or {}).get(name) or {}
    return _fmt(c.get("net_sharpe"))


def _book_row(name: str, b: dict, fund_key: str = "funding_total_pnl") -> str:
    fe = b.get("forced_exits") or {}
    fc = b.get("forced_covers") or {}
    fund = b.get(fund_key, 0.0)
    return (
        f"| {name} | {_fmt(b.get('net_sharpe'))} | {_fmt(b.get('net_sharpe_trail18m'))} "
        f"| {_cycle_cell(b, '2019-20')} | {_cycle_cell(b, '2021')} | {_cycle_cell(b, '2022')} "
        f"| {_cycle_cell(b, '2023-24')} | {_cycle_cell(b, '2025-26')} "
        f"| {_pct(b.get('maxdd'))} | {_fmt(fund, 4)} | {_fmt(b.get('ann_turnover'), 2)} "
        f"| {fe.get('n_events', b.get('forced_exit_events', ''))} "
        f"| {fc.get('n_events', b.get('forced_cover_events', ''))} |"
    )


def _cov_year_table(side: dict) -> list[str]:
    by = side.get("by_year") or {}
    lines = [
        f"**{side.get('label')}:** {_pct(side.get('pct_replayable'))} "
        f"({side.get('n_replayable')}/{side.get('n_name_days')} name-days). "
        f"Never-listed names: {side.get('n_never_listed_names', 0)}.",
        "",
        "| year | name-days | replayable | % |",
        "|------|-----------|------------|---|",
    ]
    for y in sorted(by.keys()):
        r = by[y]
        lines.append(
            f"| {y} | {r.get('n_name_days')} | {r.get('n_replayable')} | {_pct(r.get('pct_replayable'))} |"
        )
    return lines


def write_phase3c(
    path: Path,
    *,
    coverage: dict,
    books: dict,
    validation: dict,
    extra: dict,
    squeeze_cmc: list,
    squeeze_hyb: list,
    top_disagreements: list,
    discrepancy: dict | None,
) -> str:
    cmc = books.get("cmc") or {}
    hyb = books.get("hybrid") or {}
    bn = books.get("binance_only") or {}
    validated = bool(validation.get("validated"))
    verdict = "PRICES ARE VALIDATED" if validated else "PRICES ARE NOT VALIDATED"
    if validated:
        official = (
            f"BOOK-HYBRID (funding-on) is the OFFICIAL SPREAD-LS record. "
            f"Funding-off CMC numbers are deprecated (ledger footnote)."
        )
    else:
        official = (
            "Official SPREAD-LS record is SUSPENDED. No improvement work proceeds "
            "until the pricing gap is understood."
        )
    never_l = (coverage.get("long") or {}).get("never_listed") or []
    never_s = (coverage.get("short") or {}).get("never_listed") or []

    lines = [
        "# BTC-BEATER Phase 3.c — Binance replay of SPREAD-LS",
        "",
        "**BACKTEST ONLY.** Same 2.c positions (β-matched, h=14, floored PIT top-100 DV). "
        "Only pricing and native funding change. CPU only, zero GPU. COMBO untouched. "
        "No MASTER book. Phase 3.b replaced.",
        "",
        "## Addenda (verbatim, frozen before results)",
        "",
        "### 1. β-match designation with post-observation disclosure",
        "",
        f"> {PHASE3C_BETA_MATCH_DESIGNATION}",
        "",
        "### 2. House-rule correction (record only)",
        "",
        f"> {PHASE3C_HOUSE_RULE}",
        "",
        "### 3. MASTER removed from scope",
        "",
        f"> {PHASE3C_MASTER_NOTE}",
        "",
        "## Death-in-position convention (verbatim)",
        "",
        f"> {DEATH_CONVENTION}",
        "",
        "## Pre-registered validation (verbatim, before results)",
        "",
        f"> {PHASE3C_VALIDATION}",
        "",
        "## Mechanical verdict",
        "",
        f"- **{verdict}**",
        f"- Replayable-subset daily-PnL correlation CMC↔Binance = `{_fmt(validation.get('corr'), 4)}` "
        f"(need ≥ {_fmt(validation.get('need_corr'))})",
        f"- BOOK-BINANCE-ONLY net Sharpe = `{_fmt(validation.get('sharpe_binance_only'))}` vs "
        f"same-days CMC `{_fmt(validation.get('sharpe_cmc_subset'))}` "
        f"(need ≥ `{_fmt(validation.get('need_sharpe'))}`; gap `{_fmt(validation.get('sharpe_gap'))}`)",
        f"- n_days = {validation.get('n_days')}",
        f"- **{official}**",
        "",
        "Mechanical, no post-hoc adjustment.",
        "",
        "## Position identity",
        "",
        f"- Position-log sha256 = `{extra.get('position_sha256')}`",
        f"- BOOK-CMC vs engine max |daily PnL| = `{_fmt(extra.get('max_abs_daily_diff'), 12)}` "
        f"(need ≤ 1e-6)",
        f"- BOOK-CMC Sharpe `{_fmt(cmc.get('net_sharpe'))}` vs 3.x β-matched `{_fmt(extra.get('ref_sharpe'))}` "
        f"(n_days={cmc.get('n_days')} vs {extra.get('ref_n_days')}; start {cmc.get('start')} end {cmc.get('end')})",
        f"- 2.c pred cache sha256 = `{extra.get('pred_sha256')}` n_files={extra.get('pred_n_files')} "
        f"(expected `{PHASE2C_PRED_SHA256}`)",
        f"- BTC in book hits = {extra.get('btc_hits', 0)}",
        "",
        "## Coverage",
        "",
    ]
    lines += _cov_year_table(coverage.get("long") or {})
    lines += [""]
    lines += _cov_year_table(coverage.get("short") or {})
    lines += [
        "",
        f"Hybrid flagged (CMC-priced) share of name-days = `{_pct(extra.get('hybrid_flagged_share'))}`.",
        f"Spot symbols downloaded this run = {extra.get('n_spot_downloaded', 0)}; "
        f"reused = {extra.get('n_spot_reused', 0)}; attempted = {extra.get('n_spot_attempted', 0)}.",
        f"Funding events applied = {extra.get('funding_events')}; "
        f"short name-days with missing funding (treated as 0) = {extra.get('missing_funding_name_days')}.",
        "",
        "### Never-listed / unmapped names (kept at CMC in hybrid, flagged)",
        "",
        f"Longs never listed on Binance spot: {len(never_l)}. "
        f"Shorts never listed on USDT-M perp: {len(never_s)}.",
        "",
    ]
    if never_l:
        lines.append("Long sample: " + ", ".join(
            f"{x.get('symbol')}({x.get('id')})" for x in never_l[:25]
        ))
        lines.append("")
    if never_s:
        lines.append("Short sample: " + ", ".join(
            f"{x.get('symbol')}({x.get('id')})" for x in never_s[:25]
        ))
        lines.append("")

    lines += [
        "## Three books (identical positions)",
        "",
        "| book | Sharpe full | trail-18m | 2019-20 | 2021 | 2022 | 2023-24 | 2025-26 | MaxDD | funding PnL | ann TO | forced exits | forced covers |",
        "|------|-------------|-----------|---------|------|------|---------|---------|-------|-------------|--------|--------------|---------------|",
        _book_row("BOOK-CMC (funding=0)", cmc),
        _book_row("BOOK-HYBRID (funding-on)", hyb),
        _book_row("BOOK-BINANCE-ONLY", bn),
        "",
        f"Hybrid funding share of |gross| = `{_pct(hyb.get('funding_share_of_gross'))}`; "
        f"funding total PnL = `{_fmt(hyb.get('funding_total_pnl'), 4)}`.",
        "",
        "## Squeeze-days (20 largest EW top-100 up-days)",
        "",
        "| date | EW basket | BOOK-CMC | BOOK-HYBRID (funding-in) |",
        "|------|-----------|----------|--------------------------|",
    ]
    hyb_s = {r["date"]: r for r in (squeeze_hyb or [])}
    for r in squeeze_cmc or []:
        d = r.get("date")
        h = hyb_s.get(d) or {}
        lines.append(
            f"| {d} | {_pct(r.get('ew_basket'), 2)} | {_pct(r.get('spread_ls'), 2)} | {_pct(h.get('spread_ls'), 2)} |"
        )
    sl_c = [float(r["spread_ls"]) for r in (squeeze_cmc or []) if np.isfinite(r.get("spread_ls", float("nan")))]
    sl_h = [float(r["spread_ls"]) for r in (squeeze_hyb or []) if np.isfinite(r.get("spread_ls", float("nan")))]
    lines += [
        "",
        f"Squeeze-day mean CMC = `{_pct(float(np.mean(sl_c)) if sl_c else float('nan'), 2)}`; "
        f"hybrid (funding-in) = `{_pct(float(np.mean(sl_h)) if sl_h else float('nan'), 2)}`.",
        "",
        "## Largest single-day price disagreements (top 10 by |w·Δr|)",
        "",
        "| date | id | symbol | side | w | r_cmc | r_bn | Δr | w·Δr |",
        "|------|----|--------|------|---|-------|------|----|------|",
    ]
    for r in top_disagreements or []:
        lines.append(
            f"| {r.get('date')} | {r.get('id')} | {r.get('symbol', '')} | {r.get('side')} | {_fmt(r.get('w'), 4)} "
            f"| {_fmt(r.get('r_cmc'), 4)} | {_fmt(r.get('r_bn'), 4)} | {_fmt(r.get('d_r'), 4)} "
            f"| {_fmt(r.get('contrib_diff'), 5)} |"
        )
    if not top_disagreements:
        lines.append("| — | — | — | — | — | — | — | — | — |")

    if discrepancy and not validated:
        lines += [
            "",
            "## Discrepancy (NOT validated — per year and name-tier)",
            "",
            "| year | n | corr | Sharpe BN | Sharpe CMC | gap | PnL diff sum |",
            "|------|---|------|-----------|------------|-----|--------------|",
        ]
        for y, r in (discrepancy.get("by_year") or {}).items():
            lines.append(
                f"| {y} | {r.get('n')} | {_fmt(r.get('corr'), 4)} | {_fmt(r.get('sharpe_bn'))} "
                f"| {_fmt(r.get('sharpe_cmc'))} | {_fmt(r.get('sharpe_gap'))} | {_fmt(r.get('pnl_diff_sum'), 4)} |"
            )
        lines += [
            "",
            "| PIT rank tier | n | corr | Sharpe BN | Sharpe CMC | gap | PnL diff sum |",
            "|---------------|---|------|-----------|------------|-----|--------------|",
        ]
        for t, r in (discrepancy.get("by_tier") or {}).items():
            lines.append(
                f"| {t} | {r.get('n')} | {_fmt(r.get('corr'), 4)} | {_fmt(r.get('sharpe_bn'))} "
                f"| {_fmt(r.get('sharpe_cmc'))} | {_fmt(r.get('sharpe_gap'))} | {_fmt(r.get('pnl_diff_sum'), 4)} |"
            )

    lines += [
        "",
        "## Ledger",
        "",
        extra.get("ledger_note", ""),
        "",
        f"- GPU={extra.get('gpu_used', False)}. Elapsed s={_fmt(extra.get('elapsed_sec'), 1)}.",
        "- Charts: `charts/btcb_phase3c_hybrid_equity.png`, `charts/btcb_phase3c_pnl_scatter.png`.",
        "- COMBO untouched (v2.0-combo-final). Frozen BTC-BEATER v1 untouched. No MASTER book.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_hybrid_equity(hybrid_eq: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq = hybrid_eq.copy()
    eq.index = pd.DatetimeIndex(pd.to_datetime(eq.index, utc=True)).tz_convert("UTC").normalize()
    dd = eq / eq.cummax() - 1.0
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axes[0].plot(eq.index, eq.values, lw=1.3, color="#4C78A8")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("equity (log)")
    axes[0].set_title("SPREAD-LS BOOK-HYBRID — Binance prices + native funding (β-matched h=14)")
    axes[0].grid(True, alpha=0.3)
    axes[1].fill_between(dd.index, dd.values, 0.0, color="#E45756", alpha=0.55)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pnl_scatter(cmc_sub: pd.Series, bn: pd.Series, out_path: Path, corr: float | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a, b = cmc_sub.align(bn, join="inner")
    a = a.astype(float).fillna(0.0)
    b = b.astype(float).fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
    ax.scatter(a.values, b.values, s=8, alpha=0.35, color="#4C78A8", linewidths=0)
    lim = max(float(np.nanmax(np.abs(a.values))), float(np.nanmax(np.abs(b.values))), 1e-6)
    ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("CMC-priced subset daily PnL")
    ax.set_ylabel("BOOK-BINANCE-ONLY daily PnL")
    lab = "CMC vs Binance daily PnL (replayable subset)"
    if corr is not None and np.isfinite(corr):
        lab += f"  corr={corr:.4f}"
    ax.set_title(lab)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def update_numbers_ledger(path: Path, *, validated: bool, hybrid: dict, cmc: dict, extra: dict) -> str:
    text = path.read_text() if path.exists() else ""
    marker = "## BTC-BEATER SPREAD-LS (Phase 3.c Binance replay)"
    block_lines = [
        "",
        marker,
        "",
        "Production book config: β-matched, h=14, floored PIT top-100 dollar-volume. "
        "Positions from the 2.c spread cache (signals not recomputed). "
        f"COMBO overlap corr remains {extra.get('combo_corr', 0.157)} for allocation. "
        "MASTER combination book is out of scope (PI).",
        "",
    ]
    if validated:
        block_lines += [
            f"**OFFICIAL SPREAD-LS = BOOK-HYBRID (funding-on).** "
            f"Full net Sharpe `{_fmt(hybrid.get('net_sharpe'))}` / trail-18m `{_fmt(hybrid.get('net_sharpe_trail18m'))}` "
            f"/ MaxDD `{_pct(hybrid.get('maxdd'))}` / funding PnL `{_fmt(hybrid.get('funding_total_pnl'), 4)}`.",
            "",
            f"Footnote: funding-off CMC BOOK-CMC Sharpe `{_fmt(cmc.get('net_sharpe'))}` / trail "
            f"`{_fmt(cmc.get('net_sharpe_trail18m'))}` / MaxDD `{_pct(cmc.get('maxdd'))}` is **deprecated** "
            f"as of Phase 3.c (prices validated on the Binance-only subset).",
            "",
        ]
    else:
        block_lines += [
            "**OFFICIAL SPREAD-LS RECORD SUSPENDED.** Phase 3.c price validation failed. "
            "Funding-off CMC numbers are not the live record and are not replaced. "
            "No improvement work proceeds until the pricing gap is understood.",
            "",
            f"BOOK-CMC (funding-off, reference) Sharpe `{_fmt(cmc.get('net_sharpe'))}` / trail "
            f"`{_fmt(cmc.get('net_sharpe_trail18m'))}` / MaxDD `{_pct(cmc.get('maxdd'))}`.",
            f"BOOK-HYBRID (unofficial) Sharpe `{_fmt(hybrid.get('net_sharpe'))}` / trail "
            f"`{_fmt(hybrid.get('net_sharpe_trail18m'))}`.",
            "",
        ]
    block = "\n".join(block_lines) + "\n"
    if marker in text:
        pre, rest = text.split(marker, 1)
        # drop old section through EOF-ish: keep only pre
        text = pre.rstrip() + "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    path.write_text(text)
    return block
