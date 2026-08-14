"""Long-only report, JSON, and equity chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from longonly.constants import VIABILITY_CRITERION
from longonly.eval import _as_utc, cagr_maxdd
from phase_d2.metrics import _sharpe, window_slice


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


def _eq_from_rets(rets: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    r = _as_utc(rets).fillna(0.0)
    eq = (1.0 + r).cumprod()
    y = eq.to_numpy()
    if len(y) and y[0] != 0:
        y = y / y[0]
    return eq.index, y


def plot_longonly_equity(
    loh_combo: pd.Series,
    lou_combo: pd.Series,
    btc: pd.Series,
    ref_combo: pd.Series,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = [
        (loh_combo, "COMBO-LO-H"),
        (lou_combo, "COMBO-LO-U"),
        (btc, "BTC B&H (costless)"),
        (ref_combo, "Reference COMBO (LS)"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    end = None
    for rets, lab in series:
        if not isinstance(rets, pd.Series) or len(rets) == 0:
            continue
        d, y = _eq_from_rets(rets)
        end = d.max() if end is None else max(end, d.max())
        axes[0].plot(d, y, label=lab, lw=1.4)
    axes[0].set_title("Long-only COMBO vs BTC B&H vs reference COMBO")
    axes[0].set_ylabel("Equity (rebased, log)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")
    if end is not None:
        start = end - pd.Timedelta(days=int(365 * 1.5))
        for rets, lab in series:
            if not isinstance(rets, pd.Series) or len(rets) == 0:
                continue
            d, y = _eq_from_rets(rets)
            m = np.asarray((pd.DatetimeIndex(d) >= start) & (pd.DatetimeIndex(d) <= end))
            if not m.any():
                continue
            yy = np.asarray(y)[m]
            dd = pd.DatetimeIndex(d)[m]
            if len(yy) and yy[0] != 0:
                yy = yy / yy[0]
            axes[1].plot(dd, yy, label=lab, lw=1.4)
    axes[1].set_title("Trailing-18m zoom")
    axes[1].set_ylabel("Equity (rebased, log)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, which="both")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _year_cols(by_year: dict) -> list[int]:
    return sorted(int(y) for y in by_year.keys())


def _book_table_row(name: str, st: dict) -> str:
    by = st.get("net_sharpe_by_year") or {}
    years = _year_cols(by)
    ycells = " | ".join(_fmt(by.get(y)) for y in years)
    top = st.get("top5_names") or []
    top_s = ", ".join(f"{t['symbol']}={_fmt(t['pnl'], 4)}" for t in top) if top else "—"
    return (
        f"| {name} | {_fmt(st.get('net_sharpe_full'))} | {_fmt(st.get('net_sharpe_trail18m'))} "
        f"| {ycells} | {_pct(st.get('net_cagr'))} | {_pct(st.get('max_drawdown'))} "
        f"| {_fmt(st.get('avg_n_long'), 2)} | {_fmt(st.get('avg_gross_deployed'), 3)} "
        f"| {_pct(st.get('pct_flat_days'))} | {_fmt(st.get('funding_total_pnl'), 4)} "
        f"| {_fmt(st.get('ann_turnover'), 2)} | {top_s} |"
    )


def _alpha_row(name: str, st: dict) -> str:
    a = st.get("alpha") or {}
    f = a.get("full") or {}
    t = a.get("trail18m") or {}
    return (
        f"| {name} | {_fmt(f.get('alpha_ann'))} | {_fmt(f.get('beta'))} | {_fmt(f.get('nw_t_alpha'))} "
        f"| {f.get('n', '')} | {_fmt(t.get('alpha_ann'))} | {_fmt(t.get('beta'))} "
        f"| {_fmt(t.get('nw_t_alpha'))} | {t.get('n', '')} | {_fmt(st.get('corr_vs_ref_combo'))} |"
    )


def _attr_row(name: str, blob: dict) -> str:
    return (
        f"| {name} | {_fmt(blob.get('long'), 4)} | {_fmt(blob.get('short'), 4)} "
        f"| {_fmt(blob.get('hedge'), 4)} | {_fmt(blob.get('funding'), 4)} "
        f"| {_fmt(blob.get('costs'), 4)} | {_fmt(blob.get('net'), 4)} "
        f"| {_fmt(blob.get('long_share_of_net'))} | {_fmt(blob.get('long_share_of_alpha'))} "
        f"| {_fmt(blob.get('recon_gap'), 6)} |"
    )


def write_longonly_report(
    path: Path,
    *,
    frozen_hash: str,
    pred_hashes: dict,
    books: dict,
    loh_v: dict,
    lou_v: dict,
    loh_sleeve_v: dict,
    lou_sleeve_v: dict,
    attr: dict,
    benches: dict,
    extra: dict,
) -> str:
    years = []
    for st in books.values():
        years = _year_cols(st.get("net_sharpe_by_year") or {})
        if years:
            break
    yhead = " | ".join(str(y) for y in years)
    lines = [
        "# Long-only variants of the frozen COMBO system",
        "",
        "**BACKTEST AND ANALYSIS ONLY.** Portfolio layer only. No schedules, no deployments, "
        "no live components. No model retraining, no feature changes, no τ re-optimization.",
        "",
        f"**Frozen A0 SHA256:** `{frozen_hash}`",
        f"**Prediction files (reused, not recomputed):** `{pred_hashes}`",
        "**Reference book:** COMBO v2.0-combo-final (Sleeve A C0 + Sleeve B P2, causal τ). "
        "**UNCHANGED.** This evaluation does not rewrite the ledger or the system card.",
        "",
        "Sizing: long half of `_size_book` (`0.5 * tg * iv / sum(iv_longs)`). "
        "The unused short-side 50% budget is **not** dumped onto longs. Utilization floats.",
        "Exit convention matches the frozen tranche engine (`_hard_threshold_state`; "
        "`exit_hysteresis` is discarded there, as in the live COMBO books).",
        "",
        "## Pre-registered viability statements",
        "",
        f"> {VIABILITY_CRITERION}",
        "",
        "Verdicts below are mechanical. No post-hoc adjustment. "
        "Product verdicts are on **COMBO-LO-H** and **COMBO-LO-U**. "
        "Sleeve-level checks are supplementary.",
        "",
        "## Mechanical verdicts",
        "",
        f"- **LO-H (COMBO-LO-H):** **{loh_v.get('verdict')}** — "
        f"full Sharpe={_fmt(loh_v.get('sharpe_full'))} "
        f"(need ≥ {_fmt(loh_v.get('need_full'))}, pass={loh_v.get('pass_full')}); "
        f"trail-18m Sharpe={_fmt(loh_v.get('sharpe_trail18m'))} "
        f"(need ≥ {_fmt(loh_v.get('need_trail18m'))}, pass={loh_v.get('pass_trail18m')}).",
        f"- **LO-U (COMBO-LO-U):** **{lou_v.get('verdict')}** — "
        f"full alpha_ann={_fmt(lou_v.get('alpha_ann_full'))} "
        f"(need > 0, pass={lou_v.get('pass_alpha_full')}); "
        f"NW-t={_fmt(lou_v.get('nw_t_alpha_full'))} "
        f"(need ≥ {_fmt(lou_v.get('need_nw_t'))}, pass={lou_v.get('pass_nw_t')}); "
        f"trail-18m alpha_ann={_fmt(lou_v.get('alpha_ann_trail18m'))} "
        f"(need > 0, pass={lou_v.get('pass_alpha_trail18m')}).",
        "",
        "Sleeve-level (not the product verdict):",
        "",
    ]
    for k, v in (loh_sleeve_v or {}).items():
        lines.append(
            f"- LO-H {k}: {v.get('verdict')} "
            f"(full={_fmt(v.get('sharpe_full'))}, trail={_fmt(v.get('sharpe_trail18m'))})"
        )
    for k, v in (lou_sleeve_v or {}).items():
        lines.append(
            f"- LO-U {k}: {v.get('verdict')} "
            f"(α={_fmt(v.get('alpha_ann_full'))}, NW-t={_fmt(v.get('nw_t_alpha_full'))}, "
            f"α18={_fmt(v.get('alpha_ann_trail18m'))})"
        )
    lines += [
        "",
        "## Headline books",
        "",
        f"| book | full | trail-18m | {yhead} | CAGR | MaxDD | avg #longs | avg gross (alpha) | % flat | funding PnL | ann TO | top-5 name PnL |",
        f"|------|------|-----------|{'|'.join(['------'] * len(years))}|------|-------|------------|-------------------|--------|-------------|--------|----------------|",
    ]
    order = [
        "LO-H Sleeve A",
        "LO-H Sleeve B",
        "COMBO-LO-H",
        "LO-U Sleeve A",
        "LO-U Sleeve B",
        "COMBO-LO-U",
        "Reference Sleeve A",
        "Reference Sleeve B",
        "Reference COMBO",
    ]
    for name in order:
        if name in books:
            lines.append(_book_table_row(name, books[name]))
    lines += [
        "",
        "Funding PnL is the sum of daily −w·funding_rate (longs pay when the rate is positive). "
        "On long-only books this is expected to be **negative**.",
        "Avg gross (alpha) is mean Σ|w_i| over alt names only (ex-hedge). "
        "LO books do not renormalize the long half toward 1.0 when shorts are absent, "
        "so deployed gross sits near ~0.5 when every slot has ≥1 long, and lower when slots are empty.",
        "",
        "## §2 Alpha vs BTC B&H (costless) and correlation with reference COMBO",
        "",
        "OLS of the book's daily net return on BTC buy-and-hold simple returns. "
        "Newey–West t-stat on the intercept uses Bartlett weights with HAC lag = h "
        "(Sleeve A: 7; Sleeve B: 10; COMBO: 10). Alpha is annualized as daily intercept × 365.",
        "",
        "| book | α_ann full | β full | NW-t α full | n | α_ann 18m | β 18m | NW-t α 18m | n 18m | corr vs ref COMBO |",
        "|------|------------|--------|-------------|---|-----------|-------|------------|-------|-------------------|",
    ]
    for name in order:
        if name in books:
            lines.append(_alpha_row(name, books[name]))
    btc_s = benches.get("btc")
    ew_s = benches.get("ew_top20")
    lines += ["", "### Costless benchmarks (not viability inputs)", ""]
    if isinstance(btc_s, pd.Series) and len(btc_s):
        cagr, maxdd, tot = cagr_maxdd(btc_s)
        lines.append(
            f"- BTC B&H: Sharpe full={_fmt(_sharpe(btc_s))}, "
            f"trail-18m={_fmt(_sharpe(window_slice(btc_s, 'trail18m')))}, "
            f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(tot)}."
        )
    if isinstance(ew_s, pd.Series) and len(ew_s):
        cagr, maxdd, tot = cagr_maxdd(ew_s)
        lines.append(
            f"- EW PIT top-20 (daily rebalanced, costless): Sharpe full={_fmt(_sharpe(ew_s))}, "
            f"trail-18m={_fmt(_sharpe(window_slice(ew_s, 'trail18m')))}, "
            f"CAGR={_pct(cagr)}, MaxDD={_pct(maxdd)}, total={_pct(tot)}."
        )
    ref_attr = attr.get("Reference COMBO") or {}
    ref_full = ref_attr.get("full") or {}
    lines += [
        "",
        "## §3 Long/short attribution of the frozen reference book",
        "",
        "Daily net = long-leg + short-leg + hedge-leg + funding − costs. "
        "Legs are sums of simple-return units (the same units as `daily_ret` in the tranche engine). "
        "`long_share_of_net` = long-leg / net; `long_share_of_alpha` = long / (long+short).",
        "",
        f"**One-liner:** the long legs contributed **{_pct(ref_full.get('long_share_of_net'))}** "
        f"of frozen reference COMBO total net PnL "
        f"(long={_fmt(ref_full.get('long'), 4)}, net={_fmt(ref_full.get('net'), 4)}).",
        "",
        "### Full period",
        "",
        "| book | long | short | hedge | funding | costs | net | long/net | long/(L+S) | recon gap |",
        "|------|------|-------|-------|---------|-------|-----|----------|------------|-----------|",
    ]
    for name in (
        "Reference Sleeve A",
        "Reference Sleeve B",
        "Reference COMBO",
        "COMBO-LO-H",
        "COMBO-LO-U",
        "LO-H Sleeve A",
        "LO-H Sleeve B",
        "LO-U Sleeve A",
        "LO-U Sleeve B",
    ):
        blob = attr.get(name)
        if blob:
            lines.append(_attr_row(name, blob.get("full") or {}))
    lines += ["", "### Per calendar year (reference COMBO and sleeves)", ""]
    for name in ("Reference COMBO", "Reference Sleeve A", "Reference Sleeve B"):
        blob = attr.get(name) or {}
        by = blob.get("by_year") or {}
        if not by:
            continue
        lines.append(f"**{name}**")
        lines.append("")
        lines.append(
            "| year | long | short | hedge | funding | costs | net | long/net | long/(L+S) | recon gap |"
        )
        lines.append(
            "|------|------|-------|-------|---------|-------|-----|----------|------------|-----------|"
        )
        for y in sorted(by):
            lines.append(_attr_row(str(y), by[y]))
        lines.append("")
    lines += [
        "## Correlation with the reference COMBO",
        "",
        extra.get("corr_oneliner", ""),
        "",
        "A high correlation means long-only is largely a substitute for the reference book; "
        "a low correlation means it diversifies. This is a description, not a keep/kill rule.",
        "",
        "## Reference book is unchanged",
        "",
        "The frozen COMBO (v2.0-combo-final) is the reference book. "
        "LO-H / LO-U are parallel product/mandate evaluations on the same frozen A0 scores, "
        "universes, and causal median-τ. No outcome here changes the reference book, "
        "the system card, or the numbers ledger.",
        "",
        f"Elapsed seconds: {_fmt(extra.get('elapsed_sec'), 1)}. GPU used: false. "
        f"Scheduled jobs created: false.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def print_stdout(loh_v: dict, lou_v: dict, attr: dict, extra: dict) -> None:
    ref = (attr.get("Reference COMBO") or {}).get("full") or {}
    share = ref.get("long_share_of_net")
    print(
        f"LO-H COMBO: {loh_v.get('verdict')} "
        f"(full={_fmt(loh_v.get('sharpe_full'))} trail={_fmt(loh_v.get('sharpe_trail18m'))}; "
        f"need {loh_v.get('need_full')} / {loh_v.get('need_trail18m')})",
        flush=True,
    )
    print(
        f"LO-U COMBO: {lou_v.get('verdict')} "
        f"(alpha_ann={_fmt(lou_v.get('alpha_ann_full'))} nw_t={_fmt(lou_v.get('nw_t_alpha_full'))} "
        f"trail_alpha={_fmt(lou_v.get('alpha_ann_trail18m'))}; "
        f"need alpha>0, NW-t>={lou_v.get('need_nw_t')}, trail alpha>0)",
        flush=True,
    )
    print(
        f"Reference COMBO long-leg share of net PnL: {_pct(share)} "
        f"(long={_fmt(ref.get('long'), 4)}, net={_fmt(ref.get('net'), 4)}).",
        flush=True,
    )
    print(extra.get("corr_oneliner", ""), flush=True)
    print("Reference book UNCHANGED (v2.0-combo-final).", flush=True)
