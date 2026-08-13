"""Phase D decay diagnostic — diagnosis only, no fixes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline.evaluate import daily_rank_ic
from baseline.portfolio import run_tranche_portfolio

GROSS_PROXY_FORMULA = (
    "PROXY_Y = mean_daily_top20_RankIC_Y * mean_daily_CS_std(y_h7)_Y * avg_n_positions_full"
)


def run_decay_diagnostic(
    preds: pd.DataFrame,
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    pit20: pd.DataFrame,
    funding: pd.DataFrame | None,
    tau_pct: float = 60.0,
    cfg_portfolio: dict | None = None,
) -> dict:
    cfg_portfolio = cfg_portfolio or {}
    h = 7
    ycol = f"y_h{h}"
    pred = preds.copy()
    pred["date"] = pd.to_datetime(pred["date"], utc=True)
    if ycol not in pred.columns:
        pred = pred.merge(feat[["date", "symbol", ycol]], on=["date", "symbol"], how="left")
    uni = pit20.copy()
    uni["date"] = pd.to_datetime(uni["date"], utc=True)
    ev = pred.merge(uni[["date", "symbol"]], on=["date", "symbol"], how="inner")
    ic = daily_rank_ic(ev, ycol)
    disp = ev.groupby("date")[ycol].std()
    disp.index = pd.to_datetime(disp.index, utc=True)
    score_disp = ev.groupby("date")["score"].std()
    score_disp.index = pd.to_datetime(score_disp.index, utc=True)

    res = run_tranche_portfolio(
        pred,
        panel,
        feat,
        uni,
        horizon=h,
        tau_pct=float(tau_pct),
        exit_hysteresis=float(cfg_portfolio.get("exit_hysteresis", 0.6)),
        gross_limit=float(cfg_portfolio.get("gross_limit", 1.0)),
        fee_bps=float(cfg_portfolio.get("taker_fee_bps", 5.0)),
        slip_bps=float(cfg_portfolio.get("slippage_bps", 3.0)),
        lag=0,
        apply_funding=True,
        funding=funding,
    )
    daily_gross = res["daily_gross"]
    daily_cost = res["daily_cost"]
    daily_funding = res["daily_funding"]
    daily_net = res["daily_ret"]
    avg_npos = float(res.get("avg_n_positions", 0.0))

    rows = []
    years = sorted(set(ic.index.year) | set(daily_net.index.year))
    for y in years:
        if y < 2022:
            continue
        ic_y = ic[ic.index.year == y]
        disp_y = disp[disp.index.year == y]
        sd_y = score_disp[score_disp.index.year == y]
        g = daily_gross[daily_gross.index.year == y]
        c = daily_cost[daily_cost.index.year == y]
        f = daily_funding[daily_funding.index.year == y]
        n = daily_net[daily_net.index.year == y]
        if len(ic_y) < 10 or len(n) < 10:
            continue
        mean_ic = float(ic_y.mean())
        mean_disp = float(disp_y.mean())
        nonempty = float((g.abs() > 1e-12).mean()) if len(g) else 0.0
        proxy = mean_ic * mean_disp * avg_npos
        gross_pnl = float(g.sum())
        cost_pnl = float(c.sum())
        fund_pnl = float(f.sum())
        abs_gross = abs(gross_pnl) if abs(gross_pnl) > 1e-12 else np.nan
        rows.append(
            {
                "year": int(y),
                "n_ic_days": int(len(ic_y)),
                "rank_ic": mean_ic,
                "dispersion": mean_disp,
                "score_dispersion": float(sd_y.mean()) if len(sd_y) else float("nan"),
                "avg_n_positions": avg_npos,
                "pct_nonempty_book": nonempty,
                "gross_proxy": proxy,
                "gross_pnl": gross_pnl,
                "cost_drag": cost_pnl,
                "funding_pnl": fund_pnl,
                "net_pnl": float(n.sum()),
                "cost_share_of_abs_gross": float(cost_pnl / abs_gross) if np.isfinite(abs_gross) else float("nan"),
                "funding_share_of_abs_gross": float(abs(fund_pnl) / abs_gross) if np.isfinite(abs_gross) else float("nan"),
                "net_sharpe": float(n.mean() / n.std() * np.sqrt(365)) if n.std() > 0 else 0.0,
            }
        )

    if len(rows) >= 3:
        prox = np.array([r["gross_proxy"] for r in rows], float)
        gr = np.array([r["gross_pnl"] for r in rows], float)
        track = float(np.corrcoef(prox, gr)[0, 1]) if np.std(prox) > 0 and np.std(gr) > 0 else float("nan")
    else:
        track = float("nan")

    verdict, justification = _pick_verdict(rows, track)
    return {
        "formula": GROSS_PROXY_FORMULA,
        "by_year": rows,
        "proxy_vs_gross_corr": track,
        "verdict": verdict,
        "justification": justification,
        "avg_n_positions_full": avg_npos,
        "full_net_sharpe": float(res.get("net_sharpe", float("nan"))),
        "tau_pct": float(tau_pct),
    }


def _pick_verdict(rows: list[dict], track_corr: float) -> tuple[str, str]:
    if not rows:
        return "TRANSLATION_BREAK", "No yearly rows."
    by = {r["year"]: r for r in rows}
    early = [by[y] for y in (2023, 2024) if y in by] or rows[: max(1, len(rows) // 2)]
    late = [by[y] for y in (2025, 2026) if y in by] or rows[max(1, len(rows) // 2) :]

    def avg(key, xs):
        vals = [x[key] for x in xs if np.isfinite(x.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    ic_e, ic_l = avg("rank_ic", early), avg("rank_ic", late)
    disp_e, disp_l = avg("dispersion", early), avg("dispersion", late)
    fric_e = avg("cost_share_of_abs_gross", early) + avg("funding_share_of_abs_gross", early)
    fric_l = avg("cost_share_of_abs_gross", late) + avg("funding_share_of_abs_gross", late)
    sharpe_e, sharpe_l = avg("net_sharpe", early), avg("net_sharpe", late)
    nonempty_e, nonempty_l = avg("pct_nonempty_book", early), avg("pct_nonempty_book", late)
    proxy_e, proxy_l = avg("gross_proxy", early), avg("gross_proxy", late)
    gross_e, gross_l = avg("gross_pnl", early), avg("gross_pnl", late)

    ic_drop = (ic_e - ic_l) / max(abs(ic_e), 1e-6) if np.isfinite(ic_e) else 0.0
    disp_drop = (disp_e - disp_l) / max(abs(disp_e), 1e-6) if np.isfinite(disp_e) else 0.0
    fric_rise = (fric_l - fric_e) if np.isfinite(fric_e) and np.isfinite(fric_l) else 0.0

    if ic_drop >= 0.40 and ic_l < 0.5 * max(ic_e, 1e-6):
        return (
            "IC_DECAY",
            f"RankIC {ic_e:.3f}→{ic_l:.3f} (drop={ic_drop:.0%}); Sharpe {sharpe_e:.2f}→{sharpe_l:.2f}. Model ranking collapsed.",
        )
    if ic_l >= 0.6 * max(ic_e, 1e-6) and disp_drop >= 0.30:
        return (
            "DISPERSION_COLLAPSE",
            f"RankIC intact-ish ({ic_e:.3f}→{ic_l:.3f}) but y_h7 CS dispersion {disp_e:.4f}→{disp_l:.4f} "
            f"(drop={disp_drop:.0%}); proxy {proxy_e:.4f}→{proxy_l:.4f}. Ranking fine, raw material gone.",
        )
    if fric_rise >= 0.15 and sharpe_l < sharpe_e - 0.5:
        return (
            "FRICTION_SHIFT",
            f"Friction share {fric_e:.2f}→{fric_l:.2f}; Sharpe {sharpe_e:.2f}→{sharpe_l:.2f}. Costs+funding ate the edge.",
        )
    if (nonempty_l < 0.5 * max(nonempty_e, 1e-6)) or (
        np.isfinite(track_corr) and track_corr < 0.2 and abs(gross_l) < 0.3 * max(abs(gross_e), 1e-9)
    ):
        return (
            "TRANSLATION_BREAK",
            f"IC {ic_e:.3f}→{ic_l:.3f}, disp {disp_e:.4f}→{disp_l:.4f}, but nonempty {nonempty_e:.2f}→{nonempty_l:.2f}, "
            f"proxy↔gross corr={track_corr:.2f}, gross {gross_e:.3f}→{gross_l:.3f}.",
        )
    scores = {
        "IC_DECAY": ic_drop,
        "DISPERSION_COLLAPSE": disp_drop if ic_l >= 0.5 * max(ic_e, 1e-6) else 0.0,
        "FRICTION_SHIFT": max(fric_rise, 0.0),
        "TRANSLATION_BREAK": max(0.0, (nonempty_e - nonempty_l) if np.isfinite(nonempty_e) else 0.0),
    }
    winner = max(scores, key=scores.get)
    return (
        winner,
        f"Selected {winner}: ic_drop={ic_drop:.2f}, disp_drop={disp_drop:.2f}, fric_rise={fric_rise:.2f}; "
        f"IC {ic_e:.3f}→{ic_l:.3f}, disp {disp_e:.4f}→{disp_l:.4f}, Sharpe {sharpe_e:.2f}→{sharpe_l:.2f}, "
        f"proxy↔gross corr={track_corr}.",
    )


def plot_decay(diag: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = diag.get("by_year") or []
    if not rows:
        return
    years = [str(r["year"]) for r in rows]
    ics = [r["rank_ic"] for r in rows]
    disps = [r["dispersion"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax1.bar(years, ics, color="#4C78A8", alpha=0.85, label="top-20 RankIC h=7")
    ax1.set_ylabel("RankIC")
    ax2 = ax1.twinx()
    ax2.plot(years, disps, color="#F58518", marker="o", lw=2, label="CS dispersion y_h7")
    ax2.set_ylabel("Dispersion")
    if "2025" in years:
        ax1.axvline("2025", color="black", ls="--", lw=1, label="cutoff year")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax1.set_title("Phase D decay — RankIC vs return dispersion")
    ax1.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
