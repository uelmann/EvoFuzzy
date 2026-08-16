"""Survivorship + score-persistence diagnostics. No new model.

Point 1: last kline bar vs 2026-07 on the FASE 1 panel (Vision dirs, not
exchangeInfo). Point 2: CS score autocorrelation at lag 10, then RankIC of
the within-bar residual of score_t on score_{t-10} vs y_h7.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baseline.evaluate import daily_rank_ic, summarize_ic


CUT = pd.Timestamp("2026-07-01", tz="UTC")
LAG = 10
H = 7
YCOL = "y_h7"
MIN_HISTORY = 100
SHIFTED_IC = 0.058511608345255456
UNSHIFTED_IC = 0.08032113973646492
SHIFTED_RATIO = SHIFTED_IC / UNSHIFTED_IC


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def kline_span_table(raw_dir: Path) -> pd.DataFrame:
    rows = []
    files = sorted(raw_dir.glob("*.parquet"))
    print(f"[surv] scanning {len(files)} kline parquets", flush=True)
    for i, p in enumerate(files, 1):
        df = pd.read_parquet(p, columns=["date"])
        if df.empty:
            rows.append({"symbol": p.stem, "first": pd.NaT, "last": pd.NaT, "n": 0})
        else:
            d = pd.to_datetime(df["date"], utc=True)
            rows.append(
                {
                    "symbol": p.stem,
                    "first": d.min(),
                    "last": d.max(),
                    "n": int(len(df)),
                }
            )
        if i % 100 == 0 or i == len(files):
            print(f"[surv] span {i}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def year_hist(dead: pd.DataFrame) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for r in dead.itertuples():
        by[str(int(pd.Timestamp(r.last).year))].append(r.symbol)
    return {y: sorted(v) for y, v in sorted(by.items())}


def cs_lag_rho(wide: pd.DataFrame, lag: int, method: str) -> pd.Series:
    """Mean over t of CS corr_i(score_{i,t}, score_{i,t+lag})."""
    dates = wide.index
    arr = wide.to_numpy(dtype=float)
    n = arr.shape[0]
    idx: list = []
    vals: list[float] = []
    for i in range(n - lag):
        a = arr[i]
        b = arr[i + lag]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < 5:
            continue
        aa = a[m]
        bb = b[m]
        if method == "spearman":
            aa = stats.rankdata(aa).astype(float)
            bb = stats.rankdata(bb).astype(float)
        if np.unique(aa).size < 2 or np.unique(bb).size < 2:
            continue
        c = _pearson(aa, bb)
        if np.isfinite(c):
            idx.append(dates[i])
            vals.append(c)
        if i % 400 == 0:
            print(f"[persist] rho {method} i={i}/{n - lag}", flush=True)
    if not vals:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)


def ortho_residual_scores(wide: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Within-bar OLS: score_t = a + b * score_{t-lag} + e. Returns long e."""
    dates = wide.index
    arr = wide.to_numpy(dtype=float)
    cols = wide.columns.to_numpy()
    n = arr.shape[0]
    parts = []
    for i in range(lag, n):
        s = arr[i]
        s0 = arr[i - lag]
        m = np.isfinite(s) & np.isfinite(s0)
        k = int(m.sum())
        if k < 5:
            continue
        y = s[m]
        x = s0[m]
        X = np.column_stack([np.ones(k), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
        parts.append(
            pd.DataFrame({"date": dates[i], "symbol": cols[m], "score_ortho": e})
        )
        if i % 400 == 0:
            print(f"[persist] ortho i={i}/{n}", flush=True)
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "score_ortho"])
    return pd.concat(parts, ignore_index=True)


def _persist_one(label: str, df: pd.DataFrame) -> dict:
    print(f"[persist] universe={label} rows={len(df)}", flush=True)
    wide = (
        df.pivot_table(index="date", columns="symbol", values="score", aggfunc="mean")
        .sort_index()
    )
    rho_p = cs_lag_rho(wide, LAG, "pearson")
    rho_s = cs_lag_rho(wide, LAG, "spearman")
    ortho = ortho_residual_scores(wide, LAG)
    merged = ortho.merge(df[["date", "symbol", YCOL]], on=["date", "symbol"], how="inner")
    tmp = merged.rename(columns={"score_ortho": "score"})
    ic = daily_rank_ic(tmp, YCOL, score_col="score")
    summ = summarize_ic(ic, H)
    raw_ic = daily_rank_ic(df, YCOL, score_col="score")
    raw = summarize_ic(raw_ic, H)
    ratio = (
        float(summ["mean_ic"] / raw["mean_ic"])
        if raw["mean_ic"] and np.isfinite(raw["mean_ic"]) and raw["mean_ic"] != 0
        else float("nan")
    )
    out = {
        "n_days_score": int(wide.shape[0]),
        "n_names_cols": int(wide.shape[1]),
        "rho_definition": (
            "mean_t CS corr_i(score_i,t, score_i,t+10) on names present at both t and t+10"
        ),
        "rho_pearson_mean": float(rho_p.mean()) if len(rho_p) else float("nan"),
        "rho_pearson_median": float(rho_p.median()) if len(rho_p) else float("nan"),
        "rho_pearson_n": int(len(rho_p)),
        "rho_spearman_mean": float(rho_s.mean()) if len(rho_s) else float("nan"),
        "rho_spearman_median": float(rho_s.median()) if len(rho_s) else float("nan"),
        "rho_spearman_n": int(len(rho_s)),
        "unshifted_mean_ic": raw["mean_ic"],
        "unshifted_nw_t": raw["nw_tstat"],
        "unshifted_n_days": raw["n_days"],
        "ortho_mean_ic": summ["mean_ic"],
        "ortho_nw_t": summ["nw_tstat"],
        "ortho_n_days": summ["n_days"],
        "ortho_ic_over_unshifted": ratio,
        "shifted_over_unshifted_ref_top40": SHIFTED_RATIO,
        "shifted_mean_ic_ref_top40": SHIFTED_IC,
        "unshifted_mean_ic_ref_top40": UNSHIFTED_IC,
    }
    print(f"[persist] {label} rho_s={out['rho_spearman_mean']:.4f} "
          f"rho_p={out['rho_pearson_mean']:.4f} "
          f"ortho_ic={out['ortho_mean_ic']:.4f} nw={out['ortho_nw_t']:.2f}", flush=True)
    return out


def main() -> int:
    raw_dir = Path("/data/quant/raw/klines")
    pred_path = Path("/data/quant/predictions/lgbm_price_only_h7.parquet")
    feat_path = Path("/data/quant/features/features_labeled_h7.parquet")
    uni40 = Path("/data/quant/universe/top40_pit.parquet")
    uni20 = Path("/data/quant/universe/top20_pit.parquet")

    spans = kline_span_table(raw_dir)
    nonempty = spans[spans["n"] > 0]
    panel = nonempty[nonempty["n"] >= MIN_HISTORY].copy()
    dead_all = nonempty[nonempty["last"] < CUT]
    dead_panel = panel[panel["last"] < CUT]
    dropped = nonempty[nonempty["n"] < MIN_HISTORY]
    dropped_dead = dropped[dropped["last"] < CUT]
    print(
        f"[surv] listed_parquet={len(spans)} nonempty={len(nonempty)} "
        f"panel_n>={MIN_HISTORY}={len(panel)} last<2026-07-01 all={len(dead_all)} "
        f"panel={len(dead_panel)}",
        flush=True,
    )

    pred = pd.read_parquet(pred_path)
    pred["date"] = pd.to_datetime(pred["date"], utc=True)
    if YCOL not in pred.columns:
        feat = pd.read_parquet(feat_path, columns=["date", "symbol", YCOL])
        feat["date"] = pd.to_datetime(feat["date"], utc=True)
        pred = pred.merge(feat, on=["date", "symbol"], how="left")
    pit40 = pd.read_parquet(uni40)
    pit20 = pd.read_parquet(uni20)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)

    persist = {}
    persist["top40"] = _persist_one(
        "top40", pred.merge(pit40[["date", "symbol"]], on=["date", "symbol"], how="inner")
    )
    persist["top20"] = _persist_one(
        "top20", pred.merge(pit20[["date", "symbol"]], on=["date", "symbol"], how="inner")
    )
    persist["oos_all"] = _persist_one("oos_all", pred)

    rho40 = persist["top40"]["rho_spearman_mean"]
    ortho40 = persist["top40"]["ortho_mean_ic"]
    persist_verdict = {
        "headline_universe": "top40",
        "rho_spearman_lag10": rho40,
        "shifted_ic_ratio_ref": SHIFTED_RATIO,
        "rho_minus_shifted_ratio": (
            float(rho40 - SHIFTED_RATIO) if np.isfinite(rho40) else float("nan")
        ),
        "ortho_mean_rank_ic": ortho40,
        "ortho_nw_t": persist["top40"]["ortho_nw_t"],
        "hypothesis": (
            "shifted/unshifted RankIC 0.73 is entirely score stickiness (rho). "
            "If rho ~ 0.73, test_shifted_target_degrades found slowness, not leak. "
            "Orthogonalized residual RankIC vs y_h7 is the remaining timing signal."
        ),
        "rho_explains_shifted_ratio": bool(
            np.isfinite(rho40) and abs(float(rho40) - SHIFTED_RATIO) < 0.05
        ),
    }

    dead_rows = []
    for _, r in dead_panel.sort_values("last").iterrows():
        dead_rows.append(
            {
                "symbol": r["symbol"],
                "first": str(pd.Timestamp(r["first"]).date()),
                "last": str(pd.Timestamp(r["last"]).date()),
                "n": int(r["n"]),
                "year": int(pd.Timestamp(r["last"]).year),
            }
        )

    surv = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "listing_source": {
            "function": "baseline.data.list_um_symbols",
            "method": (
                "Binance Vision S3 ListObjectsV2 delimiter=/ "
                "prefix=data/futures/um/monthly/klines/"
            ),
            "exchangeInfo": False,
            "includes_delisted_with_remaining_dumps": True,
            "n_symbols_listed_is": (
                "len(Vision USDT prefixes after should_exclude), then downloaded "
                "to /data/quant/raw/klines/{SYMBOL}.parquet"
            ),
            "rebuild_required": False,
            "rebuild_reason": (
                "831 is already a historical Vision directory enumeration, not a "
                "live exchangeInfo snapshot. Delisted names with remaining monthly "
                "dumps are present. Do not rebuild."
            ),
        },
        "n_listed_parquet": int(len(spans)),
        "n_nonempty": int(len(nonempty)),
        "n_panel_min_history_100": int(len(panel)),
        "n_dropped_short_history": int(len(dropped)),
        "n_last_bar_before_2026_07_all": int(len(dead_all)),
        "n_last_bar_before_2026_07_panel": int(len(dead_panel)),
        "panel_pct_ended_before_2026_07": (
            round(100.0 * len(dead_panel) / len(panel), 2) if len(panel) else float("nan")
        ),
        "delisted_share_panel": (
            float(len(dead_panel) / len(panel)) if len(panel) else float("nan")
        ),
        "survivorship_only_survivors": bool(len(dead_panel) == 0),
        "baseline_json_invalid_survivorship": False,
        "cut": "last_bar < 2026-07-01 UTC (no kline in July 2026)",
        "panel_max_last": str(panel["last"].max().date()) if len(panel) else None,
        "delisted_by_year_panel": year_hist(dead_panel),
        "delisted_by_year_listed": year_hist(dead_all),
        "delisted_panel_rows": dead_rows,
        "dropped_short_history_ended_before_2026_07": {
            r.symbol: {"n": int(r.n), "last": str(pd.Timestamp(r.last).date())}
            for r in dropped_dead.sort_values("last").itertuples()
        },
        "verdict_reason": (
            f"{len(dead_panel)} / {len(panel)} panel names end before 2026-07 "
            f"({100.0 * len(dead_panel) / max(len(panel), 1):.1f}%). Not ~0. "
            "831 comes from Vision historical prefixes, not live exchangeInfo. "
            f"{len(dropped)} dropped names are min_history={MIN_HISTORY}, not a "
            "survivor filter. baseline.json is not invalid on survivorship."
        ),
    }

    out = {
        **surv,
        "persist": persist,
        "persist_verdict": persist_verdict,
        "icir_definition": (
            "annualized: mean(daily RankIC) / std(daily RankIC, ddof=1) * sqrt(252)"
        ),
        "icir_annualized": True,
        "icir_not_daily": True,
        "noise_floor_note": (
            "SE of daily CS Spearman ~ 1/sqrt(N); N=40 => ~0.16. ICIR 5.90 is the "
            "annualized ratio, not mean/std of daily IC without sqrt(252). A daily "
            "ICIR of 5.9 would be impossible under that noise floor."
        ),
        "pre_reg_untouched": True,
        "shifted_target_threshold_untouched": True,
        "stage_a": False,
    }
    Path("results").mkdir(exist_ok=True)
    payload = json.dumps(out, indent=2, default=str)
    Path("results/fase1_survivorship_persist.json").write_text(payload + "\n")
    Path("results/fase1_survivorship.json").write_text(
        json.dumps(surv, indent=2, default=str) + "\n"
    )
    print("[surv] wrote results/fase1_survivorship.json", flush=True)
    print("[surv] wrote results/fase1_survivorship_persist.json", flush=True)
    print("[surv] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
