"""Survivorship + score-persistence diagnostics. No new model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baseline.evaluate import daily_rank_ic, newey_west_t, summarize_ic


CUT = pd.Timestamp("2026-07-01", tz="UTC")
LAG = 10
H = 7
YCOL = "y_h7"


def _corr(x, y, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    res = stats.spearmanr(x, y)
    c = getattr(res, "correlation", None)
    if c is None:
        c = getattr(res, "statistic", np.nan)
    return float(np.asarray(c, dtype=float).reshape(-1)[0])


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
            rows.append({"symbol": p.stem, "first": d.min(), "last": d.max(), "n": int(len(df))})
        if i % 100 == 0 or i == len(files):
            print(f"[surv] span {i}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def cs_lag_rho(wide: pd.DataFrame, lag: int, method: str) -> pd.Series:
    dates = wide.index
    rows = []
    for i in range(len(dates) - lag):
        a = wide.iloc[i]
        b = wide.iloc[i + lag]
        m = a.notna() & b.notna()
        if int(m.sum()) < 5:
            continue
        c = _corr(a[m].to_numpy(), b[m].to_numpy(), method)
        if np.isfinite(c):
            rows.append((dates[i], c, int(m.sum())))
        if i % 200 == 0:
            print(f"[persist] rho {method} i={i}/{len(dates)-lag}", flush=True)
    if not rows:
        return pd.Series(dtype=float)
    idx, vals, _ = zip(*rows)
    return pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)))


def ortho_residual_scores(wide: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Within-bar OLS: score_t = a + b * score_{t-lag} + e. Returns long e."""
    dates = wide.index
    parts = []
    for i in range(lag, len(dates)):
        s = wide.iloc[i]
        s0 = wide.iloc[i - lag]
        m = s.notna() & s0.notna()
        n = int(m.sum())
        if n < 5:
            continue
        y = s[m].to_numpy(dtype=float)
        x = s0[m].to_numpy(dtype=float)
        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
        parts.append(pd.DataFrame({"date": dates[i], "symbol": s[m].index, "score_ortho": e}))
        if i % 200 == 0:
            print(f"[persist] ortho i={i}/{len(dates)}", flush=True)
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "score_ortho"])
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    raw_dir = Path("/data/quant/raw/klines")
    pred_path = Path("/data/quant/predictions/lgbm_price_only_h7.parquet")
    feat_path = Path("/data/quant/features/features_labeled_h7.parquet")
    uni40 = Path("/data/quant/universe/top40_pit.parquet")
    uni20 = Path("/data/quant/universe/top20_pit.parquet")

    spans = kline_span_table(raw_dir)
    nonempty = spans[spans["n"] > 0]
    panel = nonempty[nonempty["n"] >= 100].copy()
    dead_all = nonempty[nonempty["last"] < CUT]
    dead_panel = panel[panel["last"] < CUT]
    print(
        f"[surv] listed_parquet={len(spans)} nonempty={len(nonempty)} "
        f"panel_n>=100={len(panel)} last<2026-07-01 all={len(dead_all)} panel={len(dead_panel)}",
        flush=True,
    )

    pred = pd.read_parquet(pred_path)
    feat = pd.read_parquet(feat_path, columns=["date", "symbol", YCOL])
    pred["date"] = pd.to_datetime(pred["date"], utc=True)
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    if YCOL not in pred.columns:
        pred = pred.merge(feat, on=["date", "symbol"], how="left")
    pit40 = pd.read_parquet(uni40)
    pit20 = pd.read_parquet(uni20)
    pit40["date"] = pd.to_datetime(pit40["date"], utc=True)
    pit20["date"] = pd.to_datetime(pit20["date"], utc=True)

    persist = {}
    for label, uni in [("top40", pit40), ("top20", pit20), ("oos_all", None)]:
        print(f"[persist] universe={label}", flush=True)
        df = pred.copy()
        if uni is not None:
            df = df.merge(uni[["date", "symbol"]], on=["date", "symbol"], how="inner")
        wide = df.pivot_table(index="date", columns="symbol", values="score", aggfunc="mean").sort_index()
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
        persist[label] = {
            "n_days_score": int(wide.shape[0]),
            "n_names_cols": int(wide.shape[1]),
            "rho_pearson_mean": float(rho_p.mean()) if len(rho_p) else float("nan"),
            "rho_pearson_median": float(rho_p.median()) if len(rho_p) else float("nan"),
            "rho_pearson_n": int(len(rho_p)),
            "rho_spearman_mean": float(rho_s.mean()) if len(rho_s) else float("nan"),
            "rho_spearman_median": float(rho_s.median()) if len(rho_s) else float("nan"),
            "rho_spearman_n": int(len(rho_s)),
            "unshifted_mean_ic": raw["mean_ic"],
            "unshifted_nw_t": raw["nw_tstat"],
            "ortho_mean_ic": summ["mean_ic"],
            "ortho_nw_t": summ["nw_tstat"],
            "ortho_n_days": summ["n_days"],
            "ortho_ic_over_unshifted": ratio,
            "shifted_over_unshifted_ref_top40": 0.058511608345255456 / 0.08032113973646492,
        }
        print(f"[persist] {label} {persist[label]}", flush=True)

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
    by_year = {}
    for row in dead_rows:
        by_year.setdefault(str(row["year"]), []).append(row["symbol"])

    out = {
        "symbol_list_source": "binance_vision_s3_commonprefixes",
        "symbol_list_prefix": "data/futures/um/monthly/klines/",
        "not_exchangeInfo": True,
        "n_listed_parquet": int(len(spans)),
        "n_nonempty": int(len(nonempty)),
        "n_panel_min_history_100": int(len(panel)),
        "n_dropped_short_history": int((nonempty["n"] < 100).sum()),
        "n_last_bar_before_2026_07_all": int(len(dead_all)),
        "n_last_bar_before_2026_07_panel": int(len(dead_panel)),
        "delisted_share_panel": float(len(dead_panel) / len(panel)) if len(panel) else float("nan"),
        "survivorship_only_survivors": bool(len(dead_panel) == 0),
        "baseline_json_invalid_survivorship": False,
        "delisted_by_year_panel": {k: v for k, v in sorted(by_year.items())},
        "delisted_panel_rows": dead_rows,
        "persist": persist,
        "icir_definition": "annualized: mean(daily RankIC) / std(daily RankIC, ddof=1) * sqrt(252)",
        "icir_annualized": True,
        "icir_not_daily": True,
        "noise_floor_note": "SE of daily CS Spearman ~ 1/sqrt(N); N=40 => ~0.16. ICIR 5.90 is the annualized ratio, not mean/std of daily IC without sqrt(252).",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/fase1_survivorship_persist.json").write_text(json.dumps(out, indent=2, default=str))
    print("[surv] wrote results/fase1_survivorship_persist.json", flush=True)
    print("[surv] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
