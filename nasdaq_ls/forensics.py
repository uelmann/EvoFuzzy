"""Forensics: why NASDAQ-LS failed vs crypto A0/COMBO.

Run: python -m nasdaq_ls.forensics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baseline.data import build_pit_topn
from baseline.evaluate import daily_rank_ic, newey_west_t
from nasdaq_ls.eval import last_fold_wins, sharpe, window_from
from nasdaq_ls.prices import close_wide


def _utc(s):
    return pd.to_datetime(s, utc=True).dt.normalize() if hasattr(s, "dt") else pd.to_datetime(s, utc=True).normalize()


def _spearman_daily(df, x, y, min_n=8):
    rows = []
    for dt, g in df.groupby("date", sort=True):
        gg = g.dropna(subset=[x, y])
        if len(gg) < min_n:
            continue
        a = gg[x].to_numpy(float)
        b = gg[y].to_numpy(float)
        if np.unique(a).size < 2 or np.unique(b).size < 2:
            continue
        r = stats.spearmanr(a, b)
        c = getattr(r, "correlation", None)
        if c is None:
            c = getattr(r, "statistic", np.nan)
        c = float(np.asarray(c).reshape(-1)[0])
        if np.isfinite(c):
            rows.append((pd.Timestamp(dt), c))
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(list(vals), index=pd.DatetimeIndex(idx))


def _mean_ic(s):
    s = s.dropna()
    if len(s) < 5:
        return {"mean": float("nan"), "n": int(len(s)), "nw_t": float("nan")}
    return {"mean": float(s.mean()), "n": int(len(s)), "nw_t": float(newey_west_t(s.to_numpy(float), lag=10))}


def quintile_fwd(df, score="score", y="y_simple", nq=5):
    rows = []
    for dt, g in df.groupby("date", sort=True):
        gg = g.dropna(subset=[score, y])
        if len(gg) < nq * 3:
            continue
        try:
            q = pd.qcut(gg[score], nq, labels=False, duplicates="drop")
        except ValueError:
            continue
        tmp = gg.assign(q=q)
        for qi, part in tmp.groupby("q"):
            rows.append({"date": dt, "q": int(qi) + 1, "ret": float(part[y].mean()), "n": int(len(part))})
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    summ = long.groupby("q")["ret"].mean()
    return summ


def ls_from_score(
    df,
    rets_wide,
    top_pct=0.10,
    cost_bps=0.0,
    start="2007-01-01",
    k_fixed=None,
    qqq_r=None,
    beta_col="beta_btc_60_raw",
):
    """Equal-weight long top / short bottom, 1-day hold.

    If qqq_r is provided and beta_col exists, also reports COMBO-style overlay:
    port_beta = 0.5*mean(beta_L) - 0.5*mean(beta_S), hedged = dollar_LS - port_beta * r_QQQ.
    Residual PnL uses name-level r_i - beta_i * r_QQQ (what the label asked for).
    """
    df = df.copy()
    df["date"] = _utc(df["date"])
    cut = pd.Timestamp(start, tz="UTC").normalize()
    dates = sorted(d for d in df["date"].unique() if d >= cut)
    pnl = []
    for i, dt in enumerate(dates[:-1]):
        g = df[df["date"] == dt].dropna(subset=["score", "symbol"])
        if len(g) < 10:
            continue
        g = g.sort_values("score", ascending=False)
        k = int(k_fixed) if k_fixed is not None else max(1, int(np.ceil(top_pct * len(g))))
        k = min(k, max(1, len(g) // 3))
        longs = g["symbol"].head(k).tolist()
        shorts = g["symbol"].tail(k).tolist()
        nxt = dates[i + 1]
        if nxt not in rets_wide.index:
            continue
        row = rets_wide.loc[nxt]
        lr = [float(row[s]) for s in longs if s in row.index and np.isfinite(row[s])]
        sr = [float(row[s]) for s in shorts if s in row.index and np.isfinite(row[s])]
        if not lr or not sr:
            continue
        gross = 0.5 * float(np.mean(lr)) - 0.5 * float(np.mean(sr))
        cost = 2.0 * 0.5 * (cost_bps * 1e-4)
        bL = bS = np.nan
        if beta_col in g.columns:
            gl = g.set_index("symbol")
            bL = float(np.nanmean([gl.loc[s, beta_col] for s in longs if s in gl.index]))
            bS = float(np.nanmean([gl.loc[s, beta_col] for s in shorts if s in gl.index]))
        rq = np.nan
        if qqq_r is not None and nxt in qqq_r.index:
            rq = float(qqq_r.loc[nxt])
        port_beta = 0.5 * bL - 0.5 * bS if np.isfinite(bL) and np.isfinite(bS) else np.nan
        resid = np.nan
        if np.isfinite(port_beta) and np.isfinite(rq):
            resid = gross - port_beta * rq
        pnl.append(
            (nxt, gross - cost, 0.5 * float(np.mean(lr)), -0.5 * float(np.mean(sr)),
             len(lr), len(sr), port_beta, resid, rq, bL, bS)
        )
    if not pnl:
        return {}
    idx = pd.DatetimeIndex([p[0] for p in pnl])
    net = pd.Series([p[1] for p in pnl], index=idx)
    long_leg = pd.Series([p[2] for p in pnl], index=idx)
    short_leg = pd.Series([p[3] for p in pnl], index=idx)
    port_b = pd.Series([p[6] for p in pnl], index=idx)
    hedged = pd.Series([p[7] for p in pnl], index=idx).dropna()
    q = pd.Series([p[8] for p in pnl], index=idx)
    out = {
        "sharpe_net": sharpe(net),
        "sharpe_long": sharpe(long_leg),
        "sharpe_short": sharpe(short_leg),
        "total": float((1 + net).cumprod().iloc[-1] - 1),
        "mean_n_long": float(np.mean([p[4] for p in pnl])),
        "mean_n_short": float(np.mean([p[5] for p in pnl])),
        "n_days": int(len(net)),
        "mean_port_beta": float(port_b.mean()) if port_b.notna().any() else None,
        "mean_beta_long": float(np.nanmean([p[9] for p in pnl])),
        "mean_beta_short": float(np.nanmean([p[10] for p in pnl])),
    }
    if len(hedged) > 20:
        out["sharpe_hedged_combo_style"] = sharpe(hedged)
        out["total_hedged"] = float((1 + hedged).cumprod().iloc[-1] - 1)
    aligned = pd.concat([net.rename("ls"), q.rename("qqq")], axis=1).dropna()
    if len(aligned) > 60 and float(aligned["qqq"].std() or 0) > 0:
        out["ols_beta_ls_vs_qqq"] = float(np.cov(aligned["ls"], aligned["qqq"])[0, 1] / np.var(aligned["qqq"]))
        out["corr_ls_vs_qqq"] = float(aligned["ls"].corr(aligned["qqq"]))
    return out


def mom_12_1(panel, pit, start="2007-01-01"):
    """Classic 12-1 momentum on PIT names: skip last 21 sessions, lookback 252."""
    close = close_wide(panel)
    mom = close.shift(21) / close.shift(252) - 1.0
    rets = close.pct_change(fill_method=None)
    pit = pit.copy()
    pit["date"] = _utc(pit["date"])
    rows = []
    for dt, g in pit.groupby("date", sort=True):
        dt = pd.Timestamp(dt).normalize()
        if dt < pd.Timestamp(start, tz="UTC").normalize():
            continue
        if dt not in mom.index:
            continue
        names = g["symbol"].tolist()
        scores = mom.loc[dt]
        recs = [(s, float(scores[s])) for s in names if s in scores.index and np.isfinite(scores[s])]
        if len(recs) < 10:
            continue
        recs.sort(key=lambda x: -x[1])
        k = max(1, int(np.ceil(0.10 * len(recs))))
        longs = [s for s, _ in recs[:k]]
        shorts = [s for s, _ in recs[-k:]]
        rows.append((dt, longs, shorts))
    pnl = []
    dates = [r[0] for r in rows]
    for i, (dt, longs, shorts) in enumerate(rows[:-1]):
        nxt = rows[i + 1][0]
        if nxt not in rets.index:
            continue
        row = rets.loc[nxt]
        lr = [float(row[s]) for s in longs if s in row.index and np.isfinite(row[s])]
        sr = [float(row[s]) for s in shorts if s in row.index and np.isfinite(row[s])]
        if lr and sr:
            pnl.append((nxt, 0.5 * np.mean(lr) - 0.5 * np.mean(sr)))
    if not pnl:
        return {}
    s = pd.Series({a: b for a, b in pnl})
    return {"sharpe": sharpe(s), "total": float((1 + s).cumprod().iloc[-1] - 1), "n": int(len(s))}


def rev_21(panel, pit, start="2007-01-01"):
    """21-session reversal: short winners / long losers of last 21d."""
    close = close_wide(panel)
    past = close / close.shift(21) - 1.0
    rets = close.pct_change(fill_method=None)
    pit = pit.copy()
    pit["date"] = _utc(pit["date"])
    rows = []
    for dt, g in pit.groupby("date", sort=True):
        dt = pd.Timestamp(dt).normalize()
        if dt < pd.Timestamp(start, tz="UTC").normalize():
            continue
        if dt not in past.index:
            continue
        names = g["symbol"].tolist()
        scores = past.loc[dt]
        recs = [(s, float(scores[s])) for s in names if s in scores.index and np.isfinite(scores[s])]
        if len(recs) < 10:
            continue
        recs.sort(key=lambda x: x[1])  # low past ret = long
        k = max(1, int(np.ceil(0.10 * len(recs))))
        longs = [s for s, _ in recs[:k]]
        shorts = [s for s, _ in recs[-k:]]
        rows.append((dt, longs, shorts))
    pnl = []
    for i, (dt, longs, shorts) in enumerate(rows[:-1]):
        nxt = rows[i + 1][0]
        if nxt not in rets.index:
            continue
        row = rets.loc[nxt]
        lr = [float(row[s]) for s in longs if s in row.index and np.isfinite(row[s])]
        sr = [float(row[s]) for s in shorts if s in row.index and np.isfinite(row[s])]
        if lr and sr:
            pnl.append((nxt, 0.5 * np.mean(lr) - 0.5 * np.mean(sr)))
    if not pnl:
        return {}
    s = pd.Series({a: b for a, b in pnl})
    return {"sharpe": sharpe(s), "total": float((1 + s).cumprod().iloc[-1] - 1), "n": int(len(s))}


def main():
    out = {}
    panel = pd.read_parquet("data/nasdaq/panel.parquet")
    panel["date"] = _utc(panel["date"])
    mkt = pd.read_parquet("data/nasdaq/market.parquet")
    mkt["date"] = _utc(mkt["date"])
    market = mkt.set_index("date")["close"].sort_index()
    market = market[~market.index.duplicated(keep="last")]

    # --- data audit ---
    n_by_year = panel.groupby(panel["date"].dt.year)["symbol"].nunique().to_dict()
    n_by_date = panel.groupby("date")["symbol"].nunique()
    aapl = panel[panel.symbol == "AAPL"].sort_values("date")
    split_win = aapl[(aapl.date >= "2020-08-20") & (aapl.date <= "2020-09-10")]
    div_ratio = float((aapl["adj_close"] / aapl["close_raw"]).median())
    neg = int((panel["close"] <= 0).sum())
    dup = int(panel.duplicated(["date", "symbol"]).sum())
    # AAPL 4:1 split should NOT jump in adj_close
    pre = aapl[aapl.date == "2020-08-28"]
    post = aapl[aapl.date == "2020-08-31"]
    aapl_split = {
        "pre_adj": None if pre.empty else float(pre.adj_close.iloc[0]),
        "post_adj": None if post.empty else float(post.adj_close.iloc[0]),
        "pre_raw": None if pre.empty else float(pre.close_raw.iloc[0]),
        "post_raw": None if post.empty else float(post.close_raw.iloc[0]),
    }
    ipo = (
        panel.groupby("symbol")["date"]
        .min()
        .sort_values()
        .astype(str)
        .tail(15)
        .to_dict()
    )
    out["data"] = {
        "n_symbols": int(panel.symbol.nunique()),
        "rows": int(len(panel)),
        "min": str(panel.date.min()),
        "max": str(panel.date.max()),
        "n_names_min_date": int(n_by_date.min()),
        "n_names_max_date": int(n_by_date.max()),
        "n_names_2005_median": float(n_by_date[n_by_date.index.year == 2005].median()) if (n_by_date.index.year == 2005).any() else None,
        "n_names_2024_median": float(n_by_date[n_by_date.index.year == 2024].median()) if (n_by_date.index.year == 2024).any() else None,
        "n_by_year": {int(k): int(v) for k, v in n_by_year.items()},
        "neg_or_zero_close": neg,
        "duplicate_rows": dup,
        "aapl_median_adj_over_raw": div_ratio,
        "aapl_split_2020": aapl_split,
        "latest_ipo_like_starts": ipo,
        "adj_equals_working_close": bool((panel["close"] == panel["adj_close"]).all()),
        "adj_ne_raw_frac": float((panel["adj_close"] != panel["close_raw"]).mean()),
    }

    pit = build_pit_topn(panel, n=30, window=30)
    pit["date"] = _utc(pit["date"])
    close = close_wide(panel)
    rets = close.pct_change(fill_method=None)
    qqq_r = market.pct_change(fill_method=None)

    # --- load models ---
    def load_book(pred_path, feat_path, ycol):
        preds = pd.read_parquet(pred_path)
        preds["date"] = _utc(preds["date"])
        preds = last_fold_wins(preds)
        drop_y = [c for c in preds.columns if c.startswith("y_")]
        if drop_y:
            preds = preds.drop(columns=drop_y)
        feat = pd.read_parquet(feat_path)
        feat["date"] = _utc(feat["date"])
        cols = ["date", "symbol", ycol]
        simple = ycol.replace("y_h", "y_simple_h")
        if simple in feat.columns:
            cols.append(simple)
        extra = feat[cols].drop_duplicates(["date", "symbol"])
        df = preds.merge(extra, on=["date", "symbol"], how="left")
        df = df.merge(pit[["date", "symbol"]], on=["date", "symbol"], how="inner")
        more = [
            c
            for c in (
                "beta_btc_60_raw",
                "beta_btc_60",
                "ret_7",
                "ret_14",
                "ret_28",
                "ret_90",
                "mom_90_skip14",
                "rev_1",
            )
            if c in feat.columns
        ]
        if more:
            b = feat[["date", "symbol", *more]].drop_duplicates(["date", "symbol"])
            df = df.merge(b, on=["date", "symbol"], how="left")
        return df, simple if simple in df.columns else None

    h10, s10 = load_book("data/nasdaq/preds_h10.parquet", "data/nasdaq/features.parquet", "y_h10")
    h21, s21 = load_book("data/nasdaq21/preds_h21.parquet", "data/nasdaq21/features.parquet", "y_h21")

    def pack_signal(df, ycol, simple_col, tag):
        sub = df[df.date >= pd.Timestamp("2007-01-01", tz="UTC")].copy()
        ic_y = _mean_ic(_spearman_daily(sub, "score", ycol))
        ic_s = _mean_ic(_spearman_daily(sub, "score", simple_col)) if simple_col else {}
        ic_beta = _mean_ic(_spearman_daily(sub, "score", "beta_btc_60_raw")) if "beta_btc_60_raw" in sub.columns else {}
        # trailing 21d return vs score (is the model just momentum?)
        sub = sub.sort_values(["symbol", "date"])
        q = quintile_fwd(sub, "score", simple_col or ycol)
        qy = quintile_fwd(sub, "score", ycol)
        daily = ls_from_score(sub, rets, top_pct=0.10, cost_bps=0.0, qqq_r=qqq_r)
        daily_c = ls_from_score(sub, rets, top_pct=0.10, cost_bps=5.0, qqq_r=qqq_r)
        daily_k10 = ls_from_score(sub, rets, k_fixed=10, cost_bps=0.0, qqq_r=qqq_r)
        # oracle: rank on actual simple forward (lookahead) — upper bound of THIS universe/book
        ora = sub.copy()
        ora["score"] = ora[simple_col] if simple_col else ora[ycol]
        oracle = ls_from_score(ora, rets, top_pct=0.10, cost_bps=0.0)
        feat_ics = {}
        for col in ("ret_7", "ret_14", "ret_28", "ret_90", "mom_90_skip14", "beta_btc_60", "rev_1"):
            if col in sub.columns:
                feat_ics[col] = _mean_ic(_spearman_daily(sub, "score", col))
        # IC by year
        ic_year = {}
        ser = _spearman_daily(sub, "score", ycol)
        for y, g in ser.groupby(ser.index.year):
            ic_year[int(y)] = float(g.mean())
        return {
            "tag": tag,
            "n_rows": int(len(sub)),
            "ic_residual": ic_y,
            "ic_simple": ic_s,
            "ic_vs_beta": ic_beta,
            "quintile_simple_mean": {int(k): float(v) for k, v in q.items()} if len(q) else {},
            "quintile_resid_mean": {int(k): float(v) for k, v in qy.items()} if len(qy) else {},
            "daily_ew_ls_costless": daily,
            "daily_ew_ls_5bps": daily_c,
            "daily_ew_ls_k10_costless": daily_k10,
            "oracle_simple_ls": oracle,
            "score_vs_features": feat_ics,
            "ic_by_year": ic_year,
        }

    out["h10"] = pack_signal(h10, "y_h10", s10, "h10_k10_expanding")
    out["h21"] = pack_signal(h21, "y_h21", s21, "h21_pct10_5yroll")

    # long-only top 10% of PIT30 (no shorts) costless 1d
    def long_only_top(df, start="2007-01-01"):
        df = df[df.date >= pd.Timestamp(start, tz="UTC")].copy()
        dates = sorted(df.date.unique())
        pnl = []
        for i, dt in enumerate(dates[:-1]):
            g = df[df.date == dt].dropna(subset=["score"])
            if len(g) < 10:
                continue
            k = max(1, int(np.ceil(0.10 * len(g))))
            names = g.sort_values("score", ascending=False)["symbol"].head(k).tolist()
            nxt = dates[i + 1]
            if nxt not in rets.index:
                continue
            row = rets.loc[nxt]
            vs = [float(row[s]) for s in names if s in row.index and np.isfinite(row[s])]
            if vs:
                pnl.append((nxt, float(np.mean(vs))))
        s = pd.Series({a: b for a, b in pnl})
        q = qqq_r.reindex(s.index).fillna(0.0)
        # excess vs qqq
        xs = s - q
        return {
            "sharpe": sharpe(s),
            "sharpe_excess_qqq": sharpe(xs),
            "total": float((1 + s).cumprod().iloc[-1] - 1),
            "qqq_sharpe": sharpe(q),
            "n": int(len(s)),
        }

    out["h21_long_only_top10pct"] = long_only_top(h21)
    out["h10_long_only_top10pct"] = long_only_top(h10)

    # book beta vs QQQ: use h21 overlapping daily_ret from report json if present
    j21 = json.loads(Path("reports/nasdaq_ls21_report.json").read_text())
    j10 = json.loads(Path("reports/nasdaq_ls_report.json").read_text())
    out["recorded"] = {
        "h10_sharpe_2007": j10["factor"]["sharpe"],
        "h10_ric_2007": j10["factor"]["ric"],
        "h21_sharpe_2007": j21["factor"]["sharpe"],
        "h21_ric_2007": j21["factor"]["ric"],
        "h10_cost_drag": j10["book_2007"].get("cost_drag"),
        "h21_cost_drag": j21["book_2007"].get("cost_drag"),
        "h10_total_2007": j10["book_2007"].get("total_return"),
        "h21_total_2007": j21["book_2007"].get("total_return"),
    }

    mom_ic = None
    close_m = close_wide(panel)
    mom = close_m.shift(21) / close_m.shift(252) - 1.0
    feat21 = pd.read_parquet("data/nasdaq21/features.parquet")
    feat21["date"] = _utc(feat21["date"])
    sub21 = feat21.merge(pit[["date", "symbol"]], on=["date", "symbol"], how="inner")
    sub21 = sub21[sub21.date >= pd.Timestamp("2007-01-01", tz="UTC")].copy()
    mom_rows = []
    for dt, g in sub21.groupby("date", sort=True):
        dt = pd.Timestamp(dt).normalize()
        if dt not in mom.index:
            continue
        names = g["symbol"].tolist()
        sc = mom.loc[dt]
        g = g.copy()
        g["mom12"] = [float(sc[s]) if s in sc.index else np.nan for s in names]
        mom_rows.append(g)
    if mom_rows:
        mdf = pd.concat(mom_rows, ignore_index=True)
        mom_ic = {
            "ic_vs_simple_h21": _mean_ic(_spearman_daily(mdf, "mom12", "y_simple_h21")),
            "ic_vs_resid_h21": _mean_ic(_spearman_daily(mdf, "mom12", "y_h21")),
        }

    out["sanity_factors_on_same_panel"] = {
        "mom_12_1_ls_10pct_pit30": mom_12_1(panel, pit),
        "rev_21_ls_10pct_pit30": rev_21(panel, pit),
        "mom_12_1_rankic": mom_ic,
    }

    disp = sub21.groupby("date")["y_simple_h21"].std()
    out["cross_section"] = {
        "median_cs_std_simple_h21": float(disp.median()) if len(disp) else None,
        "median_cs_std_resid_h21": float(sub21.groupby("date")["y_h21"].std().median()),
        "mean_n_pit30": float(sub21.groupby("date").size().mean()),
    }

    Path("reports/nasdaq_ls_forensics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
