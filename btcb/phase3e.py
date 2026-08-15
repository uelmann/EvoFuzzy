"""Phase 3.e pricing-gap forensics. Same 3.c positions; no book redesign."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from btcb.binance_replay import (
    _funding_rate,
    _px,
    _ret,
    _sharpe,
    _utc_norm,
    _utc_ts,
    name_cost,
    prepare_position_log,
)
from btcb.constants import (
    ANNUALIZATION,
    LS_TRAIL_DAYS,
    PHASE3E_H,
    PHASE3E_NAME_TIERS,
    PHASE3E_RANKIC_TOL,
    PHASE3E_STALE_BN_MOVE,
    PHASE3E_STALE_CMC_BPS,
    PHASE3E_TOP_N,
)
from btcb.model import per_date_rank_ic_series
from btcb.spread_ls import trail_slice


def _year(ts) -> str:
    return str(int(pd.Timestamp(ts).year))


def name_day_table(
    plog: pd.DataFrame,
    cmc_close: pd.DataFrame,
    spot_wide: pd.DataFrame,
    perp_wide: pd.DataFrame,
    fund_wide: pd.DataFrame,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    """One row per 3.c name-day with CMC vs Binance prints and funding."""
    df = prepare_position_log(plog)
    pit2 = pit.copy()
    pit2["date"] = _utc_norm(pit2["date"])
    pit2["id"] = pit2["id"].astype(int)
    rank_map = {(_utc_ts(r.date), int(r.id)): int(r.rank) for r in pit2.itertuples(index=False)}
    recs = []
    for row in df.itertuples(index=False):
        w = float(row.w)
        if w == 0.0:
            continue
        iid = int(row.id)
        dt, nxt = _utc_ts(row.date), _utc_ts(row.nxt)
        dw = float(row.dw)
        old_w = w - dw
        r_cmc = _ret(cmc_close, dt, nxt, iid)
        if not np.isfinite(r_cmc):
            r_cmc = 0.0
        if w > 0:
            r_bn = _ret(spot_wide, dt, nxt, iid)
            side = "long"
            live = np.isfinite(r_bn)
            fund = 0.0
            fund_ok = False
        else:
            r_bn = _ret(perp_wide, dt, nxt, iid)
            side = "short"
            live = np.isfinite(r_bn)
            f = _funding_rate(fund_wide, nxt, iid)
            fund_ok = bool(np.isfinite(f))
            fund = float(f) if fund_ok else 0.0
        cost = name_cost(old_w, w)
        recs.append(
            {
                "date": dt,
                "nxt": nxt,
                "id": iid,
                "w": w,
                "dw": dw,
                "side": side,
                "r_cmc": float(r_cmc),
                "r_bn": float(r_bn) if live else float("nan"),
                "replayable": bool(live),
                "funding": fund if (live and w < 0) else 0.0,
                "funding_ok": bool(live and w < 0 and fund_ok),
                "fund_pnl": ((-w) * fund) if (live and w < 0) else 0.0,
                "cost": float(cost),
                "contrib_cmc": w * float(r_cmc),
                "contrib_bn": (w * float(r_bn)) if live else 0.0,
                "contrib_diff": (w * (float(r_bn) - float(r_cmc))) if live else 0.0,
                "rank": rank_map.get((dt, iid)),
            }
        )
    out = pd.DataFrame(recs)
    if len(out):
        out["nxt"] = _utc_norm(out["nxt"])
        out["date"] = _utc_norm(out["date"])
        out["year"] = pd.DatetimeIndex(out["nxt"]).year.astype(int).astype(str)
    return out


def replayable_daily(nd: pd.DataFrame) -> dict[str, pd.Series]:
    sub = nd[nd["replayable"]].copy()
    if sub.empty:
        z = pd.Series(dtype=float)
        return {"bn_on": z, "bn_off": z, "cmc_sub": z, "fund": z, "repricing": z}
    g = sub.groupby("nxt", sort=True)
    bn_g = g["contrib_bn"].sum()
    cmc_g = g["contrib_cmc"].sum()
    fund = g["fund_pnl"].sum()
    cost = g["cost"].sum()
    bn_off = bn_g - cost
    bn_on = bn_off + fund
    cmc_sub = cmc_g - cost
    for s in (bn_on, bn_off, cmc_sub, fund):
        s.index = _utc_norm(s.index)
    return {
        "bn_on": bn_on.sort_index(),
        "bn_off": bn_off.sort_index(),
        "cmc_sub": cmc_sub.sort_index(),
        "fund": fund.sort_index(),
        "repricing": (bn_g - cmc_g).sort_index(),
    }


def _sharpe_pack(s: pd.Series) -> dict:
    s = s.astype(float).dropna()
    s.index = _utc_norm(s.index)
    years = {}
    for y, sl in s.groupby(s.index.year):
        years[str(int(y))] = {"n": int(len(sl)), "sharpe": _sharpe(sl), "sum": float(sl.sum())}
    return {
        "n": int(len(s)),
        "sharpe": _sharpe(s),
        "sharpe_trail18m": _sharpe(trail_slice(s, LS_TRAIL_DAYS)),
        "sum": float(s.sum()) if len(s) else float("nan"),
        "by_year": years,
    }


def funding_vs_repricing(daily: dict[str, pd.Series]) -> dict:
    idx = daily["bn_off"].index.union(daily["bn_on"].index).union(daily["cmc_sub"].index)
    a = daily["bn_on"].reindex(idx).fillna(0.0).sort_index()
    b = daily["bn_off"].reindex(idx).fillna(0.0).sort_index()
    c = daily["cmc_sub"].reindex(idx).fillna(0.0).sort_index()
    years = sorted(set(int(y) for y in b.index.year))
    by_year = {}
    for y in years:
        m = b.index.year == y
        on, off, cm = a[m], b[m], c[m]
        by_year[str(y)] = {
            "n": int(m.sum()),
            "sharpe_bn_on": _sharpe(on),
            "sharpe_bn_off": _sharpe(off),
            "sharpe_cmc": _sharpe(cm),
            "d_sharpe_funding": float(_sharpe(on) - _sharpe(off)),
            "d_sharpe_repricing": float(_sharpe(off) - _sharpe(cm)),
            "funding_pnl": float(daily["fund"].reindex(on.index).fillna(0.0).sum()),
            "repricing_pnl": float((off - cm).sum()),
        }
    return {
        "sharpe_bn_on": _sharpe(a),
        "sharpe_bn_off": _sharpe(b),
        "sharpe_cmc_sub": _sharpe(c),
        "d_sharpe_funding": float(_sharpe(a) - _sharpe(b)),
        "d_sharpe_repricing": float(_sharpe(b) - _sharpe(c)),
        "funding_pnl": float(daily["fund"].sum()),
        "repricing_pnl": float((b - c).sum()),
        "by_year": by_year,
        "bn_on": _sharpe_pack(a),
        "bn_off": _sharpe_pack(b),
        "cmc_sub": _sharpe_pack(c),
    }


def by_side(nd: pd.DataFrame) -> dict:
    sub = nd[nd["replayable"]]
    out = {}
    for side in ("long", "short"):
        g = sub[sub["side"] == side]
        daily = g.groupby("nxt")["contrib_diff"].sum()
        daily.index = _utc_norm(daily.index)
        out[side] = {
            "n_name_days": int(len(g)),
            "pnl_diff_sum": float(g["contrib_diff"].sum()),
            "share_of_gap": float("nan"),
            "sharpe_of_daily_diff": _sharpe(daily) if len(daily) else float("nan"),
        }
    tot = float(sub["contrib_diff"].sum()) if len(sub) else float("nan")
    for side in out:
        s = out[side]["pnl_diff_sum"]
        out[side]["share_of_gap"] = (s / tot) if (np.isfinite(tot) and tot != 0) else float("nan")
    out["total_repricing_pnl"] = tot
    return out


def by_tier(nd: pd.DataFrame) -> dict:
    sub = nd[nd["replayable"] & nd["rank"].notna()].copy()
    tot = float(sub["contrib_diff"].sum()) if len(sub) else float("nan")
    out = {}
    for lo, hi, name in PHASE3E_NAME_TIERS:
        g = sub[(sub["rank"] >= lo) & (sub["rank"] <= hi)]
        daily = g.groupby("nxt")["contrib_diff"].sum()
        daily.index = _utc_norm(daily.index)
        s = float(g["contrib_diff"].sum()) if len(g) else 0.0
        out[name] = {
            "n_name_days": int(len(g)),
            "pnl_diff_sum": s,
            "share_of_gap": (s / tot) if (np.isfinite(tot) and tot != 0) else float("nan"),
            "sharpe_of_daily_diff": _sharpe(daily) if len(daily) else float("nan"),
        }
    out["total_repricing_pnl"] = tot
    return out


def _lag_ret(wide: pd.DataFrame, dt, iid) -> float:
    prev = _utc_ts(dt) - pd.Timedelta(days=1)
    return _ret(wide, prev, dt, iid)


def _unchanged_close(wide: pd.DataFrame, dt, iid) -> bool:
    prev = _utc_ts(dt) - pd.Timedelta(days=1)
    a = _px(wide, prev, iid)
    b = _px(wide, dt, iid)
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return False
    return abs(b / a - 1.0) < (PHASE3E_STALE_CMC_BPS * 1e-4)


def classify_stale_row(r_cmc: float, r_bn: float, r_bn_lag: float, cmc_unchanged: bool) -> str:
    stale_flat = bool(cmc_unchanged or (np.isfinite(r_cmc) and abs(r_cmc) < PHASE3E_STALE_CMC_BPS * 1e-4))
    bn_moved = bool(np.isfinite(r_bn) and abs(r_bn) >= float(PHASE3E_STALE_BN_MOVE))
    lagged = False
    if np.isfinite(r_cmc) and np.isfinite(r_bn) and np.isfinite(r_bn_lag):
        err_lag = abs(r_cmc - r_bn_lag)
        err_now = abs(r_cmc - r_bn)
        lagged = bool(
            abs(r_bn_lag) >= float(PHASE3E_STALE_BN_MOVE)
            and err_lag <= min(0.02, 0.25 * abs(r_bn_lag) + 1e-12)
            and err_now > err_lag + 0.02
        )
    if (stale_flat and bn_moved) or lagged:
        return "STALE"
    if (
        np.isfinite(r_cmc)
        and np.isfinite(r_bn)
        and abs(r_cmc) >= float(PHASE3E_STALE_BN_MOVE)
        and abs(r_bn) >= float(PHASE3E_STALE_BN_MOVE)
        and (r_cmc * r_bn) > 0
    ):
        return "LEVEL-DIFF"
    return "OTHER"


def classify_stale(nd: pd.DataFrame, cmc_close: pd.DataFrame, spot_wide: pd.DataFrame, perp_wide: pd.DataFrame) -> pd.DataFrame:
    sub = nd[nd["replayable"]].copy()
    labels = []
    lags = []
    for row in sub.itertuples(index=False):
        wide = spot_wide if row.side == "long" else perp_wide
        r_lag = _lag_ret(wide, row.date, int(row.id))
        unch = _unchanged_close(cmc_close, row.date, int(row.id))
        labels.append(classify_stale_row(float(row.r_cmc), float(row.r_bn), r_lag, unch))
        lags.append(float(r_lag) if np.isfinite(r_lag) else float("nan"))
    sub = sub.copy()
    sub["stale_class"] = labels
    sub["r_bn_lag"] = lags
    return sub


def concentration_and_stale(classified: pd.DataFrame) -> dict:
    tot = float(classified["contrib_diff"].sum()) if len(classified) else 0.0
    abs_tot = float(classified["contrib_diff"].abs().sum()) if len(classified) else 0.0
    top = classified.reindex(classified["contrib_diff"].abs().sort_values(ascending=False).index).head(int(PHASE3E_TOP_N))
    by_cls = {}
    for lab in ("STALE", "LEVEL-DIFF", "OTHER"):
        g = classified[classified["stale_class"] == lab]
        s = float(g["contrib_diff"].sum()) if len(g) else 0.0
        by_cls[lab] = {
            "n": int(len(g)),
            "pnl_diff_sum": s,
            "share_of_gap": (s / tot) if tot != 0 else float("nan"),
            "share_of_abs": (float(g["contrib_diff"].abs().sum()) / abs_tot) if abs_tot else float("nan"),
        }
    top_rows = []
    for row in top.itertuples(index=False):
        top_rows.append(
            {
                "date": str(pd.Timestamp(row.nxt).date()),
                "id": int(row.id),
                "side": row.side,
                "w": float(row.w),
                "r_cmc": float(row.r_cmc),
                "r_bn": float(row.r_bn),
                "d_r": float(row.r_bn) - float(row.r_cmc),
                "contrib_diff": float(row.contrib_diff),
                "stale_class": row.stale_class,
                "rank": int(row.rank) if row.rank is not None and np.isfinite(row.rank) else None,
            }
        )
    top_sum = float(top["contrib_diff"].sum()) if len(top) else 0.0
    top_abs = float(top["contrib_diff"].abs().sum()) if len(top) else 0.0
    return {
        "total_repricing_pnl": tot,
        "top_n": int(PHASE3E_TOP_N),
        "top_share_of_gap": (top_sum / tot) if tot != 0 else float("nan"),
        "top_share_of_abs": (top_abs / abs_tot) if abs_tot else float("nan"),
        "top_rows": top_rows,
        "by_class": by_cls,
        "stale_share_of_gap": by_cls["STALE"]["share_of_gap"],
        "n_replayable": int(len(classified)),
    }


def fwd_excess_wide(close: pd.DataFrame, btc_id: int, h: int = PHASE3E_H) -> pd.DataFrame:
    if close is None or close.empty or int(btc_id) not in close.columns:
        return pd.DataFrame()
    c = close.sort_index().copy()
    c.index = _utc_norm(c.index)
    idx = pd.date_range(c.index.min(), c.index.max(), freq="D", tz="UTC")
    c = c.reindex(idx)
    logp = np.log(c.clip(lower=1e-18))
    fwd = logp.shift(-int(h)) - logp
    btc = fwd[int(btc_id)]
    return fwd.sub(btc, axis=0)


def combine_bn_close(spot_wide: pd.DataFrame, perp_wide: pd.DataFrame) -> pd.DataFrame:
    """Spot preferred; perp fills missing ids / missing dates."""
    frames = []
    if spot_wide is not None and not spot_wide.empty:
        s = spot_wide.copy()
        s.index = _utc_norm(s.index)
        frames.append(s)
    if perp_wide is not None and not perp_wide.empty:
        p = perp_wide.copy()
        p.index = _utc_norm(p.index)
        frames.append(p)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        out = out.combine_first(extra)
    return out.sort_index()


def rankic_on_prices(
    twin: pd.DataFrame,
    pit: pd.DataFrame,
    cmc_ex: pd.DataFrame,
    bn_ex: pd.DataFrame,
    h: int = PHASE3E_H,
) -> dict:
    pr = twin.copy()
    pr["date"] = pd.to_datetime(pr["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pr["id"] = pr["id"].astype(int)
    pr = pr.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
    u = pit.copy()
    u["date"] = pd.to_datetime(u["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    u["id"] = u["id"].astype(int)
    m = pr.merge(u[["date", "id"]], on=["date", "id"], how="inner")
    m["date"] = pd.to_datetime(m["date"], utc=True).dt.tz_convert("UTC").dt.normalize()

    def _stack(ex: pd.DataFrame, col: str) -> pd.DataFrame:
        if ex is None or ex.empty:
            return pd.DataFrame(columns=["date", "id", col])
        s = ex.copy()
        s.index = _utc_norm(s.index)
        long = s.stack().rename(col).reset_index()
        long.columns = ["date", "id", col]
        long["date"] = pd.to_datetime(long["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        long["id"] = long["id"].astype(int)
        long[col] = pd.to_numeric(long[col], errors="coerce")
        return long

    both = m[["date", "id", "spread"]].merge(_stack(cmc_ex, "ex_cmc"), on=["date", "id"], how="left")
    both = both.merge(_stack(bn_ex, "ex_bn"), on=["date", "id"], how="left")
    both = both[np.isfinite(both["ex_bn"]) & np.isfinite(both["ex_cmc"]) & np.isfinite(both["spread"])]
    ic_bn = per_date_rank_ic_series(both["spread"].to_numpy(), both["ex_bn"].to_numpy(), both["date"].to_numpy())
    ic_cmc = per_date_rank_ic_series(both["spread"].to_numpy(), both["ex_cmc"].to_numpy(), both["date"].to_numpy())
    ic_bn = ic_bn.replace([np.inf, -np.inf], np.nan).dropna()
    ic_cmc = ic_cmc.replace([np.inf, -np.inf], np.nan).dropna()
    ic_bn, ic_cmc = ic_bn.align(ic_cmc, join="inner")

    def _win(s: pd.Series) -> dict:
        trail = trail_slice(s, LS_TRAIL_DAYS)
        return {
            "full": float(s.mean()) if len(s) else float("nan"),
            "trail18m": float(trail.mean()) if len(trail) else float("nan"),
            "n_full": int(len(s)),
            "n_trail": int(len(trail)),
        }

    buckets = []
    if len(both):
        tmp = both.copy()
        tmp["q"] = np.nan
        for dt, g in tmp.groupby("date"):
            if g["spread"].nunique() < 5 or len(g) < 10:
                continue
            try:
                qq = pd.qcut(g["spread"], 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            tmp.loc[g.index, "q"] = qq
        for q in range(5):
            sl = tmp[tmp["q"] == q]
            buckets.append(
                {
                    "quintile": int(q) + 1,
                    "n": int(len(sl)),
                    "mean_ex_bn": float(sl["ex_bn"].mean()) if len(sl) else float("nan"),
                    "mean_ex_cmc": float(sl["ex_cmc"].mean()) if len(sl) else float("nan"),
                }
            )
    return {
        "binance": _win(ic_bn),
        "cmc_same_names": _win(ic_cmc),
        "n_name_dates": int(len(both)),
        "n_dates": int(ic_bn.index.nunique()) if len(ic_bn) else 0,
        "horizon": int(h),
        "buckets": buckets,
        "ic_bn": ic_bn,
        "ic_cmc": ic_cmc,
    }


def signal_verdict(rankic: dict) -> dict:
    bn = rankic.get("binance") or {}
    cmc = rankic.get("cmc_same_names") or {}
    tol = float(PHASE3E_RANKIC_TOL)
    d_full = float(bn.get("full") - cmc.get("full")) if np.isfinite(bn.get("full")) and np.isfinite(cmc.get("full")) else float("nan")
    d_tr = (
        float(bn.get("trail18m") - cmc.get("trail18m"))
        if np.isfinite(bn.get("trail18m")) and np.isfinite(cmc.get("trail18m"))
        else float("nan")
    )
    ok_full = bool(np.isfinite(d_full) and d_full + 1e-12 >= -tol)
    ok_tr = bool(np.isfinite(d_tr) and d_tr + 1e-12 >= -tol)
    confirmed = bool(ok_full and ok_tr)
    return {
        "label": "SIGNAL-CONFIRMED" if confirmed else "SIGNAL-PARTLY-ARTIFACT",
        "confirmed": confirmed,
        "d_full": d_full,
        "d_trail": d_tr,
        "need_tol": tol,
        "pass_full": ok_full,
        "pass_trail": ok_tr,
        "rankic_bn_full": bn.get("full"),
        "rankic_cmc_full": cmc.get("full"),
        "rankic_bn_trail": bn.get("trail18m"),
        "rankic_cmc_trail": cmc.get("trail18m"),
    }


def funding_structure(nd: pd.DataFrame, fund_wide: pd.DataFrame, shortable: dict) -> dict:
    shorts = nd[(nd["side"] == "short") & nd["replayable"]].copy()
    by_year = {}
    if len(shorts):
        for y, g in shorts.groupby("year"):
            by_year[str(y)] = {
                "n_name_days": int(len(g)),
                "funding_pnl": float(g["fund_pnl"].sum()),
                "mean_rate": float(g["funding"].mean()) if len(g) else float("nan"),
            }
    held_num = held_den = 0.0
    univ_num = univ_den = 0.0
    n_days = 0
    if not fund_wide.empty and len(shorts):
        for nxt, g in shorts.groupby("nxt"):
            nxt = _utc_ts(nxt)
            wabs = g["w"].abs()
            f = g["funding"]
            if float(wabs.sum()) > 0:
                held_num += float((wabs * f).sum())
                held_den += float(wabs.sum())
            # universe: shortable ids with a funding print that day
            dt = _utc_ts(g["date"].iloc[0]) if "date" in g.columns else nxt
            sh = shortable.get(_utc_ts(g["date"].iloc[0]), set()) if shortable else set()
            if nxt in fund_wide.index and sh:
                row = fund_wide.loc[nxt]
                xs = []
                for iid in sh:
                    if iid in row.index:
                        v = row[iid]
                        if v is not None and np.isfinite(float(v)):
                            xs.append(float(v))
                if xs:
                    univ_num += float(np.mean(xs))
                    univ_den += 1.0
            n_days += 1
    held = (held_num / held_den) if held_den else float("nan")
    univ = (univ_num / univ_den) if univ_den else float("nan")
    return {
        "by_year": by_year,
        "funding_pnl_total": float(shorts["fund_pnl"].sum()) if len(shorts) else 0.0,
        "held_mean_rate": held,
        "universe_mean_rate": univ,
        "held_bps_day": float(held * 1e4) if np.isfinite(held) else float("nan"),
        "universe_bps_day": float(univ * 1e4) if np.isfinite(univ) else float("nan"),
        "delta_bps_day": float((held - univ) * 1e4) if np.isfinite(held) and np.isfinite(univ) else float("nan"),
        "n_short_name_days": int(len(shorts)),
        "n_days": int(n_days),
        "note": (
            "Positive funding rate: longs pay shorts. Held rate below universe "
            "means we are short cheaper (more negative) funding names."
        ),
    }


def never_listed_contribution(nd: pd.DataFrame, never_ids: set[int]) -> dict:
    longs = nd[nd["side"] == "long"]
    never = longs[longs["id"].isin(never_ids)]
    tot = float(nd["contrib_cmc"].sum()) if len(nd) else 0.0
    s = float(never["contrib_cmc"].sum()) if len(never) else 0.0
    return {
        "n_names": int(len(never_ids)),
        "n_name_days": int(len(never)),
        "pnl_cmc": s,
        "cmc_book_pnl": tot,
        "share": (s / tot) if tot != 0 else float("nan"),
    }


def gap_waterfall_shares(fr: dict, stale: dict) -> dict:
    """PnL identity: (BN_on − CMC_sub) = funding + stale_repricing + diffuse_repricing."""
    fund = float(fr.get("funding_pnl") or 0.0)
    repricing = float(fr.get("repricing_pnl") or 0.0)
    stale_pnl = float((stale.get("by_class") or {}).get("STALE", {}).get("pnl_diff_sum") or 0.0)
    diffuse = repricing - stale_pnl
    total = fund + repricing
    def _pct(x):
        return (x / total) if total != 0 else float("nan")
    return {
        "funding_pnl": fund,
        "stale_pnl": stale_pnl,
        "diffuse_pnl": diffuse,
        "total_pnl_gap": total,
        "pct_funding": _pct(fund),
        "pct_stale": _pct(stale_pnl),
        "pct_diffuse": _pct(diffuse),
        "d_sharpe_funding": fr.get("d_sharpe_funding"),
        "d_sharpe_repricing": fr.get("d_sharpe_repricing"),
    }
