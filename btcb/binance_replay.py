"""Phase 3.c: replay SPREAD-LS positions on Binance prices + native funding.

Positions come from a frozen log. Signals are not recomputed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.constants import (
    ANNUALIZATION,
    LS_LONG_BPS,
    LS_SHORT_FEE_BPS,
    LS_SHORT_SLIP_BPS,
    PHASE3C_CORR_MIN,
    PHASE3C_NAME_TIERS,
    PHASE3C_SHARPE_TOL,
)
from btcb.spread_ls import _cmc_bases, _hash_position_log, _sharpe, summarize_ls


LONG_C = LS_LONG_BPS * 1e-4
SHORT_C = (LS_SHORT_FEE_BPS + LS_SHORT_SLIP_BPS) * 1e-4


def _utc_norm(s) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(s, utc=True)).tz_convert("UTC").normalize()


def _utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()


def name_cost(old_w: float, new_w: float) -> float:
    m = float(new_w) - float(old_w)
    if m == 0.0:
        return 0.0
    if old_w >= 0 and new_w >= 0:
        return abs(m) * LONG_C
    if old_w <= 0 and new_w <= 0:
        return abs(m) * SHORT_C
    return abs(old_w) * (LONG_C if old_w > 0 else SHORT_C) + abs(new_w) * (
        LONG_C if new_w > 0 else SHORT_C
    )


def candidate_usdt_symbols(raw_symbol: str) -> list[str]:
    """Preference-ordered Binance USDT tickers for a CMC base symbol."""
    out: list[str] = []
    seen: set[str] = set()
    for base in _cmc_bases(raw_symbol):
        for cand in (base + "USDT", "1000" + base + "USDT"):
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def symbols_for_id(panel: pd.DataFrame, iid: int) -> list[str]:
    sub = panel.loc[panel["id"].astype(int) == int(iid), "symbol"]
    if sub.empty:
        return []
    # last-seen first, then historical uniques
    last = str(sub.iloc[-1])
    ordered = [last] + [str(s) for s in sub.unique() if str(s) != last]
    cands: list[str] = []
    seen: set[str] = set()
    for raw in ordered:
        for c in candidate_usdt_symbols(raw):
            if c not in seen:
                seen.add(c)
                cands.append(c)
    return cands


def pick_symbol(cands: list[str], available: set[str], nonempty: set[str] | None = None) -> str | None:
    live = nonempty if nonempty is not None else available
    for c in cands:
        if c in live:
            return c
    for c in cands:
        if c in available:
            return c
    return None


def build_id_symbol_map(
    ids: list[int],
    panel: pd.DataFrame,
    available: set[str],
    nonempty: set[str] | None = None,
) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    for iid in ids:
        out[int(iid)] = pick_symbol(symbols_for_id(panel, int(iid)), available, nonempty)
    return out


def close_wide_from_panel(kline_panel: pd.DataFrame, id_to_sym: dict[int, str | None]) -> pd.DataFrame:
    if kline_panel is None or kline_panel.empty:
        return pd.DataFrame()
    kl = kline_panel.copy()
    kl["date"] = _utc_norm(kl["date"])
    kl["symbol"] = kl["symbol"].astype(str).str.upper()
    wide = kl.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    cols = {}
    for iid, sym in id_to_sym.items():
        if not sym or sym not in wide.columns:
            continue
        cols[int(iid)] = wide[sym]
    if not cols:
        return pd.DataFrame(index=wide.index)
    out = pd.DataFrame(cols).sort_index()
    out.index = _utc_norm(out.index)
    return out


def funding_wide_from_panel(funding: pd.DataFrame, id_to_sym: dict[int, str | None]) -> pd.DataFrame:
    if funding is None or funding.empty:
        return pd.DataFrame()
    f = funding.copy()
    f["date"] = _utc_norm(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.upper()
    wide = f.pivot_table(index="date", columns="symbol", values="funding_rate", aggfunc="sum").sort_index()
    cols = {}
    for iid, sym in id_to_sym.items():
        if not sym or sym not in wide.columns:
            continue
        cols[int(iid)] = wide[sym]
    if not cols:
        return pd.DataFrame(index=wide.index)
    out = pd.DataFrame(cols).sort_index()
    out.index = _utc_norm(out.index)
    return out


def _px(wide: pd.DataFrame, when, iid) -> float:
    if wide is None or wide.empty:
        return float("nan")
    when = _utc_ts(when)
    iid = int(iid)
    if when not in wide.index or iid not in wide.columns:
        return float("nan")
    v = wide.at[when, iid]
    v = float(v) if v is not None else float("nan")
    if np.isfinite(v) and v > 0:
        return v
    return float("nan")


def _funding_rate(wide: pd.DataFrame, when, iid) -> float:
    """Funding can be negative; unlike prices, zero/negative is valid."""
    if wide is None or wide.empty:
        return float("nan")
    when = _utc_ts(when)
    iid = int(iid)
    if when not in wide.index or iid not in wide.columns:
        return float("nan")
    v = wide.at[when, iid]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _ret(wide: pd.DataFrame, dt, nxt, iid) -> float:
    a = _px(wide, dt, iid)
    b = _px(wide, nxt, iid)
    if np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0:
        return b / a - 1.0
    return float("nan")


def replayable_pair(wide: pd.DataFrame, dt, nxt, iid) -> bool:
    return np.isfinite(_ret(wide, dt, nxt, iid))


def prepare_position_log(plog: pd.DataFrame) -> pd.DataFrame:
    df = plog.copy()
    df["date"] = _utc_norm(df["date"])
    df["nxt"] = _utc_norm(df["nxt"])
    df["id"] = df["id"].astype(int)
    df["w"] = df["w"].astype(float)
    df["dw"] = df["dw"].astype(float)
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def coverage_tables(
    plog: pd.DataFrame,
    spot_wide: pd.DataFrame,
    perp_wide: pd.DataFrame,
    id_to_spot: dict[int, str | None],
    id_to_perp: dict[int, str | None],
    id_to_sym: dict,
) -> dict:
    df = prepare_position_log(plog)
    longs = df[df["w"] > 0].copy()
    shorts = df[df["w"] < 0].copy()

    def _side_cov(side: pd.DataFrame, wide: pd.DataFrame, id_map: dict[int, str | None], label: str) -> dict:
        if side.empty:
            return {
                "label": label,
                "n_name_days": 0,
                "n_replayable": 0,
                "pct_replayable": float("nan"),
                "by_year": {},
                "never_listed": [],
            }
        flags = []
        for row in side.itertuples(index=False):
            flags.append(bool(replayable_pair(wide, row.date, row.nxt, int(row.id))))
        side = side.copy()
        side["replayable"] = flags
        side["year"] = pd.DatetimeIndex(side["nxt"]).year.astype(int)
        by_year = {}
        for y, g in side.groupby("year"):
            n = int(len(g))
            k = int(g["replayable"].sum())
            by_year[str(int(y))] = {
                "n_name_days": n,
                "n_replayable": k,
                "pct_replayable": (k / n) if n else float("nan"),
            }
        n = int(len(side))
        k = int(side["replayable"].sum())
        ever = sorted(set(int(i) for i in side["id"].unique()))
        never = []
        for iid in ever:
            mapped = id_map.get(int(iid))
            if not mapped:
                never.append(
                    {
                        "id": int(iid),
                        "symbol": id_to_sym.get(int(iid)),
                        "reason": "never_listed",
                    }
                )
                continue
            sub = side[side["id"] == int(iid)]
            if int(sub["replayable"].sum()) == 0:
                never.append(
                    {
                        "id": int(iid),
                        "symbol": id_to_sym.get(int(iid)),
                        "mapped": mapped,
                        "reason": "mapped_but_no_replayable_days",
                    }
                )
        return {
            "label": label,
            "n_name_days": n,
            "n_replayable": k,
            "pct_replayable": (k / n) if n else float("nan"),
            "by_year": by_year,
            "never_listed": never,
            "n_never_listed_names": int(len(never)),
        }

    long_c = _side_cov(longs, spot_wide, id_to_spot, "long_spot")
    short_c = _side_cov(shorts, perp_wide, id_to_perp, "short_perp")
    return {"long": long_c, "short": short_c}


def _empty_stats_like(index: pd.DatetimeIndex, n_long, n_short, n_shortable, incomplete, long_gross, short_gross, cash, to_list):
    z = pd.Series(0.0, index=index, dtype=float)
    return {
        "to_list": list(to_list) if to_list is not None else [0.0] * len(index),
        "n_long": n_long.reindex(index).fillna(0) if n_long is not None else z,
        "n_short": n_short.reindex(index).fillna(0) if n_short is not None else z,
        "n_shortable": n_shortable.reindex(index).fillna(0) if n_shortable is not None else z,
        "incomplete": incomplete.reindex(index).fillna(0) if incomplete is not None else z,
        "long_gross": long_gross.reindex(index).fillna(0) if long_gross is not None else z,
        "short_gross": short_gross.reindex(index).fillna(0) if short_gross is not None else z,
        "cash": cash.reindex(index).fillna(0) if cash is not None else z,
    }


def replay_three_books(
    plog: pd.DataFrame,
    cmc_close: pd.DataFrame,
    spot_wide: pd.DataFrame,
    perp_wide: pd.DataFrame,
    fund_wide: pd.DataFrame,
    engine_packed: dict,
) -> dict:
    """Replay CMC / hybrid / binance-only from one position log.

    CMC must match the engine daily_ret (funding=0). Hybrid uses Binance
    where both entry and exit prints exist, else CMC. Binance-only keeps
    only fully replayable name-days. Funding accrues on short weights at
    the 8h daily sum, applied on nxt (same convention as COMBO).
    """
    df = prepare_position_log(plog)
    engine_rets = engine_packed["daily_ret"].copy()
    engine_rets.index = _utc_norm(engine_rets.index)
    engine_cost = engine_packed.get("daily_cost")
    if engine_cost is None:
        engine_cost = pd.Series(0.0, index=engine_rets.index)
    else:
        engine_cost = pd.Series(engine_cost, dtype=float)
        engine_cost.index = _utc_norm(engine_cost.index)

    days = df.groupby(["date", "nxt"], sort=True)
    recs = []
    disagreements = []
    missing_funding = 0
    funding_events = 0

    for (dt, nxt), g in days:
        dt = _utc_ts(dt)
        nxt = _utc_ts(nxt)
        cmc_g = 0.0
        hyb_g = 0.0
        bn_g = 0.0
        cmc_sub_g = 0.0
        hyb_fund = 0.0
        bn_fund = 0.0
        hyb_cmc_nd = 0
        hyb_bn_nd = 0
        bn_nd = 0
        sub_cost = 0.0
        n_long_bn = 0
        n_short_bn = 0

        for row in g.itertuples(index=False):
            iid = int(row.id)
            w = float(row.w)
            dw = float(row.dw)
            old_w = w - dw
            r_cmc = _ret(cmc_close, dt, nxt, iid)
            if not np.isfinite(r_cmc):
                r_cmc = 0.0
            if w > 0:
                r_bn = _ret(spot_wide, dt, nxt, iid)
                live = np.isfinite(r_bn)
            elif w < 0:
                r_bn = _ret(perp_wide, dt, nxt, iid)
                live = np.isfinite(r_bn)
            else:
                r_bn = float("nan")
                live = False

            if np.isfinite(w) and w != 0.0 and np.isfinite(r_cmc):
                cmc_g += w * r_cmc

            if w != 0.0:
                if live:
                    hyb_g += w * float(r_bn)
                    hyb_bn_nd += 1
                    if w > 0:
                        n_long_bn += 1
                    else:
                        n_short_bn += 1
                    disagreements.append(
                        {
                            "date": str(nxt.date()),
                            "id": iid,
                            "w": w,
                            "r_cmc": float(r_cmc),
                            "r_bn": float(r_bn),
                            "d_r": float(r_bn) - float(r_cmc),
                            "contrib_diff": w * (float(r_bn) - float(r_cmc)),
                            "side": "long" if w > 0 else "short",
                        }
                    )
                    # funding on short leg only, Binance-priced days
                    if w < 0:
                        f = _funding_rate(fund_wide, nxt, iid)
                        if np.isfinite(f):
                            fp = -w * f
                            hyb_fund += fp
                            bn_fund += fp
                            funding_events += 1
                        else:
                            missing_funding += 1
                    bn_g += w * float(r_bn)
                    cmc_sub_g += w * r_cmc
                    bn_nd += 1
                    sub_cost += name_cost(old_w, w)
                else:
                    hyb_g += w * r_cmc
                    hyb_cmc_nd += 1

        recs.append(
            {
                "nxt": nxt,
                "cmc_gross": cmc_g,
                "hyb_gross": hyb_g,
                "bn_gross": bn_g,
                "cmc_sub_gross": cmc_sub_g,
                "hyb_fund": hyb_fund,
                "bn_fund": bn_fund,
                "hyb_cmc_nd": hyb_cmc_nd,
                "hyb_bn_nd": hyb_bn_nd,
                "bn_nd": bn_nd,
                "sub_cost": sub_cost,
                "n_long_bn": n_long_bn,
                "n_short_bn": n_short_bn,
            }
        )

    daily = pd.DataFrame(recs).set_index("nxt").sort_index()
    daily.index = _utc_norm(daily.index)
    cost = engine_cost.reindex(daily.index).fillna(0.0)

    cmc_net = daily["cmc_gross"] - cost
    hyb_net = daily["hyb_gross"] - cost + daily["hyb_fund"]
    bn_net = daily["bn_gross"] - daily["sub_cost"] + daily["bn_fund"]
    cmc_sub_net = daily["cmc_sub_gross"] - daily["sub_cost"]

    # sanity vs engine
    eng = engine_rets.reindex(cmc_net.index)
    gap = (cmc_net - eng).abs()
    max_abs = float(gap.max()) if len(gap) else float("nan")
    if not np.isfinite(max_abs) or max_abs > 1e-8:
        # allow tiny float noise; hard fail above 1e-6
        if not np.isfinite(max_abs) or max_abs > 1e-6:
            raise RuntimeError(f"BOOK-CMC replay diverged from engine: max_abs={max_abs}")

    to_map = df.groupby("nxt")["dw"].apply(lambda s: 0.5 * float(np.abs(s).sum()))
    to_map.index = _utc_norm(to_map.index)
    to_list = [float(to_map.get(i, 0.0)) for i in daily.index]

    def _re(s, fill=0.0):
        if s is None:
            return pd.Series(fill, index=daily.index, dtype=float)
        out = pd.Series(s, dtype=float)
        out.index = _utc_norm(out.index)
        return out.reindex(daily.index).fillna(fill)

    stats_full = {
        "to_list": to_list,
        "n_long": _re(engine_packed.get("n_long")),
        "n_short": _re(engine_packed.get("n_short")),
        "n_shortable": _re(engine_packed.get("n_shortable")),
        "incomplete": _re(engine_packed.get("incomplete")),
        "long_gross": _re(engine_packed.get("long_gross")),
        "short_gross": _re(engine_packed.get("short_gross")),
        "cash": _re(engine_packed.get("cash")),
        "daily_cost": cost,
        "daily_gross": daily["cmc_gross"],
    }

    cmc_book = summarize_ls(cmc_net, {**stats_full, "daily_gross": daily["cmc_gross"], "daily_cost": cost})
    hyb_stats = {
        **stats_full,
        "daily_gross": daily["hyb_gross"],
        "daily_cost": cost,
        "daily_funding": daily["hyb_fund"],
    }
    hyb_book = summarize_ls(hyb_net, hyb_stats)
    hyb_book["daily_funding"] = daily["hyb_fund"]
    hyb_book["funding_total_pnl"] = float(daily["hyb_fund"].sum())
    hyb_book["funding_share_of_gross"] = (
        float(daily["hyb_fund"].sum() / daily["hyb_gross"].abs().sum())
        if float(daily["hyb_gross"].abs().sum()) > 0
        else float("nan")
    )
    hyb_book["hybrid_cmc_name_days"] = int(daily["hyb_cmc_nd"].sum())
    hyb_book["hybrid_bn_name_days"] = int(daily["hyb_bn_nd"].sum())
    hyb_book["hybrid_flagged_share"] = (
        float(daily["hyb_cmc_nd"].sum() / (daily["hyb_cmc_nd"].sum() + daily["hyb_bn_nd"].sum()))
        if (daily["hyb_cmc_nd"].sum() + daily["hyb_bn_nd"].sum())
        else float("nan")
    )

    bn_n_long = pd.Series(daily["n_long_bn"].to_numpy(), index=daily.index, dtype=float)
    bn_n_short = pd.Series(daily["n_short_bn"].to_numpy(), index=daily.index, dtype=float)
    bn_stats = {
        "to_list": to_list,
        "n_long": bn_n_long,
        "n_short": bn_n_short,
        "n_shortable": _re(engine_packed.get("n_shortable")),
        "incomplete": _re(engine_packed.get("incomplete")),
        "long_gross": _re(engine_packed.get("long_gross")),
        "short_gross": _re(engine_packed.get("short_gross")),
        "cash": _re(engine_packed.get("cash")),
        "daily_cost": daily["sub_cost"],
        "daily_gross": daily["bn_gross"],
        "daily_funding": daily["bn_fund"],
    }
    bn_book = summarize_ls(bn_net, bn_stats)
    bn_book["daily_funding"] = daily["bn_fund"]
    bn_book["funding_total_pnl"] = float(daily["bn_fund"].sum())

    cmc_sub_stats = {
        **bn_stats,
        "daily_gross": daily["cmc_sub_gross"],
        "daily_funding": pd.Series(0.0, index=daily.index),
    }
    cmc_sub_book = summarize_ls(cmc_sub_net, cmc_sub_stats)

    # validation on overlapping days (all days; subset may be zero-gross some days)
    a = bn_net.astype(float)
    b = cmc_sub_net.astype(float)
    corr = float(a.corr(b)) if len(a) > 5 else float("nan")
    sh_bn = _sharpe(a)
    sh_cmc = _sharpe(b)
    validated = bool(
        np.isfinite(corr)
        and corr >= float(PHASE3C_CORR_MIN)
        and np.isfinite(sh_bn)
        and np.isfinite(sh_cmc)
        and sh_bn >= (sh_cmc - float(PHASE3C_SHARPE_TOL))
    )

    ddf = pd.DataFrame(disagreements)
    if len(ddf):
        ddf["abs_contrib"] = ddf["contrib_diff"].abs()
        top = ddf.sort_values("abs_contrib", ascending=False).head(10)
        top10 = top.to_dict(orient="records")
    else:
        top10 = []

    n_hyb = int(daily["hyb_cmc_nd"].sum() + daily["hyb_bn_nd"].sum())
    return {
        "cmc": cmc_book,
        "hybrid": hyb_book,
        "binance_only": bn_book,
        "cmc_subset": cmc_sub_book,
        "daily": daily,
        "cmc_net": cmc_net,
        "hybrid_net": hyb_net,
        "bn_net": bn_net,
        "cmc_sub_net": cmc_sub_net,
        "sanity": {
            "max_abs_daily_diff": max_abs,
            "n_days": int(len(cmc_net)),
            "position_sha256": engine_packed.get("position_sha256") or _hash_position_log(df),
        },
        "validation": {
            "corr": corr,
            "sharpe_binance_only": sh_bn,
            "sharpe_cmc_subset": sh_cmc,
            "need_corr": float(PHASE3C_CORR_MIN),
            "need_sharpe": float(sh_cmc - PHASE3C_SHARPE_TOL) if np.isfinite(sh_cmc) else float("nan"),
            "sharpe_gap": float(sh_bn - sh_cmc) if (np.isfinite(sh_bn) and np.isfinite(sh_cmc)) else float("nan"),
            "validated": validated,
            "n_days": int(len(a)),
        },
        "top_disagreements": top10,
        "funding_events": int(funding_events),
        "missing_funding_name_days": int(missing_funding),
        "hybrid_flagged_share": hyb_book.get("hybrid_flagged_share"),
        "n_hybrid_name_days": n_hyb,
    }


def discrepancy_tables(
    daily: pd.DataFrame,
    plog: pd.DataFrame,
    pit: pd.DataFrame,
    spot_wide: pd.DataFrame,
    perp_wide: pd.DataFrame,
    cmc_close: pd.DataFrame,
) -> dict:
    """Per-year and per-name-tier PnL gap on the replayable subset (used if NOT validated)."""
    bn = daily["bn_gross"] - daily["sub_cost"] + daily["bn_fund"]
    cmc = daily["cmc_sub_gross"] - daily["sub_cost"]
    idx = bn.index
    years = {}
    for y, sl in bn.groupby(idx.year):
        b = sl
        c = cmc.reindex(b.index)
        years[str(int(y))] = {
            "n": int(len(b)),
            "corr": float(b.corr(c)) if len(b) > 5 else float("nan"),
            "sharpe_bn": _sharpe(b),
            "sharpe_cmc": _sharpe(c),
            "sharpe_gap": float(_sharpe(b) - _sharpe(c)),
            "pnl_diff_sum": float((b - c).sum()),
        }

    df = prepare_position_log(plog)
    pit2 = pit.copy()
    pit2["date"] = _utc_norm(pit2["date"])
    pit2["id"] = pit2["id"].astype(int)
    rank_map = {( _utc_ts(r.date), int(r.id) ): int(r.rank) for r in pit2.itertuples(index=False)}

    tier_pnl = {name: {"bn": [], "cmc": [], "idx": []} for _, _, name in PHASE3C_NAME_TIERS}
    # accumulate daily contrib by tier
    by_day_tier = defaultdict(lambda: {"bn": 0.0, "cmc": 0.0})
    for row in df.itertuples(index=False):
        w = float(row.w)
        if w == 0.0:
            continue
        iid = int(row.id)
        dt, nxt = _utc_ts(row.date), _utc_ts(row.nxt)
        r_cmc = _ret(cmc_close, dt, nxt, iid)
        if w > 0:
            r_bn = _ret(spot_wide, dt, nxt, iid)
        else:
            r_bn = _ret(perp_wide, dt, nxt, iid)
        if not np.isfinite(r_bn):
            continue
        if not np.isfinite(r_cmc):
            r_cmc = 0.0
        rk = rank_map.get((dt, iid))
        if rk is None:
            continue
        tname = None
        for lo, hi, name in PHASE3C_NAME_TIERS:
            if lo <= rk <= hi:
                tname = name
                break
        if tname is None:
            continue
        by_day_tier[(nxt, tname)]["bn"] += w * float(r_bn)
        by_day_tier[(nxt, tname)]["cmc"] += w * float(r_cmc)

    tiers = {}
    for _, _, name in PHASE3C_NAME_TIERS:
        rows = [(d, v) for (d, t), v in by_day_tier.items() if t == name]
        if not rows:
            tiers[name] = {"n": 0, "corr": float("nan"), "sharpe_bn": float("nan"), "sharpe_cmc": float("nan")}
            continue
        rows.sort(key=lambda x: x[0])
        s_bn = pd.Series([v["bn"] for _, v in rows], index=pd.DatetimeIndex([d for d, _ in rows]))
        s_cmc = pd.Series([v["cmc"] for _, v in rows], index=s_bn.index)
        tiers[name] = {
            "n": int(len(s_bn)),
            "corr": float(s_bn.corr(s_cmc)) if len(s_bn) > 5 else float("nan"),
            "sharpe_bn": _sharpe(s_bn),
            "sharpe_cmc": _sharpe(s_cmc),
            "sharpe_gap": float(_sharpe(s_bn) - _sharpe(s_cmc)),
            "pnl_diff_sum": float((s_bn - s_cmc).sum()),
        }
    return {"by_year": years, "by_tier": tiers}


def needed_spot_symbols(plog: pd.DataFrame, panel: pd.DataFrame, listed: set[str]) -> dict:
    df = prepare_position_log(plog)
    long_ids = sorted(set(int(i) for i in df.loc[df["w"] > 0, "id"].unique()))
    wanted: list[str] = []
    seen: set[str] = set()
    per_id = {}
    for iid in long_ids:
        cands = [c for c in symbols_for_id(panel, iid) if c in listed]
        per_id[int(iid)] = cands
        for c in cands:
            if c not in seen:
                seen.add(c)
                wanted.append(c)
    return {"symbols": wanted, "long_ids": long_ids, "per_id": per_id}


def replay_long_leg(
    plog: pd.DataFrame,
    cmc_close: pd.DataFrame,
    spot_wide: pd.DataFrame,
    perp_wide: pd.DataFrame,
    fund_wide: pd.DataFrame,
) -> dict:
    """Same frozen weights; PnL of the long leg only (spot, no funding).

    Weights are NOT renormalized — this is the long contribution inside the
    β-matched book (typical long gross ≈ 0.5). Short-leg and full-hybrid
    series are returned for overlay only.
    """
    df = prepare_position_log(plog)
    recs = []
    for (dt, nxt), g in df.groupby(["date", "nxt"], sort=True):
        dt = _utc_ts(dt)
        nxt = _utc_ts(nxt)
        long_g = short_g = 0.0
        long_c = short_c = 0.0
        short_f = 0.0
        n_long = n_short = 0
        lg = sg = 0.0
        for row in g.itertuples(index=False):
            iid = int(row.id)
            w = float(row.w)
            dw = float(row.dw)
            old_w = w - dw
            r_cmc = _ret(cmc_close, dt, nxt, iid)
            if not np.isfinite(r_cmc):
                r_cmc = 0.0
            if w > 0:
                r_bn = _ret(spot_wide, dt, nxt, iid)
                r = float(r_bn) if np.isfinite(r_bn) else r_cmc
                long_g += w * r
                n_long += 1
                lg += w
            elif w < 0:
                r_bn = _ret(perp_wide, dt, nxt, iid)
                r = float(r_bn) if np.isfinite(r_bn) else r_cmc
                short_g += w * r
                n_short += 1
                sg += abs(w)
                f = _funding_rate(fund_wide, nxt, iid)
                if np.isfinite(f):
                    short_f += -w * f
            if old_w >= 0 and w >= 0:
                long_c += name_cost(old_w, w)
            elif old_w <= 0 and w <= 0:
                short_c += name_cost(old_w, w)
            else:
                if old_w > 0:
                    long_c += abs(old_w) * LONG_C
                if old_w < 0:
                    short_c += abs(old_w) * SHORT_C
                if w > 0:
                    long_c += abs(w) * LONG_C
                if w < 0:
                    short_c += abs(w) * SHORT_C
        recs.append(
            {
                "nxt": nxt,
                "long_gross": long_g,
                "short_gross": short_g,
                "long_cost": long_c,
                "short_cost": short_c,
                "short_fund": short_f,
                "n_long": n_long,
                "n_short": n_short,
                "long_gross_w": lg,
                "short_gross_w": sg,
            }
        )
    daily = pd.DataFrame(recs).set_index("nxt").sort_index()
    daily.index = _utc_norm(daily.index)
    long_net = daily["long_gross"] - daily["long_cost"]
    short_net = daily["short_gross"] - daily["short_cost"] + daily["short_fund"]
    full_net = long_net + short_net
    z = pd.Series(0.0, index=daily.index)
    n_long_s = pd.Series(daily["n_long"].to_numpy(), index=daily.index, dtype=float)
    n_short_s = pd.Series(daily["n_short"].to_numpy(), index=daily.index, dtype=float)
    lg_s = pd.Series(daily["long_gross_w"].to_numpy(), index=daily.index, dtype=float)
    sg_s = pd.Series(daily["short_gross_w"].to_numpy(), index=daily.index, dtype=float)
    to_list = [0.0] * len(daily)

    def _pack(rets, n_long, n_short, lg, sg, cost, gross):
        stats = {
            "to_list": to_list,
            "n_long": n_long,
            "n_short": n_short,
            "n_shortable": z,
            "incomplete": z,
            "long_gross": lg,
            "short_gross": sg,
            "cash": (1.0 - lg - sg).clip(lower=0.0),
            "daily_cost": cost,
            "daily_gross": gross,
        }
        return summarize_ls(rets, stats)

    long_book = _pack(long_net, n_long_s, z, lg_s, z, daily["long_cost"], daily["long_gross"])
    short_book = _pack(short_net, z, n_short_s, z, sg_s, daily["short_cost"], daily["short_gross"])
    short_book["funding_total_pnl"] = float(daily["short_fund"].sum())
    full_book = _pack(
        full_net,
        n_long_s,
        n_short_s,
        lg_s,
        sg_s,
        daily["long_cost"] + daily["short_cost"],
        daily["long_gross"] + daily["short_gross"],
    )
    full_book["funding_total_pnl"] = float(daily["short_fund"].sum())
    return {
        "long": long_book,
        "short": short_book,
        "full": full_book,
        "long_net": long_net,
        "short_net": short_net,
        "full_net": full_net,
        "daily": daily,
    }


def nonempty_parquets(raw_dir: Path) -> set[str]:
    out = set()
    if not raw_dir.exists():
        return out
    for p in raw_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(p, columns=["date"] if False else None)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        out.add(p.stem.upper())
    return out
