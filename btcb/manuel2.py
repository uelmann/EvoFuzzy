"""MANUEL-2: gauss(ret14)·gauss(ret28)/gauss(std63) long-only falsification.

BACKTEST / ANALYSIS ONLY. Nothing adopted. Formula frozen verbatim.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import ndtr

from btcb.book import summarize_book
from btcb.constants import (
    ANNUALIZATION,
    LS_TRAIL_DAYS,
    MANUEL2_BEST_RULE,
    MANUEL2_CORR_MAX,
    MANUEL2_COST_BPS,
    MANUEL2_CS_DDOF,
    MANUEL2_EXTRA_STABLES,
    MANUEL2_MAXDD_MULT,
    MANUEL2_PEG_ABS_RET,
    MANUEL2_PEG_LOOKBACK,
    MANUEL2_PEG_NEEDLES,
    MANUEL2_PEG_USD_EXCEPT,
    MANUEL2_REL_SHARPE_STRONG,
    MANUEL2_RET_A,
    MANUEL2_RET_B,
    MANUEL2_ROLL_CORR,
    MANUEL2_STD_DDOF,
    MANUEL2_STD_W,
    MANUEL2_TOP_PCT,
    MANUEL2_UNIVERSE_N,
    MANUEL2_WEEKDAY,
    STABLE_OR_WRAP,
)
from btcb.oracle_ladder import _as_utc, _spearman, _utc_idx, last_close_map, listed_mask
from btcb.spread_ls import trail_slice


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION)) if len(x) and x.std() > 0 else 0.0


def tagged_peg_ids(panel: pd.DataFrame) -> tuple[set[int], list[dict]]:
    """Static pegged-asset tags from panel symbol/name/slug. Logged."""
    p = panel.copy()
    if "name" not in p.columns:
        p["name"] = p["symbol"] if "symbol" in p.columns else ""
    if "slug" not in p.columns:
        p["slug"] = p["symbol"] if "symbol" in p.columns else ""
    p = p[["id", "symbol", "name", "slug"]].copy()
    p["id"] = p["id"].astype(int)
    p["symbol"] = p["symbol"].astype(str)
    p["name"] = p["name"].astype(str)
    p["slug"] = p["slug"].astype(str)
    last = p.sort_values("id").groupby("id").tail(1)
    tagged: set[int] = set()
    rows: list[dict] = []
    needles = tuple(n.lower() for n in MANUEL2_PEG_NEEDLES)
    tag_set = set(s.upper() for s in STABLE_OR_WRAP) | set(s.upper() for s in MANUEL2_EXTRA_STABLES)
    excepts = set(s.upper() for s in MANUEL2_PEG_USD_EXCEPT)
    hist_map: dict[int, set[str]] = defaultdict(set)
    for iid, sym in zip(p["id"].to_numpy(), p["symbol"].to_numpy()):
        hist_map[int(iid)].add(str(sym).upper())
    for rec in last.itertuples(index=False):
        iid = int(rec.id)
        reasons = []
        for col, val in (("symbol", rec.symbol), ("name", rec.name), ("slug", rec.slug)):
            u = str(val).upper().strip()
            low = str(val).lower()
            if u in tag_set:
                reasons.append(f"{col}={u} in tag-set")
            if u.endswith("USD") and len(u) <= 5 and u not in excepts:
                reasons.append(f"{col}={u} USD-suffix")
            if any(n in low for n in needles):
                hit = next(n for n in needles if n in low)
                reasons.append(f"{col} needle '{hit}'")
        # any historical symbol for this id
        hist = hist_map.get(iid, set())
        for u in hist:
            if u in tag_set and f"symbol={u} in tag-set" not in reasons:
                reasons.append(f"hist-symbol={u} in tag-set")
            if u.endswith("USD") and len(u) <= 5 and u not in excepts and f"symbol={u} USD-suffix" not in reasons:
                reasons.append(f"hist-symbol={u} USD-suffix")
        if reasons:
            tagged.add(iid)
            rows.append(
                {
                    "id": iid,
                    "symbol": str(rec.symbol),
                    "name": str(rec.name),
                    "slug": str(rec.slug),
                    "reasons": reasons[:6],
                }
            )
    rows.sort(key=lambda r: r["symbol"])
    return tagged, rows


def hybrid_close_wide(cmc: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """Binance spot where a print exists, else CMC."""
    cmc = cmc.copy()
    spot = spot.copy()
    cmc.index = _utc_idx(cmc.index)
    spot.index = _utc_idx(spot.index)
    idx = cmc.index.union(spot.index).sort_values()
    cmc = cmc.reindex(idx)
    spot = spot.reindex(idx)
    cols = sorted(set(int(c) for c in cmc.columns) | set(int(c) for c in spot.columns))
    out = {}
    for c in cols:
        s = spot[c] if c in spot.columns else None
        m = cmc[c] if c in cmc.columns else None
        if s is not None and m is not None:
            out[c] = s.astype(float).combine_first(m.astype(float))
        elif s is not None:
            out[c] = s.astype(float)
        else:
            out[c] = m.astype(float)
    wide = pd.DataFrame(out).sort_index()
    wide.index = _utc_idx(wide.index)
    return wide


def cmc_close_wide(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    wide = df.pivot_table(index="date", columns="id", values="close", aggfunc="last").sort_index()
    wide.index = _utc_idx(wide.index)
    return wide


def mcap_wide(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    wide = df.pivot_table(index="date", columns="id", values="mcap", aggfunc="last").sort_index()
    wide.index = _utc_idx(wide.index)
    return wide


def price_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    px = close.copy()
    px.index = _utc_idx(px.index)
    px = px.sort_index().astype(float)
    r1 = px.pct_change()
    ret_a = px / px.shift(int(MANUEL2_RET_A)) - 1.0
    ret_b = px / px.shift(int(MANUEL2_RET_B)) - 1.0
    std_w = r1.rolling(int(MANUEL2_STD_W), min_periods=int(MANUEL2_STD_W)).std(ddof=int(MANUEL2_STD_DDOF))
    ret90 = px / px.shift(int(MANUEL2_PEG_LOOKBACK)) - 1.0
    return {"r1": r1, "ret14": ret_a, "ret28": ret_b, "std63": std_w, "ret90": ret90}


def gauss_cs(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 2:
        return pd.Series(dtype=float)
    mu = float(v.mean())
    sd = float(v.std(ddof=int(MANUEL2_CS_DDOF)))
    if not np.isfinite(sd) or sd == 0.0:
        z = np.zeros(len(v), dtype=float)
    else:
        z = (v.to_numpy(dtype=float) - mu) / sd
    return pd.Series(ndtr(z), index=v.index, dtype=float)


def scores_at(ids: list[int], feat: dict[str, pd.DataFrame], dt) -> pd.Series:
    dt = _as_utc(dt)
    ids = [int(i) for i in ids]
    out = {}
    for col in ("ret14", "ret28", "std63"):
        wide = feat[col]
        if dt not in wide.index:
            return pd.Series(dtype=float)
        row = wide.loc[dt]
        s = pd.Series({i: row[i] if i in row.index else np.nan for i in ids}, dtype=float)
        out[col] = s
    g14 = gauss_cs(out["ret14"])
    g28 = gauss_cs(out["ret28"])
    gs = gauss_cs(out["std63"])
    common = g14.index.intersection(g28.index).intersection(gs.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    den = gs.reindex(common).astype(float)
    den = den.where(den > 0.0)
    sc = g14.reindex(common).astype(float) * g28.reindex(common).astype(float) / den
    return sc.replace([np.inf, -np.inf], np.nan).dropna()


def top_pct_weights(scores: pd.Series, pct: float = MANUEL2_TOP_PCT) -> pd.Series:
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 1:
        return pd.Series(dtype=float)
    k = max(1, int(math.ceil(float(pct) * n)))
    ids = [int(i) for i in s.nlargest(k).index.tolist()]
    w = 1.0 / float(len(ids))
    return pd.Series({i: w for i in ids}, dtype=float)


def universe_at(
    mcap: pd.DataFrame,
    tagged: set[int],
    peg_heu: pd.DataFrame,
    dt,
    *,
    btc_id: int,
    btc_ex: bool,
    n: int = MANUEL2_UNIVERSE_N,
) -> list[int]:
    dt = _as_utc(dt)
    if dt not in mcap.index:
        return []
    row = mcap.loc[dt].astype(float)
    row = row[np.isfinite(row) & (row > 0)]
    if row.empty:
        return []
    drop = set(int(i) for i in tagged)
    if dt in peg_heu.index:
        heu = peg_heu.loc[dt]
        if isinstance(heu, pd.Series):
            drop |= {int(i) for i, v in heu.items() if bool(v)}
    ids = [int(i) for i in row.index if int(i) not in drop]
    if btc_ex:
        ids = [i for i in ids if i != int(btc_id)]
    if not ids:
        return []
    ranked = row.reindex(ids).dropna().sort_values(ascending=False)
    return [int(i) for i in ranked.index[: int(n)]]


def build_members(
    dates: list,
    mcap: pd.DataFrame,
    tagged: set[int],
    peg_heu: pd.DataFrame,
    btc_id: int,
    btc_ex: bool,
    n: int = MANUEL2_UNIVERSE_N,
) -> dict:
    out = {}
    last: list[int] = []
    for d in dates:
        d = _as_utc(d)
        u = universe_at(mcap, tagged, peg_heu, d, btc_id=btc_id, btc_ex=btc_ex, n=n)
        if u:
            last = u
        out[d] = list(last)
    return out


def peg_heuristic_mask(ret90: pd.DataFrame, thresh: float = MANUEL2_PEG_ABS_RET) -> pd.DataFrame:
    x = ret90.abs()
    return (x < float(thresh)) & x.notna()


def filter_members(members: dict, tagged: set[int], peg_heu: pd.DataFrame, btc_id: int, btc_ex: bool) -> dict:
    out = {}
    for d, ids in members.items():
        d = _as_utc(d)
        drop = set(int(i) for i in tagged)
        if d in peg_heu.index:
            heu = peg_heu.loc[d]
            if isinstance(heu, pd.Series):
                drop |= {int(i) for i, v in heu.items() if bool(v)}
        keep = [int(i) for i in ids if int(i) not in drop]
        if btc_ex:
            keep = [i for i in keep if i != int(btc_id)]
        out[d] = keep
    return out


def heuristic_drop_stats(peg_heu: pd.DataFrame, dates: list) -> dict:
    n_drop = []
    for d in dates:
        d = _as_utc(d)
        if d not in peg_heu.index:
            continue
        row = peg_heu.loc[d]
        n_drop.append(int(row.fillna(False).sum()) if isinstance(row, pd.Series) else 0)
    arr = np.asarray(n_drop, dtype=float)
    return {
        "median_flagged_per_date": float(np.median(arr)) if len(arr) else float("nan"),
        "mean_flagged_per_date": float(np.mean(arr)) if len(arr) else float("nan"),
        "n_dates": int(len(arr)),
    }


def is_rebalance(dt, *, mode: str, first: pd.Timestamp) -> bool:
    d = _as_utc(dt)
    if mode == "daily":
        return True
    if d == _as_utc(first):
        return True
    return int(pd.Timestamp(d).weekday()) == int(MANUEL2_WEEKDAY)


def ew_weights(ids: list[int], close: pd.DataFrame, dt) -> pd.Series:
    live = [int(i) for i in ids if listed_mask(close, dt, int(i))]
    if not live:
        return pd.Series(dtype=float)
    w = 1.0 / float(len(live))
    return pd.Series({i: w for i in live}, dtype=float)


def run_long_book(
    close: pd.DataFrame,
    btc_id: int,
    members: dict,
    feat: dict[str, pd.DataFrame],
    dates: list,
    *,
    mode: str = "daily",
    cost_bps: float = MANUEL2_COST_BPS,
    label: str = "",
    selection: str = "score",
    id_to_sym: dict | None = None,
) -> dict:
    """Always-invested long-only. selection='score' (top 5%) or 'ew' (universe EW)."""
    close = close.copy()
    close.index = _utc_idx(close.index)
    close = close.sort_index()
    last_map = last_close_map(close)
    id_to_sym = id_to_sym or {}
    dates = [_as_utc(d) for d in dates]
    dates = [d for d in dates if d in close.index]
    if len(dates) < 3:
        return {"error": "not enough dates", "label": label}
    alt_c = float(cost_bps) * 1e-4
    first = dates[0]
    prev_w = pd.Series(dtype=float)
    daily = {}
    to_list = []
    n_names = []
    forced_events = []
    contrib: dict[int, float] = defaultdict(float)
    n_form = 0
    n_empty_sel = 0
    tilt_dates, tilt_ics = [], []
    held_target = pd.Series(dtype=float)

    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        univ = [int(x) for x in members.get(dt, [])]
        if selection == "ew":
            target = ew_weights(univ, close, dt)
        else:
            if is_rebalance(dt, mode=mode, first=first):
                sc = scores_at(univ, feat, dt)
                if len(sc) >= 2:
                    ic = _spearman(sc.to_numpy(), feat["std63"].loc[dt].reindex(sc.index).to_numpy())
                    if np.isfinite(ic):
                        tilt_dates.append(dt)
                        tilt_ics.append(float(ic))
                target = top_pct_weights(sc)
                n_form += 1
                if target.empty:
                    n_empty_sel += 1
                    target = ew_weights(univ, close, dt)
                held_target = target.copy()
            else:
                target = held_target.copy()
                if target.empty:
                    target = ew_weights(univ, close, dt)
        w = target.copy() if len(target) else pd.Series(dtype=float)
        drop = []
        for iid in list(w.index):
            ld = last_map.get(int(iid))
            if ld is None or nxt > ld or not listed_mask(close, dt, int(iid)):
                drop.append(int(iid))
        if drop:
            w_drop = float(w.reindex(drop).fillna(0.0).sum())
            w = w.drop(labels=drop, errors="ignore")
            if w_drop > 0:
                forced_events.append({"date": str(dt.date()), "ids": drop, "weight": w_drop})
        if len(w):
            w = w / float(w.sum())
        else:
            w = ew_weights(univ, close, dt)
            extra_drop = [int(i) for i in list(w.index) if last_map.get(int(i)) is None or nxt > last_map.get(int(i))]
            if extra_drop:
                w = w.drop(labels=extra_drop, errors="ignore")
            if len(w):
                w = w / float(w.sum())
        idx = w.index.union(prev_w.index)
        dw = (w.reindex(idx).fillna(0.0) - prev_w.reindex(idx).fillna(0.0)).abs().sum()
        to_list.append(0.5 * float(dw))
        cost = float(dw) * alt_c
        r = 0.0
        if len(w) and dt in close.index and nxt in close.index:
            p0 = close.loc[dt]
            p1 = close.loc[nxt]
            for iid, wi in w.items():
                if iid not in p0.index or iid not in p1.index:
                    continue
                a = float(p0[iid]) if p0[iid] is not None else float("nan")
                b = float(p1[iid]) if p1[iid] is not None else float("nan")
                if np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0:
                    ri = b / a - 1.0
                    r += float(wi) * ri
                    contrib[int(iid)] += float(wi) * ri
        daily[nxt] = float(r - cost)
        n_names.append(int(len(w)))
        prev_w = w.copy()
        held_target = w.copy() if selection != "ew" else held_target

    if not daily:
        return {"error": "empty book", "label": label}
    rets = pd.Series(daily).sort_index()
    rets.index = _utc_idx(rets.index)
    btc_col = int(btc_id)
    if btc_col not in close.columns:
        raise RuntimeError("BTC missing from close")
    btc = close[btc_col].astype(float).pct_change().reindex(rets.index).fillna(0.0)
    z = pd.Series(0.0, index=rets.index)
    nn = pd.Series(n_names, index=list(daily.keys()), dtype=float).reindex(rets.index)
    packed = summarize_book(rets, btc, z, nn.fillna(0.0), to_list)
    packed["label"] = label
    packed["net_sharpe_trail18m"] = _sharpe(trail_slice(rets, LS_TRAIL_DAYS))
    packed["n_formations"] = int(n_form)
    packed["n_empty_sel"] = int(n_empty_sel)
    packed["forced_exits"] = {
        "n_events": int(len(forced_events)),
        "n_ids": int(len({i for e in forced_events for i in e["ids"]})),
        "weight_sum": float(sum(e["weight"] for e in forced_events)),
    }
    packed["forced_n"] = int(len(forced_events))
    m = rets.notna() & btc.notna()
    packed["corr_btc"] = float(rets[m].corr(btc[m])) if int(m.sum()) >= 10 else float("nan")
    packed["corr_btc_n"] = int(m.sum())
    packed["corr_btc_roll90"] = rets.rolling(int(MANUEL2_ROLL_CORR), min_periods=max(20, MANUEL2_ROLL_CORR // 3)).corr(btc)
    packed["score_vol_tilt"] = float(np.mean(tilt_ics)) if tilt_ics else float("nan")
    packed["score_vol_tilt_n"] = int(len(tilt_ics))
    tot = float(sum(contrib.values())) if contrib else 0.0
    ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    top5 = []
    for iid, c in ranked[:5]:
        top5.append(
            {
                "id": int(iid),
                "symbol": id_to_sym.get(int(iid)),
                "contrib": float(c),
                "share": float(c / tot) if tot else 0.0,
            }
        )
    packed["top5"] = top5
    packed["top5_share"] = float(sum(abs(r["contrib"]) for r in top5) / sum(abs(v) for v in contrib.values())) if contrib else float("nan")
    packed["top5_share_signed"] = float(sum(r["contrib"] for r in top5) / tot) if tot else float("nan")
    packed["total"] = packed.get("book_total")
    packed["cagr"] = packed.get("book_cagr")
    packed["sharpe"] = packed.get("book_sharpe")
    return packed


def btc_bh_book(close: pd.DataFrame, btc_id: int, idx: pd.DatetimeIndex) -> dict:
    close = close.copy()
    close.index = _utc_idx(close.index)
    idx = _utc_idx(idx)
    px = close[int(btc_id)].astype(float).reindex(idx)
    rets = px.pct_change().fillna(0.0)
    # first day 0; BH from first to last
    ones = pd.Series(1.0, index=idx)
    z = pd.Series(0.0, index=idx)
    packed = summarize_book(rets, rets, ones, z, [0.0] * len(idx))
    packed["label"] = "btc_bh"
    packed["net_sharpe_trail18m"] = _sharpe(trail_slice(rets, LS_TRAIL_DAYS))
    packed["corr_btc"] = 1.0
    packed["corr_btc_n"] = int(len(rets))
    packed["corr_btc_roll90"] = pd.Series(1.0, index=idx)
    packed["forced_exits"] = {"n_events": 0, "n_ids": 0, "weight_sum": 0.0}
    packed["forced_n"] = 0
    packed["top5"] = []
    packed["top5_share"] = float("nan")
    packed["total"] = packed.get("book_total")
    packed["cagr"] = packed.get("book_cagr")
    packed["sharpe"] = packed.get("book_sharpe")
    packed["avg_n_names"] = 1.0
    packed["ann_turnover"] = 0.0
    return packed


def align_to_index(book: dict, idx: pd.DatetimeIndex) -> dict:
    """Recompute headline stats on a common date index (fill missing with 0)."""
    idx = _utc_idx(idx)
    rets = book["daily_ret"].reindex(idx).fillna(0.0)
    btc = book["btc_ret"].reindex(idx).fillna(0.0)
    nn = book["n_names"].reindex(idx) if isinstance(book.get("n_names"), pd.Series) else pd.Series(np.nan, index=idx)
    z = pd.Series(0.0, index=idx)
    packed = summarize_book(rets, btc, z, nn.fillna(0.0), [])
    packed["label"] = book.get("label")
    packed["net_sharpe_trail18m"] = _sharpe(trail_slice(rets, LS_TRAIL_DAYS))
    m = rets.notna() & btc.notna()
    packed["corr_btc"] = float(rets[m].corr(btc[m])) if int(m.sum()) >= 10 else float("nan")
    packed["corr_btc_n"] = int(m.sum())
    packed["corr_btc_roll90"] = rets.rolling(int(MANUEL2_ROLL_CORR), min_periods=max(20, MANUEL2_ROLL_CORR // 3)).corr(btc)
    packed["forced_exits"] = book.get("forced_exits")
    packed["forced_n"] = book.get("forced_n")
    packed["ann_turnover"] = book.get("ann_turnover")
    packed["avg_n_names"] = packed.get("avg_n_names")
    packed["n_formations"] = book.get("n_formations")
    packed["top5"] = book.get("top5")
    packed["top5_share"] = book.get("top5_share")
    packed["score_vol_tilt"] = book.get("score_vol_tilt")
    packed["total"] = packed.get("book_total")
    packed["cagr"] = packed.get("book_cagr")
    packed["sharpe"] = packed.get("book_sharpe")
    packed["n_empty_sel"] = book.get("n_empty_sel")
    return packed


def mechanical_verdicts(books: dict, btc: dict) -> dict:
    """Verdicts on the highest-total of the four pre-declared books."""
    keys = [k for k in ("daily_btc_in", "daily_btc_ex", "weekly_btc_in", "weekly_btc_ex") if k in books]
    if not keys:
        return {"label": "MANUEL-2 REFUTED", "error": "no books"}
    best_key = max(keys, key=lambda k: float((books[k] or {}).get("total") or -np.inf))
    best = books[best_key] or {}
    btc_tot = float(btc.get("total") if btc.get("total") is not None else btc.get("book_total") or np.nan)
    book_tot = float(best.get("total") if best.get("total") is not None else best.get("book_total") or np.nan)
    corr = float(best.get("corr_btc") if best.get("corr_btc") is not None else np.nan)
    rel = float(best.get("rel_sharpe") if best.get("rel_sharpe") is not None else np.nan)
    mdd = float(best.get("maxdd") if best.get("maxdd") is not None else np.nan)
    btc_mdd = float(btc.get("maxdd") if btc.get("maxdd") is not None else np.nan)
    clause1 = bool(np.isfinite(book_tot) and np.isfinite(btc_tot) and book_tot >= btc_tot)
    clause2 = bool(np.isfinite(corr) and corr <= float(MANUEL2_CORR_MAX))
    confirmed = bool(clause1 and clause2)
    maxdd_ok = bool(
        np.isfinite(mdd)
        and np.isfinite(btc_mdd)
        and abs(float(mdd)) <= float(MANUEL2_MAXDD_MULT) * abs(float(btc_mdd))
    )
    rel_ok = bool(np.isfinite(rel) and rel >= float(MANUEL2_REL_SHARPE_STRONG))
    strong = bool(confirmed and rel_ok and maxdd_ok)
    if strong:
        label = "MANUEL-2 STRONG"
    elif confirmed:
        label = "MANUEL-2 CONFIRMED"
    elif clause1 != clause2:
        label = "MANUEL-2 PARTIAL"
    else:
        label = "MANUEL-2 REFUTED"
    which = None
    if label.endswith("PARTIAL"):
        which = "total ≥ BTC B&H" if clause1 else "daily corr vs BTC ≤ 0.70"
    return {
        "label": label,
        "best_book": best_key,
        "clause1_total_ge_btc": clause1,
        "clause2_corr_le_070": clause2,
        "confirmed": confirmed,
        "strong": strong,
        "partial_which": which,
        "best_total": book_tot,
        "btc_total": btc_tot,
        "best_corr": corr,
        "best_rel_sharpe": rel,
        "best_maxdd": mdd,
        "btc_maxdd": btc_mdd,
        "rel_ok": rel_ok,
        "maxdd_ok": maxdd_ok,
        "best_rule": MANUEL2_BEST_RULE,
        "nothing_adopted": True,
    }
