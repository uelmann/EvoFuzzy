"""LONG-TIDE: full-size long leg + frozen Stage-T gate, BTC parking.

Backtest only. No shorts, no funding. Gate parameters imported frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.book import summarize_book
from btcb.constants import (
    ALT_BPS,
    ANNUALIZATION,
    BLOWOFF_RET_7,
    BTC_BPS,
    DEATH_CONVENTION,
    LONGTIDE_ALT_MIN,
    LONGTIDE_BUDGET,
    LONGTIDE_CRITERION,
    LONGTIDE_CYCLE_REL_FLOOR,
    LONGTIDE_H,
    LONGTIDE_K,
    LONGTIDE_PRECONDITION,
    LONGTIDE_REL_MARGIN,
    LS_DECILE_K,
    LS_QUINTILE_K,
    LS_TRAIL_DAYS,
    NAME_CAP,
    PHASE2_CYCLES,
    REGIME_BREADTH,
    REGIME_OFF_HYSTERESIS,
)
from btcb.spread_ls import trail_slice


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _utc_idx(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert("UTC").normalize()


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION)) if len(x) and x.std() > 0 else 0.0


def read_phase3e_verdict(md_path: Path, json_path: Path | None = None) -> str:
    """Mechanical precondition: 3.e label, else unknown."""
    if json_path is not None and json_path.exists():
        blob = json.loads(json_path.read_text())
        lab = ((blob.get("verdict") or {}).get("label")) if isinstance(blob, dict) else None
        if lab:
            return str(lab)
    if md_path.exists():
        text = md_path.read_text()
        for line in text.splitlines():
            s = line.strip().replace("*", "")
            if s in ("SIGNAL-CONFIRMED", "SIGNAL-PARTLY-ARTIFACT"):
                return s
            if s.startswith("SIGNAL-CONFIRMED"):
                return "SIGNAL-CONFIRMED"
            if s.startswith("SIGNAL-PARTLY-ARTIFACT"):
                return "SIGNAL-PARTLY-ARTIFACT"
    return "UNKNOWN"


def load_phase2_preds(pred_dir: Path, horizon: int = 14) -> pd.DataFrame:
    files = sorted(pred_dir.glob(f"preds_h{horizon}_fold*.parquet"))
    if not files:
        raise RuntimeError(f"missing BTC-BEATER v1 preds in {pred_dir}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    return df


def last_close_map(close: pd.DataFrame) -> dict[int, pd.Timestamp]:
    out: dict[int, pd.Timestamp] = {}
    for iid in close.columns:
        col = close[iid].astype(float)
        ok = col[np.isfinite(col) & (col > 0)]
        if len(ok):
            out[int(iid)] = pd.Timestamp(ok.index.max()).tz_convert("UTC").normalize()
    return out


def listed_mask(close: pd.DataFrame, dt, iid: int) -> bool:
    if close is None or close.empty:
        return False
    if dt not in close.index or int(iid) not in close.columns:
        return False
    v = close.at[dt, int(iid)]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > 0)


def enrich_book(packed: dict) -> dict:
    rets = packed.get("daily_ret")
    if not isinstance(rets, pd.Series) or rets.empty:
        packed["net_sharpe_trail18m"] = float("nan")
        packed["rel_sharpe_trail18m"] = float("nan")
        packed["avg_alt_deployment"] = float("nan")
        packed["rel_total"] = float("nan")
        return packed
    packed["net_sharpe_trail18m"] = _sharpe(trail_slice(rets, LS_TRAIL_DAYS))
    rel_eq = packed.get("rel_equity")
    if isinstance(rel_eq, pd.Series) and len(rel_eq):
        packed["rel_sharpe_trail18m"] = _sharpe(trail_slice(rel_eq.pct_change().fillna(0.0), LS_TRAIL_DAYS))
        packed["rel_total"] = float(rel_eq.iloc[-1] - 1.0)
    else:
        packed["rel_sharpe_trail18m"] = float("nan")
        packed["rel_total"] = float("nan")
    ag = packed.get("alt_gross")
    wbtc = packed.get("w_btc")
    if isinstance(ag, pd.Series) and len(ag):
        packed["avg_alt_deployment"] = float(ag.mean())
    elif isinstance(wbtc, pd.Series) and len(wbtc):
        packed["avg_alt_deployment"] = float((1.0 - wbtc.clip(lower=0.0, upper=1.0)).mean())
    else:
        packed["avg_alt_deployment"] = float("nan")
    gon = packed.get("gate_on")
    if isinstance(gon, pd.Series) and len(gon):
        packed["gate_on_frac"] = float(gon.mean())
    packed.setdefault("gate_on_frac", float("nan"))
    return packed


def slice_book(book: dict, idx: pd.DatetimeIndex) -> dict:
    """Recompute headline stats on a common date index. Each book vs its own BTC."""
    rets = book["daily_ret"].reindex(idx).fillna(0.0)
    btc = book["btc_ret"].reindex(idx).fillna(0.0)
    wbtc = book["w_btc"].reindex(idx) if isinstance(book.get("w_btc"), pd.Series) else pd.Series(0.0, index=idx)
    nn = book["n_names"].reindex(idx) if isinstance(book.get("n_names"), pd.Series) else pd.Series(np.nan, index=idx)
    packed = summarize_book(rets, btc, wbtc, nn, [])
    if isinstance(book.get("gate_on"), pd.Series):
        packed["gate_on"] = book["gate_on"].reindex(idx).fillna(0.0)
    if isinstance(book.get("alt_gross"), pd.Series):
        packed["alt_gross"] = book["alt_gross"].reindex(idx).fillna(0.0)
    packed["forced_exits"] = book.get("forced_exits")
    packed["ann_turnover"] = book.get("ann_turnover")
    return enrich_book(packed)


def control_from_rets(rets: pd.Series, btc_simple: pd.Series) -> dict:
    rets = rets.dropna()
    if rets.empty:
        return {"error": "empty control"}
    btc = btc_simple.reindex(rets.index).fillna(0.0)
    z = pd.Series(0.0, index=rets.index)
    packed = summarize_book(rets, btc, z, z, [0.0] * len(rets))
    packed["avg_alt_deployment"] = 1.0
    packed["gate_on_frac"] = float("nan")
    packed["forced_exits"] = {"n_events": 0, "n_ids": 0, "weight_sum": 0.0, "cost_drag": 0.0}
    return enrich_book(packed)


def btc_bh_book(btc_simple: pd.Series, idx: pd.DatetimeIndex) -> dict:
    btc = btc_simple.reindex(idx).fillna(0.0)
    ones = pd.Series(1.0, index=idx)
    z = pd.Series(0.0, index=idx)
    packed = summarize_book(btc, btc, ones, z, [0.0] * len(idx))
    packed["avg_alt_deployment"] = 0.0
    packed["gate_on_frac"] = float("nan")
    packed["forced_exits"] = {"n_events": 0, "n_ids": 0, "weight_sum": 0.0, "cost_drag": 0.0}
    packed["rel_sharpe"] = 0.0
    packed["rel_total"] = 0.0
    packed["rel_cagr"] = 0.0
    return enrich_book(packed)


def series_corr(a: pd.Series, b: pd.Series) -> dict:
    x, y = a.align(b, join="inner")
    x = x.astype(float)
    y = y.astype(float)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 10:
        return {"corr": float("nan"), "n": int(len(x))}
    return {
        "corr": float(x.corr(y)),
        "n": int(len(x)),
        "start": str(x.index.min().date()),
        "end": str(x.index.max().date()),
    }


def run_long_tide(
    close: pd.DataFrame,
    pit100: pd.DataFrame,
    preds: pd.DataFrame,
    feat: pd.DataFrame,
    btc_id: int,
    h: int = LONGTIDE_H,
    gate_on: pd.Series | None = None,
    *,
    park_btc: bool = True,
    spot_filter: bool = True,
    budget: float = LONGTIDE_BUDGET,
    name_cap: float = NAME_CAP,
    n_hold: int = LONGTIDE_K,
    k_enter: int = LS_DECILE_K,
    k_stay: int = LS_QUINTILE_K,
    alt_bps: float = ALT_BPS,
    btc_bps: float = BTC_BPS,
    blowoff: float = BLOWOFF_RET_7,
    id_to_sym: dict | None = None,
    cycles=PHASE2_CYCLES,
) -> dict:
    """EW long-only h-tranche book. Gate OFF empties the rolling slot (→ BTC if park_btc)."""
    close = close.copy()
    close.index = _utc_idx(close.index)
    close = close.sort_index()
    last_map = last_close_map(close)
    id_to_sym = id_to_sym or {}

    pit = pit100.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit["id"] = pit["id"].astype(int)
    members = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit.groupby("date")["id"]
    }

    pr = preds.copy()
    pr["date"] = pd.to_datetime(pr["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pr["id"] = pr["id"].astype(int)
    pr = pr.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
    swide = pr.pivot(index="date", columns="id", values="spread").sort_index()
    swide.index = _utc_idx(swide.index)

    ft = feat[["date", "id", "ret_7_raw"]].copy()
    ft["date"] = pd.to_datetime(ft["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    ft["id"] = ft["id"].astype(int)
    blow = ft.pivot(index="date", columns="id", values="ret_7_raw").sort_index()
    blow.index = _utc_idx(blow.index)

    gon = None
    if gate_on is not None:
        gon = gate_on.copy()
        gon.index = _utc_idx(gon.index)

    oos_dates = [d for d in close.index if d in swide.index]
    if int(btc_id) in close.columns:
        oos_dates = [d for d in oos_dates if listed_mask(close, d, int(btc_id))]
    if len(oos_dates) < h + 5:
        return {"error": "not enough OOS dates", "horizon": h}

    alt_c = alt_bps * 1e-4
    btc_c = btc_bps * 1e-4
    slots = [pd.Series(dtype=float) for _ in range(h)]
    prev_full = pd.Series({int(btc_id): 1.0}, dtype=float) if park_btc else pd.Series(dtype=float)
    daily, btc_w, to_list, eq_dates, n_alt, gate_hist, alt_g = [], [], [], [], [], [], []
    forced_events = []

    def _ranks(dt) -> list[int]:
        names = [s for s in members.get(dt, []) if s in close.columns and int(s) != int(btc_id)]
        if spot_filter:
            names = [s for s in names if listed_mask(close, dt, int(s))]
        if dt not in swide.index:
            return []
        row = swide.loc[dt]
        scored = []
        for iid in names:
            if iid not in row.index:
                continue
            sc = float(row[iid])
            if np.isfinite(sc):
                scored.append((sc, int(iid)))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    def _picks(dt, prev_slot: pd.Series) -> list[int]:
        ordered = _ranks(dt)
        n = len(ordered)
        kd = min(int(k_enter), n)
        kq = min(int(k_stay), n)
        long_enter = set(ordered[:kd]) if kd else set()
        long_stay = set(ordered[:kq]) if kq else set()
        held = set(int(i) for i in prev_slot.index) if len(prev_slot) else set()
        kept, news = [], []
        for iid in ordered:
            if iid in held and iid in long_stay:
                kept.append(iid)
            elif iid in long_enter and iid not in held:
                r7 = np.nan
                if dt in blow.index and iid in blow.columns:
                    r7 = float(blow.loc[dt, iid])
                if np.isfinite(r7) and r7 > blowoff:
                    continue
                news.append(iid)
        out, seen = [], set()
        for iid in kept + news:
            if iid in seen:
                continue
            seen.add(iid)
            out.append(iid)
            if len(out) >= int(n_hold):
                break
        return out

    def _gate_state(dt) -> bool:
        if gon is None:
            return True
        if dt in gon.index:
            return bool(gon.loc[dt])
        asof = gon.asof(dt)
        if asof is None or (isinstance(asof, float) and not np.isfinite(asof)):
            return False
        return bool(asof)

    for i, dt in enumerate(oos_dates[:-1]):
        k = i % h
        nxt = oos_dates[i + 1]
        is_on = _gate_state(dt)
        if not is_on:
            slots[k] = pd.Series(dtype=float)
        else:
            pick = _picks(dt, slots[k])
            if pick:
                w = (float(budget) / h) / len(pick)
                slots[k] = pd.Series({s: w for s in pick}, dtype=float)
            else:
                slots[k] = pd.Series(dtype=float)
        for j in range(h):
            if not len(slots[j]):
                continue
            drop = []
            for iid in slots[j].index:
                ld = last_map.get(int(iid))
                if ld is None or nxt > ld:
                    drop.append(int(iid))
            if drop:
                w_drop = float(slots[j].reindex(drop).fillna(0.0).sum())
                slots[j] = slots[j].drop(labels=drop, errors="ignore")
                if w_drop > 0:
                    forced_events.append({"date": str(dt.date()), "ids": drop, "weight": w_drop, "slot": j})
        full = pd.Series(dtype=float)
        for sl in slots:
            if len(sl):
                full = full.add(sl, fill_value=0.0)
        if len(full):
            full = full.clip(upper=name_cap)
        alt_gross = float(full.abs().sum()) if len(full) else 0.0
        w_btc = max(0.0, 1.0 - alt_gross) if park_btc else 0.0
        applied = full.copy() if len(full) else pd.Series(dtype=float)
        if park_btc:
            applied.loc[int(btc_id)] = float(applied.get(int(btc_id), 0.0)) + w_btc

        idx = applied.index.union(prev_full.index)
        a = applied.reindex(idx).fillna(0.0)
        prev = prev_full.reindex(idx).fillna(0.0)
        dw = (a - prev).abs()
        cost = 0.0
        for s, mag in dw.items():
            cost += float(mag) * (btc_c if int(s) == int(btc_id) else alt_c)
        turnover = 0.5 * float(dw.sum())

        r = 0.0
        if nxt in close.index and dt in close.index:
            simple = close.loc[nxt] / close.loc[dt] - 1.0
            for s, wi in applied.items():
                if s in simple.index and np.isfinite(simple[s]):
                    r += float(wi) * float(simple[s])
        net = r - cost
        daily.append(net)
        btc_w.append(float(applied.get(int(btc_id), 0.0)) if park_btc else 0.0)
        to_list.append(turnover)
        eq_dates.append(nxt)
        n_alt.append(int((full > 0).sum()) if len(full) else 0)
        gate_hist.append(1.0 if is_on else 0.0)
        alt_g.append(alt_gross)
        prev_full = applied
        if i % 60 == 0:
            print(
                f"[HB] longtide h={h} park={int(park_btc)} spot={int(spot_filter)} "
                f"i={i}/{len(oos_dates)} dt={dt.date()} on={int(is_on)} "
                f"nalts={n_alt[-1]} walt={alt_gross:.2f} wbtc={btc_w[-1]:.2f} net={net:.5f}",
                flush=True,
            )

    rets = pd.Series(daily, index=pd.DatetimeIndex(eq_dates), dtype=float)
    if int(btc_id) in close.columns:
        btc_simple = close[int(btc_id)].astype(float).pct_change()
    else:
        btc_simple = pd.Series(0.0, index=close.index)
    wbtc = pd.Series(btc_w, index=rets.index)
    nn = pd.Series(n_alt, index=rets.index)
    packed = summarize_book(rets, btc_simple, wbtc, nn, to_list, cycles=cycles)
    packed["gate_on"] = pd.Series(gate_hist, index=rets.index)
    packed["alt_gross"] = pd.Series(alt_g, index=rets.index)
    n_events = len(forced_events)
    ids_forced = sorted({i for e in forced_events for i in e["ids"]})
    wsum = float(sum(e["weight"] for e in forced_events))
    packed.update(
        {
            "horizon": int(h),
            "budget": float(budget),
            "park_btc": bool(park_btc),
            "spot_filter": bool(spot_filter),
            "k_enter": int(k_enter),
            "k_stay": int(k_stay),
            "n_hold": int(n_hold),
            "forced_exits": {
                "n_events": n_events,
                "n_ids": len(ids_forced),
                "ids": ids_forced[:40],
                "symbols": [id_to_sym.get(i) for i in ids_forced[:40]],
                "weight_sum": wsum,
                "cost_drag": wsum * (alt_c + (btc_c if park_btc else 0.0)),
            },
            "regime_breadth": float(REGIME_BREADTH),
            "regime_off_hysteresis": int(REGIME_OFF_HYSTERESIS),
        }
    )
    return enrich_book(packed)


def longtide_verdicts(tide: dict, v1: dict) -> dict:
    """Mechanical VIABLE / SUPERSEDES. No post-hoc adjustment."""
    book_tot = float(tide.get("book_total") or 0.0)
    btc_tot = float(tide.get("btc_total") or 0.0)
    rel = float(tide.get("rel_sharpe") or 0.0)
    mdd = float(tide.get("maxdd") or 0.0)
    btc_mdd = float(tide.get("btc_maxdd") or 0.0)
    alt = float(tide.get("avg_alt_deployment") or 0.0)
    v1_rel = float(v1.get("rel_sharpe") or 0.0)
    a = bool(np.isfinite(book_tot) and np.isfinite(btc_tot) and book_tot >= btc_tot)
    b = bool(np.isfinite(rel) and rel > 0)
    c = bool(np.isfinite(mdd) and np.isfinite(btc_mdd) and mdd >= btc_mdd)
    viable = bool(a and b and c)
    d = bool(np.isfinite(rel) and np.isfinite(v1_rel) and rel >= v1_rel + float(LONGTIDE_REL_MARGIN))
    e = bool(np.isfinite(alt) and alt >= float(LONGTIDE_ALT_MIN))
    cycle_rels = []
    f_ok = True
    worst = float("nan")
    for name, blob in (tide.get("cycles") or {}).items():
        rs = blob.get("rel_sharpe")
        if rs is None or not np.isfinite(float(rs)):
            continue
        rs = float(rs)
        cycle_rels.append({"cycle": name, "rel_sharpe": rs})
        if not np.isfinite(worst) or rs < worst:
            worst = rs
        if rs < float(LONGTIDE_CYCLE_REL_FLOOR):
            f_ok = False
    if not cycle_rels:
        f_ok = False
    supersedes = bool(viable and d and e and f_ok)
    if viable and not supersedes:
        status = "PARALLEL-VARIANT"
    elif supersedes:
        status = "SUPERSEDES-V1"
    else:
        status = "NOT-VIABLE"
    return {
        "criterion": LONGTIDE_CRITERION,
        "death_convention": DEATH_CONVENTION,
        "a_total_ge_btc": a,
        "b_rel_sharpe_gt0": b,
        "c_maxdd_le_btc": c,
        "viable": viable,
        "d_rel_ge_v1_plus_margin": d,
        "e_alt_deployment_ge_15pct": e,
        "f_no_cycle_rel_below_floor": f_ok,
        "supersedes": supersedes,
        "status": status,
        "rel_sharpe": rel,
        "v1_rel_sharpe": v1_rel,
        "need_supersede_rel": v1_rel + float(LONGTIDE_REL_MARGIN),
        "avg_alt_deployment": alt,
        "book_total": book_tot,
        "btc_total": btc_tot,
        "maxdd": mdd,
        "btc_maxdd": btc_mdd,
        "cycle_rels": cycle_rels,
        "worst_cycle_rel": worst,
        "need_alt_min": float(LONGTIDE_ALT_MIN),
        "need_rel_margin": float(LONGTIDE_REL_MARGIN),
        "need_cycle_floor": float(LONGTIDE_CYCLE_REL_FLOOR),
        "precondition": LONGTIDE_PRECONDITION,
    }


def gate_params_ok() -> dict:
    return {
        "REGIME_BREADTH": float(REGIME_BREADTH),
        "REGIME_OFF_HYSTERESIS": int(REGIME_OFF_HYSTERESIS),
        "byte_identical": bool(float(REGIME_BREADTH) == 0.50 and int(REGIME_OFF_HYSTERESIS) == 5),
    }
