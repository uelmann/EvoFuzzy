"""Hysteresis BTC-parked book with death-in-position and anti-blowoff."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import (
    ALT_BPS,
    ANNUALIZATION,
    BTC_BPS,
    BLOWOFF_RET_7,
    NAME_CAP,
    N_HOLD,
    P_EXIT_GAP,
    PHASE2_CYCLES,
    REPLACES_MARGIN,
)


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANNUALIZATION)) if len(x) and x.std() > 0 else 0.0


def _cagr(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return float("nan")
    years = len(eq) / ANNUALIZATION
    return float(eq.iloc[-1] ** (1.0 / max(years, 1e-6)) - 1.0)


def _maxdd(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def summarize_book(rets: pd.Series, btc_simple: pd.Series, wbtc: pd.Series, n_names: pd.Series, to_list: list, cycles=PHASE2_CYCLES) -> dict:
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    eq_btc = (1.0 + btc_simple.reindex(rets.index).fillna(0.0)).cumprod()
    rel_eq = eq / eq_btc.replace(0, np.nan)
    rel_ret = rel_eq.pct_change().fillna(0.0)
    out_cycles = {}
    for name, a, b in cycles:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = rets[(rets.index >= t0) & (rets.index <= t1)]
        if sl.empty:
            continue
        eq_s = (1.0 + sl.fillna(0.0)).cumprod()
        btc_s = btc_simple.reindex(sl.index).fillna(0.0)
        eqb = (1.0 + btc_s).cumprod()
        rel = eq_s / eqb.replace(0, np.nan)
        out_cycles[name] = {
            "n": int(len(sl)),
            "book_total": float(eq_s.iloc[-1] - 1.0) if len(eq_s) else float("nan"),
            "btc_total": float(eqb.iloc[-1] - 1.0) if len(eqb) else float("nan"),
            "book_cagr": _cagr(eq_s),
            "btc_cagr": _cagr(eqb),
            "book_sharpe": _sharpe(sl),
            "rel_cagr": _cagr(rel),
            "rel_sharpe": _sharpe(rel.pct_change().fillna(0.0)),
            "maxdd": _maxdd(eq_s),
            "btc_maxdd": _maxdd(eqb),
            "avg_w_btc": float(wbtc.reindex(sl.index).mean()) if len(sl) else float("nan"),
            "avg_n_names": float(n_names.reindex(sl.index).mean()) if len(sl) else float("nan"),
        }
    book_total = float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")
    btc_total = float(eq_btc.iloc[-1] - 1.0) if len(eq_btc) else float("nan")
    return {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()) if len(rets) else None,
        "end": str(rets.index.max().date()) if len(rets) else None,
        "book_total": book_total,
        "btc_total": btc_total,
        "book_cagr": _cagr(eq),
        "btc_cagr": _cagr(eq_btc),
        "book_sharpe": _sharpe(rets),
        "btc_sharpe": _sharpe(btc_simple.reindex(rets.index)),
        "maxdd": _maxdd(eq),
        "btc_maxdd": _maxdd(eq_btc),
        "rel_cagr": _cagr(rel_eq),
        "rel_sharpe": _sharpe(rel_ret),
        "avg_w_btc": float(wbtc.mean()) if len(wbtc) else float("nan"),
        "avg_n_names": float(n_names.mean()) if len(n_names) else float("nan"),
        "ann_turnover": float(np.mean(to_list) * ANNUALIZATION) if to_list else float("nan"),
        "cycles": out_cycles,
        "daily_ret": rets,
        "btc_ret": btc_simple.reindex(rets.index),
        "equity": eq,
        "equity_btc": eq_btc,
        "rel_equity": rel_eq,
        "w_btc": wbtc,
        "n_names": n_names,
    }


def pick_median_p_enter(runs: list[dict]) -> dict:
    """House median convention: the grid point whose rel Sharpe is the median of the three."""
    ok = [r for r in runs if r.get("rel_sharpe") is not None and np.isfinite(r.get("rel_sharpe"))]
    if not ok:
        return {"error": "no finite rel_sharpe"}
    sharpes = sorted(float(r["rel_sharpe"]) for r in ok)
    med = float(np.median(sharpes))
    picked = min(ok, key=lambda r: (abs(float(r["rel_sharpe"]) - med), float(r.get("p_enter", 9))))
    return picked


def mechanical_verdicts(model: dict, naive: dict, margin: float = REPLACES_MARGIN) -> dict:
    rel = float(model.get("rel_sharpe") or 0.0)
    book_tot = float(model.get("book_total") or 0.0)
    btc_tot = float(model.get("btc_total") or 0.0)
    mdd = float(model.get("maxdd") or 0.0)
    btc_mdd = float(model.get("btc_maxdd") or 0.0)
    naive_rel = float(naive.get("rel_sharpe") or 0.0)
    a = bool(np.isfinite(rel) and rel > 0)
    b = bool(np.isfinite(book_tot) and np.isfinite(btc_tot) and book_tot >= btc_tot)
    c = bool(np.isfinite(mdd) and np.isfinite(btc_mdd) and mdd >= btc_mdd)  # MaxDD less negative or equal
    viable = bool(a and b and c)
    replaces = bool(viable and np.isfinite(naive_rel) and rel >= naive_rel + margin)
    return {
        "a_rel_sharpe_gt0": a,
        "b_total_ge_btc": b,
        "c_maxdd_le_btc": c,
        "viable": viable,
        "replaces_floor": replaces,
        "rel_sharpe": rel,
        "naive_rel_sharpe": naive_rel,
        "need_replaces": naive_rel + margin,
        "book_total": book_tot,
        "btc_total": btc_tot,
        "maxdd": mdd,
        "btc_maxdd": btc_mdd,
    }


def mechanical_verdicts_v3(model: dict) -> dict:
    """VIABLE / PRODUCT-GRADE vs BTC. Operative floor is BTC (naive v4 is record-only)."""
    from btcb.constants import PRODUCT_ALT_MIN, PRODUCT_REL_SHARPE

    rel = float(model.get("rel_sharpe") or 0.0)
    book_tot = float(model.get("book_total") or 0.0)
    btc_tot = float(model.get("btc_total") or 0.0)
    mdd = float(model.get("maxdd") or 0.0)
    btc_mdd = float(model.get("btc_maxdd") or 0.0)
    wbtc = float(model.get("avg_w_btc") or 1.0)
    alt = 1.0 - wbtc
    a = bool(np.isfinite(rel) and rel > 0)
    b = bool(np.isfinite(book_tot) and np.isfinite(btc_tot) and book_tot >= btc_tot)
    c = bool(np.isfinite(mdd) and np.isfinite(btc_mdd) and mdd >= btc_mdd)
    viable = bool(a and b and c)
    product = bool(
        viable
        and np.isfinite(rel)
        and rel >= float(PRODUCT_REL_SHARPE)
        and np.isfinite(alt)
        and alt >= float(PRODUCT_ALT_MIN)
    )
    return {
        "a_rel_sharpe_gt0": a,
        "b_total_ge_btc": b,
        "c_maxdd_le_btc": c,
        "viable": viable,
        "product_grade": product,
        "rel_sharpe": rel,
        "book_total": book_tot,
        "btc_total": btc_tot,
        "maxdd": mdd,
        "btc_maxdd": btc_mdd,
        "avg_w_btc": wbtc,
        "avg_alt": float(alt),
        "need_product_rel": float(PRODUCT_REL_SHARPE),
        "need_product_alt": float(PRODUCT_ALT_MIN),
    }


def run_hysteresis_book(
    panel: pd.DataFrame,
    pit50: pd.DataFrame,
    preds: pd.DataFrame,
    feat: pd.DataFrame,
    btc_id: int,
    p_enter: float,
    h: int,
    *,
    name_cap: float = NAME_CAP,
    n_hold: int = N_HOLD,
    alt_bps: float = ALT_BPS,
    btc_bps: float = BTC_BPS,
    blowoff: float = BLOWOFF_RET_7,
    p_exit_gap: float = P_EXIT_GAP,
) -> dict:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = df.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    last_map = {int(i): pd.Timestamp(d).tz_convert("UTC").normalize() for i, d in df.groupby("id")["date"].max().items()}
    id_to_sym = df.sort_values("date").groupby("id")["symbol"].last().to_dict()

    pit = pit50.copy()
    pit["date"] = pd.to_datetime(pit["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pit["id"] = pit["id"].astype(int)
    members = {
        pd.Timestamp(d).tz_convert("UTC").normalize(): [int(x) for x in v]
        for d, v in pit.groupby("date")["id"]
    }

    pr = preds.copy()
    pr["date"] = pd.to_datetime(pr["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    pr["id"] = pr["id"].astype(int)
    # last prediction wins on overlap
    pr = pr.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
    pwide = pr.pivot(index="date", columns="id", values="p").sort_index()
    pwide.index = pd.to_datetime(pwide.index, utc=True).tz_convert("UTC").normalize()

    ft = feat[["date", "id", "ret_7_raw"]].copy()
    ft["date"] = pd.to_datetime(ft["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    ft["id"] = ft["id"].astype(int)
    blow = ft.pivot(index="date", columns="id", values="ret_7_raw").sort_index()
    blow.index = pd.to_datetime(blow.index, utc=True).tz_convert("UTC").normalize()

    oos_dates = [d for d in close.index if d in pwide.index]
    if len(oos_dates) < h + 5:
        return {"error": "not enough OOS dates", "p_enter": p_enter, "horizon": h}

    p_exit = float(p_enter) - float(p_exit_gap)
    alt_c = alt_bps * 1e-4
    btc_c = btc_bps * 1e-4
    slots = [pd.Series(dtype=float) for _ in range(h)]
    prev_full = pd.Series({btc_id: 1.0}, dtype=float)
    daily, btc_w, to_list, eq_dates, n_alt = [], [], [], [], []
    forced_events = []

    def _qualifiers(dt, prev_slot: pd.Series) -> list[int]:
        names = [s for s in members.get(dt, []) if s in close.columns and int(s) != int(btc_id)]
        held = [int(i) for i in prev_slot.index] if len(prev_slot) else []
        kept, news = [], []
        for iid in names:
            pv = np.nan
            if dt in pwide.index and iid in pwide.columns:
                pv = float(pwide.loc[dt, iid])
            if not np.isfinite(pv):
                continue
            if iid in held:
                if pv > p_exit:
                    kept.append((pv, iid))
            else:
                r7 = np.nan
                if dt in blow.index and iid in blow.columns:
                    r7 = float(blow.loc[dt, iid])
                if np.isfinite(r7) and r7 > blowoff:
                    continue
                if pv >= p_enter:
                    news.append((pv, iid))
        kept.sort(reverse=True)
        news.sort(reverse=True)
        ordered = [i for _, i in kept] + [i for _, i in news]
        # unique preserve order
        seen = set()
        uniq = []
        for i in ordered:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq[: int(n_hold)]

    for i, dt in enumerate(oos_dates[:-1]):
        k = i % h
        nxt = oos_dates[i + 1]
        pick = _qualifiers(dt, slots[k])
        if pick:
            w = (1.0 / h) / len(pick)
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
            over = full[full > name_cap]
            if len(over):
                full = full.clip(upper=name_cap)
        alt_gross = float(full.abs().sum()) if len(full) else 0.0
        w_btc = max(0.0, 1.0 - alt_gross)
        applied = full.copy()
        applied.loc[btc_id] = float(applied.get(btc_id, 0.0)) + w_btc

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
        btc_w.append(float(applied.get(btc_id, 0.0)))
        to_list.append(turnover)
        eq_dates.append(nxt)
        n_alt.append(int((full > 0).sum()) if len(full) else 0)
        prev_full = applied
        if i % 60 == 0:
            print(
                f"[HB] book h={h} p_enter={p_enter:.2f} i={i}/{len(oos_dates)} dt={dt.date()} "
                f"nalts={n_alt[-1]} wbtc={w_btc:.2f} net={net:.5f} forced={len(forced_events)}",
                flush=True,
            )

    rets = pd.Series(daily, index=pd.DatetimeIndex(eq_dates), dtype=float)
    btc_simple = close[btc_id].pct_change()
    wbtc = pd.Series(btc_w, index=rets.index)
    nn = pd.Series(n_alt, index=rets.index)
    packed = summarize_book(rets, btc_simple, wbtc, nn, to_list)
    n_events = len(forced_events)
    ids_forced = sorted({i for e in forced_events for i in e["ids"]})
    wsum = float(sum(e["weight"] for e in forced_events))
    cost_drag = wsum * (alt_c + btc_c)
    packed.update(
        {
            "p_enter": float(p_enter),
            "p_exit": float(p_exit),
            "horizon": int(h),
            "forced_exits": {
                "n_events": n_events,
                "n_ids": len(ids_forced),
                "ids": ids_forced[:40],
                "symbols": [id_to_sym.get(i) for i in ids_forced[:40]],
                "weight_sum": wsum,
                "cost_drag": cost_drag,
            },
        }
    )
    return packed


def run_gated_book(
    panel: pd.DataFrame,
    pit50: pd.DataFrame,
    preds: pd.DataFrame,
    feat: pd.DataFrame,
    btc_id: int,
    p_enter: float,
    h: int,
    gate_on: pd.Series,
    *,
    budget: float = 0.50,
    name_cap: float = NAME_CAP,
    n_hold: int = N_HOLD,
    alt_bps: float = ALT_BPS,
    btc_bps: float = BTC_BPS,
    blowoff: float = BLOWOFF_RET_7,
    cycles=PHASE2_CYCLES,
) -> dict:
    """Fill `budget` with top-K calibrated p when gate is ON; else 100% BTC. h-tranche."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = df.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    last_map = {int(i): pd.Timestamp(d).tz_convert("UTC").normalize() for i, d in df.groupby("id")["date"].max().items()}
    id_to_sym = df.sort_values("date").groupby("id")["symbol"].last().to_dict()

    pit = pit50.copy()
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
    pwide = pr.pivot(index="date", columns="id", values="p").sort_index()
    pwide.index = pd.to_datetime(pwide.index, utc=True).tz_convert("UTC").normalize()

    ft = feat[["date", "id", "ret_7_raw"]].copy()
    ft["date"] = pd.to_datetime(ft["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    ft["id"] = ft["id"].astype(int)
    blow = ft.pivot(index="date", columns="id", values="ret_7_raw").sort_index()
    blow.index = pd.to_datetime(blow.index, utc=True).tz_convert("UTC").normalize()

    gon = gate_on.copy()
    gon.index = pd.to_datetime(gon.index, utc=True).tz_convert("UTC").normalize()

    oos_dates = [d for d in close.index if d in pwide.index]
    if len(oos_dates) < h + 5:
        return {"error": "not enough OOS dates", "p_enter": p_enter, "horizon": h}

    alt_c = alt_bps * 1e-4
    btc_c = btc_bps * 1e-4
    slots = [pd.Series(dtype=float) for _ in range(h)]
    prev_full = pd.Series({btc_id: 1.0}, dtype=float)
    daily, btc_w, to_list, eq_dates, n_alt, gate_hist = [], [], [], [], [], []
    forced_events = []
    contrib: dict[int, float] = {}

    def _picks(dt, prev_slot: pd.Series) -> list[int]:
        names = [s for s in members.get(dt, []) if s in close.columns and int(s) != int(btc_id)]
        held = set(int(i) for i in prev_slot.index) if len(prev_slot) else set()
        scored = []
        for iid in names:
            if dt not in pwide.index or iid not in pwide.columns:
                continue
            pv = float(pwide.loc[dt, iid])
            if not np.isfinite(pv) or pv < float(p_enter):
                continue
            if iid not in held:
                r7 = np.nan
                if dt in blow.index and iid in blow.columns:
                    r7 = float(blow.loc[dt, iid])
                if np.isfinite(r7) and r7 > blowoff:
                    continue
            scored.append((pv, iid))
        scored.sort(reverse=True)
        return [i for _, i in scored[: int(n_hold)]]

    for i, dt in enumerate(oos_dates[:-1]):
        k = i % h
        nxt = oos_dates[i + 1]
        is_on = bool(gon.reindex([dt]).fillna(0).iloc[0]) if dt in gon.index else bool(gon.asof(dt) if len(gon) else 0)
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
        w_btc = max(0.0, 1.0 - alt_gross)
        applied = full.copy()
        applied.loc[btc_id] = float(applied.get(btc_id, 0.0)) + w_btc

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
                    rs = float(simple[s])
                    r += float(wi) * rs
                    contrib[int(s)] = contrib.get(int(s), 0.0) + float(wi) * rs
        net = r - cost
        daily.append(net)
        btc_w.append(float(applied.get(btc_id, 0.0)))
        to_list.append(turnover)
        eq_dates.append(nxt)
        n_alt.append(int((full > 0).sum()) if len(full) else 0)
        gate_hist.append(1.0 if is_on else 0.0)
        prev_full = applied
        if i % 60 == 0:
            print(
                f"[HB] gated h={h} p={p_enter:.2f} i={i}/{len(oos_dates)} dt={dt.date()} "
                f"on={int(is_on)} nalts={n_alt[-1]} wbtc={w_btc:.2f} net={net:.5f}",
                flush=True,
            )

    rets = pd.Series(daily, index=pd.DatetimeIndex(eq_dates), dtype=float)
    btc_simple = close[btc_id].pct_change()
    wbtc = pd.Series(btc_w, index=rets.index)
    nn = pd.Series(n_alt, index=rets.index)
    packed = summarize_book(rets, btc_simple, wbtc, nn, to_list, cycles=cycles)
    n_events = len(forced_events)
    ids_forced = sorted({i for e in forced_events for i in e["ids"]})
    packed.update(
        {
            "p_enter": float(p_enter),
            "horizon": int(h),
            "budget": float(budget),
            "gate_on_frac": float(np.mean(gate_hist)) if gate_hist else float("nan"),
            "gate_on": pd.Series(gate_hist, index=rets.index),
            "contrib": contrib,
            "forced_exits": {
                "n_events": n_events,
                "n_ids": len(ids_forced),
                "ids": ids_forced[:40],
                "symbols": [id_to_sym.get(i) for i in ids_forced[:40]],
                "weight_sum": float(sum(e["weight"] for e in forced_events)),
            },
        }
    )
    return packed
