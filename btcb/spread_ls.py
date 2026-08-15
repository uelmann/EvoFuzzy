"""SPREAD-LS book: decile long/short on the 2.c spread. No BTC anywhere."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.constants import (
    ANNUALIZATION,
    BLOWOFF_RET_7,
    COMBO_OVERLAP_START,
    LS_BETA_MATCH_LOOKBACK,
    LS_BETA_WINDOW,
    LS_DECILE_K,
    LS_GROSS_LONG,
    LS_GROSS_SHORT,
    LS_LONG_BPS,
    LS_MIN_SHORTABLE,
    LS_QUINTILE_K,
    LS_REPLACE_SHARPE_GAP,
    LS_SHORT_FEE_BPS,
    LS_SHORT_SLIP_BPS,
    LS_SLEEVE_CORR_MAX,
    LS_SLEEVE_SHARPE_GAP,
    LS_SQUEEZE_N,
    LS_TRAIL_DAYS,
    LS_VIABLE_FULL,
    LS_VIABLE_TRAIL,
    NAME_CAP,
    PHASE2_CYCLES,
)
from btcb.model import merge_twin_preds


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


def _cagr(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return float("nan")
    years = len(eq) / ANNUALIZATION
    return float(eq.iloc[-1] ** (1.0 / max(years, 1e-6)) - 1.0)


def _maxdd(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def trail_slice(s: pd.Series, days: int = LS_TRAIL_DAYS) -> pd.Series:
    s = s.copy()
    s.index = _utc_idx(s.index)
    if s.empty:
        return s
    end = s.index.max()
    start = end - pd.Timedelta(days=int(days))
    return s[(s.index >= start) & (s.index <= end)]


def hash_pred_dir(pred_dir: Path) -> dict:
    files = sorted(pred_dir.glob("preds_*.parquet"))
    if not files:
        raise RuntimeError(f"empty 2.c pred cache: {pred_dir}")
    h = hashlib.sha256()
    rows = []
    for p in files:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(p.name.encode())
        h.update(digest.encode())
        rows.append({"name": p.name, "sha256": digest, "bytes": int(p.stat().st_size)})
    return {"sha256": h.hexdigest(), "n_files": int(len(files)), "files": rows}


def load_twin_from_cache(pred_dir: Path, horizon: int) -> pd.DataFrame:
    tops = sorted(pred_dir.glob(f"preds_top_h{horizon}_fold*.parquet"))
    bots = sorted(pred_dir.glob(f"preds_bot_h{horizon}_fold*.parquet"))
    if not tops or not bots:
        raise RuntimeError(
            f"missing 2.c cache for h={horizon} top={len(tops)} bot={len(bots)} in {pred_dir}"
        )
    top = pd.concat([pd.read_parquet(p) for p in tops], ignore_index=True)
    bot = pd.concat([pd.read_parquet(p) for p in bots], ignore_index=True)
    twin = merge_twin_preds(top, bot, horizon)
    twin["date"] = pd.to_datetime(twin["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    twin["id"] = twin["id"].astype(int)
    return twin


def _cmc_bases(symbol: str) -> list[str]:
    s = str(symbol).upper().strip()
    if s.endswith("USDT"):
        s = s[: -len("USDT")]
    out = [s]
    if s.startswith("1000") and len(s) > 4:
        out.append(s[4:])
    if s.endswith("2") and len(s) > 2 and s[:-1].isalpha():
        out.append(s[:-1])
    return out


def build_shortable(
    cmc_panel: pd.DataFrame,
    kline_panel: pd.DataFrame,
    btc_id: int,
) -> dict[pd.Timestamp, set[int]]:
    """date -> set of CMC ids with a live Binance USDT-M perp that day."""
    cmc = cmc_panel[["id", "symbol"]].drop_duplicates()
    cmc["id"] = cmc["id"].astype(int)
    cmc["sym"] = cmc["symbol"].astype(str).str.upper()
    sym_to_ids: dict[str, set[int]] = defaultdict(set)
    for row in cmc.itertuples(index=False):
        sym_to_ids[str(row.sym)].add(int(row.id))

    kl = kline_panel.copy()
    kl["date"] = pd.to_datetime(kl["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    kl["symbol"] = kl["symbol"].astype(str).str.upper()

    bsym_to_ids: dict[str, set[int]] = {}
    for bsym in kl["symbol"].unique():
        ids: set[int] = set()
        for base in _cmc_bases(bsym):
            ids |= sym_to_ids.get(base, set())
        if ids:
            bsym_to_ids[str(bsym)] = ids

    shortable: dict[pd.Timestamp, set[int]] = defaultdict(set)
    for bsym, g in kl.groupby("symbol", sort=False):
        ids = bsym_to_ids.get(str(bsym))
        if not ids:
            continue
        ids = {i for i in ids if int(i) != int(btc_id)}
        if not ids:
            continue
        for d in g["date"].unique():
            shortable[pd.Timestamp(d)].update(ids)
    return shortable


def trailing_beta_wide(close: pd.DataFrame, btc_id: int, window: int = LS_BETA_MATCH_LOOKBACK) -> pd.DataFrame:
    r = close.pct_change()
    if btc_id not in r.columns:
        raise RuntimeError("BTC missing from close for beta")
    xb = r[btc_id]
    var = xb.rolling(window, min_periods=max(20, window // 3)).var()
    out = {}
    for col in r.columns:
        cov = r[col].rolling(window, min_periods=max(20, window // 3)).cov(xb)
        out[col] = cov / var.replace(0, np.nan)
    return pd.DataFrame(out, index=r.index)


def ols_beta(y: pd.Series, x: pd.Series) -> float:
    a, b = y.align(x, join="inner")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if len(a) < 20 or float(b.std()) == 0:
        return float("nan")
    var = float(b.var())
    if var <= 0:
        return float("nan")
    return float(np.cov(a, b, ddof=1)[0, 1] / var)


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int = LS_BETA_WINDOW) -> pd.Series:
    a, b = y.align(x, join="inner")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    cov = a.rolling(window, min_periods=max(20, window // 3)).cov(b)
    var = b.rolling(window, min_periods=max(20, window // 3)).var()
    return cov / var.replace(0, np.nan)


def ew_basket(close: pd.DataFrame, members: dict, dates: list) -> pd.Series:
    rets = []
    idx = []
    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        ids = [s for s in members.get(dt, []) if s in close.columns]
        if not ids or nxt not in close.index or dt not in close.index:
            continue
        simple = close.loc[nxt, ids] / close.loc[dt, ids] - 1.0
        simple = simple.replace([np.inf, -np.inf], np.nan).dropna()
        if simple.empty:
            continue
        rets.append(float(simple.mean()))
        idx.append(nxt)
    return pd.Series(rets, index=pd.DatetimeIndex(idx), dtype=float)


def _size_leg(ids: list[int], budget: float, cap: float, sign: float) -> pd.Series:
    if not ids or budget <= 0:
        return pd.Series(dtype=float)
    n = len(ids)
    w = min(float(budget) / n, float(cap))
    if w <= 0:
        return pd.Series(dtype=float)
    return pd.Series({int(i): float(sign) * w for i in ids}, dtype=float)


def _scale_thinner(long_w: pd.Series, short_w: pd.Series, cap: float, long_b: float, short_b: float):
    lg = float(long_w.sum()) if len(long_w) else 0.0
    sg = float((-short_w).clip(lower=0).sum()) if len(short_w) else 0.0
    if lg + 1e-12 < long_b and len(long_w):
        room = np.maximum(0.0, cap - long_w.values)
        need = long_b - lg
        fill = min(float(room.sum()), need)
        if fill > 0 and room.sum() > 0:
            long_w = long_w + pd.Series(room * (fill / room.sum()), index=long_w.index)
    if sg + 1e-12 < short_b and len(short_w):
        mag = (-short_w).clip(lower=0)
        room = np.maximum(0.0, cap - mag.values)
        need = short_b - sg
        fill = min(float(room.sum()), need)
        if fill > 0 and room.sum() > 0:
            short_w = short_w - pd.Series(room * (fill / room.sum()), index=short_w.index)
    return long_w, short_w


def summarize_ls(rets: pd.Series, stats: dict, cycles=PHASE2_CYCLES) -> dict:
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    out_cycles = {}
    for name, a, b in cycles:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = rets[(rets.index >= t0) & (rets.index <= t1)]
        if sl.empty:
            continue
        eq_s = (1.0 + sl.fillna(0.0)).cumprod()
        out_cycles[name] = {
            "n": int(len(sl)),
            "book_total": float(eq_s.iloc[-1] - 1.0) if len(eq_s) else float("nan"),
            "book_cagr": _cagr(eq_s),
            "net_sharpe": _sharpe(sl),
            "maxdd": _maxdd(eq_s),
            "avg_n_long": float(stats["n_long"].reindex(sl.index).mean()) if len(sl) else float("nan"),
            "avg_n_short": float(stats["n_short"].reindex(sl.index).mean()) if len(sl) else float("nan"),
        }
    packed = {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()) if len(rets) else None,
        "end": str(rets.index.max().date()) if len(rets) else None,
        "book_total": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
        "book_cagr": _cagr(eq),
        "net_sharpe": _sharpe(rets),
        "net_sharpe_trail18m": _sharpe(trail_slice(rets)),
        "maxdd": _maxdd(eq),
        "ann_turnover": float(np.mean(stats["to_list"]) * ANNUALIZATION) if stats["to_list"] else float("nan"),
        "avg_n_long": float(stats["n_long"].mean()) if len(stats["n_long"]) else float("nan"),
        "avg_n_short": float(stats["n_short"].mean()) if len(stats["n_short"]) else float("nan"),
        "avg_shortable": float(stats["n_shortable"].mean()) if len(stats["n_shortable"]) else float("nan"),
        "pct_incomplete_short": float(stats["incomplete"].mean()) if len(stats["incomplete"]) else float("nan"),
        "avg_long_gross": float(stats["long_gross"].mean()) if len(stats["long_gross"]) else float("nan"),
        "avg_short_gross": float(stats["short_gross"].mean()) if len(stats["short_gross"]) else float("nan"),
        "avg_cash": float(stats["cash"].mean()) if len(stats["cash"]) else float("nan"),
        "cycles": out_cycles,
        "daily_ret": rets,
        "equity": eq,
        "n_long": stats["n_long"],
        "n_short": stats["n_short"],
        "n_shortable": stats["n_shortable"],
        "incomplete": stats["incomplete"],
        "long_gross": stats["long_gross"],
        "short_gross": stats["short_gross"],
        "cash": stats["cash"],
    }
    return packed


def mechanical_verdicts_ls(headline: dict, combo_overlap: dict) -> dict:
    full = float(headline.get("net_sharpe") or 0.0)
    trail = float(headline.get("net_sharpe_trail18m") or 0.0)
    viable = bool(np.isfinite(full) and full >= LS_VIABLE_FULL and np.isfinite(trail) and trail >= LS_VIABLE_TRAIL)
    corr = float(combo_overlap.get("corr") if combo_overlap else float("nan"))
    same = float(combo_overlap.get("ls_sharpe") if combo_overlap else float("nan"))
    cs = float(combo_overlap.get("combo_sharpe") if combo_overlap else float("nan"))
    sleeve = bool(
        viable
        and np.isfinite(corr)
        and corr < LS_SLEEVE_CORR_MAX
        and np.isfinite(same)
        and np.isfinite(cs)
        and same >= cs - LS_SLEEVE_SHARPE_GAP
    )
    replace = bool(np.isfinite(same) and np.isfinite(cs) and same >= cs + LS_REPLACE_SHARPE_GAP)
    return {
        "viable": viable,
        "sleeve_grade": sleeve,
        "replacement_candidate": replace,
        "net_sharpe_full": full,
        "net_sharpe_trail18m": trail,
        "need_full": LS_VIABLE_FULL,
        "need_trail": LS_VIABLE_TRAIL,
        "corr_combo": corr,
        "need_corr_lt": LS_SLEEVE_CORR_MAX,
        "same_window_ls": same,
        "same_window_combo": cs,
        "need_sleeve_sharpe": (cs - LS_SLEEVE_SHARPE_GAP) if np.isfinite(cs) else float("nan"),
        "need_replace_sharpe": (cs + LS_REPLACE_SHARPE_GAP) if np.isfinite(cs) else float("nan"),
        "headline_variant": "dollar_neutral",
    }


def run_spread_ls(
    panel: pd.DataFrame,
    pit100: pd.DataFrame,
    preds: pd.DataFrame,
    feat: pd.DataFrame,
    shortable: dict,
    btc_id: int,
    h: int,
    *,
    beta_matched: bool = False,
    name_cap: float = NAME_CAP,
    blowoff: float = BLOWOFF_RET_7,
) -> dict:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = df.pivot(index="date", columns="id", values="close").sort_index()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("UTC").normalize()
    last_map = {int(i): pd.Timestamp(d).tz_convert("UTC").normalize() for i, d in df.groupby("id")["date"].max().items()}
    id_to_sym = df.sort_values("date").groupby("id")["symbol"].last().to_dict()

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
    swide.index = pd.to_datetime(swide.index, utc=True).tz_convert("UTC").normalize()

    ft = feat[["date", "id", "ret_7_raw"]].copy()
    ft["date"] = pd.to_datetime(ft["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    ft["id"] = ft["id"].astype(int)
    blow = ft.pivot(index="date", columns="id", values="ret_7_raw").sort_index()
    blow.index = pd.to_datetime(blow.index, utc=True).tz_convert("UTC").normalize()

    beta_w = trailing_beta_wide(close, btc_id) if beta_matched else None

    oos_dates = [d for d in close.index if d in swide.index]
    if len(oos_dates) < h + 5:
        return {"error": "not enough OOS dates", "horizon": h, "beta_matched": beta_matched}

    long_c = LS_LONG_BPS * 1e-4
    short_c = (LS_SHORT_FEE_BPS + LS_SHORT_SLIP_BPS) * 1e-4
    long_slots = [[] for _ in range(h)]
    short_slots = [[] for _ in range(h)]
    prev_full = pd.Series(dtype=float)
    daily, eq_dates, to_list = [], [], []
    n_long_s, n_short_s, n_sh_s, inc_s = [], [], [], []
    lg_s, sg_s, cash_s = [], [], []
    forced_exits, forced_covers = [], []
    btc_hits = 0

    def _ranks(dt) -> tuple[list[int], dict[int, int]]:
        names = [s for s in members.get(dt, []) if s in close.columns and int(s) != int(btc_id)]
        scored = []
        if dt not in swide.index:
            return names, {}
        row = swide.loc[dt]
        for iid in names:
            if iid not in row.index:
                continue
            sc = float(row[iid])
            if np.isfinite(sc):
                scored.append((sc, int(iid)))
        scored.sort(reverse=True)
        ordered = [i for _, i in scored]
        rank = {iid: r + 1 for r, iid in enumerate(ordered)}
        return ordered, rank

    def _picks(dt, prev_long: list[int], prev_short: list[int]):
        ordered, rank = _ranks(dt)
        n = len(ordered)
        k_dec = min(LS_DECILE_K, n)
        k_q = min(LS_QUINTILE_K, n)
        sh_today = shortable.get(dt, set())
        n_shortable = sum(1 for iid in ordered if iid in sh_today)

        long_enter = set(ordered[:k_dec]) if k_dec else set()
        long_stay = set(ordered[:k_q]) if k_q else set()
        short_pool = ordered[-k_dec:] if k_dec else []
        short_stay_pool = set(ordered[-k_q:] if k_q else [])
        short_enter = [i for i in short_pool if i in sh_today]
        if len(short_enter) < LS_MIN_SHORTABLE:
            short_enter_set = set(short_enter)
        else:
            short_enter_set = set(short_enter)
        incomplete = bool(len(short_enter) < k_dec)

        kept_l, news_l = [], []
        held_l = set(int(i) for i in prev_long)
        for iid in ordered:
            if iid in held_l and iid in long_stay:
                kept_l.append(iid)
            elif iid in long_enter and iid not in held_l:
                r7 = np.nan
                if dt in blow.index and iid in blow.columns:
                    r7 = float(blow.loc[dt, iid])
                if np.isfinite(r7) and r7 > blowoff:
                    continue
                news_l.append(iid)
        longs = []
        seen = set()
        for iid in kept_l + news_l:
            if iid not in seen:
                seen.add(iid)
                longs.append(iid)

        kept_s, news_s = [], []
        held_s = set(int(i) for i in prev_short)
        for iid in ordered[::-1]:
            if iid not in sh_today:
                continue
            if iid in held_s and iid in short_stay_pool:
                kept_s.append(iid)
            elif iid in short_enter_set and iid not in held_s:
                news_s.append(iid)
        shorts = []
        seen = set()
        for iid in kept_s + news_s:
            if iid not in seen:
                seen.add(iid)
                shorts.append(iid)
        return longs, shorts, n_shortable, incomplete

    for i, dt in enumerate(oos_dates[:-1]):
        k = i % h
        nxt = oos_dates[i + 1]
        longs, shorts, n_sh, incomplete = _picks(dt, long_slots[k], short_slots[k])
        long_slots[k] = longs
        short_slots[k] = shorts

        def _drop_dead(slots, events, kind):
            for j in range(h):
                if not slots[j]:
                    continue
                drop = []
                for iid in slots[j]:
                    ld = last_map.get(int(iid))
                    if ld is None or nxt > ld:
                        drop.append(int(iid))
                if drop:
                    events.append({"date": str(dt.date()), "ids": drop, "slot": j, "kind": kind})
                    slots[j] = [x for x in slots[j] if int(x) not in set(drop)]

        _drop_dead(long_slots, forced_exits, "exit")
        _drop_dead(short_slots, forced_covers, "cover")

        long_w = pd.Series(dtype=float)
        short_w = pd.Series(dtype=float)
        lb = LS_GROSS_LONG / h
        sb = LS_GROSS_SHORT / h
        for sl in long_slots:
            if sl:
                long_w = long_w.add(_size_leg(sl, lb, name_cap, 1.0), fill_value=0.0)
        for sl in short_slots:
            if sl:
                short_w = short_w.add(_size_leg(sl, sb, name_cap, -1.0), fill_value=0.0)
        if len(long_w):
            long_w = long_w.clip(upper=name_cap)
        if len(short_w):
            short_w = short_w.clip(lower=-name_cap)
        if beta_matched:
            brow = beta_w.loc[dt] if (beta_w is not None and dt in beta_w.index) else pd.Series(dtype=float)

            def _leg_beta(w: pd.Series) -> float:
                acc = 0.0
                for iid, wi in w.items():
                    b = float(brow[int(iid)]) if int(iid) in brow.index else np.nan
                    if not np.isfinite(b):
                        b = 1.0
                    acc += abs(float(wi)) * b
                return acc

            bl = _leg_beta(long_w)
            bs = _leg_beta(short_w.abs() if len(short_w) else short_w)
            if bs > 1e-8 and len(short_w):
                short_w = short_w * (bl / bs)
                short_w = short_w.clip(lower=-name_cap)
        else:
            long_w, short_w = _scale_thinner(long_w, short_w, name_cap, LS_GROSS_LONG, LS_GROSS_SHORT)
            if len(long_w):
                long_w = long_w.clip(upper=name_cap)
            if len(short_w):
                short_w = short_w.clip(lower=-name_cap)

        applied = long_w.add(short_w, fill_value=0.0)
        if int(btc_id) in applied.index and abs(float(applied.get(btc_id, 0.0))) > 1e-12:
            btc_hits += 1
            raise RuntimeError(f"BTC id={btc_id} appeared in SPREAD-LS book on {dt.date()} w={applied.get(btc_id)}")

        idx = applied.index.union(prev_full.index)
        a = applied.reindex(idx).fillna(0.0)
        prev = prev_full.reindex(idx).fillna(0.0)
        dw = a - prev
        cost = 0.0
        for s, mag in dw.items():
            m = float(mag)
            if m == 0:
                continue
            # cost on traded notional; shorts pay perp costs, longs spot
            new_w = float(a.get(s, 0.0))
            old_w = float(prev.get(s, 0.0))
            # classify by post-trade sign; crossing zero pays both
            if old_w >= 0 and new_w >= 0:
                cost += abs(m) * long_c
            elif old_w <= 0 and new_w <= 0:
                cost += abs(m) * short_c
            else:
                cost += abs(old_w) * (long_c if old_w > 0 else short_c)
                cost += abs(new_w) * (long_c if new_w > 0 else short_c)
        turnover = 0.5 * float(dw.abs().sum())

        r = 0.0
        if nxt in close.index and dt in close.index:
            simple = close.loc[nxt] / close.loc[dt] - 1.0
            for s, wi in applied.items():
                if s in simple.index and np.isfinite(simple[s]):
                    r += float(wi) * float(simple[s])
        net = r - cost
        daily.append(net)
        to_list.append(turnover)
        eq_dates.append(nxt)
        n_long_s.append(int((applied > 0).sum()) if len(applied) else 0)
        n_short_s.append(int((applied < 0).sum()) if len(applied) else 0)
        n_sh_s.append(int(n_sh))
        inc_s.append(1.0 if incomplete else 0.0)
        lg = float(applied[applied > 0].sum()) if len(applied) else 0.0
        sg = float((-applied[applied < 0]).sum()) if len(applied) else 0.0
        lg_s.append(lg)
        sg_s.append(sg)
        cash_s.append(max(0.0, 1.0 - lg - sg))
        prev_full = applied
        if i % 60 == 0:
            print(
                f"[HB] LS h={h} beta_matched={int(beta_matched)} i={i}/{len(oos_dates)} "
                f"dt={dt.date()} nL={n_long_s[-1]} nS={n_short_s[-1]} sh={n_sh} "
                f"inc={int(incomplete)} net={net:.5f}",
                flush=True,
            )

    rets = pd.Series(daily, index=pd.DatetimeIndex(eq_dates), dtype=float)
    stats = {
        "to_list": to_list,
        "n_long": pd.Series(n_long_s, index=rets.index),
        "n_short": pd.Series(n_short_s, index=rets.index),
        "n_shortable": pd.Series(n_sh_s, index=rets.index),
        "incomplete": pd.Series(inc_s, index=rets.index),
        "long_gross": pd.Series(lg_s, index=rets.index),
        "short_gross": pd.Series(sg_s, index=rets.index),
        "cash": pd.Series(cash_s, index=rets.index),
    }
    packed = summarize_ls(rets, stats)
    packed.update(
        {
            "horizon": int(h),
            "beta_matched": bool(beta_matched),
            "btc_id": int(btc_id),
            "btc_in_book_hits": int(btc_hits),
            "forced_exits": {
                "n_events": len(forced_exits),
                "n_ids": len({i for e in forced_exits for i in e["ids"]}),
                "ids": sorted({i for e in forced_exits for i in e["ids"]})[:40],
                "symbols": [id_to_sym.get(i) for i in sorted({i for e in forced_exits for i in e["ids"]})[:40]],
            },
            "forced_covers": {
                "n_events": len(forced_covers),
                "n_ids": len({i for e in forced_covers for i in e["ids"]}),
                "ids": sorted({i for e in forced_covers for i in e["ids"]})[:40],
                "symbols": [id_to_sym.get(i) for i in sorted({i for e in forced_covers for i in e["ids"]})[:40]],
            },
            "id_to_sym": {int(k): v for k, v in id_to_sym.items()},
        }
    )
    return packed


def squeeze_table(ls_rets: pd.Series, basket: pd.Series, n: int = LS_SQUEEZE_N) -> list[dict]:
    a, b = ls_rets.align(basket, join="inner")
    if b.empty:
        return []
    top = b.nlargest(int(n))
    rows = []
    for dt, br in top.items():
        rows.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "ew_top100": float(br),
                "spread_ls": float(a.loc[dt]) if dt in a.index else float("nan"),
            }
        )
    return rows


def combo_overlap_stats(ls_rets: pd.Series, combo_rets: pd.Series, start: str = COMBO_OVERLAP_START) -> dict:
    t0 = _as_utc(start)
    a = ls_rets.copy()
    b = combo_rets.copy()
    a.index = _utc_idx(a.index)
    b.index = _utc_idx(b.index)
    a = a[a.index >= t0]
    b = b[b.index >= t0]
    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)
    corr = float(a.corr(b)) if len(a) > 5 else float("nan")
    return {
        "start": str(idx.min().date()) if len(idx) else None,
        "end": str(idx.max().date()) if len(idx) else None,
        "n_days": int(len(idx)),
        "ls_sharpe": _sharpe(a),
        "combo_sharpe": _sharpe(b),
        "corr": corr,
        "ls_total": float((1.0 + a).cumprod().iloc[-1] - 1.0) if len(a) else float("nan"),
        "combo_total": float((1.0 + b).cumprod().iloc[-1] - 1.0) if len(b) else float("nan"),
        "ls_daily": a,
        "combo_daily": b,
    }


def attach_beta(packed: dict, btc_simple: pd.Series) -> dict:
    rets = packed["daily_ret"]
    btc = btc_simple.reindex(rets.index)
    packed["realized_beta_full"] = ols_beta(rets, btc)
    packed["realized_beta_90d"] = rolling_ols_beta(rets, btc)
    packed["btc_ret"] = btc
    return packed
