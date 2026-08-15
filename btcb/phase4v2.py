"""Phase 4 v2 — RANK head helpers, positioning/price blocks, tail ablation.

BACKTEST / ANALYSIS ONLY. Nothing adopted.
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from baseline.data import VISION_FILE
from baseline.evaluate import newey_west_t
from btcb.binance_replay import build_id_symbol_map
from btcb.constants import (
    CS_CLIP,
    FUTURE_NULL_BIAS_MIN_VIOLATIONS,
    LS_TRAIL_DAYS,
    NULL_REPLICATES,
    NULL_SHUFFLE_SEEDS,
    ORACLE_LADDER2_MONSTER_K,
    ORACLE_LADDER_DECILE,
    ORACLE_LADDER_MIN_N,
    PHASE2_CYCLES,
    PHASE3C_NAME_TIERS,
    PHASE4V2_H,
    PHASE4V2_NW_LAG,
    PHASE4V2_OVERLAP_DELTA,
    PHASE4V2_OVERLAP_NULL_CENTER,
    PHASE4V2_PERP_COV_FROM,
    PHASE4V2_PERP_COV_MIN,
    PHASE4V2_TAIL_IC_DELTA,
    POSITIONING_COLS,
    PRICE_ADD_COLS,
    SEED,
    STOUFFER_Z_MIN,
)
from btcb.features import apply_cs_zscore
from btcb.gates import _cell_stats, metric_verdict_e1b_house
from btcb.model import FoldSpec, fit_predict_rank_fold
from btcb.oracle_ladder import _as_utc, _spearman
from btcb.oracle_ladder2 import _decile_ids, _half_ic


def _log(msg: str) -> None:
    print(f"[p4v2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.normalize()


def _own_z(s: pd.Series, window: int, min_p: int | None = None) -> pd.Series:
    mp = int(min_p if min_p is not None else max(5, window // 3))
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def rolling_ols_intercept(y: pd.Series, x: pd.Series, window: int = 60, min_p: int = 20) -> pd.Series:
    cov = y.rolling(window, min_periods=min_p).cov(x)
    var = x.rolling(window, min_periods=min_p).var()
    beta = cov / var.replace(0, np.nan)
    mu_y = y.rolling(window, min_periods=min_p).mean()
    mu_x = x.rolling(window, min_periods=min_p).mean()
    return mu_y - beta * mu_x


def trend_composite_close(close: pd.Series) -> pd.Series:
    signs = []
    for k, mp in ((20, 10), (50, 20), (100, 40), (200, 80)):
        sma = close.rolling(k, min_periods=mp).mean()
        signs.append(np.sign(close / sma.replace(0, np.nan) - 1.0))
    stacked = pd.concat(signs, axis=1)
    return stacked.mean(axis=1)


def build_price_additions(panel: pd.DataFrame, btc_id: int, ids: list[int]) -> pd.DataFrame:
    p = panel.copy()
    p["date"] = _utc(p["date"])
    p["id"] = p["id"].astype(int)
    btc = p.loc[p["id"] == int(btc_id), ["date", "close"]].drop_duplicates("date").sort_values("date")
    btc_r = np.log(btc.set_index("date")["close"].astype(float) / btc.set_index("date")["close"].astype(float).shift(1))
    keep = set(int(i) for i in ids) | {int(btc_id)}
    parts = []
    n = len(keep)
    for i, iid in enumerate(sorted(keep), 1):
        g = p.loc[p["id"] == int(iid), ["date", "close"]].drop_duplicates("date").sort_values("date")
        if g.empty:
            continue
        s = g.set_index("date")["close"].astype(float)
        r1 = np.log(s / s.shift(1))
        br = btc_r.reindex(s.index)
        out = pd.DataFrame(index=s.index)
        out["past_alpha_60"] = rolling_ols_intercept(r1, br, 60, 20)
        out["trend_composite"] = trend_composite_close(s)
        out["id"] = int(iid)
        parts.append(out.reset_index())
        if i % 200 == 0 or i == n:
            _log(f"price-additions {i}/{n}")
    if not parts:
        return pd.DataFrame(columns=["date", "id", *PRICE_ADD_COLS])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = _utc(df["date"])
    df["id"] = df["id"].astype(int)
    return df


def _parse_metrics_zip(blob: bytes, day: str, symbol: str) -> dict | None:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(zf.open(name))
        last = df.iloc[-1]
        return {
            "date": pd.Timestamp(day, tz="UTC"),
            "symbol": symbol,
            "sum_open_interest": float(pd.to_numeric(last.get("sum_open_interest"), errors="coerce")),
            "sum_open_interest_value": float(pd.to_numeric(last.get("sum_open_interest_value"), errors="coerce")),
        }
    except Exception:
        return None


def fill_metrics_gaps(
    metrics_dir: Path,
    symbols: list[str],
    start_day: str = "2020-09-01",
    end: datetime | None = None,
) -> list[dict]:
    """Download missing daily metrics zips from Binance Vision. Log every new file."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    end = end or datetime.utcnow()
    d0 = datetime.strptime(start_day, "%Y-%m-%d")
    days = []
    cur = d0
    while cur.date() <= end.date():
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    log = []
    for i, symbol in enumerate(symbols, 1):
        out = metrics_dir / f"{symbol}.parquet"
        existing = pd.DataFrame()
        have: set[str] = set()
        if out.exists():
            existing = pd.read_parquet(out)
            if not existing.empty and "date" in existing.columns:
                existing["date"] = pd.to_datetime(existing["date"], utc=True)
                have = set(existing["date"].dt.strftime("%Y-%m-%d"))
        last = max(have) if have else None
        # skip symbols already fresh enough (avoid day-by-day 404 probes)
        if last is not None and last >= (end - timedelta(days=5)).strftime("%Y-%m-%d"):
            continue
        todo = [d for d in days if d not in have]
        if not todo:
            continue
        rows = []
        n404 = 0
        for day in todo:
            url = (
                f"{VISION_FILE}/data/futures/um/daily/metrics/{symbol}/"
                f"{symbol}-metrics-{day}.zip"
            )
            try:
                r = httpx.get(url, timeout=30.0, follow_redirects=True)
                if r.status_code == 404:
                    n404 += 1
                    continue
                r.raise_for_status()
                rec = _parse_metrics_zip(r.content, day, symbol)
                if rec is not None:
                    rows.append(rec)
            except Exception:
                continue
        if rows:
            new = pd.DataFrame(rows)
            all_df = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
            all_df["date"] = pd.to_datetime(all_df["date"], utc=True)
            all_df = all_df.drop_duplicates(["date", "symbol"], keep="last").sort_values("date")
            all_df.to_parquet(out, index=False)
        rec = {
            "symbol": symbol,
            "n_new_rows": int(len(rows)),
            "n_todo": int(len(todo)),
            "n_404": int(n404),
            "source": "binance_vision_daily_metrics",
        }
        log.append(rec)
        _log(f"metrics gap {i}/{len(symbols)} {symbol} new={len(rows)} todo={len(todo)}")
    return log


def load_symbol_parquets(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
    parts = []
    for s in symbols:
        p = raw_dir / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df is None or df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper()
    return out


def _taker_from_kline_parquet(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    need_q = cols.get("taker_buy_quote") or cols.get("taker_buy_quote_asset_volume")
    need_v = cols.get("quote_volume") or cols.get("quote_asset_volume")
    if need_q is None or need_v is None:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[cols.get("date", "date")], utc=True),
            "symbol": symbol,
            "quote_volume": pd.to_numeric(df[need_v], errors="coerce"),
            "taker_buy_quote": pd.to_numeric(df[need_q], errors="coerce"),
        }
    )
    return out.dropna(subset=["date"])


def _taker_from_zips(raw_dir: Path, symbol: str) -> pd.DataFrame:
    if not raw_dir.exists():
        return pd.DataFrame()
    frames = []
    for zp in sorted(raw_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zp) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as fh:
                    df = pd.read_csv(fh, header=None)
            if df.shape[1] < 11:
                continue
            if len(df) and str(df.iloc[0, 0]).lower().startswith("open"):
                df = df.iloc[1:]
            tmp = pd.DataFrame(
                {
                    "open_time": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
                    "quote_volume": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
                    "taker_buy_quote": pd.to_numeric(df.iloc[:, 10], errors="coerce"),
                }
            ).dropna(subset=["open_time"])
            ot = tmp["open_time"].to_numpy(dtype="float64")
            ms = np.where(ot > 1e14, ot / 1000.0, ot)
            tmp["date"] = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce").normalize()
            tmp["symbol"] = symbol
            frames.append(tmp[["date", "symbol", "quote_volume", "taker_buy_quote"]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    return out.drop_duplicates(["date"]).sort_values("date")


def load_taker_panel(klines_dir: Path, symbols: list[str], cache_path: Path | None = None) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        have = set(df["symbol"].astype(str).str.upper().unique()) if not df.empty else set()
        missing = [s for s in symbols if s not in have]
        if not missing:
            return df
        _log(f"taker cache missing {len(missing)} symbols; filling")
        extra = load_taker_panel(klines_dir, missing, cache_path=None)
        if extra is not None and not extra.empty:
            df = pd.concat([df, extra], ignore_index=True)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
        return df
    parts = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        pq = klines_dir / f"{sym}.parquet"
        got = pd.DataFrame()
        if pq.exists():
            got = _taker_from_kline_parquet(pq, sym)
        if got.empty:
            got = _taker_from_zips(klines_dir / "raw" / sym, sym)
        if not got.empty:
            parts.append(got)
        if i % 50 == 0 or i == n:
            _log(f"taker {i}/{n} have={len(parts)}")
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "quote_volume", "taker_buy_quote"])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out = out.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def _symbol_feat_frame(dates, **cols) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    for k, v in cols.items():
        out[k] = v
    return out


def build_positioning_by_symbol(
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
    perp_close: pd.DataFrame,
    spot_close: pd.DataFrame,
    taker: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Raw (pre CS-z) positioning features keyed by date, symbol."""
    fund = funding.copy() if funding is not None and not funding.empty else pd.DataFrame()
    met = metrics.copy() if metrics is not None and not metrics.empty else pd.DataFrame()
    pc = perp_close.copy() if perp_close is not None and not perp_close.empty else pd.DataFrame()
    sc = spot_close.copy() if spot_close is not None and not spot_close.empty else pd.DataFrame()
    tk = taker.copy() if taker is not None and not taker.empty else pd.DataFrame()
    for df in (fund, met, pc, sc, tk):
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        if not df.empty and "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()

    first_oi: dict[str, str] = {}
    parts = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        pieces = []
        f_s = fund[fund["symbol"] == sym] if not fund.empty else fund
        m_s = met[met["symbol"] == sym] if not met.empty else met
        p_s = pc[pc["symbol"] == sym] if not pc.empty else pc
        s_s = sc[sc["symbol"] == sym] if not sc.empty else sc
        t_s = tk[tk["symbol"] == sym] if not tk.empty else tk
        for df in (f_s, m_s, p_s, s_s, t_s):
            if df is not None and not df.empty:
                pieces.append(df["date"])
        if not pieces:
            continue
        dates = pd.DatetimeIndex(sorted(pd.concat(pieces).unique())).tz_convert("UTC").normalize()
        out = pd.DataFrame({"date": dates, "symbol": sym})
        if f_s is not None and not f_s.empty:
            f = f_s.sort_values("date")[["date", "funding_rate"]].drop_duplicates("date")
            out = out.merge(f, on="date", how="left")
        else:
            out["funding_rate"] = np.nan
        out["funding_z_7"] = _own_z(out["funding_rate"], 7)
        out["funding_z_30"] = _own_z(out["funding_rate"], 30)
        out["funding_level_3d"] = out["funding_rate"].rolling(3, min_periods=1).mean()

        if m_s is not None and not m_s.empty and "sum_open_interest" in m_s.columns:
            m = m_s.sort_values("date")[["date", "sum_open_interest"]].drop_duplicates("date")
            out = out.merge(m, on="date", how="left")
        else:
            out["sum_open_interest"] = np.nan
        oi = out["sum_open_interest"].astype(float)
        oi_ok_dates = out.loc[np.isfinite(oi) & (oi > 0), "date"]
        if len(oi_ok_dates):
            first_oi[sym] = str(pd.Timestamp(oi_ok_dates.iloc[0]).date())
        log_oi = np.log(oi.replace(0, np.nan).to_numpy(dtype=float))
        log_oi = pd.Series(log_oi, index=out.index)
        out["dOI_7"] = log_oi - log_oi.shift(7)
        out["dOI_30"] = log_oi - log_oi.shift(30)

        if p_s is not None and not p_s.empty:
            p = p_s.sort_values("date")[["date", "close"]].drop_duplicates("date").rename(columns={"close": "perp_close"})
            out = out.merge(p, on="date", how="left")
        else:
            out["perp_close"] = np.nan
        if s_s is not None and not s_s.empty:
            s = s_s.sort_values("date")[["date", "close"]].drop_duplicates("date").rename(columns={"close": "spot_close"})
            out = out.merge(s, on="date", how="left")
        else:
            out["spot_close"] = np.nan
        out["basis"] = out["perp_close"] / out["spot_close"].replace(0, np.nan) - 1.0

        if t_s is not None and not t_s.empty:
            t = t_s.sort_values("date")[["date", "quote_volume", "taker_buy_quote"]].drop_duplicates("date")
            out = out.merge(t, on="date", how="left")
        else:
            out["quote_volume"] = np.nan
            out["taker_buy_quote"] = np.nan
        share = out["taker_buy_quote"] / out["quote_volume"].replace(0, np.nan)
        mu7 = share.rolling(7, min_periods=3).mean()
        mu30 = share.rolling(30, min_periods=10).mean()
        sd30 = share.rolling(30, min_periods=10).std(ddof=0)
        out["taker_imbalance_7"] = (mu7 - mu30) / sd30.replace(0, np.nan)

        keep = ["date", "symbol"] + [c for c in POSITIONING_COLS if c != "pos_missing"]
        parts.append(out[keep])
        if i % 40 == 0 or i == n:
            _log(f"positioning symbols {i}/{n}")

    if not parts:
        empty = pd.DataFrame(columns=["date", "symbol"] + [c for c in POSITIONING_COLS if c != "pos_missing"])
        return empty, {"first_oi_date": first_oi}
    raw = pd.concat(parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    return raw, {"first_oi_date": first_oi}


def map_positioning_to_ids(
    raw_sym: pd.DataFrame,
    feat: pd.DataFrame,
    id_to_perp: dict[int, str | None],
    clip: float = CS_CLIP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join symbol features onto (date, id); non-perp → 0 + pos_missing=1. CS-z continuous cols."""
    key = feat[["date", "id"]].copy()
    key["date"] = _utc(key["date"])
    key["id"] = key["id"].astype(int)
    key["perp_symbol"] = key["id"].map(lambda i: id_to_perp.get(int(i)))
    key["perp_mapped"] = key["perp_symbol"].notna() & (key["perp_symbol"].astype(str) != "") & (key["perp_symbol"].astype(str) != "None")
    raw = raw_sym.copy()
    if not raw.empty:
        raw["date"] = _utc(raw["date"])
        raw["symbol"] = raw["symbol"].astype(str).str.upper()
        key = key.merge(raw.rename(columns={"symbol": "perp_symbol"}), on=["date", "perp_symbol"], how="left")
    else:
        for c in POSITIONING_COLS:
            if c != "pos_missing":
                key[c] = np.nan

    cov = pd.DataFrame(
        {
            "date": key["date"],
            "id": key["id"],
            "perp_mapped": key["perp_mapped"].astype(bool),
            "has_funding": key["funding_z_30"].notna() if "funding_z_30" in key.columns else False,
            "has_oi": key["dOI_7"].notna() if "dOI_7" in key.columns else False,
            "has_basis": key["basis"].notna() if "basis" in key.columns else False,
            "has_taker": key["taker_imbalance_7"].notna() if "taker_imbalance_7" in key.columns else False,
        }
    )
    key["pos_missing"] = (~key["perp_mapped"]).astype(float)
    zcols = [c for c in POSITIONING_COLS if c != "pos_missing"]
    for c in zcols:
        if c not in key.columns:
            key[c] = np.nan
        key.loc[~key["perp_mapped"], c] = np.nan
    mapped = key["perp_mapped"]
    if bool(mapped.any()):
        z_part = apply_cs_zscore(key.loc[mapped].copy(), zcols, clip=clip)
        key.loc[mapped, zcols] = z_part[zcols]
    for c in zcols:
        key[c] = key[c].where(key["perp_mapped"], 0.0).fillna(0.0)
    out = key[["date", "id"] + list(POSITIONING_COLS)].copy()
    return out, cov


def join_price_additions(feat: pd.DataFrame, price_add: pd.DataFrame, clip: float = CS_CLIP) -> pd.DataFrame:
    key = feat[["date", "id"]].copy()
    key["date"] = _utc(key["date"])
    key["id"] = key["id"].astype(int)
    pa = price_add.copy()
    pa["date"] = _utc(pa["date"])
    pa["id"] = pa["id"].astype(int)
    key = key.merge(pa[["date", "id", *PRICE_ADD_COLS]], on=["date", "id"], how="left")
    key = apply_cs_zscore(key, list(PRICE_ADD_COLS), clip=clip)
    for c in PRICE_ADD_COLS:
        key[c] = key[c].fillna(0.0)
    return key[["date", "id", *PRICE_ADD_COLS]]


def coverage_tables(cov: pd.DataFrame, pit: pd.DataFrame) -> dict:
    c = cov.copy()
    c["date"] = _utc(c["date"])
    c["id"] = c["id"].astype(int)
    c["year"] = c["date"].dt.year.astype(int)
    p = pit.copy()
    p["date"] = _utc(p["date"])
    p["id"] = p["id"].astype(int)
    if "rank" not in p.columns:
        p["rank"] = p.groupby("date").cumcount() + 1
    c = c.merge(p[["date", "id", "rank"]], on=["date", "id"], how="left")

    def _tier(r):
        if not np.isfinite(r):
            return "unknown"
        r = int(r)
        for a, b, name in PHASE3C_NAME_TIERS:
            if a <= r <= b:
                return name
        return "unknown"

    c["tier"] = c["rank"].map(_tier)
    from_ts = pd.Timestamp(PHASE4V2_PERP_COV_FROM, tz="UTC")
    post = c[c["date"] >= from_ts]
    frac_perp_2021 = float(post["perp_mapped"].mean()) if len(post) else float("nan")

    def _agg(df: pd.DataFrame) -> dict:
        n = int(len(df))
        if n == 0:
            return {"n": 0, "perp": float("nan"), "funding": float("nan"), "oi": float("nan"), "basis": float("nan"), "taker": float("nan")}
        return {
            "n": n,
            "perp": float(df["perp_mapped"].mean()),
            "funding": float(df["has_funding"].mean()),
            "oi": float(df["has_oi"].mean()),
            "basis": float(df["has_basis"].mean()),
            "taker": float(df["has_taker"].mean()),
        }

    by_year = {str(y): _agg(g) for y, g in c.groupby("year")}
    by_tier = {str(t): _agg(g) for t, g in c.groupby("tier")}
    by_year_tier = {}
    for (y, t), g in c.groupby(["year", "tier"]):
        by_year_tier[f"{y}|{t}"] = _agg(g)
    return {
        "perp_coverage_top100_from_2021": frac_perp_2021,
        "n_name_days_from_2021": int(len(post)),
        "n_name_days": int(len(c)),
        "by_year": by_year,
        "by_tier": by_tier,
        "by_year_tier": by_year_tier,
        "live_coverage_ok": bool(np.isfinite(frac_perp_2021) and frac_perp_2021 >= float(PHASE4V2_PERP_COV_MIN)),
    }


def _cycle_pack(idx: pd.DatetimeIndex, values: np.ndarray) -> dict:
    ser = pd.Series(values, index=pd.DatetimeIndex(idx), dtype=float)
    out = {}
    for name, a, b in PHASE2_CYCLES:
        t0, t1 = _as_utc(a), _as_utc(b)
        sl = ser[(ser.index >= t0) & (ser.index <= t1)].dropna()
        out[name] = {"n": int(len(sl)), "mean": float(sl.mean()) if len(sl) else float("nan")}
    return out


def _window_stats(dates, values, lag: int = PHASE4V2_NW_LAG) -> dict:
    idx = pd.DatetimeIndex([_as_utc(d) for d in dates])
    ser = pd.Series(np.asarray(values, dtype=float), index=idx).replace([np.inf, -np.inf], np.nan).dropna()
    trail_cut = ser.index.max() - pd.Timedelta(days=int(LS_TRAIL_DAYS)) if len(ser) else None
    trail = ser[ser.index >= trail_cut] if trail_cut is not None else ser.iloc[0:0]

    def _one(x: pd.Series) -> dict:
        a = x.dropna()
        return {
            "n": int(len(a)),
            "mean": float(a.mean()) if len(a) else float("nan"),
            "nw_t": float(newey_west_t(a.to_numpy(), lag=int(lag))) if len(a) else float("nan"),
        }

    return {
        "full": _one(ser),
        "trail18m": _one(trail),
        "cycles": _cycle_pack(ser.index, ser.to_numpy()),
    }


def per_date_tail_metrics(
    df: pd.DataFrame,
    score_col: str,
    excess_col: str = f"excess_h{PHASE4V2_H}",
    min_n: int = ORACLE_LADDER_MIN_N,
    decile: int = ORACLE_LADDER_DECILE,
    monster_k: int = ORACLE_LADDER2_MONSTER_K,
) -> dict:
    """Per-date tail-IC / overlap / monster / RankIC on a labeled cross-section."""
    d = df[["date", "id", score_col, excess_col]].copy()
    d["date"] = _utc(d["date"])
    d["id"] = d["id"].astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d[excess_col] = pd.to_numeric(d[excess_col], errors="coerce")
    d = d.dropna(subset=[score_col, excess_col])
    dates, top_ics, bot_ics, overlaps, monsters, full_ics = [], [], [], [], [], []
    for dt, g in d.groupby("date", sort=True):
        sc = pd.Series(g.set_index("id")[score_col], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        ex = pd.Series(g.set_index("id")[excess_col], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        sc, ex = sc.align(ex, join="inner")
        if len(sc) < int(min_n):
            continue
        sig = set(_decile_ids(sc, decile=decile))
        k_real = max(1, len(ex) // int(decile))
        real = set(int(i) for i in ex.nlargest(k_real).index.tolist())
        k_m = min(int(monster_k), len(ex))
        monsters_set = set(int(i) for i in ex.nlargest(k_m).index.tolist())
        dates.append(_as_utc(dt))
        overlaps.append(float(len(sig & real) / max(len(sig), 1)))
        monsters.append(float(len(sig & monsters_set) / max(len(monsters_set), 1)))
        top_ics.append(_half_ic(sc, ex, "top"))
        bot_ics.append(_half_ic(sc, ex, "bottom"))
        full_ics.append(_spearman(sc.to_numpy(), ex.to_numpy()))

    def _pack(xs, key):
        blob = _window_stats(dates, xs)
        return {
            key: blob["full"]["mean"],
            f"{key}_nw_t": blob["full"]["nw_t"],
            f"{key}_n": blob["full"]["n"],
            f"{key}_trail": blob["trail18m"]["mean"],
            f"{key}_trail_nw_t": blob["trail18m"]["nw_t"],
            f"{key}_cycles": blob["cycles"],
        }

    out = {
        "n_dates": int(len(dates)),
        "label": score_col,
    }
    out.update(_pack(top_ics, "tail_ic_top"))
    out.update(_pack(bot_ics, "tail_ic_bot"))
    out.update(_pack(overlaps, "overlap"))
    out.update(_pack(monsters, "monster"))
    out.update(_pack(full_ics, "rankic"))
    out["bottom_minus_top"] = (
        float(out["tail_ic_bot"] - out["tail_ic_top"])
        if np.isfinite(out.get("tail_ic_bot", np.nan)) and np.isfinite(out.get("tail_ic_top", np.nan))
        else float("nan")
    )
    return out


def listed_id_set(close: pd.DataFrame, dt) -> set[int]:
    dt = _as_utc(dt)
    if dt not in close.index:
        return set()
    row = close.loc[dt]
    return set(int(i) for i, v in row.items() if np.isfinite(float(v)) and float(v) > 0)


def restrict_eval_frame(
    preds: pd.DataFrame,
    labeled: pd.DataFrame,
    close: pd.DataFrame,
    btc_id: int,
    score_col: str,
    excess_col: str = f"excess_h{PHASE4V2_H}",
) -> pd.DataFrame:
    pr = preds[["date", "id", score_col]].copy()
    pr["date"] = _utc(pr["date"])
    pr["id"] = pr["id"].astype(int)
    lab = labeled[["date", "id", excess_col]].copy()
    lab["date"] = _utc(lab["date"])
    lab["id"] = lab["id"].astype(int)
    m = pr.merge(lab, on=["date", "id"], how="inner")
    m = m[m["id"] != int(btc_id)]
    if close is None or close.empty:
        return m
    keep_rows = []
    for dt, g in m.groupby("date", sort=True):
        listed = listed_id_set(close, dt)
        if not listed:
            continue
        keep_rows.append(g[g["id"].isin(listed)])
    if not keep_rows:
        return m.iloc[0:0]
    return pd.concat(keep_rows, ignore_index=True)


def cs_rank_blend(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str, out_col: str = "blend") -> pd.DataFrame:
    x = a[["date", "id", col_a]].copy()
    y = b[["date", "id", col_b]].copy()
    x["date"] = _utc(x["date"])
    y["date"] = _utc(y["date"])
    x["id"] = x["id"].astype(int)
    y["id"] = y["id"].astype(int)
    m = x.merge(y, on=["date", "id"], how="inner")
    parts = []
    for _, g in m.groupby("date", sort=False):
        g = g.copy()
        g[out_col] = 0.5 * (
            g[col_a].rank(method="average", pct=True) + g[col_b].rank(method="average", pct=True)
        )
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else m
    return out[["date", "id", out_col]]


def preds_to_score_at(preds: pd.DataFrame, col: str, times: list) -> dict:
    pr = preds.copy()
    pr["date"] = _utc(pr["date"])
    pr["id"] = pr["id"].astype(int)
    wide = pr.pivot_table(index="date", columns="id", values=col, aggfunc="last").sort_index()
    out = {}
    for t in times:
        t = _as_utc(t)
        sl = wide.loc[wide.index <= t]
        if sl.empty:
            continue
        row = sl.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
        if len(row):
            out[t] = row
    return out


def collapse_fold_preds(preds: pd.DataFrame, score_col: str = "p") -> pd.DataFrame:
    pr = preds.copy()
    pr["date"] = _utc(pr["date"])
    pr["id"] = pr["id"].astype(int)
    if "fold_id" not in pr.columns:
        pr["fold_id"] = 0
    pr = pr.sort_values(["date", "id", "fold_id"]).drop_duplicates(["date", "id"], keep="last")
    return pr


def fold_tail_from_pred(pred: pd.DataFrame, labeled: pd.DataFrame, close, btc_id: int, score_col: str = "p") -> dict:
    if pred is None or pred.empty:
        return {"tail_ic_top": float("nan"), "overlap": float("nan"), "rankic": float("nan")}
    p = pred.copy()
    if score_col not in p.columns and "p" in p.columns:
        p[score_col] = p["p"]
    ev = restrict_eval_frame(p, labeled, close, btc_id, score_col)
    if ev.empty:
        return {"tail_ic_top": float("nan"), "overlap": float("nan"), "rankic": float("nan")}
    met = per_date_tail_metrics(ev, score_col)
    return {
        "tail_ic_top": met.get("tail_ic_top"),
        "overlap": met.get("overlap"),
        "rankic": met.get("rankic"),
        "n_dates": met.get("n_dates"),
    }


def gate_rank_tail_null(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    real_tail: dict[int, float],
    real_overlap: dict[int, float],
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    feature_cols: list[str],
    n_replicates: int = NULL_REPLICATES,
    seeds: tuple[int, ...] = NULL_SHUFFLE_SEEDS,
    ycol: str = f"y_rank_h{PHASE4V2_H}",
    cache_dir: Path | None = None,
    commit_fn=None,
) -> dict:
    import json

    from btcb.constants import NULL_K_EXCEED as KEX

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cells_ic, cells_ov = [], []
    use_seeds = list(seeds)[: int(n_replicates)]
    for fold in folds:
        ics, ovs = [], []
        for i, ss in enumerate(use_seeds):
            cached = None
            if cache_dir is not None:
                cp = cache_dir / f"fold{fold.fold_id}_seed{ss}.json"
                if cp.exists():
                    cached = json.loads(cp.read_text())
            if cached is not None:
                ics.append(cached.get("tail_ic_top"))
                ovs.append(cached.get("overlap"))
                continue
            _log(f"null fold={fold.fold_id} rep={i+1}/{len(use_seeds)} seed={ss}")
            pred, meta = fit_predict_rank_fold(
                df,
                fold,
                seed=SEED,
                shuffle_labels=True,
                shuffle_seed=int(ss),
                feature_cols=feature_cols,
                ycol=ycol,
            )
            if pred.empty or meta.get("status") != "ok":
                ics.append(float("nan"))
                ovs.append(float("nan"))
                rec = {"tail_ic_top": None, "overlap": None, "status": meta.get("status")}
            else:
                sm = fold_tail_from_pred(pred, labeled, close, btc_id, "p")
                ics.append(sm.get("tail_ic_top"))
                ovs.append(sm.get("overlap"))
                rec = {
                    "tail_ic_top": sm.get("tail_ic_top"),
                    "overlap": sm.get("overlap"),
                    "status": "ok",
                }
            if cache_dir is not None:
                (cache_dir / f"fold{fold.fold_id}_seed{ss}.json").write_text(json.dumps(rec, default=str))
                if commit_fn is not None and (i + 1) % 5 == 0:
                    commit_fn()
        st_i = _cell_stats(ics, center=0.0)
        real_i = float(real_tail.get(fold.fold_id, float("nan")))
        st_i.update(
            {
                "fold_id": fold.fold_id,
                "horizon": fold.horizon,
                "real_tail_ic_top": real_i,
                "exceeds_p95": bool(np.isfinite(real_i) and np.isfinite(st_i["p95"]) and real_i > st_i["p95"]),
            }
        )
        cells_ic.append(st_i)
        st_o = _cell_stats(ovs, center=float(PHASE4V2_OVERLAP_NULL_CENTER))
        real_o = float(real_overlap.get(fold.fold_id, float("nan")))
        st_o.update(
            {
                "fold_id": fold.fold_id,
                "horizon": fold.horizon,
                "real_overlap": real_o,
                "exceeds_p95": bool(np.isfinite(real_o) and np.isfinite(st_o["p95"]) and real_o > st_o["p95"]),
            }
        )
        cells_ov.append(st_o)
        _log(
            f"null fold={fold.fold_id} tailIC mean={st_i['mean']:.4f} p95={st_i['p95']:.4f} "
            f"real={real_i:.4f} bias_ok={st_i['bias_ok']} | overlap mean={st_o['mean']:.4f} real={real_o:.4f}"
        )
    ic_v = metric_verdict_e1b_house(cells_ic, "real_tail_ic_top", KEX, STOUFFER_Z_MIN)
    ov_v = metric_verdict_e1b_house(cells_ov, "real_overlap", KEX, STOUFFER_Z_MIN)
    return {
        "name": "rank_tail_null",
        "passed": bool(ic_v["passed"]),
        "judged": "tail_ic_top",
        "bias_min_violations": int(FUTURE_NULL_BIAS_MIN_VIOLATIONS),
        "tail_ic_top": {k: v for k, v in ic_v.items() if k != "cells"},
        "overlap": {k: v for k, v in ov_v.items() if k != "cells"},
        "tail_ic_cells": cells_ic,
        "overlap_cells": cells_ov,
        "n_replicates": int(n_replicates),
        "fold_ids": [int(f.fold_id) for f in folds],
    }


def mechanical_verdicts(grid: dict, coverage: dict, null: dict) -> dict:
    base = grid["frozen_spread"]
    rank = grid["rank"]
    blend = grid["spread_rank"]
    pos = grid["spread_pos"]
    price = grid["spread_pos_price"]

    def _pick_a():
        a_ic, b_ic = rank.get("tail_ic_top"), blend.get("tail_ic_top")
        if not np.isfinite(a_ic) and np.isfinite(b_ic):
            return "spread_rank", blend
        if np.isfinite(a_ic) and not np.isfinite(b_ic):
            return "rank", rank
        if not (np.isfinite(a_ic) and np.isfinite(b_ic)):
            return "rank", rank
        if float(b_ic) > float(a_ic):
            return "spread_rank", blend
        if float(b_ic) < float(a_ic):
            return "rank", rank
        a_ov, b_ov = rank.get("overlap"), blend.get("overlap")
        if np.isfinite(b_ov) and np.isfinite(a_ov) and float(b_ov) > float(a_ov):
            return "spread_rank", blend
        return "rank", rank

    best_name, best_a = _pick_a()
    d_ic = (
        float(best_a["tail_ic_top"]) - float(base["tail_ic_top"])
        if np.isfinite(best_a.get("tail_ic_top", np.nan)) and np.isfinite(base.get("tail_ic_top", np.nan))
        else float("nan")
    )
    d_ov = (
        float(best_a["overlap"]) - float(base["overlap"])
        if np.isfinite(best_a.get("overlap", np.nan)) and np.isfinite(base.get("overlap", np.nan))
        else float("nan")
    )
    null_pass = bool((null or {}).get("passed"))
    tail_extracts = bool(
        null_pass
        and np.isfinite(d_ic)
        and np.isfinite(d_ov)
        and d_ic >= float(PHASE4V2_TAIL_IC_DELTA)
        and d_ov >= float(PHASE4V2_OVERLAP_DELTA)
    )
    p_ic = (
        float(pos["tail_ic_top"]) - float(best_a["tail_ic_top"])
        if np.isfinite(pos.get("tail_ic_top", np.nan)) and np.isfinite(best_a.get("tail_ic_top", np.nan))
        else float("nan")
    )
    p_ov = (
        float(pos["overlap"]) - float(best_a["overlap"])
        if np.isfinite(pos.get("overlap", np.nan)) and np.isfinite(best_a.get("overlap", np.nan))
        else float("nan")
    )
    cov_ok = bool(coverage.get("live_coverage_ok"))
    positioning_live = bool(
        cov_ok
        and (
            (np.isfinite(p_ic) and p_ic >= float(PHASE4V2_TAIL_IC_DELTA))
            or (np.isfinite(p_ov) and p_ov >= float(PHASE4V2_OVERLAP_DELTA))
        )
    )
    r_ic = (
        float(price["tail_ic_top"]) - float(pos["tail_ic_top"])
        if np.isfinite(price.get("tail_ic_top", np.nan)) and np.isfinite(pos.get("tail_ic_top", np.nan))
        else float("nan")
    )
    r_ov = (
        float(price["overlap"]) - float(pos["overlap"])
        if np.isfinite(price.get("overlap", np.nan)) and np.isfinite(pos.get("overlap", np.nan))
        else float("nan")
    )
    price_live = bool(
        (np.isfinite(r_ic) and r_ic >= float(PHASE4V2_TAIL_IC_DELTA))
        or (np.isfinite(r_ov) and r_ov >= float(PHASE4V2_OVERLAP_DELTA))
    )
    best_tail = max(
        grid.items(),
        key=lambda kv: (float(kv[1].get("tail_ic_top")) if np.isfinite(kv[1].get("tail_ic_top", np.nan)) else -np.inf),
    )
    return {
        "tail_loss": "TAIL-LOSS EXTRACTS" if tail_extracts else "BARREN",
        "positioning": "POSITIONING LIVE" if positioning_live else "POSITIONING NOT LIVE",
        "price_additions": "PRICE-ADDITIONS LIVE" if price_live else "PRICE-ADDITIONS NOT LIVE",
        "best_a": best_name,
        "delta_a_vs_base_tail_ic": d_ic,
        "delta_a_vs_base_overlap": d_ov,
        "delta_pos_vs_best_a_tail_ic": p_ic,
        "delta_pos_vs_best_a_overlap": p_ov,
        "delta_price_vs_pos_tail_ic": r_ic,
        "delta_price_vs_pos_overlap": r_ov,
        "null_pass": null_pass,
        "coverage_ok": cov_ok,
        "perp_coverage_from_2021": coverage.get("perp_coverage_top100_from_2021"),
        "best_tail_signal": best_tail[0],
        "nothing_adopted": True,
    }


def id_symbol_maps(cleaned: pd.DataFrame, ids: list[int], perp_syms: set[str], spot_syms: set[str], btc_id: int):
    id_to_perp = build_id_symbol_map(ids, cleaned, perp_syms, perp_syms)
    id_to_spot = build_id_symbol_map(ids, cleaned, spot_syms, spot_syms)
    id_to_perp[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in perp_syms else id_to_perp.get(int(btc_id))
    id_to_spot[int(btc_id)] = "BTCUSDT" if "BTCUSDT" in spot_syms else id_to_spot.get(int(btc_id))
    return id_to_perp, id_to_spot
