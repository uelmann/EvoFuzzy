"""Phase D — microstructure Vision downloads + panels."""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from baseline.data import VISION_FILE, VISION_LIST, month_range

MICRO_FEATURE_COLS = [
    "funding_now",
    "funding_z_30",
    "funding_cum_7",
    "funding_cs_rank",
    "basis_z_30",
    "oi_chg_1",
    "oi_chg_7",
    "oi_turnover",
    "liq_imb_1",
    "liq_imb_7",
    "taker_imb_z",
    "ls_ratio_z",
]
MICRO_LIQ_NAN_COLS = ["liq_imb_1", "liq_imb_7"]
MICRO_FEATURE_COLS_10 = [c for c in MICRO_FEATURE_COLS if c not in MICRO_LIQ_NAN_COLS]


def _log(msg: str) -> None:
    print(f"[micro {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_zip_bytes(url: str, timeout: float = 60.0) -> bytes | None:
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def download_premium_symbol_months(symbol: str, months: list[str], dest_dir: Path) -> Path:
    """Monthly premiumIndexKlines 1d → daily parquet."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{symbol}.parquet"
    if out.exists():
        return out
    frames = []
    for ym in months:
        url = (
            f"{VISION_FILE}/data/futures/um/monthly/premiumIndexKlines/{symbol}/1d/"
            f"{symbol}-1d-{ym}.zip"
        )
        blob = _get_zip_bytes(url)
        if blob is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                df = pd.read_csv(zf.open(name))
            cols = {c.lower(): c for c in df.columns}
            tcol = cols.get("open_time") or df.columns[0]
            ccol = cols.get("close") or df.columns[4]
            tmp = pd.DataFrame(
                {
                    "open_time": pd.to_numeric(df[tcol], errors="coerce"),
                    "premium_close": pd.to_numeric(df[ccol], errors="coerce"),
                }
            )
            frames.append(tmp)
        except Exception as e:
            _log(f"premium {symbol} {ym} skip: {e}")
    if not frames:
        empty = pd.DataFrame(columns=["date", "symbol", "premium_close"])
        empty.to_parquet(out, index=False)
        return out
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["open_time"], unit="ms", utc=True).dt.floor("D")
    all_df = all_df.groupby("date", as_index=False)["premium_close"].last()
    all_df["symbol"] = symbol
    all_df.to_parquet(out, index=False)
    return out


def _date_range_days(start: str, end: datetime | None = None) -> list[str]:
    end = end or datetime.utcnow()
    d0 = datetime.strptime(start, "%Y-%m-%d")
    out = []
    cur = d0
    while cur.date() <= end.date():
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def download_metrics_symbol_days(
    symbol: str,
    start_day: str,
    dest_dir: Path,
    end: datetime | None = None,
) -> Path:
    """
    Daily metrics zips → one row per UTC day (last 5m snapshot).
    Cached per symbol; incremental append if file exists.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{symbol}.parquet"
    have_dates: set[str] = set()
    existing = pd.DataFrame()
    if out.exists():
        existing = pd.read_parquet(out)
        if not existing.empty and "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"], utc=True)
            have_dates = set(existing["date"].dt.strftime("%Y-%m-%d"))

    days = _date_range_days(start_day, end=end)
    todo = [d for d in days if d not in have_dates]
    rows = []
    t0 = time.time()
    for i, day in enumerate(todo):
        url = (
            f"{VISION_FILE}/data/futures/um/daily/metrics/{symbol}/"
            f"{symbol}-metrics-{day}.zip"
        )
        blob = _get_zip_bytes(url, timeout=30.0)
        if blob is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                df = pd.read_csv(zf.open(name))
            # last snapshot of the day
            last = df.iloc[-1]
            rows.append(
                {
                    "date": pd.Timestamp(day, tz="UTC"),
                    "symbol": symbol,
                    "sum_open_interest": float(pd.to_numeric(last.get("sum_open_interest"), errors="coerce")),
                    "sum_open_interest_value": float(
                        pd.to_numeric(last.get("sum_open_interest_value"), errors="coerce")
                    ),
                    "count_long_short_ratio": float(
                        pd.to_numeric(last.get("count_long_short_ratio"), errors="coerce")
                    ),
                    "sum_taker_long_short_vol_ratio": float(
                        pd.to_numeric(last.get("sum_taker_long_short_vol_ratio"), errors="coerce")
                    ),
                    "sum_toptrader_long_short_ratio": float(
                        pd.to_numeric(last.get("sum_toptrader_long_short_ratio"), errors="coerce")
                    ),
                }
            )
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            _log(f"metrics {symbol} {i+1}/{len(todo)} elapsed={time.time()-t0:.0f}s")
    if rows:
        new = pd.DataFrame(rows)
        if not existing.empty:
            all_df = pd.concat([existing, new], ignore_index=True)
        else:
            all_df = new
        all_df["date"] = pd.to_datetime(all_df["date"], utc=True)
        all_df = all_df.drop_duplicates(["date", "symbol"], keep="last").sort_values("date")
        all_df.to_parquet(out, index=False)
    elif not out.exists():
        empty = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "sum_open_interest",
                "sum_open_interest_value",
                "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
                "sum_toptrader_long_short_ratio",
            ]
        )
        empty.to_parquet(out, index=False)
    return out


def liquidation_availability_note() -> dict:
    """UM liquidationSnapshot is not published on data.binance.vision (only CM)."""
    return {
        "available_on_vision_um": False,
        "note": (
            "data/futures/um/*/liquidationSnapshot/ is empty on data.binance.vision; "
            "CM liquidations are not used. liq_imb_1/liq_imb_7 remain NaN."
        ),
        "coverage_pct": 0.0,
    }


def load_symbol_parquets(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
    parts = []
    for s in symbols:
        p = raw_dir / f"{s}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def coverage_report(df: pd.DataFrame, symbols: list[str], field: str) -> dict:
    if df.empty or field not in df.columns:
        return {
            "field": field,
            "n_symbols_with_data": 0,
            "coverage_pct_symbol": 0.0,
            "min_date": None,
            "max_date": None,
            "n_rows": 0,
        }
    sub = df.dropna(subset=[field])
    have = set(sub["symbol"].unique()) if not sub.empty else set()
    return {
        "field": field,
        "n_symbols_with_data": int(len(have)),
        "n_symbols_requested": int(len(symbols)),
        "coverage_pct_symbol": float(len(have) / max(len(symbols), 1)),
        "min_date": str(sub["date"].min().date()) if not sub.empty else None,
        "max_date": str(sub["date"].max().date()) if not sub.empty else None,
        "n_rows": int(len(sub)),
    }
