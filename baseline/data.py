"""Binance Vision USDT-M perpetual daily kline download + PIT universe."""

from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import httpx
import numpy as np
import pandas as pd

VISION_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION_FILE = "https://data.binance.vision"


STABLE_OR_WRAP_RE = re.compile(
    r"(USDT|USDC|BUSD|DAI|TUSD|FDUSD|USDP|USDD|EUR|AEUR|WBTC|WETH|STETH|WBETH|BTCB|RENBTC)$",
    re.I,
)


def _log(msg: str) -> None:
    print(f"[data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_um_symbols(quote: str = "USDT") -> list[str]:
    """List perpetual symbols present on data.binance.vision monthly klines."""
    symbols: list[str] = []
    marker = ""
    while True:
        params = {
            "prefix": "data/futures/um/monthly/klines/",
            "delimiter": "/",
            "max-keys": "1000",
        }
        if marker:
            params["marker"] = marker
        r = httpx.get(VISION_LIST, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        # strip namespaces
        for el in root.iter():
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        prefixes = []
        for cp in root.findall("CommonPrefixes"):
            p = cp.findtext("Prefix")
            if p:
                prefixes.append(p)
        for pref in prefixes:
            sym = pref.rstrip("/").split("/")[-1]
            if sym.endswith(quote):
                symbols.append(sym)
        truncated = (root.findtext("IsTruncated") or "false").lower() == "true"
        if not truncated:
            break
        marker = root.findtext("NextMarker") or (prefixes[-1] if prefixes else "")
        if not marker:
            break
    return sorted(set(symbols))


def should_exclude(symbol: str, exclude_bases: Iterable[str]) -> bool:
    base = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")
    if base.upper() in {b.upper() for b in exclude_bases}:
        return True
    # exclude quote-only / stable-like bases already handled; also skip non-USDT quote leftovers
    if not symbol.endswith("USDT"):
        return True
    return False


def month_range(start_month: str, end: datetime | None = None) -> list[str]:
    end = end or datetime.utcnow()
    y, m = map(int, start_month.split("-"))
    out = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def download_symbol_months(
    symbol: str,
    months: list[str],
    dest_dir: Path,
    interval: str = "1d",
) -> Path:
    """Download and cache monthly zips; return parquet path for the symbol."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_pq = dest_dir / f"{symbol}.parquet"
    if out_pq.exists():
        return out_pq

    frames: list[pd.DataFrame] = []
    raw_dir = dest_dir / "raw" / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)

    for ym in months:
        zip_name = f"{symbol}-{interval}-{ym}.zip"
        zip_path = raw_dir / zip_name
        url = (
            f"{VISION_FILE}/data/futures/um/monthly/klines/{symbol}/"
            f"{interval}/{zip_name}"
        )
        if not zip_path.exists():
            try:
                with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()
                    zip_path.write_bytes(r.read())
            except Exception as e:
                _log(f"{symbol} {ym} skip: {e}")
                continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as fh:
                    df = pd.read_csv(fh, header=None)
            # Binance kline schema (no header in vision dumps typically)
            # open_time, open, high, low, close, volume, close_time, quote_volume, ...
            if df.shape[1] < 8:
                continue
            df = df.iloc[:, :11].copy()
            df.columns = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
            ]
            frames.append(df)
        except Exception as e:
            _log(f"{symbol} {ym} parse skip: {e}")
            continue

    if not frames:
        # write empty marker parquet to avoid re-download loops
        empty = pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "symbol",
            ]
        )
        empty.to_parquet(out_pq, index=False)
        return out_pq

    all_df = pd.concat(frames, ignore_index=True)
    all_df["open_time"] = pd.to_numeric(all_df["open_time"], errors="coerce")
    all_df = all_df.dropna(subset=["open_time"])
    all_df["date"] = pd.to_datetime(all_df["open_time"], unit="ms", utc=True).dt.normalize()
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    all_df["symbol"] = symbol
    all_df = (
        all_df[["date", "open", "high", "low", "close", "volume", "quote_volume", "symbol"]]
        .dropna()
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    all_df.to_parquet(out_pq, index=False)
    return out_pq


def load_panel(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
    parts = []
    for sym in symbols:
        pq = raw_dir / f"{sym}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        raise RuntimeError("No kline data loaded")
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel["dollar_volume"] = panel["quote_volume"].astype(float)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def select_train_universe(panel: pd.DataFrame, n: int = 120) -> list[str]:
    """Top-n by median dollar volume (documented full-sample liquidity screen)."""
    med = panel.groupby("symbol")["dollar_volume"].median().sort_values(ascending=False)
    # ensure BTCUSDT included
    tops = list(med.head(n).index)
    if "BTCUSDT" not in tops:
        tops = ["BTCUSDT"] + [s for s in tops if s != "BTCUSDT"][: n - 1]
    return tops


def build_pit_topn(
    panel: pd.DataFrame,
    n: int = 20,
    window: int = 30,
) -> pd.DataFrame:
    """
    Point-in-time top-n by rolling median dollar volume using data ≤ t.
    Returns long df: date, symbol, rank, dv_med
    """
    wide = panel.pivot(index="date", columns="symbol", values="dollar_volume").sort_index()
    # rolling median with min_periods
    roll = wide.rolling(window=window, min_periods=max(5, window // 3)).median()
    rows = []
    for dt, row in roll.iterrows():
        s = row.dropna().sort_values(ascending=False)
        if s.empty:
            continue
        top = s.head(n)
        for rank, (sym, val) in enumerate(top.items(), start=1):
            rows.append({"date": dt, "symbol": sym, "rank": rank, "dv_med": float(val)})
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out
