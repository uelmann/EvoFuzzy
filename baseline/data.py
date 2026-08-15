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


def _list_vision_symbols(prefix: str, quote: str = "USDT") -> list[str]:
    """List symbols under a Binance Vision S3 prefix (CommonPrefixes)."""
    symbols: list[str] = []
    marker = ""
    while True:
        params = {
            "prefix": prefix,
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


def list_um_symbols(quote: str = "USDT") -> list[str]:
    """List perpetual symbols present on data.binance.vision monthly klines."""
    return _list_vision_symbols("data/futures/um/monthly/klines/", quote)


def list_spot_symbols(quote: str = "USDT") -> list[str]:
    """List spot symbols present on data.binance.vision monthly klines."""
    return _list_vision_symbols("data/spot/monthly/klines/", quote)


def should_exclude(symbol: str, exclude_bases: Iterable[str]) -> bool:
    if not symbol.endswith("USDT"):
        return True
    base = symbol[: -len("USDT")]
    exclude = {b.upper() for b in exclude_bases}
    if base.upper() in exclude:
        return True
    # heuristic: wrapped / stable-like names
    up = base.upper()
    if up.startswith("1000") and up.endswith(("USDC",)):
        return True
    if up in {"USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "USDD"}:
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


def _kline_zip_url(symbol: str, interval: str, ym: str, kind: str = "um") -> tuple[str, str]:
    zip_name = f"{symbol}-{interval}-{ym}.zip"
    if kind == "spot":
        url = f"{VISION_FILE}/data/spot/monthly/klines/{symbol}/{interval}/{zip_name}"
    else:
        url = f"{VISION_FILE}/data/futures/um/monthly/klines/{symbol}/{interval}/{zip_name}"
    return url, zip_name


def download_symbol_months(
    symbol: str,
    months: list[str],
    dest_dir: Path,
    interval: str = "1d",
    kind: str = "um",
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
        url, zip_name = _kline_zip_url(symbol, interval, ym, kind=kind)
        zip_path = raw_dir / zip_name
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


def download_spot_symbol_months(
    symbol: str,
    months: list[str],
    dest_dir: Path,
    interval: str = "1d",
) -> Path:
    """Download Binance spot 1d klines (Vision); cache as parquet."""
    return download_symbol_months(symbol, months, dest_dir, interval=interval, kind="spot")


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


def build_pit_topn(
    panel: pd.DataFrame,
    n: int = 20,
    window: int = 30,
) -> pd.DataFrame:
    """
    Point-in-time top-n by rolling median dollar volume using data ≤ t.
    Returns long df: date, symbol, rank, dv_med.

    Uses only trailing window ending at t (pandas rolling is causal).
    """
    wide = panel.pivot(index="date", columns="symbol", values="dollar_volume").sort_index()
    roll = wide.rolling(window=window, min_periods=max(5, window // 3)).median()
    ranks = roll.rank(axis=1, ascending=False, method="first")
    mask = ranks <= n
    long_rank = ranks.where(mask).stack(future_stack=True).rename("rank")
    long_dv = roll.where(mask).stack(future_stack=True).rename("dv_med")
    out = pd.concat([long_rank, long_dv], axis=1).dropna(how="any").reset_index()
    out.columns = ["date", "symbol", "rank", "dv_med"]
    out["rank"] = out["rank"].astype(int)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["date", "rank"]).reset_index(drop=True)
    return out


def download_funding_symbol_months(
    symbol: str,
    months: list[str],
    dest_dir: Path,
) -> Path:
    """Download and cache monthly fundingRate zips; return parquet path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_pq = dest_dir / f"{symbol}.parquet"
    if out_pq.exists():
        return out_pq

    frames: list[pd.DataFrame] = []
    raw_dir = dest_dir / "raw" / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)

    for ym in months:
        zip_name = f"{symbol}-fundingRate-{ym}.zip"
        zip_path = raw_dir / zip_name
        url = (
            f"{VISION_FILE}/data/futures/um/monthly/fundingRate/{symbol}/{zip_name}"
        )
        if not zip_path.exists():
            try:
                with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()
                    zip_path.write_bytes(r.read())
            except Exception as e:
                _log(f"funding {symbol} {ym} skip: {e}")
                continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as fh:
                    df = pd.read_csv(fh)
            # schema: calc_time, funding_interval_hours, last_funding_rate
            cols = {c.lower(): c for c in df.columns}
            tcol = cols.get("calc_time") or list(df.columns)[0]
            rcol = cols.get("last_funding_rate") or list(df.columns)[-1]
            part = pd.DataFrame(
                {
                    "funding_time": pd.to_numeric(df[tcol], errors="coerce"),
                    "funding_rate": pd.to_numeric(df[rcol], errors="coerce"),
                }
            ).dropna()
            part["symbol"] = symbol
            frames.append(part)
        except Exception as e:
            _log(f"funding {symbol} {ym} parse skip: {e}")
            continue

    if not frames:
        empty = pd.DataFrame(columns=["date", "symbol", "funding_rate", "n_events"])
        empty.to_parquet(out_pq, index=False)
        return out_pq

    all_df = pd.concat(frames, ignore_index=True)
    all_df["ts"] = pd.to_datetime(all_df["funding_time"], unit="ms", utc=True)
    all_df["date"] = all_df["ts"].dt.normalize()
    daily = (
        all_df.groupby(["date", "symbol"], as_index=False)
        .agg(funding_rate=("funding_rate", "sum"), n_events=("funding_rate", "size"))
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    daily.to_parquet(out_pq, index=False)
    return out_pq


def load_funding_panel(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["date", "symbol", "funding_rate", "n_events"])
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def funding_coverage_report(funding: pd.DataFrame, symbols: list[str]) -> dict:
    """Summarize which symbols/dates lack funding (treated as 0)."""
    if funding.empty:
        return {"n_symbols_with_funding": 0, "missing_note": "no funding files loaded"}
    have = set(funding["symbol"].unique())
    missing = sorted(set(symbols) - have)
    by_sym = funding.groupby("symbol")["date"].agg(["min", "max", "count"])
    return {
        "n_symbols_with_funding": int(len(have)),
        "n_symbols_requested": int(len(symbols)),
        "n_missing_symbols": int(len(missing)),
        "missing_symbols_sample": missing[:30],
        "span": [str(funding["date"].min().date()), str(funding["date"].max().date())],
        "median_days_per_symbol": float(by_sym["count"].median()) if len(by_sym) else 0.0,
    }


def luna_presence_report(pit: pd.DataFrame, start: str = "2021-01-01", end: str = "2022-12-31") -> dict:
    """Verify whether LUNA (or close tickers) appear in a PIT universe during 2021–2022."""
    df = pit.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    mask = (df["date"] >= pd.Timestamp(start, tz="UTC")) & (df["date"] <= pd.Timestamp(end, tz="UTC"))
    sub = df.loc[mask]
    luna_like = sorted({s for s in sub["symbol"].unique() if "LUNA" in str(s).upper()})
    days = []
    for sym in luna_like:
        dsub = sub.loc[sub["symbol"] == sym, "date"]
        if len(dsub):
            days.append(
                {
                    "symbol": sym,
                    "n_days": int(len(dsub)),
                    "first": str(dsub.min().date()),
                    "last": str(dsub.max().date()),
                    "median_rank": float(sub.loc[sub["symbol"] == sym, "rank"].median()),
                }
            )
    return {
        "window": [start, end],
        "luna_like_symbols": luna_like,
        "present": bool(luna_like),
        "details": days,
    }
