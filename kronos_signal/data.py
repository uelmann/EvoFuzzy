"""Fetch OHLCV for Kronos from Binance public API."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import requests

from . import config

# api.binance.com is geo-blocked in some regions (HTTP 451); try mirrors first.
BINANCE_KLINES_URLS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
)


def _klines_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=[
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
            "ignore",
        ],
    )
    return pd.DataFrame(
        {
            "timestamps": pd.to_datetime(df["open_time"], unit="ms", utc=True),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "amount": df["quote_volume"].astype(float),
        }
    )


def _get_klines(params: dict) -> list:
    errors: list[str] = []
    for url in BINANCE_KLINES_URLS:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                return rows
            errors.append(f"{url}: empty response")
        except Exception as exc:  # noqa: BLE001 - collect and try next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "No klines returned. Tried mirrors:\n- " + "\n- ".join(errors)
    )


def fetch_binance_klines(
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    limit: int = 1000,
) -> pd.DataFrame:
    """Return recent OHLCV bars (single request, max 1000)."""
    rows = _get_klines(
        {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    )
    return _klines_to_df(rows).reset_index(drop=True)


def fetch_binance_klines_history(
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    min_bars: int = 1000,
) -> pd.DataFrame:
    """
    Paginate backwards until at least min_bars daily candles are collected.
    Binance caps each request at 1000 rows.
    """
    chunks: list[pd.DataFrame] = []
    end_time: int | None = None
    safety = 20  # max pages

    while safety > 0:
        safety -= 1
        params: dict = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        rows = _get_klines(params)
        chunk = _klines_to_df(rows)
        chunks.append(chunk)
        earliest = int(rows[0][0])
        end_time = earliest - 1
        total = sum(len(c) for c in chunks)
        if total >= min_bars or len(rows) < 1000:
            break

    df = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["timestamps"])
        .sort_values("timestamps")
        .reset_index(drop=True)
    )
    if len(df) < min_bars:
        # Return what we have; caller may still proceed with fewer bars.
        return df
    return df.iloc[-min_bars:].reset_index(drop=True)


def prepare_windows(
    df: pd.DataFrame,
    lookback: int = config.LOOKBACK,
    pred_len: int = config.PRED_LEN,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build Kronos inputs from the most recent lookback bars + future timestamps."""
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} bars, got {len(df)}")

    hist = df.iloc[-lookback:].copy().reset_index(drop=True)
    x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = hist["timestamps"]

    last_ts = hist["timestamps"].iloc[-1]
    # Crypto daily bars are contiguous calendar days on Binance.
    y_timestamp = pd.Series(
        [last_ts + timedelta(days=i) for i in range(1, pred_len + 1)],
        name="timestamps",
    )
    return x_df, x_timestamp, y_timestamp


def window_at(
    df: pd.DataFrame,
    end_idx: int,
    lookback: int = config.LOOKBACK,
    pred_len: int = config.PRED_LEN,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, float, float]:
    """
    Build a historical window ending at end_idx (inclusive).

    Returns x_df, x_timestamp, y_timestamp, last_close, realized_horizon_close.
    Requires end_idx + pred_len < len(df) for realized close.
    """
    if end_idx - lookback + 1 < 0:
        raise ValueError("end_idx/lookback out of range")
    if end_idx + pred_len >= len(df):
        raise ValueError("Not enough future bars for realized return")

    hist = df.iloc[end_idx - lookback + 1 : end_idx + 1].copy().reset_index(drop=True)
    x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = hist["timestamps"]
    last_ts = hist["timestamps"].iloc[-1]
    y_timestamp = pd.Series(
        [last_ts + timedelta(days=i) for i in range(1, pred_len + 1)],
        name="timestamps",
    )
    last_close = float(hist["close"].iloc[-1])
    realized_close = float(df.iloc[end_idx + pred_len]["close"])
    return x_df, x_timestamp, y_timestamp, last_close, realized_close
