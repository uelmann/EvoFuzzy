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


def fetch_binance_klines(
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    limit: int = 1000,
) -> pd.DataFrame:
    """Return a DataFrame with open/high/low/close/volume/amount + timestamps."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    errors: list[str] = []
    rows = None
    for url in BINANCE_KLINES_URLS:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
            break
        except Exception as exc:  # noqa: BLE001 - collect and try next mirror
            errors.append(f"{url}: {exc}")
    if rows is None:
        raise RuntimeError(
            f"No klines returned for {symbol} {interval}. Tried mirrors:\n- "
            + "\n- ".join(errors)
        )
    if not rows:
        raise RuntimeError(f"Empty klines for {symbol} {interval}")

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
    out = pd.DataFrame(
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
    return out.reset_index(drop=True)


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
