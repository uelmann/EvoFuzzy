"""Cross-asset crypto panel from CMC historical_data.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "historical_data.csv"

# Extra junk / non-crypto-beta names to drop from ranking universe.
EXCLUDE_SYMBOLS = {
    "USDT",
    "USDC",
    "DAI",
    "TUSD",
    "FDUSD",
    "USDE",
    "USD1",
    "USDD",
    "BUSD",
    "USDG",
    "PYUSD",
    "USD",
    "EUR",
    "U",
    "STABLE",
    "XAUT",
    "PAXG",
}


def load_historical_long(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m kronos_signal.download_cmc_kucoin --max-coins 60"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["currency_symbol"] = df["currency_symbol"].astype(str).str.upper()
    df = df[~df["currency_symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    df = df.sort_values(["currency_symbol", "date"])
    df = df.drop_duplicates(["currency_symbol", "date"], keep="last")
    return df.reset_index(drop=True)


def to_wide_panels(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return date×symbol panels for OHLCV + marketCap."""
    panels = {}
    for col in ("open", "high", "low", "close", "volume", "marketCap"):
        panels[col] = (
            long_df.pivot(index="date", columns="currency_symbol", values=col)
            .sort_index()
            .astype(float)
        )
    return panels


def point_in_time_universe(
    mcap: pd.DataFrame,
    asof: pd.Timestamp,
    top_n: int = 30,
    min_history_days: int = 90,
    close: pd.DataFrame | None = None,
) -> list[str]:
    """Top-N by market cap using only info available at asof (no look-ahead)."""
    if asof not in mcap.index:
        # nearest previous date
        prev = mcap.index[mcap.index <= asof]
        if len(prev) == 0:
            return []
        asof = prev[-1]
    row = mcap.loc[asof].dropna()
    if close is not None:
        hist = close.loc[:asof]
        ok = [s for s in row.index if hist[s].dropna().shape[0] >= min_history_days]
        row = row.loc[row.index.intersection(ok)]
    return list(row.nlargest(top_n).index)


def amount_proxy(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Kronos 'amount' ≈ close * volume when turnover missing."""
    return close * volume


def symbol_ohlcv_window(
    panels: dict[str, pd.DataFrame],
    symbol: str,
    end_date: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame | None:
    """Last `lookback` daily bars ending at end_date for one symbol (Kronos-ready)."""
    close = panels["close"]
    if symbol not in close.columns or end_date not in close.index:
        return None
    hist = close.loc[:end_date, symbol].dropna()
    if len(hist) < lookback:
        return None
    idx = hist.index[-lookback:]
    out = pd.DataFrame(
        {
            "timestamps": idx,
            "open": panels["open"].loc[idx, symbol].values,
            "high": panels["high"].loc[idx, symbol].values,
            "low": panels["low"].loc[idx, symbol].values,
            "close": panels["close"].loc[idx, symbol].values,
            "volume": panels["volume"].loc[idx, symbol].values,
        }
    )
    out["amount"] = out["close"] * out["volume"]
    if out[["open", "high", "low", "close"]].isna().any().any():
        return None
    return out.reset_index(drop=True)
