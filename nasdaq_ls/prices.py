"""Price column for returns: Yahoo Adj Close."""

from __future__ import annotations

import pandas as pd


def return_price_col(df: pd.DataFrame) -> str:
    if "adj_close" in df.columns and pd.to_numeric(df["adj_close"], errors="coerce").notna().any():
        return "adj_close"
    return "close"


def close_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Session-date × symbol Adj Close (total return)."""
    col = return_price_col(panel)
    close = panel.pivot(index="date", columns="symbol", values=col).sort_index()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True)).normalize()
    return close
