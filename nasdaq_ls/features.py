"""A0 features with a stock-market close series instead of BTCUSDT."""

from __future__ import annotations

import pandas as pd

from baseline.features import FEATURE_COLS, apply_cs_zscore, features_for_symbol


def build_feature_panel(
    panel: pd.DataFrame,
    market_close: pd.Series,
    clip: float = 5.0,
    zscore: bool = True,
) -> pd.DataFrame:
    mkt = pd.to_numeric(market_close, errors="coerce").sort_index()
    mkt.index = pd.to_datetime(mkt.index, utc=True).normalize()
    mkt = mkt[~mkt.index.duplicated(keep="last")].ffill()
    parts = []
    n_sym = panel["symbol"].nunique()
    for i, (sym, g) in enumerate(panel.groupby("symbol", sort=False), start=1):
        if len(g) < 50:
            continue
        gg = g.copy()
        gg["date"] = pd.to_datetime(gg["date"], utc=True).dt.normalize()
        if "adj_close" in gg.columns:
            gg["close"] = pd.to_numeric(gg["adj_close"], errors="coerce")
        parts.append(features_for_symbol(gg, mkt))
        if i % 25 == 0 or i == n_sym:
            print(f"[features] {i}/{n_sym} symbols", flush=True)
    if not parts:
        raise RuntimeError("no stock features")
    feat = pd.concat(parts, ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
    if zscore:
        feat = apply_cs_zscore(feat, clip=clip)
    return feat


__all__ = ["FEATURE_COLS", "build_feature_panel", "apply_cs_zscore"]
