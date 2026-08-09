"""Cross-sectional Kronos scores for crypto panel (zero-shot or fine-tuned)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config
from .cross_asset_bt import ScoreFn
from .panel_data import symbol_ohlcv_window


def kronos_mean_return_score_fn(
    predictor: Any,
    lookback: int = 90,
    pred_len: int = 10,
    n_paths: int = 5,
    T: float = 0.6,
    top_p: float = 0.9,
) -> ScoreFn:
    """
    Score(symbol) = mean predicted horizon return (close[-1] path avg / last_close - 1).

    Matches the spirit of Kronos A-share signal (predicted price change), then ranked
    cross-sectionally by the long/short backtest.
    """
    from .forecast import forecast_close_paths

    def _score(asof: pd.Timestamp, symbols: list[str], panels: dict[str, pd.DataFrame]) -> pd.Series:
        scores: dict[str, float] = {}
        for sym in symbols:
            win = symbol_ohlcv_window(panels, sym, asof, lookback=lookback)
            if win is None or len(win) < lookback:
                continue
            x_df = win[["open", "high", "low", "close", "volume", "amount"]]
            x_ts = pd.to_datetime(win["timestamps"], utc=True)
            # Future calendar stubs: next pred_len days (timestamps only for Kronos).
            last = x_ts.iloc[-1]
            y_ts = pd.Series(
                pd.date_range(last + pd.Timedelta(days=1), periods=pred_len, freq="D", tz="UTC")
            )
            try:
                closes = forecast_close_paths(
                    predictor,
                    x_df,
                    x_ts,
                    y_ts,
                    pred_len=pred_len,
                    n_paths=n_paths,
                    T=T,
                    top_p=top_p,
                    verbose=False,
                )
                last_close = float(x_df["close"].iloc[-1])
                pred = float(np.mean(closes[:, -1]))
                scores[sym] = pred / last_close - 1.0
            except Exception:
                continue
        return pd.Series(scores, dtype=float)

    return _score


def default_zero_shot_predictor(device: str | None = None, kronos_root: str | None = None):
    from .forecast import load_predictor

    return load_predictor(
        model_id=config.MODEL_ID,
        tokenizer_id=config.TOKENIZER_ID,
        max_context=config.MAX_CONTEXT,
        device=device,
        kronos_root=kronos_root,
    )
