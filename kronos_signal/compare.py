"""Compare raw Kronos rule vs walk-forward meta-model on a steps JSON + OHLCV."""

from __future__ import annotations

import json
from pathlib import Path

from .data import fetch_binance_klines_history
from .features import steps_to_frame
from .meta_model import raw_rule_on_frame, walk_forward_meta


def compare_raw_vs_meta(
    steps: list[dict],
    *,
    ohlcv=None,
    min_train: int = 40,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
) -> dict:
    if ohlcv is None:
        ohlcv = fetch_binance_klines_history(min_bars=2000)
    frame = steps_to_frame(steps, ohlcv)
    raw = raw_rule_on_frame(frame)
    # Meta evaluates from min_train onward; also report raw on the same slice for fair compare
    raw_aligned = raw_rule_on_frame(frame.iloc[min_train:].reset_index(drop=True))
    meta = walk_forward_meta(
        frame,
        min_train=min_train,
        proba_long=proba_long,
        proba_short=proba_short,
    )
    return {
        "n_feature_rows": len(frame),
        "feature_start": str(frame.iloc[0]["asof"]),
        "feature_end": str(frame.iloc[-1]["asof"]),
        "raw_full": raw.to_dict(),
        "raw_aligned": raw_aligned.to_dict(),
        "meta": meta.to_dict(),
    }


def compare_from_json(
    path: str | Path = Path(__file__).with_name("last_backtest.json"),
    **kwargs,
) -> dict:
    data = json.loads(Path(path).read_text())
    return compare_raw_vs_meta(data["steps"], **kwargs)
