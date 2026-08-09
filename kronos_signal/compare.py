"""Compare raw Kronos rule vs walk-forward meta-model variants."""

from __future__ import annotations

import json
from pathlib import Path

from .data import fetch_binance_klines_history
from .features import steps_to_frame
from .meta_model import raw_rule_on_frame, walk_forward_meta


# Feature set that produced the first positive meta backtest.
BASELINE_META_FEATURES = [
    "kronos_p_up",
    "kronos_mean_r",
    "kronos_abs_mean_r",
    "kronos_edge",
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "mom_20",
    "drawdown_60",
    "range_10",
]


def compare_raw_vs_meta(
    steps: list[dict],
    *,
    ohlcv=None,
    min_train: int = 40,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
    model_type: str = "logistic",
    embargo_steps: int = 0,
    supervised_p_up: dict[str, float] | None = None,
    rich: bool = False,
) -> dict:
    if ohlcv is None:
        ohlcv = fetch_binance_klines_history(min_bars=2000)
    frame = steps_to_frame(steps, ohlcv, supervised_p_up=supervised_p_up)
    raw = raw_rule_on_frame(frame)
    raw_aligned = raw_rule_on_frame(frame.iloc[min_train:].reset_index(drop=True))

    feat_cols = None if rich else list(BASELINE_META_FEATURES)
    if supervised_p_up is not None:
        feat_cols = list(feat_cols or BASELINE_META_FEATURES) + ["sup_p_up"]

    meta = walk_forward_meta(
        frame,
        min_train=min_train,
        proba_long=proba_long,
        proba_short=proba_short,
        model_type=model_type,
        embargo_steps=embargo_steps,
        feature_cols=feat_cols,
        name=f"meta_{model_type}_emb{embargo_steps}",
    )
    out = {
        "n_feature_rows": len(frame),
        "feature_start": str(frame.iloc[0]["asof"]),
        "feature_end": str(frame.iloc[-1]["asof"]),
        "feature_cols": feat_cols or list(frame.columns),
        "raw_full": raw.to_dict(),
        "raw_aligned": raw_aligned.to_dict(),
        "meta": meta.to_dict(),
    }
    from .features import FEATURE_COLS

    market_cols = [c for c in (feat_cols or FEATURE_COLS) if not c.startswith("kronos_") and c != "sup_p_up"]
    out["meta_market_only"] = walk_forward_meta(
        frame,
        min_train=min_train,
        proba_long=proba_long,
        proba_short=proba_short,
        model_type=model_type,
        embargo_steps=embargo_steps,
        name="meta_market_only",
        feature_cols=market_cols,
    ).to_dict()
    return out


def compare_from_json(
    path: str | Path = Path(__file__).with_name("last_backtest.json"),
    **kwargs,
) -> dict:
    data = json.loads(Path(path).read_text())
    return compare_raw_vs_meta(data["steps"], **kwargs)
