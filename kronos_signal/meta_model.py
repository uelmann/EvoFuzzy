"""
Walk-forward meta-model on Kronos + market features.

Uses expanding window with purge/embargo and LightGBM (fallback: logistic).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import StepResult, summarize_steps
from .features import active_feature_cols

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:  # noqa: BLE001
    HAS_LGBM = False


@dataclass
class MetaBacktestResult:
    name: str
    n_steps: int
    n_long: int
    n_short: int
    n_hold: int
    hit_rate: float | None
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    avg_active_return: float | None
    start: str
    end: str
    min_train: int
    proba_long: float
    proba_short: float
    model_type: str = "logistic"
    embargo_steps: int = 1
    steps: list[dict] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _make_model(model_type: str = "auto"):
    if model_type == "auto":
        model_type = "lightgbm" if HAS_LGBM else "logistic"
    if model_type == "lightgbm":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm not installed")
        # Conservative params for small financial samples
        return LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=8,
            max_depth=3,
            min_child_samples=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        ), "lightgbm"
    return (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        C=0.5,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "logistic",
    )


def _train_mask(t: int, embargo_steps: int) -> np.ndarray:
    """
    Indices [0, t) usable for training at decision t.
    Embargo drops the last `embargo_steps` labels to reduce serial leakage.
    With non-overlapping 5d steps, overlapping-label purge is already mostly handled;
    embargo is an extra safety buffer.
    """
    end = max(0, t - embargo_steps)
    mask = np.zeros(t, dtype=bool)
    if end > 0:
        mask[:end] = True
    return mask


def walk_forward_meta(
    frame: pd.DataFrame,
    *,
    min_train: int = 40,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
    lookback: int = 400,
    pred_len: int = 5,
    n_paths: int = 10,
    step: int = 5,
    tau: float = 0.0,
    model_type: str = "auto",
    embargo_steps: int = 1,
    name: str | None = None,
    feature_cols: list[str] | None = None,
) -> MetaBacktestResult:
    """Expanding-window walk-forward with embargoed training labels."""
    if len(frame) <= min_train:
        raise ValueError(f"Need more than min_train={min_train} rows, got {len(frame)}")

    feat_cols = feature_cols or active_feature_cols(frame)
    feat_cols = [c for c in feat_cols if c in frame.columns]
    if not feat_cols:
        raise ValueError("No usable feature columns")
    X = frame[feat_cols].to_numpy(dtype=float)
    y = frame["y_up"].to_numpy(dtype=int)
    realized = frame["realized_return"].to_numpy(dtype=float)

    # Impute non-finite
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    steps: list[StepResult] = []
    used_model = None
    for t in range(min_train, len(frame)):
        mask = _train_mask(t, embargo_steps)
        if mask.sum() < max(10, min_train // 2):
            # not enough clean train rows yet
            signal, proba, strat, correct = "HOLD", 0.5, 0.0, None
        else:
            model, used_model = _make_model(model_type)
            model.fit(X[:t][mask], y[:t][mask])
            proba = float(model.predict_proba(X[t : t + 1])[0, 1])
            real = float(realized[t])
            if proba >= proba_long:
                signal, strat, correct = "LONG", real, real > 0
            elif proba <= proba_short:
                signal, strat, correct = "SHORT", -real, real < 0
            else:
                signal, strat, correct = "HOLD", 0.0, None

        real = float(realized[t])
        steps.append(
            StepResult(
                asof=str(frame.iloc[t]["asof"]),
                signal=signal,
                p_up=proba,
                mean_return=proba - 0.5,
                realized_return=real,
                strategy_return=float(strat),
                last_close=float("nan"),
                realized_close=float("nan"),
                correct=correct,
            )
        )

    bh = float(np.prod(1.0 + realized[min_train:]) - 1.0)
    summary = summarize_steps(
        steps,
        first_close=100.0,
        last_close=100.0 * (1.0 + bh),
        lookback=lookback,
        pred_len=pred_len,
        n_paths=n_paths,
        step=step,
        tau=tau,
    )
    return MetaBacktestResult(
        name=name or f"meta_{used_model or model_type}_purge",
        n_steps=summary.n_steps,
        n_long=summary.n_long,
        n_short=summary.n_short,
        n_hold=summary.n_hold,
        hit_rate=summary.hit_rate,
        total_return=summary.total_return,
        buy_hold_return=bh,
        max_drawdown=summary.max_drawdown,
        avg_active_return=summary.avg_active_return,
        start=summary.start,
        end=summary.end,
        min_train=min_train,
        proba_long=proba_long,
        proba_short=proba_short,
        model_type=used_model or model_type,
        embargo_steps=embargo_steps,
        steps=summary.steps,
    )


def raw_rule_on_frame(frame: pd.DataFrame) -> MetaBacktestResult:
    """Baseline: original Kronos raw LONG/HOLD/SHORT on the same rows."""
    steps: list[StepResult] = []
    for _, row in frame.iterrows():
        sig = row["raw_signal"]
        real = float(row["realized_return"])
        if sig == "LONG":
            strat, correct = real, real > 0
        elif sig == "SHORT":
            strat, correct = -real, real < 0
        else:
            strat, correct = 0.0, None
        steps.append(
            StepResult(
                asof=str(row["asof"]),
                signal=sig,
                p_up=float(row["kronos_p_up"]),
                mean_return=float(row["kronos_mean_r"]),
                realized_return=real,
                strategy_return=float(strat),
                last_close=float("nan"),
                realized_close=float("nan"),
                correct=correct,
            )
        )
    bh = float(np.prod(1.0 + frame["realized_return"].to_numpy()) - 1.0)
    summary = summarize_steps(
        steps,
        first_close=100.0,
        last_close=100.0 * (1.0 + bh),
        lookback=400,
        pred_len=5,
        n_paths=10,
        step=5,
        tau=0.005,
    )
    return MetaBacktestResult(
        name="raw_kronos_rule",
        n_steps=summary.n_steps,
        n_long=summary.n_long,
        n_short=summary.n_short,
        n_hold=summary.n_hold,
        hit_rate=summary.hit_rate,
        total_return=summary.total_return,
        buy_hold_return=bh,
        max_drawdown=summary.max_drawdown,
        avg_active_return=summary.avg_active_return,
        start=summary.start,
        end=summary.end,
        min_train=0,
        proba_long=0.6,
        proba_short=0.4,
        model_type="raw_rule",
        embargo_steps=0,
        steps=summary.steps,
    )
