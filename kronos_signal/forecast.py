"""Kronos Monte Carlo forecasts (per-path closes, not averaged)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import config


def _ensure_kronos_on_path(kronos_root: str | Path | None = None) -> Path:
    candidates = []
    if kronos_root is not None:
        candidates.append(Path(kronos_root))
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "vendor" / "Kronos",
            Path("/opt/Kronos"),
        ]
    )
    for root in candidates:
        if (root / "model").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise FileNotFoundError(
        "Kronos model package not found. Clone "
        "https://github.com/shiyu-coder/Kronos into vendor/Kronos or /opt/Kronos."
    )


def load_predictor(
    model_id: str = config.MODEL_ID,
    tokenizer_id: str = config.TOKENIZER_ID,
    max_context: int = config.MAX_CONTEXT,
    device: str | None = None,
    kronos_root: str | Path | None = None,
):
    _ensure_kronos_on_path(kronos_root)
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def forecast_close_paths(
    predictor: Any,
    df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int = config.PRED_LEN,
    n_paths: int = config.N_PATHS,
    T: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    verbose: bool = False,
) -> np.ndarray:
    """
    Return shape (n_paths, pred_len) of forecasted close prices.

    Upstream KronosPredictor.predict() averages sample_count paths; we call
    sample_count=1 repeatedly so each Monte Carlo path is kept.
    """
    _ensure_kronos_on_path()
    from model.kronos import auto_regressive_inference, calc_time_stamps

    price_cols = ["open", "high", "low", "close"]
    data = df.copy()
    if "volume" not in data.columns:
        data["volume"] = 0.0
    if "amount" not in data.columns:
        data["amount"] = data["volume"] * data[price_cols].mean(axis=1)

    x = data[price_cols + ["volume", "amount"]].values.astype(np.float32)
    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    device = predictor.device
    x_t = torch.from_numpy(x_norm[np.newaxis, :].astype(np.float32)).to(device)
    x_stamp_t = torch.from_numpy(x_stamp[np.newaxis, :].astype(np.float32)).to(device)
    y_stamp_t = torch.from_numpy(y_stamp[np.newaxis, :].astype(np.float32)).to(device)

    close_idx = price_cols.index("close")
    path_closes = []
    for i in range(n_paths):
        if verbose:
            print(f"Kronos path {i + 1}/{n_paths}", flush=True)
        preds = auto_regressive_inference(
            predictor.tokenizer,
            predictor.model,
            x_t,
            x_stamp_t,
            y_stamp_t,
            predictor.max_context,
            pred_len,
            predictor.clip,
            T,
            0,
            top_p,
            1,
            False,
        )
        # preds: (1, seq, feats), already denormalized? No — still normalized space.
        preds = preds[:, -pred_len:, :]
        preds = preds * (x_std + 1e-5) + x_mean
        path_closes.append(preds[0, :, close_idx])

    return np.stack(path_closes, axis=0)
