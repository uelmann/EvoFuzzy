"""Official Kronos FT inference + TopkDropout backtest (mirrors finetune/qlib_test.py)."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .forecast import _ensure_kronos_on_path
from .official_config import OfficialConfig
from .official_topk_bt import load_full_panel, panels_from_full, roc_scores, topk_dropout_backtest


class OfficialTestDataset(Dataset):
    """Sliding windows like Kronos QlibTestDataset."""

    def __init__(self, data: dict, config: OfficialConfig, start: str | None = None, end: str | None = None):
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.indices: list[tuple[str, int, pd.Timestamp]] = []
        self.data: dict[str, pd.DataFrame] = {}

        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None

        print("Building inference indices...", flush=True)
        for symbol, sdf in data.items():
            df = sdf.reset_index()
            if "date" in df.columns:
                df = df.rename(columns={"date": "datetime"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df["minute"] = df["datetime"].dt.minute
            df["hour"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            # keep only model features + time + datetime
            keep = ["datetime"] + self.feature_list + self.time_feature_list
            df = df[keep].dropna(subset=self.feature_list)
            self.data[symbol] = df

            num_samples = len(df) - self.window_size + 1
            if num_samples <= 0:
                continue
            for i in range(num_samples):
                ts = df.iloc[i + self.config.lookback_window - 1]["datetime"]
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts:
                    continue
                self.indices.append((symbol, i, ts))
        print(f"Inference windows: {len(self.indices)}", flush=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, start_idx, timestamp = self.indices[idx]
        df = self.data[symbol]
        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window
        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]
        x = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)
        return (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(y_stamp),
            symbol,
            timestamp,
        )


def _collate(batch):
    xs, xst, yst, symbols, timestamps = zip(*batch)
    return torch.stack(xs), torch.stack(xst), torch.stack(yst), list(symbols), list(timestamps)


def generate_ft_scores(
    cfg: OfficialConfig,
    device: str = "cuda",
    kronos_root: str | None = None,
    signal: str = "mean",
) -> dict[str, pd.DataFrame]:
    """Return dict of score DataFrames (last/mean/max/min) like upstream qlib_test."""
    _ensure_kronos_on_path(kronos_root)
    from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference

    # Prefer full panel (has history for lookback into train); fall back to test pickle
    full_path = Path(cfg.dataset_path) / "full_panel.pkl"
    test_path = Path(cfg.dataset_path) / "test_data.pkl"
    with open(full_path if full_path.exists() else test_path, "rb") as f:
        raw = pickle.load(f)
    # Drop marketCap for model features if present
    data = {}
    for sym, df in raw.items():
        cols = [c for c in cfg.feature_list if c in df.columns]
        data[sym] = df[cols].dropna()

    tokenizer = KronosTokenizer.from_pretrained(cfg.finetuned_tokenizer_path).to(device).eval()
    model = Kronos.from_pretrained(cfg.finetuned_predictor_path).to(device).eval()

    bt_start, bt_end = cfg.backtest_time_range
    # Include lookback warm-up before backtest start for scoring continuity
    warm = (pd.Timestamp(bt_start, tz="UTC") - pd.Timedelta(days=cfg.lookback_window + 5)).strftime("%Y-%m-%d")
    dataset = OfficialTestDataset(data, cfg, start=warm, end=bt_end)
    bs = max(1, cfg.backtest_batch_size // max(cfg.inference_sample_count, 1))
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    results = defaultdict(list)
    n_done = 0
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps in loader:
            preds = auto_regressive_inference(
                tokenizer,
                model,
                x.to(device),
                x_stamp.to(device),
                y_stamp.to(device),
                max_context=cfg.max_context,
                pred_len=cfg.predict_window,
                clip=cfg.clip,
                T=cfg.inference_T,
                top_k=cfg.inference_top_k,
                top_p=cfg.inference_top_p,
                sample_count=cfg.inference_sample_count,
                verbose=False,
            )
            preds = preds[:, -cfg.predict_window :, :]
            last_day_close = x[:, -1, 3].numpy()
            signals = {
                "last": preds[:, -1, 3] - last_day_close,
                "mean": np.mean(preds[:, :, 3], axis=1) - last_day_close,
                "max": np.max(preds[:, :, 3], axis=1) - last_day_close,
                "min": np.min(preds[:, :, 3], axis=1) - last_day_close,
            }
            for i in range(len(symbols)):
                for sig_type, sig_values in signals.items():
                    results[sig_type].append((timestamps[i], symbols[i], float(sig_values[i])))
            n_done += len(symbols)
            if n_done % 500 < bs:
                print(f"[infer] scored {n_done}/{len(dataset)}", flush=True)

    prediction_dfs = {}
    for sig_type, records in results.items():
        df = pd.DataFrame(records, columns=["datetime", "instrument", "score"])
        pivot = df.pivot_table(index="datetime", columns="instrument", values="score")
        prediction_dfs[sig_type] = pivot.sort_index()
    return prediction_dfs


def _bt_summary(bt: dict) -> dict:
    return {
        "total_return": bt["total_return"],
        "max_drawdown": bt["max_drawdown"],
        "sharpe": bt["sharpe"],
        "btc_total_return": bt["btc_total_return"],
        "n_days": bt["n_days"],
    }


def run_official_ft_backtest(
    root: Path,
    device: str = "cuda",
    kronos_root: str | None = None,
    predictor_size: str = "small",
    signal: str = "mean",
) -> dict:
    cfg = OfficialConfig(root=root, predictor_size=predictor_size, epochs=30)
    tok = Path(cfg.finetuned_tokenizer_path)
    pred = Path(cfg.finetuned_predictor_path)
    if not tok.exists() or not pred.exists():
        raise FileNotFoundError(f"Missing FT checkpoints:\n  {tok}\n  {pred}")

    print(f"Loading FT models from {tok} / {pred}", flush=True)
    scores_map = generate_ft_scores(cfg, device=device, kronos_root=kronos_root)

    data = load_full_panel(cfg)
    panels = panels_from_full(data)
    close, mcap = panels["close"], panels["marketCap"]

    results = {}
    for name, score_df in scores_map.items():
        bt = topk_dropout_backtest(score_df, close, cfg, universe_mcap=mcap)
        results[f"kronos_ft_{name}"] = _bt_summary(bt)
        print(f"[bt] kronos_ft_{name}: ret={bt['total_return']:.2%} DD={bt['max_drawdown']:.2%}", flush=True)

    roc = roc_scores(close, window=cfg.predict_window)
    results["roc_baseline"] = _bt_summary(topk_dropout_backtest(roc, close, cfg, mcap))

    out = {
        "recipe": "official_ft_topk_backtest",
        "signal_primary": signal,
        "primary": results.get(f"kronos_ft_{signal}"),
        "all_signals": results,
        "config": {
            "lookback": cfg.lookback_window,
            "predict_window": cfg.predict_window,
            "topk": cfg.backtest_n_symbol_hold,
            "n_drop": cfg.backtest_n_symbol_drop,
            "backtest_range": cfg.backtest_time_range,
            "tokenizer": cfg.finetuned_tokenizer_path,
            "predictor": cfg.finetuned_predictor_path,
            "inference_T": cfg.inference_T,
            "sample_count": cfg.inference_sample_count,
        },
    }
    save = Path(root) / "last_official_ft_bt.json"
    # also persist score pivots for reuse
    score_path = Path(root) / "ft_prediction_scores.pkl"
    with open(score_path, "wb") as f:
        pickle.dump(scores_map, f)
    save.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {save}", flush=True)
    return out
