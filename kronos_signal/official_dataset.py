"""Kronos finetune Dataset — same sampling logic as Kronos/finetune/dataset.py."""

from __future__ import annotations

import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .official_config import OfficialConfig


class OfficialKronosDataset(Dataset):
    def __init__(self, data_type: str, config: OfficialConfig | None = None):
        self.config = config or OfficialConfig()
        if data_type not in ("train", "val"):
            raise ValueError("data_type must be train or val")
        self.data_type = data_type
        self.py_rng = random.Random(self.config.seed)

        if data_type == "train":
            path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(path, "rb") as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        self.indices: list[tuple[str, int]] = []
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            # normalize datetime column name
            if "date" in df.columns:
                df = df.rename(columns={"date": "datetime"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "datetime"})
            series_len = len(df)
            num_samples = series_len - self.window + 1
            if num_samples <= 0:
                continue
            dt = pd_to_datetime(df["datetime"])
            df["minute"] = dt.dt.minute
            df["hour"] = dt.dt.hour
            df["weekday"] = dt.dt.weekday
            df["day"] = dt.dt.day
            df["month"] = dt.dt.month
            self.data[symbol] = df[self.feature_list + self.time_feature_list]
            for i in range(num_samples):
                self.indices.append((symbol, i))

        self.n_samples = min(self.n_samples, max(len(self.indices), 1))
        print(
            f"[{data_type.upper()}] {len(self.indices)} windows, using {self.n_samples}/epoch"
        )

    def set_epoch_seed(self, epoch: int) -> None:
        self.py_rng.seed(self.config.seed + epoch)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        past_len = self.config.lookback_window
        past_x = x[:past_len]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)
        return torch.from_numpy(x), torch.from_numpy(x_stamp)


def pd_to_datetime(series):
    import pandas as pd

    return pd.to_datetime(series, utc=True)
