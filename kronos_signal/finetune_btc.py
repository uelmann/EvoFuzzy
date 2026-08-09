"""
Simplified single-GPU Kronos predictor fine-tune on BTC daily.

Trains ONLY on bars before `train_end` (strict no leakage into backtest window).
Saves a HuggingFace-style folder with model weights.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .forecast import _ensure_kronos_on_path


class BtcWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        lookback: int = 90,
        pred_window: int = 10,
        clip: float = 5.0,
        seed: int = 100,
        n_samples: int = 2000,
    ):
        self.lookback = lookback
        self.pred_window = pred_window
        self.window = lookback + pred_window + 1
        self.clip = clip
        self.n_samples = n_samples
        self.rng = random.Random(seed)

        feat = df[["open", "high", "low", "close", "volume", "amount"]].to_numpy(np.float32)
        stamps = pd.DataFrame(
            {
                "minute": df["timestamps"].dt.minute,
                "hour": df["timestamps"].dt.hour,
                "weekday": df["timestamps"].dt.weekday,
                "day": df["timestamps"].dt.day,
                "month": df["timestamps"].dt.month,
            }
        ).to_numpy(np.float32)
        self.feat = feat
        self.stamps = stamps
        self.max_start = len(df) - self.window
        if self.max_start < 1:
            raise ValueError("Not enough bars for fine-tune windows")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = self.rng.randint(0, self.max_start)
        end = start + self.window
        x = self.feat[start:end].copy()
        stamp = self.stamps[start:end].copy()
        past = x[: self.lookback]
        mean = past.mean(axis=0)
        std = past.std(axis=0)
        x = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)
        return torch.from_numpy(x), torch.from_numpy(stamp)


def finetune_predictor_on_btc(
    df_train: pd.DataFrame,
    *,
    save_dir: str | Path,
    model_id: str = config.MODEL_ID,
    tokenizer_id: str = config.TOKENIZER_ID,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 4e-5,
    lookback: int = 90,
    pred_window: int = 10,
    n_samples: int = 1500,
    device: str | None = None,
    kronos_root: str | Path | None = None,
) -> dict:
    _ensure_kronos_on_path(kronos_root)
    from model import Kronos, KronosTokenizer

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    ds = BtcWindowDataset(
        df_train,
        lookback=lookback,
        pred_window=pred_window,
        n_samples=n_samples,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    history = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_x, batch_stamp in loader:
            batch_x = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            logits = model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"[finetune] epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}", flush=True)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(save_dir))
    # keep tokenizer id reference for reload
    (save_dir / "tokenizer_id.txt").write_text(tokenizer_id)
    return {
        "save_dir": str(save_dir),
        "epochs": epochs,
        "n_train_bars": len(df_train),
        "history": history,
        "device": device,
    }
