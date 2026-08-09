"""
Supervised direction head on frozen/lightly-tuned Kronos representations.

Objective: P(close[t+pred_len] > close[t]) from lookback context embedding.
Trains only on bars before the backtest start (no leakage).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .forecast import _ensure_kronos_on_path


class DirectionHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx_last: torch.Tensor) -> torch.Tensor:
        return self.net(ctx_last).squeeze(-1)


class DirectionWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        lookback: int = 90,
        pred_len: int = 5,
        clip: float = 5.0,
        seed: int = 100,
        n_samples: int = 2000,
    ):
        self.lookback = lookback
        self.pred_len = pred_len
        self.clip = clip
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.feat = df[["open", "high", "low", "close", "volume", "amount"]].to_numpy(
            np.float32
        )
        self.stamps = pd.DataFrame(
            {
                "minute": df["timestamps"].dt.minute,
                "hour": df["timestamps"].dt.hour,
                "weekday": df["timestamps"].dt.weekday,
                "day": df["timestamps"].dt.day,
                "month": df["timestamps"].dt.month,
            }
        ).to_numpy(np.float32)
        self.closes = df["close"].to_numpy(np.float64)
        self.max_start = len(df) - lookback - pred_len
        if self.max_start < 1:
            raise ValueError("Not enough bars for supervised windows")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = self.rng.randint(0, self.max_start)
        end = start + self.lookback
        x = self.feat[start:end].copy()
        stamp = self.stamps[start:end].copy()
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)
        c0 = self.closes[end - 1]
        c1 = self.closes[end - 1 + self.pred_len]
        y = 1.0 if c1 > c0 else 0.0
        return (
            torch.from_numpy(x),
            torch.from_numpy(stamp),
            torch.tensor(y, dtype=torch.float32),
        )


def _context_last(tokenizer, model, x, stamp):
    """Encode window and return last transformer context vector."""
    with torch.no_grad():
        token_seq_0, token_seq_1 = tokenizer.encode(x, half=True)
    # decode_s1 returns (s1_logits, context)
    _, ctx = model.decode_s1(token_seq_0, token_seq_1, stamp)
    return ctx[:, -1, :]


def train_supervised_direction(
    df_train: pd.DataFrame,
    *,
    save_dir: str | Path,
    model_id: str = config.MODEL_ID,
    tokenizer_id: str = config.TOKENIZER_ID,
    epochs: int = 8,
    batch_size: int = 8,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-5,
    lookback: int = 90,
    pred_len: int = 5,
    n_samples: int = 2000,
    unfreeze_last_n: int = 2,
    device: str | None = None,
    kronos_root: str | Path | None = None,
) -> dict:
    _ensure_kronos_on_path(kronos_root)
    from model import Kronos, KronosTokenizer

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id).to(device)
    model = Kronos.from_pretrained(model_id).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # Freeze backbone, optionally unfreeze last N transformer blocks
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        for block in model.transformer[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.norm.parameters():
            p.requires_grad = True

    head = DirectionHead(model.d_model).to(device)
    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_backbone},
        {"params": head.parameters(), "lr": lr_head},
    ]
    optim = torch.optim.AdamW(params, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = DirectionWindowDataset(
        df_train, lookback=lookback, pred_len=pred_len, n_samples=n_samples
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    history = []
    model.train()
    head.train()
    for epoch in range(epochs):
        losses, accs = [], []
        for batch_x, batch_stamp, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)
            batch_y = batch_y.to(device)

            # Discrete tokens from frozen tokenizer; grads flow via embedding(ids)
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            # decode_s1 path with grad through transformer
            x = model.embedding([token_seq_0, token_seq_1])
            x = x + model.time_emb(batch_stamp)
            x = model.token_drop(x)
            for layer in model.transformer:
                x = layer(x, key_padding_mask=None)
            x = model.norm(x)
            ctx_last = x[:, -1, :]
            logits = head(ctx_last)
            loss = loss_fn(logits, batch_y)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 3.0
            )
            optim.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                accs.append(float((pred == batch_y).float().mean().item()))
            losses.append(float(loss.item()))

        hist = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "acc": float(np.mean(accs)),
        }
        history.append(hist)
        print(
            f"[sup-ft] epoch {epoch + 1}/{epochs} loss={hist['loss']:.4f} acc={hist['acc']:.3f}",
            flush=True,
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    head.eval()
    model.save_pretrained(str(save_dir / "kronos"))
    torch.save(head.state_dict(), save_dir / "direction_head.pt")
    meta = {
        "tokenizer_id": tokenizer_id,
        "d_model": model.d_model,
        "lookback": lookback,
        "pred_len": pred_len,
        "history": history,
        "n_train_bars": len(df_train),
    }
    (save_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2))
    return {"save_dir": str(save_dir), **meta, "device": device}


def load_supervised_bundle(
    save_dir: str | Path,
    *,
    device: str | None = None,
    kronos_root: str | Path | None = None,
):
    import json

    _ensure_kronos_on_path(kronos_root)
    from model import Kronos, KronosTokenizer

    save_dir = Path(save_dir)
    meta = json.loads((save_dir / "meta.json").read_text())
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(meta["tokenizer_id"]).to(device)
    model = Kronos.from_pretrained(str(save_dir / "kronos")).to(device)
    head = DirectionHead(meta["d_model"]).to(device)
    head.load_state_dict(torch.load(save_dir / "direction_head.pt", map_location=device))
    tokenizer.eval()
    model.eval()
    head.eval()
    return tokenizer, model, head, meta, device


@torch.no_grad()
def predict_supervised_p_up_loaded(
    df_window: pd.DataFrame,
    tokenizer,
    model,
    head,
    meta: dict,
    device: str,
    clip: float = 5.0,
) -> float:
    lookback = meta["lookback"]
    hist = df_window.iloc[-lookback:]
    x = hist[["open", "high", "low", "close", "volume", "amount"]].to_numpy(np.float32)
    stamp = pd.DataFrame(
        {
            "minute": hist["timestamps"].dt.minute,
            "hour": hist["timestamps"].dt.hour,
            "weekday": hist["timestamps"].dt.weekday,
            "day": hist["timestamps"].dt.day,
            "month": hist["timestamps"].dt.month,
        }
    ).to_numpy(np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    x = np.clip((x - mean) / (std + 1e-5), -clip, clip)
    x_t = torch.from_numpy(x[None, ...]).to(device)
    s_t = torch.from_numpy(stamp[None, ...]).to(device)
    token_seq_0, token_seq_1 = tokenizer.encode(x_t, half=True)
    _, ctx = model.decode_s1(token_seq_0, token_seq_1, s_t)
    logit = head(ctx[:, -1, :])
    return float(torch.sigmoid(logit).item())


def predict_supervised_p_up(
    df_window: pd.DataFrame,
    *,
    save_dir: str | Path,
    device: str | None = None,
    kronos_root: str | Path | None = None,
    clip: float = 5.0,
) -> float:
    tokenizer, model, head, meta, device = load_supervised_bundle(
        save_dir, device=device, kronos_root=kronos_root
    )
    return predict_supervised_p_up_loaded(
        df_window, tokenizer, model, head, meta, device, clip=clip
    )


def supervised_rule_backtest(
    steps: list[dict],
    sup_p_up: dict[str, float],
    *,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
    min_train: int = 40,
) -> dict:
    """Trade supervised head alone on the aligned meta window."""
    from .backtest import StepResult, summarize_steps

    eval_steps = steps[min_train:]
    out_steps: list[StepResult] = []
    for s in eval_steps:
        asof = str(pd.Timestamp(s["asof"]))
        proba = float(sup_p_up.get(asof, sup_p_up.get(str(s["asof"]), 0.5)))
        real = float(s["realized_return"])
        if proba >= proba_long:
            signal, strat, correct = "LONG", real, real > 0
        elif proba <= proba_short:
            signal, strat, correct = "SHORT", -real, real < 0
        else:
            signal, strat, correct = "HOLD", 0.0, None
        out_steps.append(
            StepResult(
                asof=str(s["asof"]),
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
    reals = np.array([s["realized_return"] for s in eval_steps], dtype=float)
    bh = float(np.prod(1.0 + reals) - 1.0) if len(reals) else 0.0
    summary = summarize_steps(
        out_steps,
        first_close=100.0,
        last_close=100.0 * (1.0 + bh),
        lookback=400,
        pred_len=5,
        n_paths=0,
        step=5,
        tau=0.0,
    )
    d = summary.to_dict()
    d["name"] = "supervised_head_rule"
    d["buy_hold_return"] = bh
    return d
