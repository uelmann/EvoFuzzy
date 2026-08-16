"""Torch backward + tiny synthetic fold train. No market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .constants import N_FEATURES, SEED, UNIVERSE_N
from .pack import PackedPanel
from .torch_loss import path_loss_torch
from .torch_model import FuzzyXNet
from .train import train_fold


def _toy_packed(n_days: int = 40, n_assets: int = UNIVERSE_N, seed: int = SEED) -> PackedPanel:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_days, n_assets, N_FEATURES)).astype(np.float32).clip(-5, 5)
    mask = np.ones((n_days, n_assets), dtype=bool)
    mask[:, -2:] = False
    ret = (0.03 * np.tanh(X[:, :, 0]) + rng.normal(0, 0.02, size=(n_days, n_assets))).astype(np.float32)
    dates = pd.date_range("2022-01-07", periods=n_days, freq="7D", tz="UTC")
    return PackedPanel(
        symbols=[f"S{i}" for i in range(n_assets)],
        reb_dates=pd.DatetimeIndex(dates),
        X=X,
        mask=mask,
        ret_h7=ret,
        ret_1=ret / 7.0,
    )


def run() -> dict:
    torch.manual_seed(SEED)
    p = _toy_packed()
    model = FuzzyXNet(seed=SEED)
    x = torch.from_numpy(p.X).float()
    m = torch.from_numpy(p.mask)
    r = torch.from_numpy(p.ret_h7).float()
    out = model(x, m)
    # FLAT prior: init logits prefer FLAT over LONG/SHORT
    logits = out["logits"]
    assert float(logits[..., 2].mean().detach()) > float(logits[..., 0].mean().detach())
    assert float(logits[..., 2].mean().detach()) > float(logits[..., 1].mean().detach())
    stats = path_loss_torch(out["soft_pos"], r, mask=m)
    assert "mean_pnl" in stats and "active" in stats
    stats["loss"].backward()
    grad = float(sum(float(t.grad.abs().sum()) for t in model.parameters() if t.grad is not None))
    assert grad > 0.0
    res = train_fold(
        p,
        p.reb_dates[0],
        p.reb_dates[24],
        p.reb_dates[25],
        p.reb_dates[-1],
        seed=SEED,
        max_epochs=5,
        patience=5,
        inner_holdout_days=40,
    )
    assert res.status == "ok"
    return {
        "n_params": model.n_params(),
        "grad_l1": grad,
        "init_loss": float(stats["loss"].detach()),
        "train_status": res.status,
        "best_epoch": res.best_epoch,
        "best_val": res.best_val,
        "epochs": len(res.history),
    }


def main() -> None:
    r = run()
    print(
        f"FuzzyX torch smoke  params={r['n_params']}  grad={r['grad_l1']:.3f}  "
        f"init_loss={r['init_loss']:.4f}  train={r['train_status']}  "
        f"best_epoch={r['best_epoch']}  best_val={r['best_val']:.4f}"
    )


if __name__ == "__main__":
    main()
