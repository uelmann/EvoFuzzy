"""Torch backward + tiny synthetic fold train. No market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .constants import N_FEATURES, SEED, UNIVERSE_N
from .pack import PackedPanel
from .torch_loss import path_loss_torch, portfolio_net
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
    stats = path_loss_torch(out["soft_pos"], r, mask=m)
    assert "mean_pnl" in stats and "core" in stats
    # torch core = corr(wealth, t) * wealth[-1] must match numpy
    with torch.no_grad():
        port, _ = portfolio_net(out["soft_pos"].detach(), r, mask=m)
        st = np.cumprod(1.0 + port.cpu().numpy())
        np_c = float(np.corrcoef(st, np.arange(st.size))[1, 0]) if np.std(st) > 1e-12 else 0.0
        np_core = np_c * float(st[-1]) if np.isfinite(np_c) else 0.0
        torch_core = float(stats["core"].detach().cpu())
        if np.isfinite(np_core):
            assert abs(np_core - torch_core) < 1e-4, (np_core, torch_core)
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
