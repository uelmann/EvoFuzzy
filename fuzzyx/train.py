"""Walk-forward Adam trainer. Early-stop on inner-holdout path loss."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from .constants import (
    INNER_HOLDOUT_DAYS,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
)
from .pack import PackedPanel, slice_packed
from .torch_loss import path_loss_torch
from .torch_model import FuzzyXNet


@dataclass
class FoldTrainResult:
    model_state: dict
    best_epoch: int
    best_val: float
    history: list[dict]
    n_params: int
    elapsed: float
    status: str


def _to_torch(p: PackedPanel, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(p.X).to(device=device, dtype=torch.float32)
    m = torch.from_numpy(p.mask).to(device=device)
    r = torch.from_numpy(p.ret_h7).to(device=device, dtype=torch.float32)
    return x, m, r


def train_fold(
    packed: PackedPanel,
    train_start,
    train_end,
    val_start,
    val_end,
    seed: int = SEED,
    lr: float = LR,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    inner_holdout_days: int = INNER_HOLDOUT_DAYS,
    device: str | None = None,
) -> FoldTrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    tr_all = slice_packed(packed, train_start, train_end)
    va = slice_packed(packed, val_start, val_end)
    if tr_all.X.shape[0] < 8 or va.X.shape[0] < 4:
        return FoldTrainResult({}, -1, float("nan"), [], 0, 0.0, "empty")

    cut = pd_ts(train_end) - pd_td(inner_holdout_days)
    inner = slice_packed(tr_all, train_start, cut)
    hold = slice_packed(tr_all, cut + pd_td(1), train_end)
    if inner.X.shape[0] < 6 or hold.X.shape[0] < 3:
        inner, hold = tr_all, va

    x_tr, m_tr, r_tr = _to_torch(inner, device)
    x_ho, m_ho, r_ho = _to_torch(hold, device)
    x_va, m_va, r_va = _to_torch(va, device)

    model = FuzzyXNet(seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    t0 = time.time()
    best = 1e9
    best_state = None
    best_epoch = -1
    stale = 0
    history = []

    def _eval(xt, mt, rt, train_mode: bool) -> float:
        if train_mode:
            model.train()
        else:
            model.eval()
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            out = model(xt, mt)
            stats = path_loss_torch(out["soft_pos"], rt, mask=mt)
            if train_mode:
                opt.zero_grad(set_to_none=True)
                stats["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            return float(stats["loss"].detach().cpu())

    for epoch in range(int(max_epochs)):
        tr_loss = _eval(x_tr, m_tr, r_tr, True)
        ho_loss = _eval(x_ho, m_ho, r_ho, False)
        va_loss = _eval(x_va, m_va, r_va, False)
        history.append({"epoch": epoch, "train": tr_loss, "hold": ho_loss, "val": va_loss})
        if ho_loss < best - 1e-6:
            best = ho_loss
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    return FoldTrainResult(
        model_state={k: v.detach().cpu() for k, v in best_state.items()},
        best_epoch=best_epoch,
        best_val=float(best),
        history=history,
        n_params=model.n_params(),
        elapsed=time.time() - t0,
        status="ok",
    )


def pd_ts(x):
    import pandas as pd

    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def pd_td(days: int):
    import pandas as pd

    return pd.Timedelta(days=int(days))


@torch.no_grad()
def predict_packed(model: FuzzyXNet, packed: PackedPanel, device: str | None = None) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    x, m, r = _to_torch(packed, str(device))
    out = model(x, m)
    stats = path_loss_torch(out["soft_pos"], r, mask=m)
    hard_stats = path_loss_torch(out["hard_pos"], r, mask=m)
    return {
        "logits": out["logits"].cpu().numpy(),
        "soft_pos": out["soft_pos"].cpu().numpy(),
        "hard_pos": out["hard_pos"].cpu().numpy(),
        "soft_loss": {k: float(v.cpu()) if torch.is_tensor(v) else float(v) for k, v in stats.items()},
        "hard_loss": {k: float(v.cpu()) if torch.is_tensor(v) else float(v) for k, v in hard_stats.items()},
        "dates": packed.reb_dates,
        "symbols": packed.symbols,
        "mask": packed.mask,
        "ret_h7": packed.ret_h7,
    }
