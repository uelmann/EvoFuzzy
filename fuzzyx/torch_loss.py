"""Differentiable weekly path loss.

v1c: loss = −corr(st_r, arange(T)). st_r is net weekly portfolio return.
No DD multipliers, no occupancy nuke, no pay-to-play.
"""

from __future__ import annotations

import torch

from .constants import (
    ACTIVE_LAMBDA,
    BIAS_LAMBDA,
    GROSS_LIMIT,
    LEVER_UP,
    MAXDD_CAP,
    SLIPPAGE_BPS,
    TAKER_FEE_BPS,
    TURN_LAMBDA,
)


def _weights(pos: torch.Tensor, mask: torch.Tensor | None, gross_limit: float = GROSS_LIMIT) -> torch.Tensor:
    p = pos
    if mask is not None:
        p = torch.where(mask, p, torch.zeros_like(p))
    gross = p.abs().sum(dim=-1, keepdim=True)
    if LEVER_UP:
        return p / gross.clamp(min=1e-8) * gross_limit
    return p / gross.clamp(min=1.0) * gross_limit


def portfolio_net(
    pos: torch.Tensor,
    asset_ret: torch.Tensor,
    mask: torch.Tensor | None = None,
    fee_bps: float = TAKER_FEE_BPS,
    slip_bps: float = SLIPPAGE_BPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = _weights(pos, mask)
    r = asset_ret
    if mask is not None:
        r = torch.where(mask, r, torch.zeros_like(r))
    pnl = (w * r).sum(dim=-1)
    prev = torch.zeros_like(w)
    prev[1:] = w[:-1]
    cost = (w - prev).abs().sum(dim=-1) * (fee_bps + slip_bps) / 1e4
    return pnl - cost, w


def _pearson_vs_arange(x: torch.Tensor) -> torch.Tensor:
    """np.corrcoef(x, np.arange(len(x)))[1, 0], differentiable. Constant x → 0."""
    n = x.numel()
    t = torch.arange(n, device=x.device, dtype=x.dtype)
    vx = x - x.mean()
    vt = t - t.mean()
    denom = vx.norm() * vt.norm()
    return (vx * vt).sum() / denom.clamp(min=1e-8)


def path_loss_torch(
    pos: torch.Tensor,
    asset_ret: torch.Tensor,
    mask: torch.Tensor | None = None,
    turn_lambda: float = TURN_LAMBDA,
    bias_lambda: float = BIAS_LAMBDA,
    active_lambda: float = ACTIVE_LAMBDA,
) -> dict[str, torch.Tensor]:
    port, w = portfolio_net(pos, asset_ret, mask=mask)
    equity = torch.cumprod(1.0 + port.clamp(min=-0.95, max=5.0), dim=0)
    peak = torch.cummax(equity, dim=0).values
    dd = equity / peak.clamp(min=1e-12) - 1.0
    maxdd = (-dd.min()).clamp(max=MAXDD_CAP)
    under = torch.sigmoid((-dd - 1e-4) / 0.02)
    ddur = under.mean()
    # v1c train core: corr(st_r, arange). Equity trend kept as diagnostic only.
    core = _pearson_vs_arange(port)
    trend_eq = _pearson_vs_arange(equity)

    if mask is not None:
        m = mask.to(dtype=pos.dtype)
        denom_n = m.sum().clamp(min=1.0)
        long_f = ((pos > 0.05).to(pos.dtype) * m).sum() / denom_n
        short_f = ((pos < -0.05).to(pos.dtype) * m).sum() / denom_n
        traded_f = ((pos.abs() > 0.05).to(pos.dtype) * m).sum() / denom_n
        active = (torch.sigmoid((pos.abs() - 0.05) / 0.05) * m).sum() / denom_n
    else:
        long_f = (pos > 0.05).float().mean()
        short_f = (pos < -0.05).float().mean()
        traded_f = (pos.abs() > 0.05).float().mean()
        active = torch.sigmoid((pos.abs() - 0.05) / 0.05).mean()

    turn = (w[1:] - w[:-1]).abs().sum(dim=-1).mean() if w.shape[0] > 1 else w.new_zeros(())
    bias = w.mean().abs()
    loss = -core + turn_lambda * turn + bias_lambda * bias + active_lambda * active
    return {
        "loss": loss,
        "core": core.detach(),
        "trend": core.detach(),
        "trend_equity": trend_eq.detach(),
        "maxdd": maxdd.detach(),
        "ddur": ddur.detach(),
        "long_frac": long_f.detach(),
        "short_frac": short_f.detach(),
        "traded_frac": traded_f.detach(),
        "active": active.detach(),
        "turnover": turn.detach() if torch.is_tensor(turn) else turn,
        "bias": bias.detach(),
        "mean_pnl": port.mean().detach(),
    }
