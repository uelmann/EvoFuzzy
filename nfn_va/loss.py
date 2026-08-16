"""Variant A loss: ListNet on winsorized magnitudes + Huber on standardized size + L1(e)."""

from __future__ import annotations

from nfn_va.constants import HUBER_DELTA, L1_LAMBDA, MAG_COEF, RANK_COEF, TAU


def variant_a_loss(score, y_win, y_win_z, model, tau: float = TAU):
    """L = 1.0 L_rank + 0.5 L_mag + L1(e). Both label forms: winsorized mag (rank) and z (Huber)."""
    import torch
    import torch.nn.functional as F

    l_rank = listnet_mag(score, y_win, tau=float(tau))
    l_mag = huber_mag(score, y_win_z, delta=float(HUBER_DELTA))
    l1 = model.exponents().sum()
    total = float(RANK_COEF) * l_rank + float(MAG_COEF) * l_mag + float(L1_LAMBDA) * l1
    return total, {
        "l_rank": float(l_rank.detach().cpu()),
        "l_mag": float(l_mag.detach().cpu()),
        "l1_e": float(l1.detach().cpu()),
        "loss": float(total.detach().cpu()),
    }


def listnet_mag(score, y_win, tau: float = 1.0):
    """ListNet CE: softmax(score/τ) vs softmax(winsorized magnitude/τ)."""
    import torch
    import torch.nn.functional as F

    n = int(score.shape[0])
    if n < 2:
        return score.sum() * 0.0
    finite = torch.isfinite(score) & torch.isfinite(y_win)
    if int(finite.sum()) < 2:
        return score.sum() * 0.0
    s = torch.where(finite, score, torch.full_like(score, -1e9))
    y = torch.where(finite, y_win, torch.full_like(y_win, -1e9))
    t = max(float(tau), 1e-6)
    logp = F.log_softmax(s / t, dim=0)
    q = F.softmax(y / t, dim=0)
    q = torch.where(finite, q, torch.zeros_like(q))
    z = q.sum().clamp(min=1e-12)
    q = q / z
    return -(q * logp).sum()


def huber_mag(score, y_win_z, delta: float = 1.0):
    import torch
    import torch.nn.functional as F

    finite = torch.isfinite(score) & torch.isfinite(y_win_z)
    if int(finite.sum()) < 2:
        return score.sum() * 0.0
    return F.huber_loss(score[finite], y_win_z[finite], delta=float(delta), reduction="mean")
