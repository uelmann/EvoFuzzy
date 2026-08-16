"""NFN losses: class-weighted BCE (top) + listwise softmax CE + L1(e)."""

from __future__ import annotations

from nfn.constants import L1_LAMBDA, LISTWISE_COEF, LISTWISE_K, POS_WEIGHT_TOP


def nfn_loss(logit_top, logit_bot, y_top, y_bot, excess, model):
    import torch
    import torch.nn.functional as F

    pos_w = torch.tensor(float(POS_WEIGHT_TOP), device=logit_top.device, dtype=logit_top.dtype)
    bce_top = F.binary_cross_entropy_with_logits(logit_top, y_top, pos_weight=pos_w)
    bce_bot = F.binary_cross_entropy_with_logits(logit_bot, y_bot)
    listwise = listwise_topk_ce(logit_top, excess, k=int(LISTWISE_K))
    l1 = model.exponents().sum()
    total = bce_top + bce_bot + float(LISTWISE_COEF) * listwise + float(L1_LAMBDA) * l1
    return total, {
        "bce_top": float(bce_top.detach().cpu()),
        "bce_bot": float(bce_bot.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "l1_e": float(l1.detach().cpu()),
        "loss": float(total.detach().cpu()),
    }


def listwise_topk_ce(logits, excess, k: int = 10):
    """Uniform target on the date's realized top-k excess names (ListNet)."""
    import torch
    import torch.nn.functional as F

    n = int(logits.shape[0])
    if n < 2:
        return logits.sum() * 0.0
    kk = min(int(k), n)
    finite = torch.isfinite(excess)
    if int(finite.sum()) < 2:
        return logits.sum() * 0.0
    ex = excess.clone()
    ex = torch.where(finite, ex, torch.full_like(ex, -1e9))
    top_idx = torch.topk(ex, k=kk, largest=True).indices
    target = torch.zeros_like(logits)
    target[top_idx] = 1.0 / float(kk)
    logp = F.log_softmax(logits, dim=0)
    return -(target * logp).sum()
