"""Torch FuzzyX (DeepSets). Differentiable twin of fuzzyx.model.FuzzyX."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    D_MODEL,
    FEATURE_COLS,
    FLAT_INIT_BIAS,
    GATE_TEMPERATURE,
    MF_INIT_CENTERS,
    MF_INIT_SIGMA,
    N_FEATURES,
    N_MFS,
    N_RULES,
    POS_TEMPERATURE,
    SEED,
)


def market_token(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        mu = x.mean(dim=-2)
        sd = x.std(dim=-2, unbiased=False)
    else:
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = m.sum(dim=-2).clamp(min=1.0)
        mu = (x * m).sum(dim=-2) / denom
        var = ((x - mu.unsqueeze(-2)) ** 2 * m).sum(dim=-2) / denom
        sd = torch.sqrt(var.clamp(min=0.0))
    return torch.cat([mu, sd], dim=-1)


def soft_positions(logits: torch.Tensor, temperature: float = POS_TEMPERATURE) -> torch.Tensor:
    """P(long) − P(short). v1c: notebook-style signed score, not P_L²−P_S²."""
    p = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    return p[..., 0] - p[..., 1]


def hard_positions(logits: torch.Tensor) -> torch.Tensor:
    idx = logits.argmax(dim=-1)
    out = torch.zeros_like(logits[..., 0])
    out = torch.where(idx == 0, torch.ones_like(out), out)
    out = torch.where(idx == 1, -torch.ones_like(out), out)
    return out


class FuzzyXNet(nn.Module):
    """Membership → market gate → AND/OR rules → DeepSets CS residual."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_mfs: int = N_MFS,
        n_rules: int = N_RULES,
        d_model: int = D_MODEL,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_mfs = int(n_mfs)
        self.n_rules = int(n_rules)
        self.d_model = int(d_model)
        n_market = n_features * 2
        d_in = n_features + n_features * n_mfs + n_rules + 3

        g = torch.Generator()
        g.manual_seed(int(seed))

        mu = torch.tensor(MF_INIT_CENTERS, dtype=torch.float32).repeat(n_features, 1)
        self.mu = nn.Parameter(mu)
        self.log_sigma = nn.Parameter(torch.full((n_features, n_mfs), float(np.log(MF_INIT_SIGMA))))

        sel = torch.randn(n_rules, n_features, 3, generator=g) * 0.25
        sel[..., 0] += 0.35
        self.selector_logits = nn.Parameter(sel)
        self.mf_logits = nn.Parameter(torch.randn(n_rules, n_features, n_mfs, generator=g) * 0.15)
        self.head_logits = nn.Parameter(torch.randn(n_rules, 3, generator=g) * 0.15)

        scale = 1.0 / max(n_market, 1) ** 0.5
        self.gate_W = nn.Parameter(torch.randn(n_features, n_market, generator=g) * scale)
        self.gate_b = nn.Parameter(torch.zeros(n_features))

        s = 1.0 / max(d_in, 1) ** 0.5
        self.enc_W_in = nn.Parameter(torch.randn(d_in, d_model, generator=g) * s)
        self.enc_b_in = nn.Parameter(torch.zeros(d_model))
        d_z = 3 * d_model + n_market
        s2 = 1.0 / max(d_z, 1) ** 0.5
        self.enc_W_ff = nn.Parameter(torch.randn(d_z, d_model, generator=g) * s2)
        self.enc_b_ff = nn.Parameter(torch.zeros(d_model))
        self.enc_W_out = nn.Parameter(torch.randn(d_model, 3, generator=g) * (d_model**-0.5))
        b_out = torch.zeros(3)
        b_out[2] = float(FLAT_INIT_BIAS)
        self.enc_b_out = nn.Parameter(b_out)

        self.gate_temperature = GATE_TEMPERATURE
        self.pos_temperature = POS_TEMPERATURE

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def rule_sheet(self, feature_names: list[str] | None = None, top_k: int = 4) -> list[str]:
        names = feature_names or list(FEATURE_COLS)
        sel = F.softmax(self.selector_logits.detach(), dim=-1).cpu().numpy()
        mf_w = F.softmax(self.mf_logits.detach(), dim=-1).cpu().numpy()
        head = F.softmax(self.head_logits.detach(), dim=-1).cpu().numpy()
        mf_names = ("LOW", "MID", "HIGH")
        act_names = ("LONG", "SHORT", "FLAT")
        lines = []
        for r in range(self.n_rules):
            mass = sel[r, :, 1] + sel[r, :, 2]
            order = np.argsort(-mass)
            lits = []
            for f in order:
                if mass[f] < 0.28 and len(lits) >= 1:
                    break
                mode = int(np.argmax(sel[r, f]))
                if mode == 0:
                    mode = 1 if sel[r, f, 1] >= sel[r, f, 2] else 2
                k = int(np.argmax(mf_w[r, f]))
                tag = "NOT " if mode == 2 else ""
                lits.append(f"{tag}{names[f]} IS {mf_names[k]}")
                if len(lits) >= top_k:
                    break
            if lits:
                lines.append(f"R{r:02d} {act_names[int(np.argmax(head[r]))]}: " + " AND ".join(lits))
        return lines

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = x.float()
        mkt = market_token(x, mask)
        sig = self.log_sigma.exp().clamp(min=1e-4)
        memb = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.mu) / sig) ** 2).clamp(0.0, 1.0)

        alpha = F.softmax((mkt @ self.gate_W.T + self.gate_b) / max(self.gate_temperature, 1e-6), dim=-1)
        alpha = alpha * self.n_features
        gated = memb * alpha.unsqueeze(-2).unsqueeze(-1)

        sel = F.softmax(self.selector_logits, dim=-1)
        mf_w = F.softmax(self.mf_logits, dim=-1)
        head = F.softmax(self.head_logits, dim=-1)
        mu_lit = torch.einsum("...nfk,rfk->...nrf", gated, mf_w)
        and_w, nand_w = sel[..., 1], sel[..., 2]
        term = (1.0 - and_w - nand_w) + and_w * mu_lit + nand_w * (1.0 - mu_lit)
        firings = term.clamp(min=1e-8, max=1.0).prod(dim=-1)
        rule_scores = torch.einsum("...nr,rc->...nc", firings, head)

        tokens = torch.cat(
            [x, gated.reshape(*x.shape[:-1], -1), firings, rule_scores],
            dim=-1,
        )
        h = torch.tanh(tokens @ self.enc_W_in + self.enc_b_in)
        if mask is None:
            c = h.mean(dim=-2)
        else:
            m = mask.unsqueeze(-1).to(dtype=h.dtype)
            c = (h * m).sum(dim=-2) / m.sum(dim=-2).clamp(min=1.0)
        ones = torch.ones(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
        z = torch.cat(
            [h, c.unsqueeze(-2) * ones, h - c.unsqueeze(-2), mkt.unsqueeze(-2) * ones],
            dim=-1,
        )
        hid = torch.tanh(z @ self.enc_W_ff + self.enc_b_ff)
        logits = hid @ self.enc_W_out + self.enc_b_out
        if mask is not None:
            flat = torch.zeros_like(logits)
            flat[..., 2] = 8.0
            logits = torch.where(mask.unsqueeze(-1), logits, flat)
        return {
            "logits": logits,
            "soft_pos": soft_positions(logits, self.pos_temperature),
            "hard_pos": hard_positions(logits),
            "alpha": alpha,
            "firings": firings,
        }
