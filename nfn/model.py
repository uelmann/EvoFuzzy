"""NEURO-FUZZY NET v0 — learnable sigmoidal memberships, log-space rules, FiLM, twin heads."""

from __future__ import annotations

import numpy as np

from btcb.constants import STAGE_S_COLS
from nfn.constants import (
    FILM_HIDDEN,
    FILM_M_DIM,
    LOG_EPS,
    MEMBERSHIP_C_INIT,
    MEMBERSHIP_S_INIT,
    MEMBERSHIP_S_MIN,
    N_FEATURES,
    N_INIT_PRIMITIVES,
    N_MEMBERSHIPS,
    N_PRIMITIVES,
    N_RULES,
)


def _softplus_inv(y: float) -> float:
    y = float(y)
    return float(np.log(np.expm1(max(y, 1e-12))))


class NeuroFuzzyNet:
    """Thin wrapper so the module imports without torch at freeze-check time."""

    pass


def build_nfn(seed: int):
    import torch
    from torch import nn

    class NFN(nn.Module):
        def __init__(self, seed: int):
            super().__init__()
            g = torch.Generator()
            g.manual_seed(int(seed))
            n_j = int(N_FEATURES)
            n_k = int(N_MEMBERSHIPS)
            n_p = int(N_PRIMITIVES)
            n_r = int(N_RULES)
            c0 = torch.tensor(MEMBERSHIP_C_INIT, dtype=torch.float32).view(1, n_k).repeat(n_j, 1)
            self.c = nn.Parameter(c0.clone())
            # s = s_min + softplus(s_raw); init s = 1.0
            s_raw0 = _softplus_inv(float(MEMBERSHIP_S_INIT) - float(MEMBERSHIP_S_MIN))
            self.s_raw = nn.Parameter(torch.full((n_j, n_k), s_raw0))
            e_raw = torch.full((n_r, n_p), -12.0)
            for r in range(n_r):
                idx = torch.randperm(n_p, generator=g)[: int(N_INIT_PRIMITIVES)]
                e_raw[r, idx] = _softplus_inv(1.0)
            self.e_raw = nn.Parameter(e_raw)
            self.w = nn.Parameter(torch.zeros(n_r))
            self.film = nn.Sequential(
                nn.Linear(int(FILM_M_DIM), int(FILM_HIDDEN)),
                nn.ReLU(),
                nn.Linear(int(FILM_HIDDEN), n_r * 2),
            )
            nn.init.zeros_(self.film[-1].weight)
            bias = torch.zeros(n_r * 2)
            bias[:n_r] = 1.0  # γ=1, β=0 at init
            self.film[-1].bias.data.copy_(bias)
            self.head_top = nn.Linear(n_r, 1)
            self.head_bot = nn.Linear(n_r, 1)
            nn.init.zeros_(self.head_top.bias)
            nn.init.zeros_(self.head_bot.bias)
            self.c_init = c0.detach().clone()
            self.s_init = torch.full((n_j, n_k), float(MEMBERSHIP_S_INIT))
            self.feat_names = list(STAGE_S_COLS)

        def scales(self):
            import torch.nn.functional as F

            return float(MEMBERSHIP_S_MIN) + F.softplus(self.s_raw)

        def exponents(self):
            import torch.nn.functional as F

            return F.softplus(self.e_raw)

        def primitives(self, z):
            import torch
            import torch.nn.functional as F

            # z: [n, J]
            s = self.scales().clamp(min=float(MEMBERSHIP_S_MIN))
            # μ[n, J, K]
            mu = torch.sigmoid((z.unsqueeze(-1) - self.c.unsqueeze(0)) / s.unsqueeze(0))
            mu = mu.reshape(z.shape[0], -1)
            return torch.cat([mu, 1.0 - mu], dim=-1)

        def forward(self, z, m):
            import torch

            prim = self.primitives(z).clamp(min=0.0, max=1.0)
            e = self.exponents()
            logp = (prim + float(LOG_EPS)).log()
            # r: [n, R]
            r = torch.exp(logp @ e.t())
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(z.shape[0], -1)
            gb = self.film(m)
            gamma, beta = gb.chunk(2, dim=-1)
            wt = gamma * self.w.unsqueeze(0) + beta
            h = r * wt
            logit_top = self.head_top(h).squeeze(-1)
            logit_bot = self.head_bot(h).squeeze(-1)
            return logit_top, logit_bot, {"r": r, "gamma": gamma, "beta": beta, "h": h}

        def n_params(self) -> int:
            return int(sum(p.numel() for p in self.parameters()))

    model = NFN(seed)
    return model
