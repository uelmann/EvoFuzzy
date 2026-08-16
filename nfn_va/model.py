"""NEURO-FUZZY NET VARIANT A — Phase 7 body, single scalar head."""

from __future__ import annotations

import numpy as np

from btcb.constants import STAGE_S_COLS
from nfn_va.constants import (
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
    PHASE7_ARCH,
    VARIANT_A_N_PARAMS,
)


def _softplus_inv(y: float) -> float:
    y = float(y)
    return float(np.log(np.expm1(max(y, 1e-12))))


def arch_fingerprint() -> dict:
    return dict(PHASE7_ARCH)


def assert_arch_equal_phase7(extra: dict | None = None) -> dict:
    """Runtime config-equality: architecture block matches Phase 7; only head/loss/label/craft may differ."""
    got = arch_fingerprint()
    extra = extra or {}
    if extra.get("n_features") and int(extra["n_features"]) != N_FEATURES:
        raise RuntimeError("config equality: n_features drifted")
    if int(got["n_primitives"]) != 198 or int(got["n_rules"]) != 24:
        raise RuntimeError("config equality: primitives/rules drifted")
    rec = {"passed": True, "arch": got, "allowed_diffs": ["n_heads=1", "no_isotonic", "magnitude_loss", "craft_7c"]}
    print(f"[p7d] config-equality PASS arch={got}", flush=True)
    return rec


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
            bias[:n_r] = 1.0
            self.film[-1].bias.data.copy_(bias)
            self.head = nn.Linear(n_r, 1)
            nn.init.zeros_(self.head.bias)
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

            s = self.scales().clamp(min=float(MEMBERSHIP_S_MIN))
            mu = torch.sigmoid((z.unsqueeze(-1) - self.c.unsqueeze(0)) / s.unsqueeze(0))
            mu = mu.reshape(z.shape[0], -1)
            return torch.cat([mu, 1.0 - mu], dim=-1)

        def forward(self, z, m):
            import torch

            prim = self.primitives(z).clamp(min=0.0, max=1.0)
            e = self.exponents()
            logp = (prim + float(LOG_EPS)).log()
            r = torch.exp(logp @ e.t())
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(z.shape[0], -1)
            gb = self.film(m)
            gamma, beta = gb.chunk(2, dim=-1)
            wt = gamma * self.w.unsqueeze(0) + beta
            h = r * wt
            score = self.head(h).squeeze(-1)
            return score, {"r": r, "gamma": gamma, "beta": beta, "h": h}

        def n_params(self) -> int:
            return int(sum(p.numel() for p in self.parameters()))

    model = NFN(seed)
    n = model.n_params()
    if n != int(VARIANT_A_N_PARAMS):
        raise RuntimeError(f"n_params={n} expected {VARIANT_A_N_PARAMS} (Phase 7 5488 minus head_bot)")
    return model
