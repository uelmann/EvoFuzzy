"""Differentiable fuzzy rule bank: soft IGNORE/AND/NAND + product AND + prob-OR.

Each rule is a sparse conjunction over selected feature-membership literals.
Rule heads vote Long / Short / Flat. This is the KANFIS/ANDRE-style layer:
linear in F, not the exponential ANFIS grid.
"""

from __future__ import annotations

import numpy as np

from .constants import N_FEATURES, N_MFS, N_RULES
from .membership import softmax


def product_tnorm(values: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Soft conjunction: prod_i (1 - w_i + w_i * v_i). w=0 → ignore."""
    v = np.clip(values, 0.0, 1.0)
    w = np.clip(weights, 0.0, 1.0)
    return np.prod(1.0 - w + w * v, axis=axis)


def probabilistic_or(firings: np.ndarray, axis: int = -1) -> np.ndarray:
    """Soft disjunction: 1 - prod_i (1 - f_i)."""
    f = np.clip(firings, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - f, axis=axis)


class RuleBank:
    """R rules over F features × K memberships.

    selector_logits[r, f, :] → softmax over {IGNORE, AND, NAND}
    mf_logits[r, f, :] → softmax over K memberships
    head_logits[r, :] → softmax over {LONG, SHORT, FLAT}
    """

    IGNORE, AND, NAND = 0, 1, 2
    LONG, SHORT, FLAT = 0, 1, 2

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_mfs: int = N_MFS,
        n_rules: int = N_RULES,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = int(n_features)
        self.n_mfs = int(n_mfs)
        self.n_rules = int(n_rules)
        # Mild IGNORE bias so rules start sparse but not empty.
        self.selector_logits = rng.normal(0.0, 0.25, size=(n_rules, n_features, 3))
        self.selector_logits[..., self.IGNORE] += 0.35
        self.mf_logits = rng.normal(0.0, 0.15, size=(n_rules, n_features, n_mfs))
        self.head_logits = rng.normal(0.0, 0.15, size=(n_rules, 3))

    def selectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sel = softmax(self.selector_logits, axis=-1)
        mf = softmax(self.mf_logits, axis=-1)
        head = softmax(self.head_logits, axis=-1)
        return sel, mf, head

    def fire(self, memberships: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """memberships: (..., N, F, K) → firings (..., N, R), scores (..., N, 3)."""
        m = np.asarray(memberships, dtype=np.float64)
        if m.ndim < 3:
            raise ValueError("memberships must be (..., F, K) or (..., N, F, K)")
        sel, mf_w, head = self.selectors()
        # Selected membership per rule/feature: (R, F)
        # m: (..., N, F, K) ; mf_w: (R, F, K)
        # literal μ_{r,f} = sum_k mf_w[r,f,k] * m[..., n, f, k]
        mu = np.einsum("...nfk,rfk->...nrf", m, mf_w)
        one_mu = 1.0 - mu
        and_w = sel[..., self.AND]  # (R, F)
        nand_w = sel[..., self.NAND]
        # Per-feature contribution to the conjunction, then product over F.
        # and: use μ; nand: use 1-μ; ignore: 1
        term = (1.0 - and_w - nand_w) + and_w * mu + nand_w * one_mu
        firings = np.prod(np.clip(term, 1e-8, 1.0), axis=-1)  # (..., N, R)
        scores = np.einsum("...nr,rc->...nc", firings, head)
        return firings, scores

    def describe(self, feature_names: list[str], top_k: int = 4) -> list[str]:
        """Human-readable rules after argmax of selectors (eval-time crisp view)."""
        sel, mf_w, head = self.selectors()
        mf_names = ("LOW", "MID", "HIGH")
        act_names = ("LONG", "SHORT", "FLAT")
        lines = []
        for r in range(self.n_rules):
            mass = sel[r, :, self.AND] + sel[r, :, self.NAND]
            order = np.argsort(-mass)
            lits = []
            for f in order:
                if mass[f] < 0.28 and len(lits) >= 1:
                    break
                mode = int(np.argmax(sel[r, f]))
                if mode == self.IGNORE:
                    mode = self.AND if sel[r, f, self.AND] >= sel[r, f, self.NAND] else self.NAND
                k = int(np.argmax(mf_w[r, f]))
                tag = "NOT " if mode == self.NAND else ""
                lits.append(f"{tag}{feature_names[f]} IS {mf_names[k]}")
                if len(lits) >= top_k:
                    break
            if not lits:
                continue
            action = act_names[int(np.argmax(head[r]))]
            lines.append(f"R{r:02d} {action}: " + " AND ".join(lits))
        return lines

    def n_params(self) -> int:
        return int(self.selector_logits.size + self.mf_logits.size + self.head_logits.size)
