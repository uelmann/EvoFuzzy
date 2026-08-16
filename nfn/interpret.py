"""Plain-language rule dump from learned exponents + membership movement."""

from __future__ import annotations

import numpy as np

from btcb.constants import STAGE_S_COLS
from nfn.constants import N_FEATURES, N_MEMBERSHIPS, N_RULES


MEM_LABEL = ("low", "mid", "high")


def decode_primitive(p: int) -> dict:
    n_mu = int(N_FEATURES) * int(N_MEMBERSHIPS)
    complement = bool(int(p) >= n_mu)
    q = int(p) - n_mu if complement else int(p)
    feat = q // int(N_MEMBERSHIPS)
    k = q % int(N_MEMBERSHIPS)
    name = STAGE_S_COLS[feat] if 0 <= feat < len(STAGE_S_COLS) else f"j{feat}"
    lab = MEM_LABEL[k] if 0 <= k < 3 else str(k)
    token = f"(1-μ_{lab}({name}))" if complement else f"μ_{lab}({name})"
    return {"feature": name, "k": int(k), "complement": complement, "token": token}


def top_rules(exponents: np.ndarray, weights: np.ndarray, n_rules: int = 8, n_prim: int = 6) -> list[dict]:
    e = np.asarray(exponents, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = []
    for k in range(min(int(N_RULES), e.shape[0])):
        row = e[k]
        strength = float(np.sum(row))
        idx = np.argsort(-row)[: int(n_prim)]
        terms = []
        for p in idx:
            if row[int(p)] < 0.05:
                continue
            rec = decode_primitive(int(p))
            rec["e"] = float(row[int(p)])
            terms.append(rec)
        formula = " * ".join(f"{t['token']}^{t['e']:.2f}" for t in terms) if terms else "(empty)"
        out.append(
            {
                "rule": int(k),
                "weight": float(w[k]) if k < len(w) else 0.0,
                "l1_e": strength,
                "terms": terms,
                "formula": f"r_{k} = {formula}",
            }
        )
    out.sort(key=lambda r: abs(r["weight"]) * (r["l1_e"] + 1e-9), reverse=True)
    return out[: int(n_rules)]


def membership_movement(c, s, c0, s0, feat_names: list[str], top_n: int = 20) -> list[dict]:
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
    c0 = np.asarray(c0, dtype=float)
    s0 = np.asarray(s0, dtype=float)
    move = np.abs(c - c0) + np.abs(s - s0)
    per_feat = move.max(axis=1)
    order = np.argsort(-per_feat)
    rows = []
    for j in order[: int(top_n)]:
        rows.append(
            {
                "feature": feat_names[int(j)] if int(j) < len(feat_names) else str(j),
                "movement": float(per_feat[int(j)]),
                "c": c[int(j)].tolist(),
                "s": s[int(j)].tolist(),
                "c_init": c0[int(j)].tolist(),
                "s_init": s0[int(j)].tolist(),
            }
        )
    return rows
