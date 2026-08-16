"""PROJECT FORGE F0 — evolutionary strategy miner (compounding fitness).

BACKTEST / ANALYSIS ONLY. Generic primitives only. No PI-formula seeds.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.constants import (
    ANNUALIZATION,
    FORGE_BINARY_OPS,
    FORGE_COST_BPS,
    FORGE_CS_DDOF,
    FORGE_CUBE_START,
    FORGE_DIVERSITY_CORR,
    FORGE_EARLY_STOP_GENS,
    FORGE_ELITE,
    FORGE_GENS,
    FORGE_HEADLINE_K,
    FORGE_HEADLINE_REBAL,
    FORGE_HEADLINE_UNIV,
    FORGE_HL_K,
    FORGE_INIT_DEPTH,
    FORGE_JUDGE_END,
    FORGE_K_SET,
    FORGE_KNIFE_SPREAD,
    FORGE_LAMBDA_C,
    FORGE_MAX_CONCURRENCY,
    FORGE_MAX_DEPTH,
    FORGE_MAX_NODES,
    FORGE_MAXDD_MULT,
    FORGE_MCAP_N,
    FORGE_N_CHAMPIONS,
    FORGE_NULL_GENS,
    FORGE_P_CROSSOVER,
    FORGE_P_POINT,
    FORGE_P_SUBTREE,
    FORGE_PDIV_EPS,
    FORGE_POP,
    FORGE_PRIMITIVES,
    FORGE_REBAL_SET,
    FORGE_REL_SHARPE_STRONG,
    FORGE_RET_K,
    FORGE_SEED,
    FORGE_SELECT_TOP,
    FORGE_STD_DDOF,
    FORGE_STD_K,
    FORGE_TOURNAMENT,
    FORGE_TS_K,
    FORGE_TS_OPS,
    FORGE_UNARY_OPS,
    FORGE_WEEKDAY,
    MANUEL2_PEG_ABS_RET,
    STABLE_OR_WRAP,
)
from btcb.manuel2 import (
    build_members,
    cmc_close_wide,
    hybrid_close_wide,
    mcap_wide,
    peg_heuristic_mask,
    tagged_peg_ids,
)
from btcb.oracle_ladder import _as_utc, _utc_idx, ffill_members

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*_a, **_k):
        def _wrap(fn):
            return fn

        if _a and callable(_a[0]) and not _k:
            return _a[0]
        return _wrap


_PRIM_SET = frozenset(FORGE_PRIMITIVES)
_UNARY_SET = frozenset(FORGE_UNARY_OPS)
_BINARY_SET = frozenset(FORGE_BINARY_OPS)
_TS_SET = frozenset(FORGE_TS_OPS)
_TS_K_SET = frozenset(int(k) for k in FORGE_TS_K)


def _ndtr(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(np.asarray(z, dtype=np.float64))


def _clean_arr(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(y), y, np.nan)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

def node_count(tree) -> int:
    k = tree[0]
    if k == "T":
        return 1
    if k in ("U", "S"):
        return 1 + node_count(tree[-1])
    if k == "B":
        return 1 + node_count(tree[2]) + node_count(tree[3])
    raise ValueError(f"bad node {k}")


def tree_depth(tree) -> int:
    k = tree[0]
    if k == "T":
        return 1
    if k in ("U", "S"):
        return 1 + tree_depth(tree[-1])
    if k == "B":
        return 1 + max(tree_depth(tree[2]), tree_depth(tree[3]))
    raise ValueError(f"bad node {k}")


def formula_str(tree) -> str:
    k = tree[0]
    if k == "T":
        return str(tree[1])
    if k == "U":
        return f"{tree[1]}({formula_str(tree[2])})"
    if k == "S":
        return f"{tree[1]}_{int(tree[2])}({formula_str(tree[3])})"
    if k == "B":
        return f"{tree[1]}({formula_str(tree[2])}, {formula_str(tree[3])})"
    raise ValueError(f"bad node {k}")


def _tok(s: str, i: int) -> tuple[str, int]:
    n = len(s)
    while i < n and s[i] in " \t":
        i += 1
    if i >= n:
        return "", i
    if s[i] in "(),":
        return s[i], i + 1
    j = i
    while j < n and (s[j].isalnum() or s[j] in "_"):
        j += 1
    return s[i:j], j


def parse_formula(s: str):
    s = str(s).strip()

    def parse_at(i: int):
        tok, i = _tok(s, i)
        if not tok:
            raise ValueError("empty formula")
        nxt, j = _tok(s, i)
        if nxt != "(":
            if tok not in _PRIM_SET:
                raise ValueError(f"unknown primitive {tok}")
            return ("T", tok), i
        i = j
        if tok in _UNARY_SET:
            child, i = parse_at(i)
            rp, i = _tok(s, i)
            if rp != ")":
                raise ValueError(f"expected ) after unary {tok}")
            return ("U", tok, child), i
        if tok in _BINARY_SET:
            left, i = parse_at(i)
            cm, i = _tok(s, i)
            if cm != ",":
                raise ValueError(f"expected comma in {tok}")
            right, i = parse_at(i)
            rp, i = _tok(s, i)
            if rp != ")":
                raise ValueError(f"expected ) after binary {tok}")
            return ("B", tok, left, right), i
        if "_" in tok:
            op, _, ks = tok.rpartition("_")
            if op in _TS_SET and ks.isdigit() and int(ks) in _TS_K_SET:
                child, i = parse_at(i)
                rp, i = _tok(s, i)
                if rp != ")":
                    raise ValueError(f"expected ) after {tok}")
                return ("S", op, int(ks), child), i
        raise ValueError(f"unknown operator {tok}")

    tree, i = parse_at(0)
    rest, _ = _tok(s, i)
    if rest:
        raise ValueError(f"trailing {rest}")
    if node_count(tree) > int(FORGE_MAX_NODES) or tree_depth(tree) > int(FORGE_MAX_DEPTH):
        raise ValueError("formula exceeds depth/nodes")
    return tree


def walk_tokens(tree) -> tuple[list[str], list[str]]:
    prims: list[str] = []
    ops: list[str] = []

    def rec(n):
        k = n[0]
        if k == "T":
            prims.append(n[1])
        elif k == "U":
            ops.append(n[1])
            rec(n[2])
        elif k == "S":
            ops.append(n[1])
            rec(n[3])
        elif k == "B":
            ops.append(n[1])
            rec(n[2])
            rec(n[3])

    rec(tree)
    return prims, ops


def _locs(tree) -> list[tuple]:
    out = [()]

    def rec(n, path):
        k = n[0]
        if k == "T":
            return
        if k == "U":
            p = path + (0,)
            out.append(p)
            rec(n[2], p)
        elif k == "S":
            p = path + (0,)
            out.append(p)
            rec(n[3], p)
        elif k == "B":
            p0 = path + (0,)
            p1 = path + (1,)
            out.append(p0)
            rec(n[2], p0)
            out.append(p1)
            rec(n[3], p1)

    rec(tree, ())
    return out


def _get(tree, path):
    n = tree
    for i in path:
        k = n[0]
        if k in ("U", "S"):
            n = n[-1]
        elif k == "B":
            n = n[2] if i == 0 else n[3]
        else:
            raise ValueError("bad path")
    return n


def _set(tree, path, new):
    if not path:
        return new
    i, rest = path[0], path[1:]
    k = tree[0]
    if k == "U":
        return ("U", tree[1], _set(tree[2], rest, new))
    if k == "S":
        return ("S", tree[1], tree[2], _set(tree[3], rest, new))
    if k == "B":
        if i == 0:
            return ("B", tree[1], _set(tree[2], rest, new), tree[3])
        return ("B", tree[1], tree[2], _set(tree[3], rest, new))
    raise ValueError("bad path")


def _valid(tree) -> bool:
    try:
        return node_count(tree) <= int(FORGE_MAX_NODES) and tree_depth(tree) <= int(FORGE_MAX_DEPTH)
    except Exception:
        return False


def gen_tree(rng: np.random.Generator, max_depth: int, full: bool):
    n_term = len(FORGE_PRIMITIVES)
    n_fun = len(FORGE_UNARY_OPS) + len(FORGE_BINARY_OPS) + len(FORGE_TS_OPS)
    p_term = n_term / float(n_term + n_fun)

    def rec(depth: int):
        if depth >= max_depth or (not full and depth >= 1 and rng.random() < p_term):
            return ("T", FORGE_PRIMITIVES[int(rng.integers(n_term))])
        u = int(rng.integers(n_fun))
        if u < len(FORGE_UNARY_OPS):
            op = FORGE_UNARY_OPS[u]
            return ("U", op, rec(depth + 1))
        u -= len(FORGE_UNARY_OPS)
        if u < len(FORGE_BINARY_OPS):
            op = FORGE_BINARY_OPS[u]
            return ("B", op, rec(depth + 1), rec(depth + 1))
        op = FORGE_TS_OPS[u - len(FORGE_BINARY_OPS)]
        k = int(FORGE_TS_K[int(rng.integers(len(FORGE_TS_K)))])
        return ("S", op, k, rec(depth + 1))

    for _ in range(40):
        t = rec(1)
        if _valid(t):
            return t
    return ("T", "ret_14")


def init_population(rng: np.random.Generator, n: int) -> list:
    d0, d1 = int(FORGE_INIT_DEPTH[0]), int(FORGE_INIT_DEPTH[1])
    depths = list(range(d0, d1 + 1))
    pop = []
    for i in range(int(n)):
        depth = depths[i % len(depths)]
        full = ((i // len(depths)) % 2) == 0
        pop.append(gen_tree(rng, depth, full))
    return pop


def crossover(rng: np.random.Generator, a, b):
    for _ in range(12):
        pa = _locs(a)
        pb = _locs(b)
        sa = _get(a, pa[int(rng.integers(len(pa)))])
        sb = _get(b, pb[int(rng.integers(len(pb)))])
        ca = _set(a, pa[int(rng.integers(len(pa)))], sb)
        cb = _set(b, pb[int(rng.integers(len(pb)))], sa)
        if _valid(ca) and _valid(cb):
            return ca, cb
    return a, b


def subtree_mutate(rng: np.random.Generator, tree):
    for _ in range(12):
        locs = _locs(tree)
        path = locs[int(rng.integers(len(locs)))]
        grow_d = int(rng.integers(1, 5))
        new = gen_tree(rng, grow_d, full=False)
        cand = _set(tree, path, new)
        if _valid(cand):
            return cand
    return tree


def point_mutate(rng: np.random.Generator, tree):
    locs = _locs(tree)
    path = locs[int(rng.integers(len(locs)))]
    node = _get(tree, path)
    k = node[0]
    if k == "T":
        name = FORGE_PRIMITIVES[int(rng.integers(len(FORGE_PRIMITIVES)))]
        new = ("T", name)
    elif k == "U":
        op = FORGE_UNARY_OPS[int(rng.integers(len(FORGE_UNARY_OPS)))]
        new = ("U", op, node[2])
    elif k == "B":
        op = FORGE_BINARY_OPS[int(rng.integers(len(FORGE_BINARY_OPS)))]
        new = ("B", op, node[2], node[3])
    else:
        op = FORGE_TS_OPS[int(rng.integers(len(FORGE_TS_OPS)))]
        kk = int(FORGE_TS_K[int(rng.integers(len(FORGE_TS_K)))])
        new = ("S", op, kk, node[3])
    cand = _set(tree, path, new)
    return cand if _valid(cand) else tree


# ---------------------------------------------------------------------------
# Array operators
# ---------------------------------------------------------------------------

def _pdiv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _clean_arr(a)
    b = _clean_arr(b)
    out = np.empty_like(a, dtype=np.float64)
    small = np.abs(b) < float(FORGE_PDIV_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(small, 1.0, a / b)
    return _clean_arr(out)


def _cs_rank_impl(x, mask):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float64)
    out[:] = np.nan
    for t in range(T):
        n = 0
        idx = np.empty(N, dtype=np.int32)
        val = np.empty(N, dtype=np.float64)
        for j in range(N):
            if mask[t, j] and np.isfinite(x[t, j]):
                idx[n] = j
                val[n] = x[t, j]
                n += 1
        if n < 2:
            continue
        nf = float(n)
        for a in range(n):
            lt = 0
            eq = 0
            va = val[a]
            for b in range(n):
                vb = val[b]
                if vb < va:
                    lt += 1
                elif vb == va:
                    eq += 1
            avg = lt + 0.5 * (eq + 1.0)
            out[t, idx[a]] = avg / nf
    return out


def _cs_z_impl(x, mask, ddof):
    T, N = x.shape
    out = np.empty((T, N), dtype=np.float64)
    out[:] = np.nan
    for t in range(T):
        n = 0
        s = 0.0
        idx = np.empty(N, dtype=np.int32)
        val = np.empty(N, dtype=np.float64)
        for j in range(N):
            if mask[t, j] and np.isfinite(x[t, j]):
                idx[n] = j
                val[n] = x[t, j]
                s += x[t, j]
                n += 1
        if n < 2:
            continue
        mu = s / float(n)
        var = 0.0
        for a in range(n):
            d = val[a] - mu
            var += d * d
        denom = float(n - ddof)
        sd = np.sqrt(var / denom) if denom > 0.0 else 0.0
        for a in range(n):
            if sd == 0.0 or not np.isfinite(sd):
                out[t, idx[a]] = 0.0
            else:
                out[t, idx[a]] = (val[a] - mu) / sd
    return out


if _HAS_NUMBA:
    _cs_rank_impl = njit(cache=True)(_cs_rank_impl)
    _cs_z_impl = njit(cache=True)(_cs_z_impl)


def _cs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = _clean_arr(x)
    m = np.ascontiguousarray(mask, dtype=np.bool_)
    return _cs_rank_impl(np.ascontiguousarray(x, dtype=np.float64), m)


def _cs_z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = _clean_arr(x)
    m = np.ascontiguousarray(mask, dtype=np.bool_)
    return _cs_z_impl(np.ascontiguousarray(x, dtype=np.float64), m, int(FORGE_CS_DDOF))


def _ts_mean(x: np.ndarray, k: int) -> np.ndarray:
    x = _clean_arr(x)
    inf = np.isfinite(x)
    x0 = np.where(inf, x, 0.0)
    c = np.cumsum(x0, axis=0)
    n = np.cumsum(inf.astype(np.int32), axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if k <= 0 or k > T:
        return out
    s = c[k - 1 :] - np.vstack([np.zeros((1, N), dtype=c.dtype), c[: T - k]])
    cnt = n[k - 1 :] - np.vstack([np.zeros((1, N), dtype=n.dtype), n[: T - k]])
    good = cnt == k
    with np.errstate(invalid="ignore", divide="ignore"):
        out[k - 1 :] = np.where(good, s / np.maximum(cnt, 1), np.nan)
    return out


def _ts_std(x: np.ndarray, k: int) -> np.ndarray:
    x = _clean_arr(x)
    inf = np.isfinite(x)
    x0 = np.where(inf, x, 0.0)
    x2 = x0 * x0
    c = np.cumsum(x0, axis=0)
    c2 = np.cumsum(x2, axis=0)
    n = np.cumsum(inf.astype(np.int32), axis=0)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if k <= 1 or k > T:
        return out
    s = c[k - 1 :] - np.vstack([np.zeros((1, N), dtype=c.dtype), c[: T - k]])
    s2 = c2[k - 1 :] - np.vstack([np.zeros((1, N), dtype=c2.dtype), c2[: T - k]])
    cnt = n[k - 1 :] - np.vstack([np.zeros((1, N), dtype=n.dtype), n[: T - k]])
    good = cnt == k
    # ddof=1
    var = (s2 - (s * s) / np.maximum(cnt, 1)) / np.maximum(cnt - 1, 1)
    var = np.where(var < 0, 0.0, var)
    out[k - 1 :] = np.where(good, np.sqrt(var), np.nan)
    return out


def _ts_minmax(x: np.ndarray, k: int, op: str) -> np.ndarray:
    x = _clean_arr(x)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if k <= 0 or k > T:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.full((k - 1, N), np.nan, dtype=np.float64)
    z = np.vstack([pad, x])
    w = sliding_window_view(z, k, axis=0)
    cnt = np.isfinite(w).sum(axis=-1)
    if op == "ts_max":
        filled = np.where(np.isfinite(w), w, -np.inf)
        val = filled.max(axis=-1)
    else:
        filled = np.where(np.isfinite(w), w, np.inf)
        val = filled.min(axis=-1)
    out = np.where(cnt == k, val, np.nan)
    return out[:T]


def _ts_rank(x: np.ndarray, k: int) -> np.ndarray:
    x = _clean_arr(x)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if k <= 0 or k > T:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.full((k - 1, N), np.nan, dtype=np.float64)
    z = np.vstack([pad, x])
    w = sliding_window_view(z, k, axis=0)[:T]
    cur = w[:, :, -1]
    finite = np.isfinite(w)
    cnt = finite.sum(axis=-1)
    le = finite & (w <= cur[:, :, None])
    nle = le.sum(axis=-1)
    good = (cnt == k) & np.isfinite(cur)
    out = np.where(good, nle / np.maximum(cnt, 1), np.nan)
    return out


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    x = _clean_arr(x)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k <= 0 or k >= x.shape[0]:
        return out
    out[k:] = x[:-k]
    return out


def eval_tree(tree, prims: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    k = tree[0]
    if k == "T":
        return _clean_arr(prims[tree[1]])
    if k == "U":
        x = eval_tree(tree[2], prims, mask)
        op = tree[1]
        if op == "abs":
            return np.abs(x)
        if op == "neg":
            return -x
        if op == "log1pabs":
            return np.log1p(np.abs(x))
        if op == "rank_cs":
            return _cs_rank(x, mask)
        if op == "z_cs":
            return _cs_z(x, mask)
        if op == "phi_cs":
            return _ndtr(_cs_z(x, mask))
        raise ValueError(op)
    if k == "S":
        x = eval_tree(tree[3], prims, mask)
        op, kk = tree[1], int(tree[2])
        if op == "ts_mean":
            return _ts_mean(x, kk)
        if op == "ts_std":
            return _ts_std(x, kk)
        if op == "ts_max":
            return _ts_minmax(x, kk, "ts_max")
        if op == "ts_min":
            return _ts_minmax(x, kk, "ts_min")
        if op == "ts_rank":
            return _ts_rank(x, kk)
        if op == "lag":
            return _lag(x, kk)
        raise ValueError(op)
    if k == "B":
        a = eval_tree(tree[2], prims, mask)
        b = eval_tree(tree[3], prims, mask)
        op = tree[1]
        if op == "add":
            return _clean_arr(a + b)
        if op == "sub":
            return _clean_arr(a - b)
        if op == "mul":
            return _clean_arr(a * b)
        if op == "pdiv":
            return _pdiv(a, b)
        if op == "min":
            return _clean_arr(np.minimum(a, b))
        if op == "max":
            return _clean_arr(np.maximum(a, b))
        raise ValueError(op)
    raise ValueError(k)


# ---------------------------------------------------------------------------
# Book engine
# ---------------------------------------------------------------------------

def _book_daily_impl(
    score,
    univ_idx,
    n_univ,
    fwd,
    rebal,
    listed,
    k,
    cost_side,
):
    T = score.shape[0]
    N = score.shape[1]
    maxu = univ_idx.shape[1]
    w = np.zeros(N, dtype=np.float64)
    rets = np.full(T - 1, np.nan, dtype=np.float64)
    n_names = np.zeros(T - 1, dtype=np.float64)
    turnover = np.zeros(T - 1, dtype=np.float64)
    forced_n = 0
    forced_w = 0.0
    kk = int(k)
    cside = float(cost_side) * 1e-4
    for t in range(T - 1):
        w_prev = w.copy()
        dead_w = 0.0
        for j in range(N):
            if w[j] > 0.0:
                ok = listed[t, j] and np.isfinite(fwd[t, j])
                if not ok:
                    dead_w += w[j]
                    w[j] = 0.0
        if dead_w > 0.0:
            forced_n += 1
            forced_w += dead_w
        s = 0.0
        for j in range(N):
            s += w[j]
        if s > 0.0:
            for j in range(N):
                w[j] = w[j] / s
        do_rebal = bool(rebal[t]) or (s == 0.0)
        if do_rebal:
            nu = int(n_univ[t])
            nval = 0
            idxs = np.empty(maxu, dtype=np.int32)
            scs = np.empty(maxu, dtype=np.float64)
            for u in range(nu):
                j = int(univ_idx[t, u])
                if j < 0:
                    continue
                sc = score[t, j]
                if listed[t, j] and np.isfinite(sc) and np.isfinite(fwd[t, j]):
                    idxs[nval] = j
                    scs[nval] = sc
                    nval += 1
            for j in range(N):
                w[j] = 0.0
            if nval <= 0:
                n2 = 0
                for u in range(nu):
                    j = int(univ_idx[t, u])
                    if j < 0:
                        continue
                    if listed[t, j] and np.isfinite(fwd[t, j]):
                        idxs[n2] = j
                        n2 += 1
                if n2 > 0:
                    ww = 1.0 / float(n2)
                    for ii in range(n2):
                        w[int(idxs[ii])] = ww
            else:
                take = kk if kk < nval else nval
                taken = np.zeros(nval, dtype=np.uint8)
                for _t in range(take):
                    best = -1
                    bestv = -1.0e300
                    for i in range(nval):
                        if taken[i] == 0 and scs[i] > bestv:
                            bestv = scs[i]
                            best = i
                    if best < 0:
                        break
                    taken[best] = 1
                ntake = 0
                for i in range(nval):
                    if taken[i] == 1:
                        ntake += 1
                if ntake > 0:
                    ww = 1.0 / float(ntake)
                    for i in range(nval):
                        if taken[i] == 1:
                            w[int(idxs[i])] = ww
        dw = 0.0
        nn = 0.0
        r = 0.0
        for j in range(N):
            dw += abs(w[j] - w_prev[j])
            if w[j] > 0.0:
                nn += 1.0
                f = fwd[t, j]
                if np.isfinite(f):
                    r += w[j] * f
        cost = cside * dw
        rets[t] = r - cost
        n_names[t] = nn
        turnover[t] = 0.5 * dw
    return rets, n_names, turnover, forced_n, forced_w


book_daily = njit(cache=True)(_book_daily_impl) if _HAS_NUMBA else _book_daily_impl


def ann_log_wealth(rets: np.ndarray) -> float:
    r = np.asarray(rets, dtype=np.float64)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return float("-inf")
    if np.any(r <= -0.999999):
        return float("-inf")
    return float(np.log1p(r).sum() * (float(ANNUALIZATION) / float(r.size)))


def make_rebal(n: int, is_monday: np.ndarray, mode: str) -> np.ndarray:
    out = np.zeros(n, dtype=np.bool_)
    if mode == "daily":
        out[:] = True
        return out
    out[:] = is_monday[:n]
    if n:
        out[0] = True
    return out


@dataclass
class Cube:
    dates: pd.DatetimeIndex
    ids: np.ndarray
    prims: dict
    mask_dv: np.ndarray
    mask_mcap: np.ndarray
    fwd: np.ndarray
    listed: np.ndarray
    is_monday: np.ndarray
    vol_bucket: np.ndarray
    univ_dv_idx: np.ndarray
    n_dv: np.ndarray
    univ_mcap_idx: np.ndarray
    n_mcap: np.ndarray
    btc_id: int
    btc_col: int
    meta: dict = field(default_factory=dict)

    def slice_window(self, start: str, end: str) -> "Cube":
        dates = self.dates
        t0 = _as_utc(start)
        t1 = _as_utc(end)
        form = np.asarray((dates >= t0) & (dates <= t1))
        if not bool(form.any()):
            raise RuntimeError(f"empty window {start}→{end}")
        idx = np.flatnonzero(form)
        i0, i_last = int(idx[0]), int(idx[-1])
        j1 = min(i_last + 2, len(dates))
        sl = slice(i0, j1)
        return Cube(
            dates=dates[sl],
            ids=self.ids,
            prims={k: v[sl] for k, v in self.prims.items()},
            mask_dv=self.mask_dv[sl],
            mask_mcap=self.mask_mcap[sl],
            fwd=self.fwd[sl],
            listed=self.listed[sl],
            is_monday=self.is_monday[sl],
            vol_bucket=self.vol_bucket[sl],
            univ_dv_idx=self.univ_dv_idx[sl],
            n_dv=self.n_dv[sl],
            univ_mcap_idx=self.univ_mcap_idx[sl],
            n_mcap=self.n_mcap[sl],
            btc_id=self.btc_id,
            btc_col=self.btc_col,
            meta={**self.meta, "window": (str(start), str(end))},
        )


def pack_univ(mask: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    T, _N = mask.shape
    idx = np.full((T, int(max_n)), -1, dtype=np.int32)
    n = np.zeros(T, dtype=np.int32)
    for t in range(T):
        js = np.flatnonzero(mask[t])
        k = min(int(js.size), int(max_n))
        if k:
            idx[t, :k] = js[:k].astype(np.int32)
        n[t] = k
    return idx, n


def _univ_pack(cube: Cube, univ: str):
    if univ == "dv100":
        return cube.univ_dv_idx, cube.n_dv, cube.mask_dv
    if univ == "mcap50":
        return cube.univ_mcap_idx, cube.n_mcap, cube.mask_mcap
    raise ValueError(univ)


def btc_simple_from_cube(cube: Cube) -> pd.Series:
    px = np.asarray(cube.prims["close"][:, cube.btc_col], dtype=float)
    if px.size < 2:
        return pd.Series(dtype=float)
    r = px[1:] / px[:-1] - 1.0
    idx = _utc_idx(cube.dates[1 : 1 + len(r)])
    return pd.Series(r, index=idx, dtype=float)


def summarize_net_book(packed: dict, cube: Cube) -> dict:
    from btcb.book import summarize_book

    rets = packed["daily_ret"]
    btc = btc_simple_from_cube(cube).reindex(rets.index).fillna(0.0)
    z = pd.Series(0.0, index=rets.index)
    nn = packed.get("n_names")
    if not isinstance(nn, pd.Series):
        nn = pd.Series(np.nan, index=rets.index)
    to = packed.get("turnover")
    to_list = list(np.asarray(to, dtype=float)) if isinstance(to, pd.Series) else []
    out = summarize_book(rets.fillna(0.0), btc, z, nn.reindex(rets.index).fillna(0.0), to_list)
    out["label"] = f"{packed.get('univ')}_{packed.get('rebal')}_k{packed.get('k')}"
    out["ann_log_wealth"] = packed.get("ann_log_wealth")
    out["forced_n"] = packed.get("forced_n")
    out["forced_weight_sum"] = packed.get("forced_weight_sum")
    out["total"] = out.get("book_total")
    out["cagr"] = out.get("book_cagr")
    out["sharpe"] = out.get("book_sharpe")
    rel_eq = out.get("rel_equity")
    if isinstance(rel_eq, pd.Series) and len(rel_eq):
        out["rel_total"] = float(rel_eq.iloc[-1] - 1.0)
    m = rets.notna() & btc.notna()
    out["corr_btc"] = float(rets[m].corr(btc[m])) if int(m.sum()) >= 10 else float("nan")
    return out


def run_ew_book(cube: Cube, univ: str, rebal_mode: str = "daily") -> dict:
    T, N = cube.fwd.shape
    score = np.zeros((T, N), dtype=np.float64)
    k = 100 if univ == "dv100" else int(FORGE_MCAP_N)
    return run_book(score, cube, k=k, rebal_mode=rebal_mode, univ=univ)


def run_book(
    score: np.ndarray,
    cube: Cube,
    *,
    k: int,
    rebal_mode: str,
    univ: str,
    cost_bps: float = FORGE_COST_BPS,
) -> dict:
    uidx, n_u, _mask = _univ_pack(cube, univ)
    rebal = make_rebal(score.shape[0], cube.is_monday, rebal_mode)
    rets, nn, to, fn, fw = book_daily(
        np.ascontiguousarray(score, dtype=np.float64),
        np.ascontiguousarray(uidx, dtype=np.int32),
        np.ascontiguousarray(n_u, dtype=np.int32),
        np.ascontiguousarray(cube.fwd, dtype=np.float64),
        np.ascontiguousarray(rebal, dtype=np.bool_),
        np.ascontiguousarray(cube.listed, dtype=np.bool_),
        int(k),
        float(cost_bps),
    )
    if cube.dates.size >= 2:
        idx = cube.dates[1 : 1 + len(rets)]
    else:
        idx = cube.dates[:0]
    s = pd.Series(np.asarray(rets, dtype=float), index=_utc_idx(idx), dtype=float)
    return {
        "daily_ret": s,
        "n_names": pd.Series(np.asarray(nn, dtype=float), index=s.index),
        "turnover": pd.Series(np.asarray(to, dtype=float), index=s.index),
        "ann_log_wealth": ann_log_wealth(np.asarray(rets)),
        "forced_n": int(fn),
        "forced_weight_sum": float(fw),
        "k": int(k),
        "rebal": str(rebal_mode),
        "univ": str(univ),
    }


P_SPEC = tuple(
    (int(k), str(rb), str(u))
    for u in ("dv100", "mcap50")
    for rb in FORGE_REBAL_SET
    for k in FORGE_K_SET
)
assert len(P_SPEC) == 12, len(P_SPEC)


def fitness_details(tree, cube: Cube) -> dict:
    nodes = node_count(tree)
    try:
        score_dv = eval_tree(tree, cube.prims, cube.mask_dv)
        score_mc = eval_tree(tree, cube.prims, cube.mask_mcap)
    except Exception:
        return {
            "fitness": float("-inf"),
            "median": float("-inf"),
            "spread": float("inf"),
            "nodes": int(nodes),
            "ann": [float("-inf")] * 12,
            "discarded": True,
            "reason": "eval",
        }
    anns = []
    headline = None
    for k, rb, univ in P_SPEC:
        sc = score_dv if univ == "dv100" else score_mc
        packed = run_book(sc, cube, k=k, rebal_mode=rb, univ=univ)
        anns.append(float(packed["ann_log_wealth"]))
        if k == int(FORGE_HEADLINE_K) and rb == FORGE_HEADLINE_REBAL and univ == FORGE_HEADLINE_UNIV:
            headline = packed
    arr = np.asarray(anns, dtype=float)
    if not np.all(np.isfinite(arr)):
        return {
            "fitness": float("-inf"),
            "median": float("-inf"),
            "spread": float("inf"),
            "nodes": int(nodes),
            "ann": [float(x) if np.isfinite(x) else None for x in arr],
            "discarded": True,
            "reason": "nan_book",
            "headline": headline,
        }
    spread = float(arr.max() - arr.min())
    if spread > float(FORGE_KNIFE_SPREAD):
        return {
            "fitness": float("-inf"),
            "median": float(np.median(arr)),
            "spread": spread,
            "nodes": int(nodes),
            "ann": [float(x) for x in arr],
            "discarded": True,
            "reason": "knife",
            "headline": headline,
        }
    med = float(np.median(arr))
    fit = med - float(FORGE_LAMBDA_C) * (float(nodes) / float(FORGE_MAX_NODES))
    return {
        "fitness": float(fit),
        "median": med,
        "spread": spread,
        "nodes": int(nodes),
        "ann": [float(x) for x in arr],
        "discarded": False,
        "reason": "",
        "headline": headline,
    }


def fitness_value(tree, cube: Cube) -> float:
    return float(fitness_details(tree, cube)["fitness"])


# ---------------------------------------------------------------------------
# GP loop
# ---------------------------------------------------------------------------

_WORKER_CUBE: Cube | None = None


def _worker_fit(tree) -> dict:
    assert _WORKER_CUBE is not None
    d = fitness_details(tree, _WORKER_CUBE)
    d.pop("headline", None)
    d["expr"] = formula_str(tree)
    return d


def _n_jobs(requested: int | None = None) -> int:
    cpu = int(os.cpu_count() or 1)
    n = int(requested) if requested else cpu
    return max(1, min(n, int(FORGE_MAX_CONCURRENCY), cpu))


def eval_population(
    pop: list,
    cube: Cube,
    *,
    n_jobs: int | None = None,
    ping=None,
    chunk: int = 64,
) -> list[dict]:
    global _WORKER_CUBE
    _WORKER_CUBE = cube
    jobs = _n_jobs(n_jobs)
    out: list[dict] = [None] * len(pop)  # type: ignore
    if pop:
        _worker_fit(pop[0])
        if ping:
            ping("engine warmup")
    if jobs <= 1 or len(pop) <= 4:
        for i, t in enumerate(pop):
            out[i] = _worker_fit(t)
            if ping and (i % 8 == 0):
                ping(f"eval {i}/{len(pop)}")
        return out
    import multiprocessing as mp

    ctx = mp.get_context("fork")

    def _init():
        global _WORKER_CUBE
        _WORKER_CUBE = cube

    with ctx.Pool(processes=jobs, initializer=_init) as pool:
        for i0 in range(0, len(pop), int(chunk)):
            part = pop[i0 : i0 + int(chunk)]
            res = pool.map(_worker_fit, part, chunksize=max(1, len(part) // max(jobs, 1) or 1))
            out[i0 : i0 + len(res)] = res
            if ping:
                ping(f"eval {min(i0 + len(res), len(pop))}/{len(pop)}")
    return out


def tournament(rng: np.random.Generator, fits: np.ndarray, k: int) -> int:
    n = int(fits.size)
    k = min(int(k), n)
    ix = rng.integers(0, n, size=k)
    best = int(ix[0])
    bv = fits[best]
    for i in ix[1:]:
        i = int(i)
        v = fits[i]
        if v > bv or (v == bv and i < best):
            best = i
            bv = v
    return best


def evolve(
    cube: Cube,
    *,
    n_pop: int = FORGE_POP,
    n_gen: int = FORGE_GENS,
    seed: int = FORGE_SEED,
    checkpoint_path: Path | None = None,
    ping=None,
    n_jobs: int | None = None,
    t0: float | None = None,
    budget_sec: float | None = None,
    label: str = "mine",
) -> dict:
    rng = np.random.default_rng(int(seed))
    t0 = time.time() if t0 is None else float(t0)
    start_gen = 0
    pop = None
    history = []
    best_fit = float("-inf")
    flat = 0
    budget_flag = False
    if checkpoint_path and Path(checkpoint_path).exists():
        ck = json.loads(Path(checkpoint_path).read_text())
        if ck.get("label") == label:
            start_gen = int(ck.get("generation", -1)) + 1
            history = list(ck.get("history") or [])
            budget_flag = bool(ck.get("budget_flag"))
            exprs = [p["expr"] for p in ck.get("population") or []]
            if exprs:
                pop = [parse_formula(e) for e in exprs]
                last_fits = [p.get("fitness") for p in ck.get("population") or []]
                if last_fits:
                    best_fit = float(np.nanmax(np.asarray(last_fits, dtype=float)))
            rng_path = Path(checkpoint_path).with_suffix(".rng")
            if rng_path.exists():
                import pickle

                rng = np.random.default_rng()
                rng.bit_generator.state = pickle.loads(rng_path.read_bytes())
            if ping:
                ping(f"resume {label} gen={start_gen} n={len(pop or [])}")
    if pop is None:
        pop = init_population(rng, int(n_pop))
        start_gen = 0

    scored = None
    for gen in range(start_gen, int(n_gen)):
        if budget_sec is not None and (time.time() - t0) >= float(budget_sec):
            budget_flag = True
            if ping:
                ping(f"BUDGET flag at gen={gen}")
            break
        scored = eval_population(pop, cube, n_jobs=n_jobs, ping=ping)
        fits = np.asarray([s["fitness"] for s in scored], dtype=float)
        fits = np.where(np.isfinite(fits), fits, -1e18)
        gi = int(np.argmax(fits))
        gfit = float(fits[gi])
        finite = fits[np.isfinite(fits) & (fits > -1e17)]
        rec = {
            "generation": gen,
            "best": gfit,
            "median": float(np.median(finite)) if finite.size else float("-inf"),
            "mean": float(np.mean(finite)) if finite.size else float("-inf"),
            "n_finite": int(finite.size),
            "best_expr": scored[gi]["expr"],
            "best_nodes": scored[gi]["nodes"],
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)
        if ping:
            ping(
                f"{label} gen={gen}/{n_gen - 1} best={gfit:.6f} med={rec['median']:.6f} "
                f"finite={rec['n_finite']} nodes={rec['best_nodes']}"
            )
        if gfit > best_fit + 1e-12:
            best_fit = gfit
            flat = 0
        else:
            flat += 1
        if checkpoint_path:
            _write_ckpt(
                checkpoint_path,
                generation=gen,
                pop=pop,
                scored=scored,
                history=history,
                label=label,
                budget_flag=budget_flag,
                seed=int(seed),
                rng=rng,
            )
        if flat >= int(FORGE_EARLY_STOP_GENS) and gen + 1 >= int(FORGE_EARLY_STOP_GENS):
            if ping:
                ping(f"{label} early stop gen={gen} flat={flat}")
            break
        if gen + 1 >= int(n_gen):
            break
        elite_i = gi
        next_pop = [pop[elite_i]]
        n = len(pop)
        while len(next_pop) < n:
            u = float(rng.random())
            if u < float(FORGE_P_CROSSOVER):
                i = tournament(rng, fits, int(FORGE_TOURNAMENT))
                j = tournament(rng, fits, int(FORGE_TOURNAMENT))
                a, b = crossover(rng, pop[i], pop[j])
                next_pop.append(a)
                if len(next_pop) < n:
                    next_pop.append(b)
            elif u < float(FORGE_P_CROSSOVER) + float(FORGE_P_SUBTREE):
                i = tournament(rng, fits, int(FORGE_TOURNAMENT))
                next_pop.append(subtree_mutate(rng, pop[i]))
            else:
                i = tournament(rng, fits, int(FORGE_TOURNAMENT))
                next_pop.append(point_mutate(rng, pop[i]))
        pop = next_pop[:n]

    if scored is None:
        scored = eval_population(pop, cube, n_jobs=n_jobs, ping=ping)
    return {
        "population": pop,
        "scored": scored,
        "history": history,
        "budget_flag": budget_flag,
        "label": label,
        "elapsed_s": time.time() - t0,
    }


def _write_ckpt(path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pop = kw["pop"]
    scored = kw["scored"]
    rows = []
    for t, s in zip(pop, scored):
        rows.append(
            {
                "expr": s.get("expr") or formula_str(t),
                "fitness": s.get("fitness"),
                "median": s.get("median"),
                "spread": s.get("spread"),
                "nodes": s.get("nodes"),
                "discarded": s.get("discarded"),
            }
        )
    payload = {
        "generation": int(kw["generation"]),
        "label": kw["label"],
        "best_fitness": None if not rows else max((r["fitness"] if r["fitness"] is not None and np.isfinite(r["fitness"]) else -1e18) for r in rows),
        "best_formula": None if not rows else max(rows, key=lambda r: r["fitness"] if r["fitness"] is not None and np.isfinite(r["fitness"]) else -1e18)["expr"],
        "population": rows,
        "history": kw["history"],
        "budget_flag": bool(kw["budget_flag"]),
        "seed": int(kw["seed"]),
        "n_pop": len(rows),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)
    rng = kw.get("rng")
    if rng is not None:
        import pickle

        path.with_suffix(".rng").write_bytes(pickle.dumps(rng.bit_generator.state))


# ---------------------------------------------------------------------------
# SELECT / null / census
# ---------------------------------------------------------------------------

def select_champions(
    pop: list,
    mine_scored: list[dict],
    select_cube: Cube,
    *,
    ping=None,
    n_jobs: int | None = None,
) -> dict:
    order = sorted(
        range(len(mine_scored)),
        key=lambda i: (
            mine_scored[i]["fitness"] if np.isfinite(mine_scored[i]["fitness"]) else -1e18
        ),
        reverse=True,
    )
    top_i = []
    seen = set()
    for i in order:
        expr = mine_scored[i].get("expr") or formula_str(pop[i])
        if expr in seen:
            continue
        if not np.isfinite(mine_scored[i]["fitness"]):
            continue
        seen.add(expr)
        top_i.append(i)
        if len(top_i) >= int(FORGE_SELECT_TOP):
            break
    trees = [pop[i] for i in top_i]
    if ping:
        ping(f"SELECT scoring n={len(trees)}")
    sel_scored = eval_population(trees, select_cube, n_jobs=n_jobs, ping=ping)
    headline_rets = []
    for t, s, mi in zip(trees, sel_scored, top_i):
        det = fitness_details(t, select_cube)
        hl = det.get("headline") or {}
        headline_rets.append(hl.get("daily_ret"))
        s["select_fitness"] = det["fitness"]
        s["select_median"] = det["median"]
        s["select_spread"] = det["spread"]
        s["mine_fitness"] = mine_scored[mi]["fitness"]
        s["expr"] = formula_str(t)
        s["nodes"] = node_count(t)

    # greedy diversity on headline SELECT daily PnL
    ranked = sorted(
        range(len(trees)),
        key=lambda i: sel_scored[i]["select_fitness"] if np.isfinite(sel_scored[i]["select_fitness"]) else -1e18,
        reverse=True,
    )
    chosen: list[int] = []
    skipped = []
    for i in ranked:
        r = headline_rets[i]
        if r is None or len(r) < 10 or not np.isfinite(sel_scored[i]["select_fitness"]):
            skipped.append({"expr": sel_scored[i]["expr"], "reason": "no_headline"})
            continue
        ok = True
        corrs = []
        for j in chosen:
            a, b = r.align(headline_rets[j], join="inner")
            if len(a) < 10:
                continue
            c = float(pd.Series(a).corr(pd.Series(b)))
            corrs.append(c)
            if np.isfinite(c) and c >= float(FORGE_DIVERSITY_CORR):
                ok = False
                break
        if ok:
            chosen.append(i)
            sel_scored[i]["max_pairwise_corr"] = float(max(corrs)) if corrs else 0.0
        else:
            skipped.append(
                {
                    "expr": sel_scored[i]["expr"],
                    "reason": "corr",
                    "corr": float(max(corrs)) if corrs else None,
                }
            )
        if len(chosen) >= int(FORGE_N_CHAMPIONS):
            break
    champs = []
    for rank, i in enumerate(chosen, start=1):
        champs.append(
            {
                "rank": rank,
                "expr": sel_scored[i]["expr"],
                "nodes": sel_scored[i]["nodes"],
                "mine_fitness": float(sel_scored[i]["mine_fitness"]),
                "select_fitness": float(sel_scored[i]["select_fitness"]),
                "select_median": sel_scored[i].get("select_median"),
                "select_spread": sel_scored[i].get("select_spread"),
                "max_pairwise_corr": sel_scored[i].get("max_pairwise_corr"),
            }
        )
    decay = []
    for j, i in enumerate(top_i[:20]):
        decay.append(
            {
                "rank_mine": j + 1,
                "expr": formula_str(pop[i]),
                "nodes": node_count(pop[i]),
                "mine_fitness": mine_scored[i]["fitness"],
                "select_fitness": sel_scored[j]["select_fitness"] if j < len(sel_scored) else None,
            }
        )
    return {
        "champions": champs,
        "top50": [
            {
                "expr": sel_scored[i]["expr"],
                "nodes": sel_scored[i]["nodes"],
                "mine_fitness": sel_scored[i]["mine_fitness"],
                "select_fitness": sel_scored[i]["select_fitness"],
            }
            for i in range(len(sel_scored))
        ],
        "decay_top20": decay,
        "skipped": skipped,
    }


def ingredient_census(rows: list[dict]) -> dict:
    from collections import Counter

    pc, oc = Counter(), Counter()
    for r in rows:
        try:
            t = parse_formula(r["expr"])
        except Exception:
            continue
        p, o = walk_tokens(t)
        pc.update(p)
        oc.update(o)
    return {
        "n_formulas": len(rows),
        "primitives": pc.most_common(),
        "operators": oc.most_common(),
        "one_liner": _census_line(pc, oc),
    }


def _census_line(pc, oc) -> str:
    top_p = ", ".join(f"{n}×{c}" for n, c in pc.most_common(5)) or "none"
    top_o = ", ".join(f"{n}×{c}" for n, c in oc.most_common(5)) or "none"
    return f"primitives[{top_p}] operators[{top_o}]"


def shuffle_fwd_vol(cube: Cube, seed_seq=(FORGE_SEED, 7)) -> Cube:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_seq)))
    fwd = np.array(cube.fwd, copy=True)
    T, N = fwd.shape
    vb = cube.vol_bucket
    for t in range(T):
        m = np.isfinite(fwd[t])
        if int(m.sum()) < 2:
            continue
        buckets = vb[t]
        for b in range(5):
            ix = np.flatnonzero(m & (buckets == b))
            if ix.size < 2:
                continue
            perm = rng.permutation(ix.size)
            fwd[t, ix] = fwd[t, ix][perm]
    out = Cube(
        dates=cube.dates,
        ids=cube.ids,
        prims=cube.prims,
        mask_dv=cube.mask_dv,
        mask_mcap=cube.mask_mcap,
        fwd=fwd,
        listed=cube.listed,
        is_monday=cube.is_monday,
        vol_bucket=cube.vol_bucket,
        univ_dv_idx=cube.univ_dv_idx,
        n_dv=cube.n_dv,
        univ_mcap_idx=cube.univ_mcap_idx,
        n_mcap=cube.n_mcap,
        btc_id=cube.btc_id,
        btc_col=cube.btc_col,
        meta={**cube.meta, "null": True},
    )
    return out


def null_select_floor(
    null_pop: list,
    null_scored: list[dict],
    select_cube: Cube,
    *,
    ping=None,
    n_jobs: int | None = None,
) -> dict:
    order = sorted(
        range(len(null_scored)),
        key=lambda i: null_scored[i]["fitness"] if np.isfinite(null_scored[i]["fitness"]) else -1e18,
        reverse=True,
    )
    take = [i for i in order if np.isfinite(null_scored[i]["fitness"])][: int(FORGE_SELECT_TOP)]
    trees = [null_pop[i] for i in take]
    if not trees:
        return {"best_select_fitness": float("-inf"), "n": 0, "best_expr": None}
    if ping:
        ping(f"null SELECT n={len(trees)}")
    scored = eval_population(trees, select_cube, n_jobs=n_jobs, ping=ping)
    best_i = int(np.argmax([s["fitness"] if np.isfinite(s["fitness"]) else -1e18 for s in scored]))
    return {
        "best_select_fitness": scored[best_i]["fitness"],
        "best_expr": scored[best_i]["expr"],
        "best_nodes": scored[best_i]["nodes"],
        "n": len(scored),
        "top": [
            {"expr": s["expr"], "select_fitness": s["fitness"], "nodes": s["nodes"]}
            for s in sorted(scored, key=lambda x: x["fitness"] if np.isfinite(x["fitness"]) else -1e18, reverse=True)[:10]
        ],
    }


def champion_passes(book: dict, btc: dict) -> dict:
    book_tot = float(book.get("book_total") if book.get("book_total") is not None else book.get("total") or np.nan)
    btc_tot = float(btc.get("book_total") if btc.get("book_total") is not None else btc.get("total") or np.nan)
    rel = float(book.get("rel_sharpe") if book.get("rel_sharpe") is not None else np.nan)
    mdd = float(book.get("maxdd") if book.get("maxdd") is not None else np.nan)
    btc_mdd = float(btc.get("maxdd") if btc.get("maxdd") is not None else np.nan)
    a = bool(np.isfinite(book_tot) and np.isfinite(btc_tot) and book_tot >= btc_tot)
    b = bool(np.isfinite(rel) and rel > 0.0)
    c = bool(
        np.isfinite(mdd)
        and np.isfinite(btc_mdd)
        and abs(float(mdd)) <= float(FORGE_MAXDD_MULT) * abs(float(btc_mdd))
    )
    return {
        "a_total_ge_btc": a,
        "b_rel_sharpe_gt0": b,
        "c_maxdd_le": c,
        "pass": bool(a and b and c),
        "rel_sharpe": rel,
        "book_total": book_tot,
        "btc_total": btc_tot,
        "maxdd": mdd,
        "btc_maxdd": btc_mdd,
        "rel_strong": bool(np.isfinite(rel) and rel >= float(FORGE_REL_SHARPE_STRONG)),
    }


def mechanical_forge_verdict(per_champ: list[dict]) -> dict:
    n_pass = int(sum(1 for p in per_champ if p.get("pass")))
    strong_one = any(p.get("pass") and p.get("rel_strong") for p in per_champ)
    if n_pass >= 2 or (n_pass >= 1 and strong_one):
        label = "FORGE-STRONG"
    elif n_pass >= 1:
        label = "FORGE-ALIVE"
    else:
        label = "FORGE-DEAD"
    return {"label": label, "n_pass": n_pass, "n": len(per_champ)}


# ---------------------------------------------------------------------------
# Cubes I/O + prepare
# ---------------------------------------------------------------------------

def cube_dir(root: Path | None = None) -> Path:
    return Path(root or "/data/quant/btcb/forge")


def save_cube(cube: Cube, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dates": cube.dates.asi8,
        "ids": cube.ids,
        "mask_dv": cube.mask_dv,
        "mask_mcap": cube.mask_mcap,
        "fwd": cube.fwd,
        "listed": cube.listed,
        "is_monday": cube.is_monday,
        "vol_bucket": cube.vol_bucket,
        "univ_dv_idx": cube.univ_dv_idx,
        "n_dv": cube.n_dv,
        "univ_mcap_idx": cube.univ_mcap_idx,
        "n_mcap": cube.n_mcap,
        "btc_id": np.asarray(cube.btc_id),
        "btc_col": np.asarray(cube.btc_col),
    }
    for name, arr in cube.prims.items():
        payload[f"p_{name}"] = arr
    np.savez_compressed(path, **payload)
    meta = {
        "ids": [int(i) for i in cube.ids],
        "n_dates": int(len(cube.dates)),
        "n_ids": int(len(cube.ids)),
        "start": str(cube.dates.min().date()) if len(cube.dates) else None,
        "end": str(cube.dates.max().date()) if len(cube.dates) else None,
        "btc_id": int(cube.btc_id),
        "primitives": list(cube.prims),
        **(cube.meta or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_cube(path: Path) -> Cube:
    z = np.load(path, allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"], utc=True)).tz_convert("UTC").normalize()
    ids = np.asarray(z["ids"], dtype=np.int64)
    prims = {}
    for k in z.files:
        if k.startswith("p_"):
            prims[k[2:]] = np.asarray(z[k])
    missing = [p for p in FORGE_PRIMITIVES if p not in prims]
    if missing:
        raise RuntimeError(f"cube missing primitives {missing}")
    btc_id = int(np.asarray(z["btc_id"]).reshape(-1)[0])
    btc_col = int(np.asarray(z["btc_col"]).reshape(-1)[0])
    meta = {}
    jp = Path(path).with_suffix(".json")
    if jp.exists():
        meta = json.loads(jp.read_text())
    return Cube(
        dates=dates,
        ids=ids,
        prims=prims,
        mask_dv=np.asarray(z["mask_dv"], dtype=bool),
        mask_mcap=np.asarray(z["mask_mcap"], dtype=bool),
        fwd=np.asarray(z["fwd"]),
        listed=np.asarray(z["listed"], dtype=bool),
        is_monday=np.asarray(z["is_monday"], dtype=bool),
        vol_bucket=np.asarray(z["vol_bucket"], dtype=np.int8),
        univ_dv_idx=np.asarray(z["univ_dv_idx"], dtype=np.int32),
        n_dv=np.asarray(z["n_dv"], dtype=np.int32),
        univ_mcap_idx=np.asarray(z["univ_mcap_idx"], dtype=np.int32),
        n_mcap=np.asarray(z["n_mcap"], dtype=np.int32),
        btc_id=btc_id,
        btc_col=btc_col,
        meta=meta,
    )


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    w = df.pivot_table(index="date", columns="id", values=col, aggfunc="last").sort_index()
    w.index = _utc_idx(w.index)
    w.columns = [int(c) for c in w.columns]
    return w


def _align(wide: pd.DataFrame, index, columns) -> pd.DataFrame:
    w = wide.reindex(index=index, columns=columns)
    w.index = _utc_idx(w.index)
    return w


def _to_arr(df: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(df.to_numpy(dtype=np.float32))


def vol_buckets_from_std(std30: np.ndarray, listed: np.ndarray) -> np.ndarray:
    from btcb.model import vol_bucket_ids

    T, N = std30.shape
    out = np.full((T, N), -1, dtype=np.int8)
    for t in range(T):
        v = std30[t].astype(float)
        v = np.where(listed[t], v, np.nan)
        out[t] = vol_bucket_ids(v, n_bins=5).astype(np.int8)
    return out


def _pit_members_all(pit: pd.DataFrame) -> dict:
    """Floored PIT membership as-is (BTC kept if present)."""
    df = pit.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    df["id"] = df["id"].astype(int)
    out: dict = {}
    for d, v in df.groupby("date")["id"]:
        out[_as_utc(d)] = [int(x) for x in v]
    return out


def prepare_cubes(
    *,
    panel: pd.DataFrame,
    cleaned: pd.DataFrame,
    pit: pd.DataFrame,
    close: pd.DataFrame,
    btc_id: int,
    pos_path: Path | None = None,
    ping=None,
) -> Cube:
    def hb(msg):
        if ping:
            ping(msg)
        else:
            print(f"[HB] {msg}", flush=True)

    end = _as_utc(FORGE_JUDGE_END)
    start_cube = _as_utc(FORGE_CUBE_START)
    close = close.copy()
    close.index = _utc_idx(close.index)
    close = close[(close.index >= start_cube) & (close.index <= end)].sort_index()
    btc_ok = close[int(btc_id)].astype(float)
    close = close.loc[np.isfinite(btc_ok) & (btc_ok > 0)].copy()
    dates = close.index
    hb(f"hybrid close {close.shape} {dates.min().date()}→{dates.max().date()}")

    tagged, _tag_rows = tagged_peg_ids(cleaned)
    cmc_px = cmc_close_wide(cleaned).reindex(dates)
    mcap = mcap_wide(cleaned).reindex(dates)
    feat_cmc = {"ret90": cmc_px / cmc_px.shift(90) - 1.0}
    peg_heu = peg_heuristic_mask(feat_cmc["ret90"], thresh=MANUEL2_PEG_ABS_RET)
    mem_mcap = build_members(list(dates), mcap, tagged, peg_heu, int(btc_id), btc_ex=False, n=int(FORGE_MCAP_N))
    pit_mem = ffill_members(_pit_members_all(pit), list(dates))

    union = set()
    for d, ids in pit_mem.items():
        union.update(int(i) for i in ids)
    for d, ids in mem_mcap.items():
        union.update(int(i) for i in ids)
    union.add(int(btc_id))
    # restrict to columns that exist in close or mcap
    all_cols = set(int(c) for c in close.columns) | set(int(c) for c in mcap.columns)
    ids = np.asarray(sorted(i for i in union if i in all_cols), dtype=np.int64)
    hb(f"cube ids={len(ids)} union_raw={len(union)}")

    close_u = _align(close, dates, ids)
    mcap_u = _align(mcap, dates, ids)
    listed = np.isfinite(close_u.to_numpy(dtype=float)) & (close_u.to_numpy(dtype=float) > 0)
    px = close_u.astype(float)
    r1 = px.pct_change(fill_method=None)
    fwd = (px.shift(-1) / px - 1.0).to_numpy(dtype=np.float64)

    prims: dict[str, np.ndarray] = {}
    prims["close"] = _to_arr(px)
    dv_w = _align(_wide(cleaned, "dv") if "dv" in cleaned.columns else _wide(cleaned, "volume"), dates, ids)
    prims["volume"] = _to_arr(dv_w)
    prims["mcap"] = _to_arr(mcap_u)
    for k in FORGE_RET_K:
        prims[f"ret_{int(k)}"] = _to_arr(px / px.shift(int(k)) - 1.0)
    for k in FORGE_STD_K:
        prims[f"std_{int(k)}"] = _to_arr(
            r1.rolling(int(k), min_periods=int(k)).std(ddof=int(FORGE_STD_DDOF))
        )

    cmc_c = _align(cmc_px, dates, ids)
    high = _align(_wide(cleaned, "high"), dates, ids) if "high" in cleaned.columns else cmc_c
    low = _align(_wide(cleaned, "low"), dates, ids) if "low" in cleaned.columns else cmc_c
    for k in FORGE_HL_K:
        hh = high.rolling(int(k), min_periods=int(k)).max()
        ll = low.rolling(int(k), min_periods=int(k)).min()
        prims[f"dist_high_{int(k)}"] = _to_arr(cmc_c / hh.replace(0, np.nan) - 1.0)
        prims[f"dist_low_{int(k)}"] = _to_arr(cmc_c / ll.replace(0, np.nan) - 1.0)

    ath = px.expanding(min_periods=1).max()
    prims["dist_ath"] = _to_arr(px / ath.replace(0, np.nan) - 1.0)
    amihud = (r1.abs() / dv_w.replace(0, np.nan)).rolling(14, min_periods=5).mean()
    prims["amihud_14"] = _to_arr(amihud)
    prims["turnover"] = _to_arr(dv_w / mcap_u.replace(0, np.nan))

    first = cleaned.copy()
    first["date"] = pd.to_datetime(first["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    first["id"] = first["id"].astype(int)
    first_map = first.groupby("id")["date"].min().to_dict()
    age = np.full((len(dates), len(ids)), np.nan, dtype=np.float32)
    dnp = pd.DatetimeIndex(dates).tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[D]")
    for j, iid in enumerate(ids):
        fd = first_map.get(int(iid))
        if fd is None:
            continue
        fts = pd.Timestamp(fd)
        if fts.tzinfo is not None:
            fts = fts.tz_convert("UTC").tz_localize(None)
        fd64 = np.datetime64(fts.normalize().to_datetime64()).astype("datetime64[D]")
        age[:, j] = (dnp - fd64).astype(np.float32)
    prims["age"] = age

    rank_src = cleaned.copy()
    rank_src["date"] = pd.to_datetime(rank_src["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    rank_src["id"] = rank_src["id"].astype(int)
    rank_src = rank_src[~rank_src["symbol"].astype(str).str.upper().isin(STABLE_OR_WRAP)]
    rank_src["mcap_rank"] = rank_src.groupby("date")["mcap"].rank(ascending=False, method="first")
    rk = rank_src.pivot_table(index="date", columns="id", values="mcap_rank", aggfunc="last")
    prims["mcap_rank"] = _to_arr(_align(rk, dates, ids))

    # positioning
    for name in ("funding_z_7", "funding_z_30", "dOI_7", "dOI_30", "basis", "taker_imb_7"):
        prims[name] = np.zeros((len(dates), len(ids)), dtype=np.float32)
    prims["pos_missing"] = np.ones((len(dates), len(ids)), dtype=np.float32)
    pos_file = Path(pos_path) if pos_path else Path("/data/quant/btcb/phase4v2/positioning_id.parquet")
    if pos_file.exists():
        hb(f"loading positioning {pos_file}")
        pos = pd.read_parquet(pos_file)
        pos["date"] = pd.to_datetime(pos["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        pos["id"] = pos["id"].astype(int)
        pos = pos.rename(columns={"taker_imbalance_7": "taker_imb_7"})
        id_set = set(int(i) for i in ids)
        pos = pos[pos["id"].isin(id_set)]
        mapped = int(len(pos))
        for src in ("funding_z_7", "funding_z_30", "dOI_7", "dOI_30", "basis", "taker_imb_7", "pos_missing"):
            if src not in pos.columns:
                continue
            w = pos.pivot_table(index="date", columns="id", values=src, aggfunc="last")
            arr = _to_arr(_align(w, dates, ids))
            if src == "pos_missing":
                cur = prims["pos_missing"]
                prims["pos_missing"] = np.where(np.isfinite(arr), arr, cur).astype(np.float32)
            else:
                cur = prims[src]
                prims[src] = np.where(np.isfinite(arr), arr, cur).astype(np.float32)
        hb(f"positioning mapped rows={mapped}")
    else:
        hb("positioning cache absent; all pos_missing=1")

    T, N = listed.shape
    mask_dv = np.zeros((T, N), dtype=bool)
    mask_mcap = np.zeros((T, N), dtype=bool)
    col_ix = {int(i): j for j, i in enumerate(ids)}
    for ti, d in enumerate(dates):
        for iid in pit_mem.get(d, []):
            j = col_ix.get(int(iid))
            if j is not None:
                mask_dv[ti, j] = True
        for iid in mem_mcap.get(d, []):
            j = col_ix.get(int(iid))
            if j is not None:
                mask_mcap[ti, j] = True
    # require listed to trade
    mask_dv &= listed
    mask_mcap &= listed

    univ_dv_idx, n_dv = pack_univ(mask_dv, 100)
    univ_mcap_idx, n_mcap = pack_univ(mask_mcap, int(FORGE_MCAP_N))
    is_monday = np.asarray([int(pd.Timestamp(d).weekday()) == int(FORGE_WEEKDAY) for d in dates], dtype=bool)
    std30 = prims["std_30"].astype(np.float64)
    vb = vol_buckets_from_std(std30, listed)
    btc_col = int(col_ix[int(btc_id)])
    cube = Cube(
        dates=dates,
        ids=ids,
        prims=prims,
        mask_dv=mask_dv,
        mask_mcap=mask_mcap,
        fwd=np.ascontiguousarray(fwd, dtype=np.float64),
        listed=listed,
        is_monday=is_monday,
        vol_bucket=vb,
        univ_dv_idx=univ_dv_idx,
        n_dv=n_dv,
        univ_mcap_idx=univ_mcap_idx,
        n_mcap=n_mcap,
        btc_id=int(btc_id),
        btc_col=btc_col,
        meta={
            "n_tagged_peg": int(len(tagged)),
            "mean_dv": float(mask_dv.sum(axis=1).mean()) if T else 0.0,
            "mean_mcap": float(mask_mcap.sum(axis=1).mean()) if T else 0.0,
        },
    )
    hb(f"cube ready T={T} N={N} mean_dv={cube.meta['mean_dv']:.1f} mean_mcap={cube.meta['mean_mcap']:.1f}")
    return cube


def make_synthetic_cube(
    T: int = 40,
    N: int = 12,
    seed: int = 0,
) -> Cube:
    rng = np.random.default_rng(int(seed))
    dates = pd.date_range("2020-01-01", periods=T, freq="D", tz="UTC").normalize()
    ids = np.arange(1, N + 1, dtype=np.int64)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=(T, N)), axis=0)
    close = np.clip(close, 1.0, None)
    r1 = np.empty((T, N))
    r1[0] = np.nan
    r1[1:] = close[1:] / close[:-1] - 1.0
    fwd = np.full((T, N), np.nan)
    fwd[:-1] = r1[1:]
    listed = np.ones((T, N), dtype=bool)
    listed[-3:, -1] = False
    prims = {}
    for name in FORGE_PRIMITIVES:
        prims[name] = rng.normal(0, 1, size=(T, N)).astype(np.float32)
    prims["close"] = close.astype(np.float32)
    prims["pos_missing"] = np.ones((T, N), dtype=np.float32)
    mask_dv = np.zeros((T, N), dtype=bool)
    mask_mcap = np.zeros((T, N), dtype=bool)
    mask_dv[:, : min(8, N)] = True
    mask_mcap[:, : min(6, N)] = True
    mask_dv &= listed
    mask_mcap &= listed
    univ_dv_idx, n_dv = pack_univ(mask_dv, 100)
    univ_mcap_idx, n_mcap = pack_univ(mask_mcap, 50)
    is_monday = np.asarray([d.weekday() == 0 for d in dates], dtype=bool)
    vb = vol_buckets_from_std(np.abs(prims["std_30"]).astype(np.float64), listed)
    return Cube(
        dates=dates,
        ids=ids,
        prims=prims,
        mask_dv=mask_dv,
        mask_mcap=mask_mcap,
        fwd=fwd,
        listed=listed,
        is_monday=is_monday,
        vol_bucket=vb,
        univ_dv_idx=univ_dv_idx,
        n_dv=n_dv,
        univ_mcap_idx=univ_mcap_idx,
        n_mcap=n_mcap,
        btc_id=1,
        btc_col=0,
        meta={"synthetic": True},
    )
