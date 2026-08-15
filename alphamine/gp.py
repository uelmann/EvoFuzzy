"""Small genetic program over causal OHLCV ops."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from alphamine.arrays import MarketArrays, _map_date_i, attach_matrix, period_mask, y_matrix
from alphamine.constants import (
    BINARY_OPS,
    BTC_SYMBOL,
    FIELDS,
    GP_BUDGET_SEC,
    GP_CX,
    GP_ELITE,
    GP_GENS,
    GP_KEEP,
    GP_MAX_ABS_CORR,
    GP_MAX_DEPTH,
    GP_MIN_CS,
    GP_MIN_HOLDOUT_IC,
    GP_MUT,
    GP_POP,
    GP_TOURNAMENT,
    GP_WINDOWS,
    TS_BINARY_OPS,
    TS_UNARY_OPS,
    UNARY_OPS,
    YCOL,
)
from baseline.features import FEATURE_COLS


@dataclass
class Node:
    kind: str
    name: str
    window: int | None
    children: tuple["Node", ...]

    def to_obj(self) -> list:
        return [
            self.kind,
            self.name,
            self.window,
            [c.to_obj() for c in self.children],
        ]

    def to_str(self) -> str:
        if self.kind == "field":
            return str(self.name)
        if self.kind == "un":
            return f"{self.name}({self.children[0].to_str()})"
        if self.kind == "ts":
            return f"{self.name}:{int(self.window)}({self.children[0].to_str()})"
        if self.kind == "bin":
            return f"{self.name}({self.children[0].to_str()},{self.children[1].to_str()})"
        if self.kind == "tsb":
            return (
                f"{self.name}:{int(self.window)}("
                f"{self.children[0].to_str()},{self.children[1].to_str()})"
            )
        return "?"


def node_from_obj(obj: list) -> Node:
    kind, name, window, kids = obj[0], obj[1], obj[2], obj[3]
    return Node(str(kind), str(name), None if window is None else int(window), tuple(node_from_obj(k) for k in kids))


def clone(n: Node) -> Node:
    return Node(n.kind, n.name, n.window, tuple(clone(c) for c in n.children))


def _finite(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def _rolling(a: np.ndarray, w: int, how: str) -> np.ndarray:
    minp = max(2, int(w) // 2)
    df = pd.DataFrame(a)
    r = df.rolling(int(w), min_periods=minp)
    if how == "mean":
        out = r.mean()
    elif how == "std":
        out = r.std()
    elif how == "max":
        out = r.max()
    elif how == "min":
        out = r.min()
    elif how == "sum":
        out = r.sum()
    elif how == "rank":
        out = r.rank(pct=True)
    else:
        raise ValueError(how)
    return _finite(out.to_numpy(dtype=float))


def _delay(a: np.ndarray, w: int) -> np.ndarray:
    out = np.roll(a, int(w), axis=0)
    out[: int(w), :] = np.nan
    return _finite(out)


def eval_node(node: Node, arr: MarketArrays, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = node.to_str()
    hit = cache.get(key)
    if hit is not None:
        return hit
    if node.kind == "field":
        out = arr.field(node.name).astype(float, copy=False)
    elif node.kind == "un":
        x = eval_node(node.children[0], arr, cache)
        op = node.name
        if op == "abs":
            out = np.abs(x)
        elif op == "neg":
            out = -x
        elif op == "sign":
            out = np.sign(x)
        elif op == "log":
            ax = np.abs(x)
            out = np.full_like(x, np.nan)
            m = ax > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                out[m] = np.log(ax[m])
        elif op == "cs_rank":
            out = pd.DataFrame(x).rank(axis=1, pct=True, na_option="keep").to_numpy(dtype=float)
        else:
            raise ValueError(op)
        out = _finite(out)
    elif node.kind == "ts":
        x = eval_node(node.children[0], arr, cache)
        w = int(node.window)
        op = node.name
        if op == "delay":
            out = _delay(x, w)
        elif op == "delta":
            out = _finite(x - _delay(x, w))
        elif op == "ts_mean":
            out = _rolling(x, w, "mean")
        elif op == "ts_std":
            out = _rolling(x, w, "std")
        elif op == "ts_max":
            out = _rolling(x, w, "max")
        elif op == "ts_min":
            out = _rolling(x, w, "min")
        elif op == "ts_sum":
            out = _rolling(x, w, "sum")
        elif op == "ts_rank":
            out = _rolling(x, w, "rank")
        else:
            raise ValueError(op)
    elif node.kind == "bin":
        a = eval_node(node.children[0], arr, cache)
        b = eval_node(node.children[1], arr, cache)
        op = node.name
        if op == "add":
            out = a + b
        elif op == "sub":
            out = a - b
        elif op == "mul":
            out = a * b
        elif op == "div":
            out = np.full_like(a, np.nan)
            m = np.abs(b) > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                out[m] = a[m] / b[m]
        else:
            raise ValueError(op)
        out = _finite(out)
    elif node.kind == "tsb":
        a = eval_node(node.children[0], arr, cache)
        b = eval_node(node.children[1], arr, cache)
        w = int(node.window)
        da = pd.DataFrame(a)
        db = pd.DataFrame(b)
        minp = max(3, w // 2)
        cols = []
        for j in range(a.shape[1]):
            cols.append(da.iloc[:, j].rolling(w, min_periods=minp).corr(db.iloc[:, j]))
        out = _finite(pd.concat(cols, axis=1).to_numpy(dtype=float))
    else:
        raise ValueError(node.kind)
    cache[key] = out
    return out


def mean_rank_ic_mat(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, min_n: int = GP_MIN_CS) -> float:
    ics = []
    t_n = pred.shape[0]
    for t in range(t_n):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(y[t])
        if int(m.sum()) < int(min_n):
            continue
        x = pred[t, m]
        yy = y[t, m]
        if np.unique(np.round(x, 12)).size < 2 or np.unique(np.round(yy, 12)).size < 2:
            continue
        res = stats.spearmanr(x, yy)
        corr = getattr(res, "correlation", None)
        if corr is None:
            corr = getattr(res, "statistic", np.nan)
        c = float(np.asarray(corr, dtype=float).reshape(-1)[0])
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return 0.0
    return float(np.mean(ics))


def _rand_field(rng: np.random.Generator) -> Node:
    return Node("field", str(rng.choice(FIELDS)), None, ())


def random_node(rng: np.random.Generator, depth: int, max_depth: int) -> Node:
    if depth >= max_depth or (depth >= 1 and rng.random() < 0.35):
        return _rand_field(rng)
    w = int(rng.choice(GP_WINDOWS))
    u = rng.random()
    if u < 0.40:
        return Node("ts", str(rng.choice(TS_UNARY_OPS)), w, (random_node(rng, depth + 1, max_depth),))
    if u < 0.62:
        return Node("un", str(rng.choice(UNARY_OPS)), None, (random_node(rng, depth + 1, max_depth),))
    if u < 0.90:
        return Node(
            "bin",
            str(rng.choice(BINARY_OPS)),
            None,
            (random_node(rng, depth + 1, max_depth), random_node(rng, depth + 1, max_depth)),
        )
    return Node(
        "tsb",
        str(rng.choice(TS_BINARY_OPS)),
        w,
        (random_node(rng, depth + 1, max_depth), random_node(rng, depth + 1, max_depth)),
    )


def _all_nodes(n: Node) -> list[Node]:
    out = [n]
    for c in n.children:
        out.extend(_all_nodes(c))
    return out


def _replace_once(n: Node, target: Node, repl: Node, seen: dict) -> Node:
    if id(n) in seen:
        return seen[id(n)]
    if n is target:
        seen[id(n)] = repl
        return repl
    kids = tuple(_replace_once(c, target, repl, seen) for c in n.children)
    out = Node(n.kind, n.name, n.window, kids)
    seen[id(n)] = out
    return out


def mutate(n: Node, rng: np.random.Generator) -> Node:
    nodes = _all_nodes(n)
    tgt = nodes[int(rng.integers(0, len(nodes)))]
    repl = random_node(rng, 0, max(1, GP_MAX_DEPTH - 1))
    return _replace_once(n, tgt, clone(repl), {})


def crossover(a: Node, b: Node, rng: np.random.Generator) -> Node:
    na = _all_nodes(a)
    nb = _all_nodes(b)
    tgt = na[int(rng.integers(0, len(na)))]
    src = nb[int(rng.integers(0, len(nb)))]
    return _replace_once(a, tgt, clone(src), {})


def tournament(pop: list[Node], fit: list[float], rng: np.random.Generator, k: int) -> Node:
    idx = rng.choice(len(pop), size=min(int(k), len(pop)), replace=False)
    best = max(idx, key=lambda i: fit[int(i)])
    return pop[int(best)]


def _gather_col(feat: pd.DataFrame, arr: MarketArrays, mat: np.ndarray) -> np.ndarray:
    out = np.full(len(feat), np.nan, dtype=float)
    di = _map_date_i(feat["date"], arr)
    sj = feat["symbol"].map(arr.sym_to_j)
    ok = di.notna() & sj.notna()
    if bool(ok.any()):
        out[ok.to_numpy()] = mat[di[ok].astype(int).to_numpy(), sj[ok].astype(int).to_numpy()]
    return out


def _mean_abs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 30:
        return 1.0
    x, y = a[m], b[m]
    if np.unique(np.round(x, 12)).size < 2 or np.unique(np.round(y, 12)).size < 2:
        return 1.0
    res = stats.spearmanr(x, y)
    corr = getattr(res, "correlation", None)
    if corr is None:
        corr = getattr(res, "statistic", np.nan)
    c = float(np.asarray(corr, dtype=float).reshape(-1)[0])
    return float(abs(c)) if np.isfinite(c) else 1.0


def mine_fold(
    arr: MarketArrays,
    feat: pd.DataFrame,
    fold,
    *,
    seed: int,
    inner_holdout_days: int,
    ycol: str = YCOL,
    budget_sec: float = GP_BUDGET_SEC,
) -> list[dict]:
    rng = np.random.default_rng(int(seed) + 17 * int(fold.fold_id) + 901)
    ymat = y_matrix(arr, feat, ycol)
    cut = pd.Timestamp(fold.train_end) - pd.Timedelta(days=int(inner_holdout_days))
    mask_fit = period_mask(arr, feat, fold.train_start, cut)
    mask_ho = period_mask(arr, feat, cut + pd.Timedelta(days=1), fold.train_end)
    if int(mask_fit.sum()) < 200 or int(mask_ho.sum()) < 50:
        mask_fit = period_mask(arr, feat, fold.train_start, fold.train_end)
        mask_ho = mask_fit
    cache: dict[str, np.ndarray] = {}
    pop = [random_node(rng, 0, GP_MAX_DEPTH) for _ in range(int(GP_POP))]
    fit = [0.0] * len(pop)
    t0 = time.time()

    def _score(n: Node) -> float:
        try:
            mat = eval_node(n, arr, cache)
        except Exception:
            return -1.0
        return mean_rank_ic_mat(mat, ymat, mask_fit)

    for i, n in enumerate(pop):
        fit[i] = _score(n)
    gen_done = 0
    for g in range(int(GP_GENS)):
        if time.time() - t0 > float(budget_sec):
            break
        ranked = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
        new_pop = [clone(pop[i]) for i in ranked[: int(GP_ELITE)]]
        new_fit = [fit[i] for i in ranked[: int(GP_ELITE)]]
        while len(new_pop) < int(GP_POP):
            if time.time() - t0 > float(budget_sec):
                break
            p1 = tournament(pop, fit, rng, GP_TOURNAMENT)
            child = clone(p1)
            if rng.random() < float(GP_CX):
                p2 = tournament(pop, fit, rng, GP_TOURNAMENT)
                child = crossover(child, p2, rng)
            if rng.random() < float(GP_MUT):
                child = mutate(child, rng)
            new_pop.append(child)
            new_fit.append(_score(child))
        pop, fit = new_pop, new_fit
        gen_done = g + 1

    # unique by string, scored on holdout
    seen: set[str] = set()
    ranked_nodes: list[tuple[float, Node, np.ndarray]] = []
    for n, f_fit in sorted(zip(pop, fit), key=lambda kv: kv[1], reverse=True):
        s = n.to_str()
        if s in seen or n.kind == "field":
            continue
        seen.add(s)
        try:
            mat = eval_node(n, arr, cache)
        except Exception:
            continue
        ho = mean_rank_ic_mat(mat, ymat, mask_ho)
        ranked_nodes.append((ho, n, mat))
    ranked_nodes.sort(key=lambda t: t[0], reverse=True)

    ho_feat = feat.copy()
    ho_feat["date"] = pd.to_datetime(ho_feat["date"], utc=True)
    lo = pd.Timestamp(cut)
    hi = pd.Timestamp(fold.train_end)
    if lo.tzinfo is None:
        lo = lo.tz_localize("UTC")
    if hi.tzinfo is None:
        hi = hi.tz_localize("UTC")
    ho_rows = ho_feat[(ho_feat["date"] > lo) & (ho_feat["date"] <= hi) & (ho_feat["symbol"] != BTC_SYMBOL)]
    if ho_rows.empty:
        ho_rows = ho_feat[ho_feat["symbol"] != BTC_SYMBOL]
    a0_hold = {c: ho_rows[c].to_numpy(dtype=float) for c in FEATURE_COLS if c in ho_rows.columns}

    kept: list[dict] = []
    kept_vals: list[np.ndarray] = []
    for ho_ic, n, mat in ranked_nodes:
        if len(kept) >= int(GP_KEEP):
            break
        if ho_ic < float(GP_MIN_HOLDOUT_IC):
            continue
        col_vals = _gather_col(ho_rows, arr, mat)
        bad = False
        for c, av in a0_hold.items():
            if _mean_abs_spearman(col_vals, av) >= float(GP_MAX_ABS_CORR):
                bad = True
                break
        if bad:
            continue
        for prev in kept_vals:
            if _mean_abs_spearman(col_vals, prev) >= float(GP_MAX_ABS_CORR):
                bad = True
                break
        if bad:
            continue
        kept.append(
            {
                "expr": n.to_str(),
                "obj": n.to_obj(),
                "holdout_ic": float(ho_ic),
                "fit_ic": float(mean_rank_ic_mat(mat, ymat, mask_fit)),
            }
        )
        kept_vals.append(col_vals)

    print(
        f"[gp] fold={fold.fold_id} gens={gen_done} kept={len(kept)} "
        f"elapsed={time.time() - t0:.1f}s best_ho={kept[0]['holdout_ic'] if kept else float('nan')}",
        flush=True,
    )
    return kept


def apply_formulas(
    feat: pd.DataFrame,
    arr: MarketArrays,
    formulas: list[dict],
    clip: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    out = feat
    cols: list[str] = []
    cache: dict[str, np.ndarray] = {}
    for i, spec in enumerate(formulas):
        node = node_from_obj(spec["obj"])
        mat = eval_node(node, arr, cache)
        col = f"gp_{i}"
        out = attach_matrix(out, arr, mat, col, clip=clip)
        cols.append(col)
    return out, cols
