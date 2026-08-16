"""Walk-forward NFN training. Hygiene: ES floor epoch 8, UNDERTRAINED if best < 10."""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from btcb.constants import INNER_HOLDOUT_CALENDAR_DAYS
from btcb.model import FoldSpec, apply_calibrator, fit_isotonic
from nfn.constants import (
    ADAMW_LR,
    ADAMW_WD,
    ES_FLOOR_EPOCH,
    ES_PATIENCE,
    MAX_EPOCHS,
    UNDERTRAINED_BEST_LT,
)
from nfn.data import PackedPanel, date_index_window
from nfn.loss import nfn_loss
from nfn.model import build_nfn
from nfn.warmstart import apply_warmstart


def _half_ic_top(score: np.ndarray, excess: np.ndarray, min_n: int = 8) -> float:
    s = np.asarray(score, dtype=float)
    e = np.asarray(excess, dtype=float)
    m = np.isfinite(s) & np.isfinite(e)
    s, e = s[m], e[m]
    n = len(s)
    if n < 2 * int(min_n):
        return float("nan")
    order = pd.Series(s).rank(ascending=False, method="first").to_numpy()
    mask = order <= n / 2.0
    if int(mask.sum()) < int(min_n):
        return float("nan")
    a, b = s[mask], e[mask]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _mean_tail_ic(pack: PackedPanel, date_ids: np.ndarray, scores: np.ndarray) -> float:
    ics = []
    for di in date_ids:
        a, b = int(pack.starts[di]), int(pack.ends[di])
        if b - a < 16:
            continue
        v = _half_ic_top(scores[a:b], pack.excess[a:b])
        if np.isfinite(v):
            ics.append(v)
    return float(np.mean(ics)) if ics else float("nan")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    out = 1.0 / (1.0 + np.exp(-x))
    out = np.where(np.isfinite(np.asarray(x, dtype=float)), out, np.nan)
    return out


def _fit_ir(raw: np.ndarray, y: np.ndarray):
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(raw) & np.isfinite(y)
    if int(m.sum()) < 50:
        return None
    return fit_isotonic(raw[m], y[m])


def _safe_calibrate(ir, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    out = np.full(raw.shape, np.nan, dtype=float)
    m = np.isfinite(raw)
    if int(m.sum()) == 0:
        return out
    out[m] = apply_calibrator(ir, raw[m])
    return out


def _forward_numpy(model, pack: PackedPanel, date_ids: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    logit_t = np.full(len(pack.z), np.nan, dtype=np.float64)
    logit_b = np.full(len(pack.z), np.nan, dtype=np.float64)
    gammas, betas, dates = [], [], []
    model.eval()
    with torch.no_grad():
        for di in date_ids:
            a, b = int(pack.starts[di]), int(pack.ends[di])
            if b <= a:
                continue
            z = torch.from_numpy(pack.z[a:b]).to(device)
            m = torch.from_numpy(pack.m[a:b]).to(device)
            lt, lb, extra = model(z, m)
            logit_t[a:b] = lt.detach().cpu().numpy()
            logit_b[a:b] = lb.detach().cpu().numpy()
            g = extra["gamma"].mean(dim=0).detach().cpu().numpy()
            bt = extra["beta"].mean(dim=0).detach().cpu().numpy()
            gammas.append(g)
            betas.append(bt)
            dates.append(pd.Timestamp(pack.dates[int(di)]))
    return logit_t, logit_b, {"gamma": gammas, "beta": betas, "dates": dates}


def train_one_fold(
    pack: PackedPanel,
    fold: FoldSpec,
    *,
    seed: int,
    warm_blob: dict | None = None,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    device: str = "cpu",
    max_epochs: int = MAX_EPOCHS,
) -> tuple[pd.DataFrame, dict]:
    import torch

    t0 = time.time()
    torch.manual_seed(int(seed) + int(fold.fold_id) + (0 if shuffle_seed is None else int(shuffle_seed)))
    np.random.seed(int(seed) + int(fold.fold_id) + (0 if shuffle_seed is None else int(shuffle_seed)))

    tr_ids = date_index_window(pack, fold.train_start, fold.train_end)
    va_ids = date_index_window(pack, fold.val_start, fold.val_end)
    if len(tr_ids) < 20 or len(va_ids) < 5:
        return pd.DataFrame(), {
            "fold_id": fold.fold_id,
            "status": "empty",
            "n_train_dates": int(len(tr_ids)),
            "n_val_dates": int(len(va_ids)),
            "elapsed": time.time() - t0,
        }

    cut = pd.Timestamp(fold.train_end)
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    cut = (cut - pd.Timedelta(days=int(INNER_HOLDOUT_CALENDAR_DAYS))).tz_convert("UTC").normalize()
    idx_dates = pd.DatetimeIndex(pack.dates)
    if idx_dates.tz is None:
        idx_dates = idx_dates.tz_localize("UTC")
    else:
        idx_dates = idx_dates.tz_convert("UTC")
    inner_tr = np.asarray([i for i in tr_ids if pd.Timestamp(idx_dates[int(i)]).tz_convert("UTC").normalize() <= cut], dtype=np.int32)
    inner_ho = np.asarray([i for i in tr_ids if pd.Timestamp(idx_dates[int(i)]).tz_convert("UTC").normalize() > cut], dtype=np.int32)
    if len(inner_tr) < 10 or len(inner_ho) < 5:
        split = max(1, int(len(tr_ids) * 0.85))
        inner_tr, inner_ho = tr_ids[:split], tr_ids[split:]

    y_top = pack.y_top.copy()
    y_bot = pack.y_bot.copy()
    excess = pack.excess.copy()
    if shuffle_labels:
        ss = int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017
        rng = np.random.default_rng(ss)
        for di in np.concatenate([inner_tr, inner_ho]):
            a, b = int(pack.starts[di]), int(pack.ends[di])
            n = b - a
            if n <= 1:
                continue
            vol = pack.vol[a:b]
            from btcb.model import vol_bucket_ids

            buckets = vol_bucket_ids(vol)
            for bv in np.unique(buckets):
                loc = np.flatnonzero(buckets == bv)
                if len(loc) > 1:
                    perm = rng.permutation(len(loc))
                    idx = a + loc
                    y_top[idx] = y_top[idx][perm]
                    y_bot[idx] = y_bot[idx][perm]
                    excess[idx] = excess[idx][perm]

    model = build_nfn(int(seed) + int(fold.fold_id))
    n_ws = 0
    if warm_blob is not None and not shuffle_labels:
        n_ws = apply_warmstart(model, warm_blob)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(ADAMW_LR), weight_decay=float(ADAMW_WD))

    best_ic = -1e9
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stall = 0
    train_curve, ho_curve = [], []
    undertrained = False

    rng_ep = np.random.default_rng(int(seed) * 1009 + int(fold.fold_id))
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = inner_tr.copy()
        rng_ep.shuffle(order)
        ep_loss = []
        for di in order:
            a, b = int(pack.starts[di]), int(pack.ends[di])
            if b - a < 8:
                continue
            z = torch.from_numpy(pack.z[a:b]).to(device)
            m = torch.from_numpy(pack.m[a:b]).to(device)
            yt = torch.from_numpy(y_top[a:b]).to(device)
            yb = torch.from_numpy(y_bot[a:b]).to(device)
            ex = torch.from_numpy(excess[a:b]).to(device)
            opt.zero_grad(set_to_none=True)
            lt, lb, _ = model(z, m)
            loss, parts = nfn_loss(lt, lb, yt, yb, ex, model)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss.append(parts["loss"])

        lt_tr, lb_tr, _ = _forward_numpy(model, pack, inner_tr, device)
        lt_ho, lb_ho, _ = _forward_numpy(model, pack, inner_ho, device)
        sc_tr = _sigmoid(lt_tr) - _sigmoid(lb_tr)
        sc_ho = _sigmoid(lt_ho) - _sigmoid(lb_ho)
        ic_tr = _mean_tail_ic(pack, inner_tr, sc_tr)
        ic_ho = _mean_tail_ic(pack, inner_ho, sc_ho)
        train_curve.append({"epoch": epoch, "loss": float(np.mean(ep_loss) if ep_loss else np.nan), "tail_ic": ic_tr})
        ho_curve.append({"epoch": epoch, "tail_ic": ic_ho})
        print(
            f"[nfn {time.strftime('%H:%M:%S')}] fold={fold.fold_id} seed={seed} "
            f"epoch={epoch}/{max_epochs} loss={train_curve[-1]['loss']:.4f} "
            f"train_tailIC={ic_tr:.4f} holdout_tailIC={ic_ho:.4f}",
            flush=True,
        )
        improved = bool(np.isfinite(ic_ho) and ic_ho > best_ic + 1e-6)
        if epoch == 1 or improved:
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            if np.isfinite(ic_ho):
                best_ic = float(ic_ho)
            stall = 0
        else:
            stall += 1
        if epoch >= int(ES_FLOOR_EPOCH) and stall >= int(ES_PATIENCE):
            print(f"[nfn] fold={fold.fold_id} ES stop epoch={epoch} best={best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    if best_epoch < int(UNDERTRAINED_BEST_LT):
        undertrained = True

    lt_ho, lb_ho, _ = _forward_numpy(model, pack, inner_ho, device)
    p_top_ho = _sigmoid(lt_ho)
    p_bot_ho = _sigmoid(lb_ho)
    # isotonic on inner-holdout; skipped for shuffled (house)
    ir_top = None if shuffle_labels else _fit_ir(p_top_ho, y_top)
    ir_bot = None if shuffle_labels else _fit_ir(p_bot_ho, y_bot)

    lt_va, lb_va, film = _forward_numpy(model, pack, va_ids, device)
    p_top = _safe_calibrate(ir_top, _sigmoid(lt_va))
    p_bot = _safe_calibrate(ir_bot, _sigmoid(lb_va))
    rows = []
    for di in va_ids:
        a, b = int(pack.starts[di]), int(pack.ends[di])
        dt = pd.Timestamp(pack.dates[int(di)])
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        dt = dt.tz_convert("UTC").normalize()
        for i in range(a, b):
            if not (np.isfinite(p_top[i]) and np.isfinite(p_bot[i])):
                continue
            rows.append(
                {
                    "date": dt,
                    "id": int(pack.ids[i]),
                    "fold_id": int(fold.fold_id),
                    "p_top": float(p_top[i]),
                    "p_bot": float(p_bot[i]),
                    "spread": float(p_top[i] - p_bot[i]),
                }
            )
    pred = pd.DataFrame(rows)
    membership = {
        "c": model.c.detach().cpu().numpy().tolist(),
        "s": model.scales().detach().cpu().numpy().tolist(),
        "c_init": model.c_init.cpu().numpy().tolist(),
        "s_init": model.s_init.cpu().numpy().tolist(),
        "feat_names": list(model.feat_names),
    }
    e = model.exponents().detach().cpu().numpy()
    meta = {
        "fold_id": int(fold.fold_id),
        "status": "ok",
        "best_epoch": int(best_epoch),
        "best_holdout_tail_ic": float(best_ic) if np.isfinite(best_ic) and best_ic > -1e8 else float("nan"),
        "undertrained": bool(undertrained),
        "n_train_dates": int(len(inner_tr)),
        "n_holdout_dates": int(len(inner_ho)),
        "n_val_dates": int(len(va_ids)),
        "n_val_rows": int(len(pred)),
        "n_params": int(model.n_params()),
        "warmstart_rules": int(n_ws),
        "train_curve": train_curve,
        "holdout_curve": ho_curve,
        "membership": membership,
        "exponents": e.astype(float).tolist(),
        "rule_w": model.w.detach().cpu().numpy().astype(float).tolist(),
        "film": {
            "dates": [str(pd.Timestamp(x).date()) for x in film["dates"]],
            "gamma_mean": [float(np.mean(g)) for g in film["gamma"]],
            "beta_mean": [float(np.mean(b)) for b in film["beta"]],
            "gamma_last": film["gamma"][-1].astype(float).tolist() if film["gamma"] else [],
            "beta_last": film["beta"][-1].astype(float).tolist() if film["beta"] else [],
        },
        "elapsed": time.time() - t0,
        "gpu": False,
    }
    print(
        f"[nfn] fold={fold.fold_id} done best_epoch={best_epoch} undertrained={undertrained} "
        f"val_rows={len(pred)} elapsed={meta['elapsed']:.1f}s",
        flush=True,
    )
    return pred, meta


def train_walkforward(
    pack: PackedPanel,
    folds: list[FoldSpec],
    *,
    seed: int,
    warm_blob: dict | None = None,
    device: str = "cpu",
    cache_dir=None,
    cache_ver: str = "p7v1",
    commit_fn=None,
) -> tuple[pd.DataFrame, list[dict]]:
    import json
    from pathlib import Path

    parts, metas = [], []
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        pred, meta = None, None
        if cache_dir is not None:
            pq = cache_dir / f"preds_seed{int(seed)}_fold{fold.fold_id}.parquet"
            js = cache_dir / f"meta_seed{int(seed)}_fold{fold.fold_id}.json"
            if pq.exists() and js.exists():
                try:
                    meta = json.loads(js.read_text())
                    if meta.get("cache_ver") == cache_ver and meta.get("status") == "ok":
                        pred = pd.read_parquet(pq)
                        meta["cache_hit"] = True
                        print(f"[nfn] fold={fold.fold_id} seed={seed} CACHE HIT rows={len(pred)}", flush=True)
                    else:
                        pred, meta = None, None
                except Exception as exc:
                    print(f"[nfn] fold={fold.fold_id} cache unreadable: {exc}", flush=True)
                    pred, meta = None, None
        if meta is None:
            pred, meta = train_one_fold(pack, fold, seed=seed, warm_blob=warm_blob, device=device)
            meta["cache_ver"] = cache_ver
            meta["cache_hit"] = False
            if cache_dir is not None and pred is not None and not pred.empty:
                pred.to_parquet(cache_dir / f"preds_seed{int(seed)}_fold{fold.fold_id}.parquet", index=False)
                (cache_dir / f"meta_seed{int(seed)}_fold{fold.fold_id}.json").write_text(json.dumps(meta, default=str))
                if commit_fn is not None:
                    commit_fn()
        metas.append(meta)
        if pred is not None and not pred.empty:
            parts.append(pred)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out, metas
