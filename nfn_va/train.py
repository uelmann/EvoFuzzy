"""Walk-forward Variant A training. 7.c craft: trail-3 SWA, 5-init bag, fold warm-start."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from btcb.model import FoldSpec
from nfn_va.constants import (
    ADAMW_LR,
    ADAMW_LR_MIN,
    ADAMW_WD,
    CACHE_VER,
    ES_FLOOR_EPOCH,
    ES_PATIENCE,
    GRAD_CLIP,
    HORIZON,
    INNER_HOLDOUT_DATES,
    MAX_EPOCHS,
    N_INITS,
    SWA_K,
    TRAIL_WINDOW,
    UNDERTRAINED_BEST_LT,
)
from nfn_va.data import PackedPanel, date_index_window
from nfn_va.loss import variant_a_loss
from nfn_va.model import build_nfn
from nfn_va.warmstart import apply_warmstart


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


def _mean_tail_ic(
    pack: PackedPanel,
    date_ids: np.ndarray,
    scores: np.ndarray,
    excess: np.ndarray | None = None,
) -> float:
    ex = pack.excess if excess is None else excess
    ics = []
    for di in date_ids:
        a, b = int(pack.starts[di]), int(pack.ends[di])
        if b - a < 16:
            continue
        v = _half_ic_top(scores[a:b], ex[a:b])
        if np.isfinite(v):
            ics.append(v)
    return float(np.mean(ics)) if ics else float("nan")


def _inner_split(pack: PackedPanel, tr_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last 120 train dates as holdout, purged by h+3 from the train side."""
    if len(tr_ids) < 30:
        split = max(1, int(len(tr_ids) * 0.85))
        return tr_ids[:split], tr_ids[split:]
    n_ho = min(int(INNER_HOLDOUT_DATES), max(5, len(tr_ids) // 5))
    ho = tr_ids[-n_ho:]
    idx = pd.DatetimeIndex(pack.dates)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.normalize()
    ho_start = pd.Timestamp(idx[int(ho[0])]).tz_convert("UTC").normalize()
    cut = (ho_start - pd.Timedelta(days=int(HORIZON) + 3)).normalize()
    inner_tr = np.asarray(
        [i for i in tr_ids if pd.Timestamp(idx[int(i)]).tz_convert("UTC").normalize() <= cut],
        dtype=np.int32,
    )
    if len(inner_tr) < 10:
        inner_tr = tr_ids[: max(1, len(tr_ids) - n_ho)]
    return inner_tr, ho


def _shuffle_vol(pack: PackedPanel, date_ids: np.ndarray, rng, y_win, y_z, y_rank, excess):
    from btcb.model import vol_bucket_ids

    for di in date_ids:
        a, b = int(pack.starts[di]), int(pack.ends[di])
        n = b - a
        if n <= 1:
            continue
        buckets = vol_bucket_ids(pack.vol[a:b])
        for bv in np.unique(buckets):
            loc = np.flatnonzero(buckets == bv)
            if len(loc) > 1:
                perm = rng.permutation(len(loc))
                idx = a + loc
                y_win[idx] = y_win[idx][perm]
                y_z[idx] = y_z[idx][perm]
                y_rank[idx] = y_rank[idx][perm]
                excess[idx] = excess[idx][perm]
    return y_win, y_z, y_rank, excess


def _forward_numpy(
    model,
    pack: PackedPanel,
    date_ids: np.ndarray,
    device,
    z_t=None,
    m_t=None,
) -> tuple[np.ndarray, dict]:
    import torch

    scores = np.full(len(pack.z), np.nan, dtype=np.float64)
    gammas, betas, dates = [], [], []
    model.eval()
    with torch.inference_mode():
        for di in date_ids:
            a, b = int(pack.starts[di]), int(pack.ends[di])
            if b <= a:
                continue
            z = (z_t[a:b] if z_t is not None else torch.from_numpy(pack.z[a:b])).to(device)
            m = (m_t[a:b] if m_t is not None else torch.from_numpy(pack.m[a:b])).to(device)
            sc, extra = model(z, m)
            scores[a:b] = sc.detach().cpu().numpy()
            g = extra["gamma"].mean(dim=0).detach().cpu().numpy()
            bt = extra["beta"].mean(dim=0).detach().cpu().numpy()
            gammas.append(g)
            betas.append(bt)
            dates.append(pd.Timestamp(pack.dates[int(di)]))
    return scores, {"gamma": gammas, "beta": betas, "dates": dates}


def _average_states(states: list[dict]):
    import torch

    if not states:
        return None
    avg = {}
    n = float(len(states))
    for k in states[0]:
        avg[k] = sum(s[k] for s in states) / n
    return avg


def _cosine_lr(epoch: int, n_epochs: int, lr0: float, lr_min: float) -> float:
    if n_epochs <= 1:
        return float(lr_min)
    t = (int(epoch) - 1) / float(n_epochs - 1)
    return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + np.cos(np.pi * t)))


def train_one_init(
    pack: PackedPanel,
    fold: FoldSpec,
    *,
    seed: int,
    init_id: int,
    warm_state: dict | None,
    warm_blob: dict | None,
    y_win: np.ndarray,
    y_z: np.ndarray,
    device: str,
    max_epochs: int,
    excess: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    import torch

    t0 = time.time()
    torch.manual_seed(int(seed) * 1009 + int(fold.fold_id) * 17 + int(init_id))
    np.random.seed(int(seed) * 1009 + int(fold.fold_id) * 17 + int(init_id))

    tr_ids = date_index_window(pack, fold.train_start, fold.train_end)
    va_ids = date_index_window(pack, fold.val_start, fold.val_end)
    if len(tr_ids) < 20 or len(va_ids) < 5:
        return np.array([]), {"status": "empty", "init_id": int(init_id), "fold_id": int(fold.fold_id)}

    inner_tr, inner_ho = _inner_split(pack, tr_ids)
    excess_np = pack.excess if excess is None else excess
    model = build_nfn(int(seed) + 10_000 * int(init_id) + int(fold.fold_id))
    n_ws = 0
    if warm_state is not None:
        model.load_state_dict(warm_state)
    elif warm_blob is not None and int(init_id) == 0:
        n_ws = apply_warmstart(model, warm_blob)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(ADAMW_LR), weight_decay=float(ADAMW_WD))

    z_t = torch.from_numpy(pack.z)
    m_t = torch.from_numpy(pack.m)
    yw_t = torch.from_numpy(np.ascontiguousarray(y_win))
    yz_t = torch.from_numpy(np.ascontiguousarray(y_z))

    epoch_states = []
    ho_ics = []
    train_curve, ho_curve = [], []
    rng_ep = np.random.default_rng(int(seed) * 7919 + int(fold.fold_id) * 13 + int(init_id))
    stall = 0
    best_trail = -1e9
    selected_epoch = 0

    for epoch in range(1, int(max_epochs) + 1):
        lr = _cosine_lr(epoch, int(max_epochs), float(ADAMW_LR), float(ADAMW_LR_MIN))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        order = inner_tr.copy()
        rng_ep.shuffle(order)
        ep_loss = []
        for di in order:
            a, b = int(pack.starts[di]), int(pack.ends[di])
            if b - a < 8:
                continue
            z = z_t[a:b]
            m = m_t[a:b]
            yw = yw_t[a:b]
            yz = yz_t[a:b]
            opt.zero_grad(set_to_none=True)
            sc, _ = model(z, m)
            loss, parts = variant_a_loss(sc, yw, yz, model)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))
            opt.step()
            ep_loss.append(parts["loss"])

        sc_ho, _ = _forward_numpy(model, pack, inner_ho, device, z_t=z_t, m_t=m_t)
        ic_ho = _mean_tail_ic(pack, inner_ho, sc_ho, excess=excess_np)
        ho_ics.append(ic_ho)
        epoch_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        w = int(TRAIL_WINDOW)
        window = ho_ics[-w:]
        finite = [x for x in window if np.isfinite(x)]
        trail = float(np.mean(finite)) if finite else float("nan")
        train_curve.append({"epoch": epoch, "loss": float(np.mean(ep_loss) if ep_loss else np.nan), "lr": lr})
        ho_curve.append({"epoch": epoch, "tail_ic": ic_ho, "trail3": trail})
        print(
            f"[p7d {time.strftime('%H:%M:%S')}] fold={fold.fold_id} seed={seed} init={init_id} "
            f"epoch={epoch}/{max_epochs} loss={train_curve[-1]['loss']:.4f} "
            f"holdout_tailIC={ic_ho:.4f} trail3={trail:.4f}",
            flush=True,
        )
        improved = bool(np.isfinite(trail) and trail > best_trail + 1e-6)
        if epoch == 1 or improved:
            selected_epoch = int(epoch)
            if np.isfinite(trail):
                best_trail = float(trail)
            stall = 0
        else:
            stall += 1
        if epoch >= int(ES_FLOOR_EPOCH) and stall >= int(ES_PATIENCE):
            print(
                f"[p7d] fold={fold.fold_id} init={init_id} ES stop epoch={epoch} selected={selected_epoch}",
                flush=True,
            )
            break

    # SWA over the best 3 epochs by trailing-mean (fallback: last 3)
    scored = []
    for i, rec in enumerate(ho_curve):
        t = rec.get("trail3")
        scored.append((float(t) if t is not None and np.isfinite(t) else -1e9, i))
    scored.sort(reverse=True)
    pick = [i for _, i in scored[: int(SWA_K)]]
    if not pick:
        pick = list(range(max(0, len(epoch_states) - int(SWA_K)), len(epoch_states)))
    swa = _average_states([epoch_states[i] for i in pick if 0 <= i < len(epoch_states)])
    if swa is not None:
        model.load_state_dict(swa)

    sc_va, film = _forward_numpy(model, pack, va_ids, device, z_t=z_t, m_t=m_t)
    under = bool(selected_epoch < int(UNDERTRAINED_BEST_LT))
    window_lo = max(1, selected_epoch - int(TRAIL_WINDOW) + 1)
    meta = {
        "fold_id": int(fold.fold_id),
        "init_id": int(init_id),
        "status": "ok",
        "selected_epoch": int(selected_epoch),
        "selected_epoch_window": [int(window_lo), int(selected_epoch)],
        "best_holdout_trail": float(best_trail) if best_trail > -1e8 else float("nan"),
        "undertrained": under,
        "n_train_dates": int(len(inner_tr)),
        "n_holdout_dates": int(len(inner_ho)),
        "n_val_dates": int(len(va_ids)),
        "n_params": int(model.n_params()),
        "warmstart_rules": int(n_ws),
        "train_curve": train_curve,
        "holdout_curve": ho_curve,
        "swa_epochs": [int(i + 1) for i in pick],
        "membership": {
            "c": model.c.detach().cpu().numpy().tolist(),
            "s": model.scales().detach().cpu().numpy().tolist(),
            "c_init": model.c_init.cpu().numpy().tolist(),
            "s_init": model.s_init.cpu().numpy().tolist(),
            "feat_names": list(model.feat_names),
        },
        "exponents": model.exponents().detach().cpu().numpy().astype(float).tolist(),
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
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "val_scores": sc_va,
        "va_ids": va_ids,
    }
    print(
        f"[p7d] fold={fold.fold_id} init={init_id} selected={selected_epoch} "
        f"UNDERTRAINED={under} elapsed={meta['elapsed']:.1f}s",
        flush=True,
    )
    return sc_va, meta


def _rows_from_scores(pack: PackedPanel, va_ids: np.ndarray, scores: np.ndarray, fold_id: int) -> pd.DataFrame:
    rows = []
    for di in va_ids:
        a, b = int(pack.starts[di]), int(pack.ends[di])
        dt = pd.Timestamp(pack.dates[int(di)])
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        dt = dt.tz_convert("UTC").normalize()
        for i in range(a, b):
            if not np.isfinite(scores[i]):
                continue
            rows.append({"date": dt, "id": int(pack.ids[i]), "fold_id": int(fold_id), "score": float(scores[i])})
    return pd.DataFrame(rows)


def train_one_fold(
    pack: PackedPanel,
    fold: FoldSpec,
    *,
    seed: int,
    prev_states: list[dict] | None = None,
    warm_blob: dict | None = None,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    device: str = "cpu",
    max_epochs: int = MAX_EPOCHS,
    n_inits: int = N_INITS,
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
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

    y_win = pack.y_win.copy()
    y_z = pack.y_win_z.copy()
    y_rank = pack.y_rank01.copy()
    excess = pack.excess.copy()
    if shuffle_labels:
        ss = int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017
        rng = np.random.default_rng(ss)
        inner_tr, inner_ho = _inner_split(pack, tr_ids)
        y_win, y_z, y_rank, excess = _shuffle_vol(
            pack, np.concatenate([inner_tr, inner_ho]), rng, y_win, y_z, y_rank, excess
        )

    bag = np.zeros(len(pack.z), dtype=np.float64)
    n_ok = 0
    init_metas = []
    out_states = []
    ics = []
    for init_id in range(int(n_inits)):
        wst = None
        if prev_states is not None and init_id < len(prev_states) and prev_states[init_id] is not None:
            wst = prev_states[init_id]
        sc, meta = train_one_init(
            pack,
            fold,
            seed=int(seed),
            init_id=int(init_id),
            warm_state=wst,
            warm_blob=warm_blob if (prev_states is None and not shuffle_labels) else None,
            y_win=y_win,
            y_z=y_z,
            device=device,
            max_epochs=int(max_epochs),
            excess=excess,
        )
        st = meta.pop("state", None)
        va_sc = meta.pop("val_scores", None)
        meta.pop("va_ids", None)
        init_metas.append(meta)
        if st is not None:
            out_states.append(st)
        else:
            out_states.append(None)
        if meta.get("status") == "ok" and va_sc is not None and len(va_sc):
            bag = np.where(np.isfinite(va_sc), bag + va_sc, bag)
            n_ok += 1
            ics.append(meta.get("best_holdout_trail"))

    if n_ok == 0:
        return pd.DataFrame(), {
            "fold_id": int(fold.fold_id),
            "status": "empty",
            "elapsed": time.time() - t0,
            "inits": init_metas,
        }
    bag = bag / float(n_ok)
    pred = _rows_from_scores(pack, va_ids, bag, int(fold.fold_id))
    finite_ics = [float(x) for x in ics if x is not None and np.isfinite(x)]
    init_spread = float(max(finite_ics) - min(finite_ics)) if len(finite_ics) >= 2 else 0.0
    selected = [int(m.get("selected_epoch") or 0) for m in init_metas]
    under = bool(any(m.get("undertrained") for m in init_metas))
    # representative membership from init 0
    last = init_metas[0] if init_metas else {}
    meta = {
        "fold_id": int(fold.fold_id),
        "status": "ok",
        "selected_epoch": int(np.median(selected)) if selected else 0,
        "selected_epoch_window": last.get("selected_epoch_window"),
        "init_spread_trail_ic": init_spread,
        "n_inits_ok": int(n_ok),
        "undertrained": under,
        "n_train_dates": last.get("n_train_dates"),
        "n_holdout_dates": last.get("n_holdout_dates"),
        "n_val_dates": last.get("n_val_dates"),
        "n_val_rows": int(len(pred)),
        "n_params": last.get("n_params"),
        "warmstart_rules": int(sum(m.get("warmstart_rules") or 0 for m in init_metas)),
        "inits": [{k: v for k, v in m.items() if k not in ("membership", "exponents", "film", "train_curve", "holdout_curve")} for m in init_metas],
        "membership": last.get("membership"),
        "exponents": last.get("exponents"),
        "rule_w": last.get("rule_w"),
        "film": last.get("film"),
        "train_curve": last.get("train_curve"),
        "holdout_curve": last.get("holdout_curve"),
        "elapsed": time.time() - t0,
        "gpu": False,
        "_states": out_states,
    }
    print(
        f"[p7d] fold={fold.fold_id} bagged n_inits={n_ok} spread={init_spread:.4f} "
        f"UNDERTRAINED={under} rows={len(pred)} elapsed={meta['elapsed']:.1f}s",
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
    cache_ver: str = CACHE_VER,
    commit_fn=None,
    n_inits: int = N_INITS,
) -> tuple[pd.DataFrame, list[dict]]:
    import json
    from pathlib import Path

    parts, metas = [], []
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    prev_states = None
    for fold in folds:
        pred, meta = None, None
        if cache_dir is not None:
            pq = cache_dir / f"preds_seed{int(seed)}_fold{fold.fold_id}.parquet"
            js = cache_dir / f"meta_seed{int(seed)}_fold{fold.fold_id}.json"
            st_path = cache_dir / f"state_seed{int(seed)}_fold{fold.fold_id}.pt"
            if pq.exists() and js.exists():
                try:
                    meta = json.loads(js.read_text())
                    if meta.get("cache_ver") == cache_ver and meta.get("status") == "ok":
                        pred = pd.read_parquet(pq)
                        meta["cache_hit"] = True
                        if st_path.exists():
                            import torch

                            prev_states = torch.load(st_path, map_location="cpu")
                        print(f"[p7d] fold={fold.fold_id} seed={seed} CACHE HIT rows={len(pred)}", flush=True)
                    else:
                        pred, meta = None, None
                except Exception as exc:
                    print(f"[p7d] fold={fold.fold_id} cache unreadable: {exc}", flush=True)
                    pred, meta = None, None
        if meta is None:
            pred, meta = train_one_fold(
                pack,
                fold,
                seed=seed,
                prev_states=prev_states,
                warm_blob=warm_blob,
                device=device,
                n_inits=int(n_inits),
            )
            states = meta.pop("_states", None)
            prev_states = states
            meta["cache_ver"] = cache_ver
            meta["cache_hit"] = False
            if cache_dir is not None and pred is not None and not pred.empty:
                pred.to_parquet(cache_dir / f"preds_seed{int(seed)}_fold{fold.fold_id}.parquet", index=False)
                slim = {k: v for k, v in meta.items() if k not in ("_states",)}
                (cache_dir / f"meta_seed{int(seed)}_fold{fold.fold_id}.json").write_text(json.dumps(slim, default=str))
                if states is not None:
                    import torch

                    torch.save(states, cache_dir / f"state_seed{int(seed)}_fold{fold.fold_id}.pt")
                if commit_fn is not None:
                    commit_fn()
        else:
            meta.pop("_states", None)
        metas.append(meta)
        if pred is not None and not pred.empty:
            parts.append(pred)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out, metas
