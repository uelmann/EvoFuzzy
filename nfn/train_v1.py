"""Phase 7.c NFN v1 trainer. Architecture unchanged; six craft items only."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from nfn.constants_v1 import (
    ADAMW_LR,
    ADAMW_LR_MIN,
    ADAMW_WD,
    ES_FLOOR_EPOCH,
    ES_PATIENCE,
    GRAD_CLIP,
    MAX_EPOCHS,
    N_BAG,
    SWA_K,
    TRAILING_K,
)
from nfn.craft_math import init_seed, pick_swa_epochs, split_inner_holdout, swa_average, trailing_mean
from nfn.data import PackedPanel, date_index_window
from nfn.loss import nfn_loss
from nfn.model import build_nfn
from nfn.train import (
    _fit_ir,
    _forward_numpy,
    _mean_tail_ic,
    _safe_calibrate,
    _sigmoid,
)


def _cpu_state(model) -> dict:
    import torch

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _shuffle_labels(pack: PackedPanel, date_ids: np.ndarray, seed: int):
    y_top = pack.y_top.copy()
    y_bot = pack.y_bot.copy()
    excess = pack.excess.copy()
    rng = np.random.default_rng(int(seed))
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
                y_top[idx] = y_top[idx][perm]
                y_bot[idx] = y_bot[idx][perm]
                excess[idx] = excess[idx][perm]
    return y_top, y_bot, excess


def train_one_fold_v1(
    pack: PackedPanel,
    fold,
    *,
    seed: int,
    bag: int = 0,
    prev_state: dict | None = None,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    device: str = "cpu",
    max_epochs: int = MAX_EPOCHS,
) -> tuple[pd.DataFrame, dict, dict | None]:
    import torch

    t0 = time.time()
    torch.manual_seed(init_seed(seed, bag) + int(fold.fold_id) + (0 if shuffle_seed is None else int(shuffle_seed)))
    np.random.seed((init_seed(seed, bag) + int(fold.fold_id)) % (2**31 - 1))

    tr_ids = date_index_window(pack, fold.train_start, fold.train_end)
    va_ids = date_index_window(pack, fold.val_start, fold.val_end)
    if len(tr_ids) < 20 or len(va_ids) < 5:
        return pd.DataFrame(), {
            "fold_id": int(fold.fold_id),
            "seed": int(seed),
            "bag": int(bag),
            "status": "empty",
            "n_train_dates": int(len(tr_ids)),
            "n_val_dates": int(len(va_ids)),
            "elapsed": time.time() - t0,
            "warmstart_from_prev_fold": bool(prev_state is not None),
        }, None

    inner_tr, inner_ho = split_inner_holdout(tr_ids, horizon=int(fold.horizon))
    y_top, y_bot, excess = pack.y_top, pack.y_bot, pack.excess
    if shuffle_labels:
        ss = int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017
        y_top, y_bot, excess = _shuffle_labels(pack, np.concatenate([inner_tr, inner_ho]), ss)

    model = build_nfn(init_seed(seed, bag) if prev_state is None else init_seed(seed, bag) + 17)
    if prev_state is not None:
        model.load_state_dict(prev_state)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(ADAMW_LR), weight_decay=float(ADAMW_WD))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(max_epochs), eta_min=float(ADAMW_LR_MIN)
    )

    epoch_states: list[dict] = []
    ho_ics: list[float] = []
    train_curve, ho_curve = [], []
    best_trail = -1e9
    best_epoch = 0
    stall = 0
    rng_ep = np.random.default_rng(init_seed(seed, bag) * 13 + int(fold.fold_id))

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))
            opt.step()
            ep_loss.append(parts["loss"])
        sched.step()

        lt_tr, lb_tr, _ = _forward_numpy(model, pack, inner_tr, device)
        lt_ho, lb_ho, _ = _forward_numpy(model, pack, inner_ho, device)
        sc_tr = _sigmoid(lt_tr) - _sigmoid(lb_tr)
        sc_ho = _sigmoid(lt_ho) - _sigmoid(lb_ho)
        ic_tr = _mean_tail_ic(pack, inner_tr, sc_tr)
        ic_ho = _mean_tail_ic(pack, inner_ho, sc_ho)
        epoch_states.append(_cpu_state(model))
        ho_ics.append(float(ic_ho) if np.isfinite(ic_ho) else float("nan"))
        trails = trailing_mean(ho_ics, TRAILING_K)
        trail = trails[-1]
        train_curve.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(ep_loss) if ep_loss else np.nan),
                "tail_ic": ic_tr,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        ho_curve.append({"epoch": epoch, "tail_ic": ic_ho, "trail3": trail})
        print(
            f"[nfn-v1 {time.strftime('%H:%M:%S')}] fold={fold.fold_id} seed={seed} bag={bag} "
            f"epoch={epoch}/{max_epochs} loss={train_curve[-1]['loss']:.4f} "
            f"train_tailIC={ic_tr:.4f} holdout_tailIC={ic_ho:.4f} trail3={trail:.4f}",
            flush=True,
        )
        improved = bool(np.isfinite(trail) and trail > best_trail + 1e-6)
        if epoch == 1 or improved:
            best_epoch = int(epoch)
            if np.isfinite(trail):
                best_trail = float(trail)
            stall = 0
        else:
            stall += 1
        if epoch >= int(ES_FLOOR_EPOCH) and stall >= int(ES_PATIENCE):
            print(
                f"[nfn-v1] fold={fold.fold_id} ES stop epoch={epoch} best_trail_epoch={best_epoch}",
                flush=True,
            )
            break

    trails = trailing_mean(ho_ics, TRAILING_K)
    swa_epochs = pick_swa_epochs(trails, SWA_K)
    swa_states = [epoch_states[e - 1] for e in swa_epochs if 1 <= e <= len(epoch_states)]
    if not swa_states:
        swa_states = [epoch_states[-1]]
        swa_epochs = [len(epoch_states)]
    swa_state = swa_average(swa_states)
    model.load_state_dict(swa_state)
    swa_span = int(max(swa_epochs) - min(swa_epochs)) if len(swa_epochs) >= 2 else 0

    lt_ho, lb_ho, _ = _forward_numpy(model, pack, inner_ho, device)
    p_top_ho = _sigmoid(lt_ho)
    p_bot_ho = _sigmoid(lb_ho)
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
                    "bag": int(bag),
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
        "seed": int(seed),
        "bag": int(bag),
        "status": "ok",
        "best_epoch": int(best_epoch),
        "selected_epoch": int(best_epoch),
        "selected_epoch_window": [int(x) for x in swa_epochs],
        "swa_span": int(swa_span),
        "best_holdout_trail3": float(best_trail) if np.isfinite(best_trail) and best_trail > -1e8 else float("nan"),
        "best_holdout_tail_ic": float(ho_ics[best_epoch - 1]) if 1 <= best_epoch <= len(ho_ics) else float("nan"),
        "n_train_dates": int(len(inner_tr)),
        "n_holdout_dates": int(len(inner_ho)),
        "n_val_dates": int(len(va_ids)),
        "n_val_rows": int(len(pred)),
        "n_params": int(model.n_params()),
        "warmstart_from_prev_fold": bool(prev_state is not None),
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
        "n_epochs_ran": int(len(epoch_states)),
        "es_floor": int(ES_FLOOR_EPOCH),
        "es_patience": int(ES_PATIENCE),
        "max_epochs": int(max_epochs),
        "n_bag": int(N_BAG),
        "lr": float(ADAMW_LR),
        "grad_clip": float(GRAD_CLIP),
    }
    print(
        f"[nfn-v1] fold={fold.fold_id} seed={seed} bag={bag} done window={swa_epochs} "
        f"span={swa_span} val_rows={len(pred)} elapsed={meta['elapsed']:.1f}s",
        flush=True,
    )
    return pred, meta, swa_state


def train_chain_v1(
    pack: PackedPanel,
    folds: list,
    *,
    seed: int,
    bag: int,
    device: str = "cpu",
    cache_dir=None,
    cache_ver: str = "p7cv1",
    commit_fn=None,
    cold_folds: tuple[int, ...] = (),
) -> tuple[pd.DataFrame, list[dict]]:
    import json
    from pathlib import Path

    import torch

    parts, metas = [], []
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    prev_state = None
    for fold in folds:
        pred, meta = None, None
        swa_path = None
        if cache_dir is not None:
            pq = cache_dir / f"preds_seed{int(seed)}_bag{int(bag)}_fold{fold.fold_id}.parquet"
            js = cache_dir / f"meta_seed{int(seed)}_bag{int(bag)}_fold{fold.fold_id}.json"
            swa_path = cache_dir / f"swa_seed{int(seed)}_bag{int(bag)}_fold{fold.fold_id}.pt"
            if pq.exists() and js.exists() and swa_path.exists():
                try:
                    meta = json.loads(js.read_text())
                    if meta.get("cache_ver") == cache_ver and meta.get("status") == "ok":
                        pred = pd.read_parquet(pq)
                        meta["cache_hit"] = True
                        try:
                            prev_state = torch.load(swa_path, map_location="cpu", weights_only=True)
                        except TypeError:
                            prev_state = torch.load(swa_path, map_location="cpu")
                        print(
                            f"[nfn-v1] fold={fold.fold_id} seed={seed} bag={bag} CACHE HIT rows={len(pred)}",
                            flush=True,
                        )
                    else:
                        pred, meta = None, None
                except Exception as exc:
                    print(f"[nfn-v1] fold={fold.fold_id} cache unreadable: {exc}", flush=True)
                    pred, meta = None, None
        if meta is None:
            use_prev = prev_state if int(fold.fold_id) not in set(int(x) for x in cold_folds) else None
            if int(fold.fold_id) == 0:
                use_prev = None
            pred, meta, swa_state = train_one_fold_v1(
                pack,
                fold,
                seed=int(seed),
                bag=int(bag),
                prev_state=use_prev,
                device=device,
            )
            meta["cache_ver"] = cache_ver
            meta["cache_hit"] = False
            prev_state = swa_state
            if cache_dir is not None and pred is not None and not pred.empty:
                pred.to_parquet(cache_dir / f"preds_seed{int(seed)}_bag{int(bag)}_fold{fold.fold_id}.parquet", index=False)
                (cache_dir / f"meta_seed{int(seed)}_bag{int(bag)}_fold{fold.fold_id}.json").write_text(
                    json.dumps(meta, default=str)
                )
                torch.save(swa_state, cache_dir / f"swa_seed{int(seed)}_bag{int(bag)}_fold{fold.fold_id}.pt")
                if commit_fn is not None:
                    commit_fn()
        else:
            if prev_state is None and swa_path is not None and swa_path.exists():
                try:
                    prev_state = torch.load(swa_path, map_location="cpu", weights_only=True)
                except TypeError:
                    prev_state = torch.load(swa_path, map_location="cpu")
        metas.append(meta)
        if pred is not None and not pred.empty:
            parts.append(pred)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out, metas
