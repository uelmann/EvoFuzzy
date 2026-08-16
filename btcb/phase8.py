"""Phase 8 MODEL-ZOO: CS-ATTN, TabPFN v2, ridge-on-ranks. Backtest only."""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from btcb.constants import (
    CS_CLIP,
    FUTURE_NULL_BIAS_MIN_VIOLATIONS,
    NULL_SHUFFLE_SEEDS,
    PHASE8_ATTN_CAP,
    PHASE8_ATTN_CLIP,
    PHASE8_ATTN_D_MODEL,
    PHASE8_ATTN_DATE_BATCH,
    PHASE8_ATTN_ES_FLOOR,
    PHASE8_ATTN_FF,
    PHASE8_ATTN_HEADS,
    PHASE8_ATTN_LAYERS,
    PHASE8_ATTN_LR,
    PHASE8_ATTN_LR_MIN,
    PHASE8_ATTN_PATIENCE,
    PHASE8_ATTN_SWA,
    PHASE8_ATTN_WD,
    PHASE8_DIAG_DATES,
    PHASE8_H,
    PHASE8_INNER_HOLDOUT_DATES,
    PHASE8_LINEAR_CEILING_RATIO,
    PHASE8_NULL_FOLD_IDS,
    PHASE8_NULL_K_EXCEED,
    PHASE8_NULL_REPLICATES,
    PHASE8_ORTH_CORR,
    PHASE8_ORTH_RANKIC,
    PHASE8_OVERLAP_DELTA,
    PHASE8_RANKIC_DELTA,
    PHASE8_RIDGE_ALPHAS,
    PHASE8_SEEDS,
    PHASE8_SUBSAMPLE_OFFSET,
    PHASE8_SUBSAMPLE_STRIDE,
    PHASE8_TABPFN_CONTEXT_CAP,
    PHASE8_TABPFN_N_ESTIMATORS,
    PHASE8_TAIL_IC_DELTA,
    STAGE_S_COLS,
    STOUFFER_Z_MIN,
)
from btcb.features import apply_cs_zscore
from btcb.gates import metric_verdict_e1b_house
from btcb.model import (
    FoldSpec,
    apply_calibrator,
    fit_isotonic,
    mean_per_date_rank_ic,
    shuffle_labels_frame,
)
from btcb.phase4b import cell_stats_vol_matched, fold_tail_pack, vol_col_name
from btcb.phase4v2 import _utc, collapse_fold_preds, per_date_tail_metrics, restrict_eval_frame
from btcb.oracle_ladder import _as_utc, _spearman


def _log(msg: str) -> None:
    print(f"[p8 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fold_to_dict(fold: FoldSpec) -> dict:
    return {
        "fold_id": int(fold.fold_id),
        "train_start": str(fold.train_start),
        "train_end": str(fold.train_end),
        "purge_end": str(fold.purge_end),
        "embargo_end": str(fold.embargo_end),
        "val_start": str(fold.val_start),
        "val_end": str(fold.val_end),
        "horizon": int(fold.horizon),
    }


def fold_from_dict(d: dict) -> FoldSpec:
    def _ts(x):
        t = pd.Timestamp(x)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t.normalize()

    return FoldSpec(
        fold_id=int(d["fold_id"]),
        train_start=_ts(d["train_start"]),
        train_end=_ts(d["train_end"]),
        purge_end=_ts(d["purge_end"]),
        embargo_end=_ts(d["embargo_end"]),
        val_start=_ts(d["val_start"]),
        val_end=_ts(d["val_end"]),
        horizon=int(d["horizon"]),
    )


def inner_holdout_dates(train_dates: list, horizon: int = PHASE8_H, n_ho: int = PHASE8_INNER_HOLDOUT_DATES):
    dts = sorted(pd.to_datetime(train_dates, utc=True))
    if len(dts) < max(20, int(n_ho) + 5):
        n_ho = max(5, len(dts) // 5)
    ho = dts[-int(n_ho) :]
    ho_start = ho[0]
    cut = ho_start - pd.Timedelta(days=int(horizon))
    tr = [d for d in dts if d <= cut]
    if not tr:
        tr = dts[: max(1, len(dts) - len(ho))]
        ho = dts[len(tr) :]
    return tr, ho


def subsample_oos_dates(dates, stride: int = PHASE8_SUBSAMPLE_STRIDE, offset: int = PHASE8_SUBSAMPLE_OFFSET):
    dts = sorted(pd.to_datetime(pd.unique(dates), utc=True))
    keep = [d for i, d in enumerate(dts) if (i % int(stride)) == int(offset)]
    return pd.DatetimeIndex(keep).tz_convert("UTC").normalize()


def restrict_dates(df: pd.DataFrame, dates) -> pd.DataFrame:
    if dates is None:
        return df
    idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).tz_convert("UTC").normalize()
    d = df.copy()
    d["date"] = _utc(d["date"])
    return d[d["date"].isin(set(idx))].copy()


def prepare_feature_frame(labeled: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    cols = list(feature_cols or STAGE_S_COLS)
    out = labeled.copy()
    out["date"] = _utc(out["date"])
    out["id"] = out["id"].astype(int)
    out = apply_cs_zscore(out, cols, clip=CS_CLIP)
    for c in cols:
        out[f"rk_{c}"] = out.groupby("date", sort=False)[c].rank(method="average", pct=True)
    ex = f"excess_h{PHASE8_H}"
    if ex in out.columns:
        out["y_rank_pct"] = out.groupby("date", sort=False)[ex].rank(method="average", pct=True)
    return out


def _date_groups(df: pd.DataFrame) -> dict:
    d = df.sort_values(["date", "id"])
    out = {}
    for dt, g in d.groupby("date", sort=True):
        out[_as_utc(dt)] = g
    return out


def _rankic_np(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) < 8 or np.unique(p).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(_spearman(p, y))


def mean_date_rankic_groups(groups: dict, dates, score_col: str, ycol: str) -> float:
    ics = []
    for dt in dates:
        g = groups.get(_as_utc(dt))
        if g is None or g.empty:
            continue
        v = _rankic_np(g[score_col], g[ycol] if ycol in g.columns else g.get("excess_h14"))
        if np.isfinite(v):
            ics.append(v)
    return float(np.mean(ics)) if ics else float("nan")


# ---------------------------------------------------------------------------
# Arm A — CS-ATTN
# ---------------------------------------------------------------------------

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class CoinSetAttn:
    """Thin wrapper so the class can be built after torch import."""

    def __init__(self, n_in: int = 33):
        torch, nn, F = _torch()

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = nn.LayerNorm(PHASE8_ATTN_D_MODEL)
                self.attn = nn.MultiheadAttention(
                    PHASE8_ATTN_D_MODEL,
                    PHASE8_ATTN_HEADS,
                    dropout=0.0,
                    batch_first=True,
                )
                self.n2 = nn.LayerNorm(PHASE8_ATTN_D_MODEL)
                self.ff = nn.Sequential(
                    nn.Linear(PHASE8_ATTN_D_MODEL, PHASE8_ATTN_FF),
                    nn.GELU(),
                    nn.Linear(PHASE8_ATTN_FF, PHASE8_ATTN_D_MODEL),
                )

            def forward(self, x, key_padding_mask, need_weights=False):
                h = self.n1(x)
                kw = dict(key_padding_mask=key_padding_mask, need_weights=need_weights)
                try:
                    a, w = self.attn(h, h, h, average_attn_weights=False if need_weights else True, **kw)
                except TypeError:
                    a, w = self.attn(h, h, h, **kw)
                x = x + a
                x = x + self.ff(self.n2(x))
                return x, w

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(n_in, PHASE8_ATTN_D_MODEL)
                self.blocks = nn.ModuleList([_Block() for _ in range(PHASE8_ATTN_LAYERS)])
                self.head_top = nn.Linear(PHASE8_ATTN_D_MODEL, 1)
                self.head_bot = nn.Linear(PHASE8_ATTN_D_MODEL, 1)

            def forward(self, x, key_padding_mask, need_weights=False):
                h = self.embed(x)
                weights = []
                for blk in self.blocks:
                    h, w = blk(h, key_padding_mask, need_weights=need_weights)
                    if need_weights:
                        weights.append(w)
                top = self.head_top(h).squeeze(-1)
                bot = self.head_bot(h).squeeze(-1)
                return top, bot, weights

        self.torch = torch
        self.nn = nn
        self.F = F
        self.net = _Net()

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def to(self, device):
        self.net.to(device)
        return self


def attn_config_dump(n_params: int) -> dict:
    return {
        "n_in": 33,
        "d_model": PHASE8_ATTN_D_MODEL,
        "n_heads": PHASE8_ATTN_HEADS,
        "n_layers": PHASE8_ATTN_LAYERS,
        "dim_feedforward": PHASE8_ATTN_FF,
        "dropout": 0.0,
        "positional_encoding": False,
        "temporal_encoder": False,
        "lr": PHASE8_ATTN_LR,
        "lr_min": PHASE8_ATTN_LR_MIN,
        "weight_decay": PHASE8_ATTN_WD,
        "clip": PHASE8_ATTN_CLIP,
        "es_floor": PHASE8_ATTN_ES_FLOOR,
        "patience": PHASE8_ATTN_PATIENCE,
        "cap": PHASE8_ATTN_CAP,
        "swa_window": PHASE8_ATTN_SWA,
        "inner_holdout_dates": PHASE8_INNER_HOLDOUT_DATES,
        "seeds": list(PHASE8_SEEDS),
        "date_batch": PHASE8_ATTN_DATE_BATCH,
        "optimizer": "AdamW",
        "scheduler": "cosine",
    }


class CSPanel:
    """Padded [n_dates × n_max × F] cache so CS-ATTN epochs do not regroup pandas."""

    def __init__(self, dates, X, yt, yb, yx, ids, valid, vol):
        self.dates = list(dates)
        self.date_to_i = {_as_utc(d): i for i, d in enumerate(self.dates)}
        self.X = X
        self.yt = yt
        self.yb = yb
        self.yx = yx
        self.ids = ids
        self.valid = valid  # True = real coin
        self.vol = vol

    def idx_for(self, dates) -> np.ndarray:
        out = []
        for d in dates:
            i = self.date_to_i.get(_as_utc(d))
            if i is not None:
                out.append(i)
        return np.asarray(out, dtype=np.int64)

    def clone_labels(self):
        return self.yt.copy(), self.yb.copy(), self.yx.copy()


def build_cs_panel(df: pd.DataFrame, feat_cols: list[str], vol_col: str | None = None) -> CSPanel:
    d = df.copy()
    d["date"] = _utc(d["date"])
    d["id"] = d["id"].astype(int)
    dates = sorted(d["date"].unique())
    sizes = d.groupby("date").size()
    nmax = int(sizes.max()) if len(sizes) else 1
    n_d, n_f = len(dates), len(feat_cols)
    X = np.zeros((n_d, nmax, n_f), dtype=np.float32)
    yt = np.full((n_d, nmax), np.nan, dtype=np.float32)
    yb = np.full((n_d, nmax), np.nan, dtype=np.float32)
    yx = np.full((n_d, nmax), np.nan, dtype=np.float32)
    ids = np.full((n_d, nmax), -1, dtype=np.int64)
    valid = np.zeros((n_d, nmax), dtype=bool)
    vol = np.full((n_d, nmax), np.nan, dtype=np.float32)
    ytc, ybc, exc = f"y_h{PHASE8_H}", f"y_bot_h{PHASE8_H}", f"excess_h{PHASE8_H}"
    vc = vol_col if vol_col and vol_col in d.columns else None
    for i, (dt, g) in enumerate(d.groupby("date", sort=True)):
        n = len(g)
        x = np.nan_to_num(g[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X[i, :n] = x
        if ytc in g.columns:
            yt[i, :n] = g[ytc].to_numpy(dtype=np.float32)
        if ybc in g.columns:
            yb[i, :n] = g[ybc].to_numpy(dtype=np.float32)
        if exc in g.columns:
            yx[i, :n] = g[exc].to_numpy(dtype=np.float32)
        ids[i, :n] = g["id"].to_numpy(dtype=np.int64)
        valid[i, :n] = True
        if vc:
            vol[i, :n] = pd.to_numeric(g[vc], errors="coerce").to_numpy(dtype=np.float32)
    return CSPanel(dates, X, yt, yb, yx, ids, valid, vol)


def shuffle_panel_labels(panel: CSPanel, date_idx: np.ndarray, rng: np.random.Generator):
    from btcb.model import vol_bucket_ids

    yt, yb, yx = panel.clone_labels()
    for i in date_idx:
        m = np.flatnonzero(panel.valid[i])
        if len(m) <= 1:
            continue
        arr = np.stack([yt[i, m], yb[i, m], yx[i, m]], axis=1)
        buckets = vol_bucket_ids(panel.vol[i, m])
        for b in np.unique(buckets):
            idx = np.flatnonzero(buckets == b)
            if len(idx) > 1:
                perm = rng.permutation(len(idx))
                arr[idx] = arr[idx][perm]
        yt[i, m], yb[i, m], yx[i, m] = arr[:, 0], arr[:, 1], arr[:, 2]
    return yt, yb, yx


def _pack_dates(groups: dict, dates, feat_cols: list[str], device, torch):
    blobs = []
    nmax = 1
    for dt in dates:
        g = groups.get(_as_utc(dt))
        if g is None or g.empty:
            continue
        x = np.nan_to_num(g[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        nmax = max(nmax, x.shape[0])
        blobs.append(
            (
                _as_utc(dt),
                x,
                g[f"y_h{PHASE8_H}"].to_numpy(dtype=np.float32) if f"y_h{PHASE8_H}" in g.columns else np.full(len(g), np.nan, np.float32),
                g[f"y_bot_h{PHASE8_H}"].to_numpy(dtype=np.float32) if f"y_bot_h{PHASE8_H}" in g.columns else np.full(len(g), np.nan, np.float32),
                g[f"excess_h{PHASE8_H}"].to_numpy(dtype=np.float32) if f"excess_h{PHASE8_H}" in g.columns else np.full(len(g), np.nan, np.float32),
                g["id"].to_numpy(dtype=np.int64),
            )
        )
    if not blobs:
        return None
    bsz = len(blobs)
    X = np.zeros((bsz, nmax, len(feat_cols)), dtype=np.float32)
    Yt = np.full((bsz, nmax), np.nan, dtype=np.float32)
    Yb = np.full((bsz, nmax), np.nan, dtype=np.float32)
    Yx = np.full((bsz, nmax), np.nan, dtype=np.float32)
    I = np.full((bsz, nmax), -1, dtype=np.int64)
    M = np.ones((bsz, nmax), dtype=bool)
    dts = []
    for i, (dt, x, yt, yb, yx, iid) in enumerate(blobs):
        n = x.shape[0]
        X[i, :n] = x
        Yt[i, :n] = yt
        Yb[i, :n] = yb
        Yx[i, :n] = yx
        I[i, :n] = iid
        M[i, :n] = False
        dts.append(dt)
    return {
        "X": torch.from_numpy(X).to(device),
        "yt": Yt,
        "yb": Yb,
        "yx": Yx,
        "ids": I,
        "pad": torch.from_numpy(M).to(device),
        "dates": dts,
        "n": nmax,
    }


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _swa_state(states: list) -> dict:
    if not states:
        return {}
    avg = copy.deepcopy(states[0])
    n = float(len(states))
    for k in avg:
        if hasattr(avg[k], "float"):
            s = states[0][k].float()
            for st in states[1:]:
                s = s + st[k].float()
            avg[k] = (s / n).to(states[0][k].dtype)
    return avg


def train_cs_attn_fold(
    panel: CSPanel,
    fold: FoldSpec,
    feat_cols: list[str],
    seed: int,
    init_state: dict | None = None,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict, dict]:
    torch, nn, F = _torch()
    torch.manual_seed(int(seed) + int(fold.fold_id))
    np.random.seed(int(seed) + int(fold.fold_id))
    t0 = time.time()
    wrapper = CoinSetAttn(n_in=len(feat_cols))
    wrapper.to(device)
    net = wrapper.net
    if init_state:
        net.load_state_dict(init_state)
    n_params = wrapper.n_params()

    all_dates = panel.dates
    train_dates = [d for d in all_dates if fold.train_start <= d <= fold.train_end]
    val_dates = [d for d in all_dates if fold.val_start <= d <= fold.val_end]
    tr_dates, ho_dates = inner_holdout_dates(train_dates, fold.horizon, PHASE8_INNER_HOLDOUT_DATES)
    tr_i = panel.idx_for(tr_dates)
    ho_i = panel.idx_for(ho_dates)
    val_i = panel.idx_for(val_dates)

    yt, yb, yx = panel.yt, panel.yb, panel.yx
    if shuffle_labels:
        rng = np.random.default_rng(int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017)
        sh_i = np.concatenate([tr_i, ho_i]) if len(tr_i) or len(ho_i) else np.array([], dtype=np.int64)
        yt, yb, yx = shuffle_panel_labels(panel, sh_i, rng)

    X_t = torch.from_numpy(panel.X)
    pad_t = torch.from_numpy(~panel.valid)  # True = PAD
    opt = torch.optim.AdamW(net.parameters(), lr=PHASE8_ATTN_LR, weight_decay=PHASE8_ATTN_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE8_ATTN_CAP, eta_min=PHASE8_ATTN_LR_MIN)
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    recent = deque(maxlen=PHASE8_ATTN_SWA)
    best_trail = -1e18
    best_states = None
    bad = 0
    history = []
    rng_dates = np.random.default_rng(int(seed) + 17)

    def _slice(idx):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return None
        return (
            X_t[idx].to(device),
            pad_t[idx].to(device),
            yt[idx],
            yb[idx],
            yx[idx],
            panel.ids[idx],
            [panel.dates[int(i)] for i in idx],
        )

    def _eval(idx):
        net.eval()
        ics = []
        with torch.inference_mode():
            for chunk in _batches(list(idx), PHASE8_ATTN_DATE_BATCH):
                sl = _slice(chunk)
                if sl is None:
                    continue
                xb, pad, _yt, _yb, _yx, _ids, _dts = sl
                top, bot, _ = net(xb, pad, need_weights=False)
                p = (torch.sigmoid(top) - torch.sigmoid(bot)).cpu().numpy()
                padn = pad.cpu().numpy()
                for i in range(p.shape[0]):
                    m = ~padn[i]
                    ics.append(_rankic_np(p[i, m], _yx[i, m]))
        ics = [x for x in ics if np.isfinite(x)]
        return float(np.mean(ics)) if ics else float("nan")

    def _predict(idx, y_top_src, y_bot_src) -> pd.DataFrame:
        net.eval()
        rows = []
        with torch.inference_mode():
            for chunk in _batches(list(idx), PHASE8_ATTN_DATE_BATCH):
                sl = _slice(chunk)
                if sl is None:
                    continue
                xb, pad, _yt, _yb, _yx, ids, dts = sl
                top, bot, _ = net(xb, pad, need_weights=False)
                pt = torch.sigmoid(top).cpu().numpy()
                pb = torch.sigmoid(bot).cpu().numpy()
                padn = pad.cpu().numpy()
                # y sources: use the arrays passed in (work labels for ho, original for val)
                for i, dt in enumerate(dts):
                    m = np.flatnonzero(~padn[i])
                    di = panel.date_to_i[_as_utc(dt)]
                    for j in m:
                        rows.append(
                            {
                                "date": dt,
                                "id": int(ids[i, j]),
                                "p_top": float(pt[i, j]),
                                "p_bot": float(pb[i, j]),
                                "p_raw": float(pt[i, j] - pb[i, j]),
                                f"y_h{PHASE8_H}": float(y_top_src[di, j]) if np.isfinite(y_top_src[di, j]) else np.nan,
                                f"y_bot_h{PHASE8_H}": float(y_bot_src[di, j]) if np.isfinite(y_bot_src[di, j]) else np.nan,
                            }
                        )
        return pd.DataFrame(rows)

    for epoch in range(1, PHASE8_ATTN_CAP + 1):
        net.train()
        order = list(tr_i)
        rng_dates.shuffle(order)
        losses = []
        for chunk in _batches(order, PHASE8_ATTN_DATE_BATCH):
            sl = _slice(chunk)
            if sl is None:
                continue
            xb, pad, ytt, ybb, _yx, _ids, _dts = sl
            opt.zero_grad(set_to_none=True)
            top, bot, _ = net(xb, pad, need_weights=False)
            ytt_t = torch.from_numpy(np.asarray(ytt, dtype=np.float32)).to(device)
            ybb_t = torch.from_numpy(np.asarray(ybb, dtype=np.float32)).to(device)
            good = (~pad) & torch.isfinite(ytt_t) & torch.isfinite(ybb_t)
            if int(good.sum()) < 8:
                continue
            loss = bce(top[good], ytt_t[good].clamp(0, 1)) + bce(bot[good], ybb_t[good].clamp(0, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), PHASE8_ATTN_CLIP)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val = _eval(ho_i)
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        recent.append(state)
        history.append(val)
        if epoch >= PHASE8_ATTN_SWA:
            trail = float(np.nanmean(history[-PHASE8_ATTN_SWA :]))
        else:
            trail = float(val) if np.isfinite(val) else -1e18
        if np.isfinite(trail) and trail > best_trail:
            best_trail = trail
            best_states = list(recent)
            bad = 0
        elif epoch >= PHASE8_ATTN_ES_FLOOR:
            bad += 1
        _log(
            f"cs-attn fold={fold.fold_id} seed={seed} epoch={epoch} "
            f"loss={np.mean(losses) if losses else float('nan'):.4f} ho_rankic={val:.4f} "
            f"trail={trail:.4f} bad={bad}"
        )
        if epoch >= PHASE8_ATTN_ES_FLOOR and bad >= PHASE8_ATTN_PATIENCE:
            break

    swa = _swa_state(best_states or list(recent))
    if swa:
        net.load_state_dict(swa)
    pred_val = _predict(val_i, panel.yt, panel.yb)
    pred_ho = _predict(ho_i, yt, yb)
    for d in (pred_val, pred_ho):
        if not d.empty:
            d["fold_id"] = int(fold.fold_id)
            d["seed"] = int(seed)
    meta = {
        "fold_id": int(fold.fold_id),
        "seed": int(seed),
        "status": "ok" if not pred_val.empty else "empty",
        "n_params": n_params,
        "epochs": int(len(history)),
        "best_trail_rankic": float(best_trail) if np.isfinite(best_trail) else None,
        "elapsed": time.time() - t0,
        "n_train_dates": int(len(tr_dates)),
        "n_ho_dates": int(len(ho_dates)),
        "n_val_dates": int(len(val_dates)),
        "warm_start": bool(init_state is not None),
    }
    return pred_val, {"ho": pred_ho, "state": swa, "n_params": n_params}, meta


def bag_and_calibrate(seed_vals: list[pd.DataFrame], seed_hos: list[pd.DataFrame], ho_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if not seed_vals:
        return pd.DataFrame()
    val = seed_vals[0][["date", "id", "fold_id"]].copy()
    val["date"] = _utc(val["date"])
    val["id"] = val["id"].astype(int)
    pt, pb = [], []
    for d in seed_vals:
        x = d.copy()
        x["date"] = _utc(x["date"])
        x["id"] = x["id"].astype(int)
        m = val.merge(x[["date", "id", "p_top", "p_bot"]], on=["date", "id"], how="left")
        pt.append(m["p_top"].to_numpy(dtype=float))
        pb.append(m["p_bot"].to_numpy(dtype=float))
    val["p_top"] = np.nanmean(np.vstack(pt), axis=0)
    val["p_bot"] = np.nanmean(np.vstack(pb), axis=0)

    ho_parts = [d for d in seed_hos if d is not None and not d.empty]
    ir_top = ir_bot = None
    if ho_parts:
        ho = ho_parts[0][["date", "id"]].copy()
        ho["date"] = _utc(ho["date"])
        ho["id"] = ho["id"].astype(int)
        hpt, hpb = [], []
        for d in ho_parts:
            x = d.copy()
            x["date"] = _utc(x["date"])
            x["id"] = x["id"].astype(int)
            m = ho.merge(x[["date", "id", "p_top", "p_bot"]], on=["date", "id"], how="left")
            hpt.append(m["p_top"].to_numpy(dtype=float))
            hpb.append(m["p_bot"].to_numpy(dtype=float))
        ho["p_top"] = np.nanmean(np.vstack(hpt), axis=0)
        ho["p_bot"] = np.nanmean(np.vstack(hpb), axis=0)
        ycols = [f"y_h{PHASE8_H}", f"y_bot_h{PHASE8_H}"]
        if all(c in ho_parts[0].columns for c in ycols):
            lab = ho_parts[0][["date", "id", *ycols]].copy()
            lab["date"] = _utc(lab["date"])
            lab["id"] = lab["id"].astype(int)
            ho = ho.merge(lab, on=["date", "id"], how="inner")
        elif ho_frame is not None and not ho_frame.empty:
            lab = ho_frame[["date", "id", *ycols]].copy()
            lab["date"] = _utc(lab["date"])
            lab["id"] = lab["id"].astype(int)
            ho = ho.merge(lab, on=["date", "id"], how="inner")
        if f"y_h{PHASE8_H}" in ho.columns:
            ir_top = fit_isotonic(ho["p_top"].to_numpy(), ho[f"y_h{PHASE8_H}"].to_numpy())
            ir_bot = fit_isotonic(ho["p_bot"].to_numpy(), ho[f"y_bot_h{PHASE8_H}"].to_numpy())
    val["p_top"] = apply_calibrator(ir_top, val["p_top"].to_numpy())
    val["p_bot"] = apply_calibrator(ir_bot, val["p_bot"].to_numpy())
    val["signal"] = val["p_top"] - val["p_bot"]
    return val


def train_cs_attn_all_folds(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    feat_cols: list[str],
    seeds: tuple[int, ...] = PHASE8_SEEDS,
    ping=None,
    single_seed: int | None = None,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    panel = build_cs_panel(df, feat_cols, vol_col=vol_col_name(df) if "yz_vol_30" in df.columns or "yz_vol_30_raw" in df.columns else None)
    use_seeds = (int(single_seed),) if single_seed is not None else tuple(int(s) for s in seeds)
    warm = {s: None for s in use_seeds}
    parts = []
    seed_parts = []
    metas = []
    n_params = None
    fold_states = {}
    for fold in folds:
        if ping:
            ping(f"cs-attn fold={fold.fold_id}")
        sval, sho = [], []
        for seed in use_seeds:
            pred, extra, meta = train_cs_attn_fold(
                panel,
                fold,
                feat_cols,
                seed=seed,
                init_state=warm[seed] if not shuffle_labels else None,
                shuffle_labels=shuffle_labels,
                shuffle_seed=shuffle_seed,
            )
            warm[seed] = extra.get("state")
            n_params = extra.get("n_params") or n_params
            metas.append(meta)
            if seed == 42 and extra.get("state"):
                fold_states[int(fold.fold_id)] = extra.get("state")
            if pred is None or pred.empty:
                continue
            sval.append(pred)
            seed_parts.append(pred)
            sho.append(extra.get("ho"))
        bag = bag_and_calibrate(sval, sho, None)
        if not bag.empty:
            parts.append(bag)
            _log(f"cs-attn fold={fold.fold_id} bagged rows={len(bag)}")
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    seed_df = pd.concat(seed_parts, ignore_index=True) if seed_parts else pd.DataFrame()
    return preds, {
        "meta": metas,
        "n_params": n_params,
        "seed_preds": seed_df,
        "fold_states": fold_states,
        "config": attn_config_dump(int(n_params or 0)),
    }


def attention_diagnostics(
    df: pd.DataFrame,
    groups: dict,
    folds: list[FoldSpec],
    feat_cols: list[str],
    states: dict,
    sample_dates: list,
    score_df: pd.DataFrame,
) -> dict:
    torch, nn, F = _torch()
    device = "cpu"
    wrapper = CoinSetAttn(n_in=len(feat_cols))
    wrapper.to(device)
    net = wrapper.net
    rows = []
    peers = []
    score_df = score_df.copy()
    score_df["date"] = _utc(score_df["date"])
    for dt in sample_dates:
        dt = _as_utc(dt)
        fold = None
        for f in folds:
            if f.val_start <= dt <= f.val_end:
                fold = f
        if fold is None:
            continue
        st = states.get(int(fold.fold_id))
        if not st:
            continue
        net.load_state_dict(st)
        net.eval()
        pack = _pack_dates(groups, [dt], feat_cols, device, torch)
        if pack is None:
            continue
        with torch.no_grad():
            top, bot, weights = net(pack["X"], pack["pad"], need_weights=True)
        pad = pack["pad"].cpu().numpy()[0]
        m = ~pad
        n = int(m.sum())
        if n < 2:
            continue
        ents = []
        attn_mean = None
        for w in weights:
            if w is None:
                continue
            ww = w.detach().cpu().numpy()
            if ww.ndim == 4:
                ww = ww[0]
            if ww.ndim == 2:
                ww = ww[None, ...]
            # ww [heads, L, L]
            sub = ww[:, m][:, :, m]
            sub = np.clip(sub, 1e-12, 1.0)
            ent = -(sub * np.log(sub)).sum(axis=-1) / np.log(n)
            ents.append(float(ent.mean()))
            am = sub.mean(axis=0)
            attn_mean = am if attn_mean is None else attn_mean + am
        if attn_mean is None:
            continue
        attn_mean = attn_mean / max(len(weights), 1)
        ids = pack["ids"][0, m]
        p = (torch.sigmoid(top) - torch.sigmoid(bot)).cpu().numpy()[0, m]
        q = int(np.nanargmax(p))
        order = np.argsort(-attn_mean[q])
        top5 = []
        for j in order[:5]:
            top5.append({"id": int(ids[j]), "weight": float(attn_mean[q, j]), "is_self": bool(j == q)})
        rows.append(
            {
                "date": str(dt.date()),
                "n_coins": n,
                "mean_entropy": float(np.mean(ents)) if ents else None,
                "self_weight": float(attn_mean[q, q]),
                "max_id": int(ids[q]),
            }
        )
        peers.append({"date": str(dt.date()), "query_id": int(ids[q]), "top5": top5, "mean_entropy": rows[-1]["mean_entropy"]})
        _log(f"attn diag {dt.date()} entropy={rows[-1]['mean_entropy']:.3f} self_w={rows[-1]['self_weight']:.3f}")
    mean_ent = float(np.nanmean([r["mean_entropy"] for r in rows])) if rows else float("nan")
    collapse = float(np.nanmean([r["self_weight"] for r in rows])) if rows else float("nan")
    return {
        "n_dates": int(len(rows)),
        "mean_entropy": mean_ent,
        "mean_self_weight": collapse,
        "per_date": rows,
        "top5_peers": peers,
        "collapse_to_self": bool(np.isfinite(collapse) and collapse >= 0.50),
    }


def collect_seed42_states(df, folds, feat_cols, ping=None):
    """Retrain is expensive; states are returned from the live training warm dict instead."""
    return {}


# ---------------------------------------------------------------------------
# Arm C — ridge on ranks
# ---------------------------------------------------------------------------

def ridge_config_dump() -> dict:
    return {
        "features": "cs_percentile_ranks_of_STAGE_S_COLS",
        "target": "cs_percentile_rank_of_excess_h14",
        "model": "sklearn.linear_model.Ridge",
        "fit_intercept": True,
        "alpha_grid": list(PHASE8_RIDGE_ALPHAS),
        "alpha_rule": "max inner-ho RankIC; ties → larger alpha",
        "inner_holdout_dates": PHASE8_INNER_HOLDOUT_DATES,
    }


def _xy_ridge(g: pd.DataFrame, feat_rank_cols: list[str]):
    x = g[feat_rank_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.5, posinf=0.5, neginf=0.5)
    y = g["y_rank_pct"].to_numpy(dtype=float) if "y_rank_pct" in g.columns else g[f"excess_h{PHASE8_H}"].to_numpy(dtype=float)
    return x, y


def fit_ridge_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    feat_rank_cols: list[str],
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    d = df.copy()
    d["date"] = _utc(d["date"])
    train = d[(d["date"] >= fold.train_start) & (d["date"] <= fold.train_end)].copy()
    valid = d[(d["date"] >= fold.val_start) & (d["date"] <= fold.val_end)].copy()
    if train.empty or valid.empty:
        return pd.DataFrame(), {"fold_id": fold.fold_id, "status": "empty"}
    if shuffle_labels:
        rng = np.random.default_rng(int(shuffle_seed) if shuffle_seed is not None else 90_017)
        cols = [f"y_h{PHASE8_H}", f"y_bot_h{PHASE8_H}", f"excess_h{PHASE8_H}", "y_rank_pct"]
        train = shuffle_labels_frame(train, cols, rng, mode="vol_matched")
    tr_dates, ho_dates = inner_holdout_dates(sorted(train["date"].unique()), fold.horizon)
    inner_tr = train[train["date"].isin(set(tr_dates))]
    inner_ho = train[train["date"].isin(set(ho_dates))]
    best_a, best_ic = None, -1e18
    grid_scores = []
    for a in PHASE8_RIDGE_ALPHAS:
        m = np.isfinite(inner_tr["y_rank_pct"].to_numpy()) if "y_rank_pct" in inner_tr.columns else np.isfinite(inner_tr[f"excess_h{PHASE8_H}"].to_numpy())
        X, y = _xy_ridge(inner_tr.loc[m], feat_rank_cols)
        if len(y) < 50 or np.unique(np.round(y, 6)).size < 5:
            grid_scores.append({"alpha": a, "ho_rankic": None})
            continue
        mdl = Ridge(alpha=float(a), fit_intercept=True)
        mdl.fit(X, np.nan_to_num(y, nan=np.nanmedian(y)))
        ho = inner_ho.copy()
        Xh, _ = _xy_ridge(ho, feat_rank_cols)
        ho["pred"] = mdl.predict(Xh)
        ic = mean_per_date_rank_ic(ho["pred"].to_numpy(), ho[f"excess_h{PHASE8_H}"].to_numpy(), ho["date"].to_numpy())
        grid_scores.append({"alpha": float(a), "ho_rankic": float(ic) if np.isfinite(ic) else None})
        if np.isfinite(ic) and (ic > best_ic or (abs(ic - best_ic) < 1e-12 and (best_a is None or a > best_a))):
            best_ic, best_a = ic, float(a)
    if best_a is None:
        best_a = float(PHASE8_RIDGE_ALPHAS[-1])
    m = np.isfinite(train["y_rank_pct"].to_numpy()) if "y_rank_pct" in train.columns else np.isfinite(train[f"excess_h{PHASE8_H}"].to_numpy())
    X, y = _xy_ridge(train.loc[m], feat_rank_cols)
    mdl = Ridge(alpha=float(best_a), fit_intercept=True)
    mdl.fit(X, np.nan_to_num(y, nan=np.nanmedian(y)))
    Xv, _ = _xy_ridge(valid, feat_rank_cols)
    pred = valid[["date", "id"]].copy()
    if "symbol" in valid.columns:
        pred["symbol"] = valid["symbol"].to_numpy()
    pred["signal"] = mdl.predict(Xv)
    pred["fold_id"] = int(fold.fold_id)
    meta = {
        "fold_id": int(fold.fold_id),
        "status": "ok",
        "alpha": float(best_a),
        "ho_rankic": float(best_ic) if np.isfinite(best_ic) else None,
        "grid": grid_scores,
        "elapsed": time.time() - t0,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
    }
    _log(f"ridge fold={fold.fold_id} alpha={best_a} ho_rankic={best_ic:.4f} rows={len(pred)}")
    return pred, meta


def train_ridge_all_folds(df: pd.DataFrame, folds: list[FoldSpec], feat_rank_cols: list[str], **kw) -> tuple[pd.DataFrame, dict]:
    parts, metas = [], []
    for fold in folds:
        pred, meta = fit_ridge_fold(df, fold, feat_rank_cols, **kw)
        metas.append(meta)
        if pred is not None and not pred.empty:
            parts.append(pred)
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return preds, {"meta": metas, "config": ridge_config_dump()}


# ---------------------------------------------------------------------------
# Arm B — TabPFN
# ---------------------------------------------------------------------------

def tabpfn_config_dump(extra: dict | None = None) -> dict:
    d = {
        "context_cap": PHASE8_TABPFN_CONTEXT_CAP,
        "n_estimators": PHASE8_TABPFN_N_ESTIMATORS,
        "features": "STAGE_S_COLS (33)",
        "targets": ["y_h14 top-quintile", "y_bot_h14 bottom-quintile"],
        "signal": "p_top - p_bot",
        "fine_tune": False,
        "gradient_steps": 0,
        "subsample": "even across train dates, seed 42, one context per fold",
        "inference": "batched per fold (all val rows one predict_proba per head)",
    }
    if extra:
        d.update(extra)
    return d


def stratified_context(train: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    dates = sorted(train["date"].unique())
    if not dates:
        return train.iloc[0:0]
    n_dates = len(dates)
    per = max(1, int(cap) // n_dates)
    rng = np.random.default_rng(int(seed))
    parts = []
    total = 0
    for dt in dates:
        g = train[train["date"] == dt]
        if g.empty:
            continue
        k = min(len(g), per)
        if len(g) <= k:
            take = g
        else:
            idx = rng.choice(len(g), size=k, replace=False)
            take = g.iloc[idx]
        parts.append(take)
        total += len(take)
        if total >= int(cap):
            break
    out = pd.concat(parts, ignore_index=True) if parts else train.iloc[0:0]
    if len(out) > int(cap):
        out = out.iloc[: int(cap)]
    if len(out) < int(cap):
        leftover = train.merge(out[["date", "id"]], on=["date", "id"], how="left", indicator=True)
        leftover = leftover[leftover["_merge"] == "left_only"]
        need = int(cap) - len(out)
        if not leftover.empty and need > 0:
            extra = leftover.sample(n=min(need, len(leftover)), random_state=int(seed) + 7)
            out = pd.concat([out, extra.drop(columns=["_merge"], errors="ignore")], ignore_index=True)
    return out.head(int(cap))


def _tabpfn_pos_col(proba: np.ndarray, classes) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    if proba.ndim == 1:
        return proba
    if 1 in classes or 1.0 in classes:
        j = classes.index(1 if 1 in classes else 1.0)
        return proba[:, j]
    return proba[:, -1]


def fit_tabpfn_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    feat_cols: list[str],
    device: str,
    query_dates=None,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    ping=None,
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    meta = {"fold_id": int(fold.fold_id), "status": "ok", "device": device}
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        meta.update({"status": "unavailable", "error": f"import: {e}"})
        return pd.DataFrame(), meta
    d = df.copy()
    d["date"] = _utc(d["date"])
    train = d[(d["date"] >= fold.train_start) & (d["date"] <= fold.train_end)].copy()
    valid = d[(d["date"] >= fold.val_start) & (d["date"] <= fold.val_end)].copy()
    if query_dates is not None:
        qset = set(pd.to_datetime(query_dates, utc=True))
        valid = valid[valid["date"].isin(qset)].copy()
    if train.empty or valid.empty:
        meta["status"] = "empty"
        return pd.DataFrame(), meta
    if shuffle_labels:
        rng = np.random.default_rng(int(shuffle_seed) if shuffle_seed is not None else 90_017)
        train = shuffle_labels_frame(
            train,
            [f"y_h{PHASE8_H}", f"y_bot_h{PHASE8_H}", f"excess_h{PHASE8_H}", "y_rank_pct"],
            rng,
            mode="vol_matched",
        )
    ytop = f"y_h{PHASE8_H}"
    ybot = f"y_bot_h{PHASE8_H}"
    tr_ok = train.dropna(subset=feat_cols + [ytop, ybot])
    ctx = stratified_context(tr_ok, PHASE8_TABPFN_CONTEXT_CAP, seed=42 + int(fold.fold_id))
    meta["n_context"] = int(len(ctx))
    meta["n_query"] = int(len(valid))
    meta["n_query_dates"] = int(valid["date"].nunique())
    Xc = np.nan_to_num(ctx[feat_cols].to_numpy(dtype=np.float32), nan=0.0)
    Xq = np.nan_to_num(valid[feat_cols].to_numpy(dtype=np.float32), nan=0.0)
    times = {}
    try:
        t_fit = time.time()
        def _mk(seed: int):
            kw = dict(device=device, n_estimators=int(PHASE8_TABPFN_N_ESTIMATORS), random_state=int(seed))
            try:
                return TabPFNClassifier(**kw, ignore_pretraining_limits=False)
            except TypeError:
                return TabPFNClassifier(**kw)

        clf_t = _mk(42)
        clf_t.fit(Xc, ctx[ytop].to_numpy().astype(int))
        times["fit_top_sec"] = time.time() - t_fit
        t_pred = time.time()
        pt = _tabpfn_pos_col(clf_t.predict_proba(Xq), getattr(clf_t, "classes_", [0, 1]))
        times["pred_top_sec"] = time.time() - t_pred
        t_fit = time.time()
        clf_b = _mk(43)
        clf_b.fit(Xc, ctx[ybot].to_numpy().astype(int))
        times["fit_bot_sec"] = time.time() - t_fit
        t_pred = time.time()
        pb = _tabpfn_pos_col(clf_b.predict_proba(Xq), getattr(clf_b, "classes_", [0, 1]))
        times["pred_bot_sec"] = time.time() - t_pred
    except Exception as e:
        meta.update({"status": "error", "error": str(e), "elapsed": time.time() - t0})
        _log(f"tabpfn fold={fold.fold_id} ERROR {e}")
        return pd.DataFrame(), meta
    pred = valid[["date", "id"]].copy()
    pred["p_top"] = pt
    pred["p_bot"] = pb
    pred["signal"] = pred["p_top"] - pred["p_bot"]
    pred["fold_id"] = int(fold.fold_id)
    n_dates = max(int(valid["date"].nunique()), 1)
    pred_sec = float(times.get("pred_top_sec", 0) + times.get("pred_bot_sec", 0))
    meta.update(
        {
            "times": times,
            "pred_sec_total": pred_sec,
            "pred_sec_per_date": pred_sec / n_dates,
            "elapsed": time.time() - t0,
            "tabpfn_version": __import__("tabpfn").__version__ if hasattr(__import__("tabpfn"), "__version__") else "unknown",
        }
    )
    _log(
        f"tabpfn fold={fold.fold_id} ctx={meta['n_context']} query={meta['n_query']} "
        f"pred_s={pred_sec:.1f} per_date={meta['pred_sec_per_date']:.3f}s elapsed={meta['elapsed']:.1f}s"
    )
    if ping:
        ping(f"tabpfn fold={fold.fold_id} elapsed={meta['elapsed']:.0f}s")
    return pred, meta


def train_tabpfn_all_folds(df, folds, feat_cols, device="cuda", query_dates=None, ping=None, **kw):
    parts, metas = [], []
    t0 = time.time()
    for fold in folds:
        pred, meta = fit_tabpfn_fold(df, fold, feat_cols, device, query_dates=query_dates, ping=ping, **kw)
        metas.append(meta)
        if pred is not None and not pred.empty:
            parts.append(pred)
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    total_pred = float(sum((m.get("pred_sec_total") or 0) for m in metas))
    n_dates = int(preds["date"].nunique()) if not preds.empty else 0
    extra = {
        "device": device,
        "total_pred_sec": total_pred,
        "total_elapsed_sec": time.time() - t0,
        "n_dates": n_dates,
        "mean_pred_sec_per_date": (total_pred / n_dates) if n_dates else None,
        "n_folds_ok": int(sum(1 for m in metas if m.get("status") == "ok")),
    }
    return preds, {"meta": metas, "config": tabpfn_config_dump(extra), "wall": extra}


# ---------------------------------------------------------------------------
# Judgment, correlation, null
# ---------------------------------------------------------------------------

def seed_dispersion(seed_preds: pd.DataFrame, labeled, close, btc_id, dates=None) -> dict:
    if seed_preds is None or seed_preds.empty:
        return {}
    out = {}
    sp = seed_preds.copy()
    sp["date"] = _utc(sp["date"])
    sp["signal"] = sp["p_top"] - sp["p_bot"]
    for seed, g in sp.groupby("seed"):
        g = collapse_fold_preds(g, "signal")
        g = restrict_dates(g, dates)
        ev = restrict_eval_frame(g, labeled, close, btc_id, "signal")
        met = per_date_tail_metrics(ev, "signal") if not ev.empty else {}
        out[str(int(seed))] = {
            "rankic": met.get("rankic"),
            "tail_ic_top": met.get("tail_ic_top"),
            "overlap": met.get("overlap"),
            "n_dates": met.get("n_dates"),
        }
    if out:
        rics = [v["rankic"] for v in out.values() if v.get("rankic") is not None and np.isfinite(v["rankic"])]
        out["rankic_mean"] = float(np.mean(rics)) if rics else None
        out["rankic_std"] = float(np.std(rics, ddof=1)) if len(rics) > 1 else 0.0
    return out


def signal_corr_matrix(frames: dict, labeled, close, btc_id, dates=None) -> dict:
    """frames: name -> (df, col). Mean per-date Spearman."""
    packed = {}
    for name, (df, col) in frames.items():
        if df is None or df.empty:
            continue
        d = restrict_dates(df, dates)
        ev = restrict_eval_frame(d, labeled, close, btc_id, col)
        if ev.empty:
            continue
        packed[name] = ev.rename(columns={col: name})[["date", "id", name]]
    names = list(packed.keys())
    mat = {a: {b: None for b in names} for a in names}
    n_dates = 0
    if len(names) < 2:
        return {"names": names, "matrix": mat, "n_dates": 0}
    m = packed[names[0]]
    for n in names[1:]:
        m = m.merge(packed[n], on=["date", "id"], how="inner")
    ics = {a: {b: [] for b in names} for a in names}
    for dt, g in m.groupby("date", sort=True):
        n_dates += 1
        for i, a in enumerate(names):
            ics[a][a].append(1.0)
            for b in names[i + 1 :]:
                c = _spearman(g[a].to_numpy(), g[b].to_numpy())
                if np.isfinite(c):
                    ics[a][b].append(c)
                    ics[b][a].append(c)
    for a in names:
        for b in names:
            xs = ics[a][b]
            mat[a][b] = float(np.mean(xs)) if xs else None
    return {"names": names, "matrix": mat, "n_dates": int(n_dates)}


def _delta(a, b) -> float:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return float("nan")
    return fa - fb


def mechanical_verdicts(grid: dict, null: dict | None, best_arm: str | None, corr: dict) -> dict:
    base = grid.get("frozen_spread") or {}
    arms = ("cs_attn", "tabpfn", "ridge")
    out = {
        "best_arm": best_arm,
        "linear_ceiling": False,
        "linear_ceiling_text": None,
        "orthogonal": [],
        "arms": {},
    }
    br = float(base.get("rankic") or np.nan)
    rr = float((grid.get("ridge") or {}).get("rankic") or np.nan)
    if np.isfinite(br) and np.isfinite(rr) and br != 0:
        out["linear_ceiling"] = bool(rr >= PHASE8_LINEAR_CEILING_RATIO * br)
        out["ridge_rankic"] = rr
        out["frozen_rankic"] = br
        out["linear_ceiling_text"] = (
            f"LINEAR-CEILING {'YES' if out['linear_ceiling'] else 'NO'}: "
            f"Arm C RankIC={rr:.4f} vs frozen RankIC={br:.4f} "
            f"(ratio={rr / br:.3f}, threshold={PHASE8_LINEAR_CEILING_RATIO})"
        )
        if out["linear_ceiling"]:
            out["linear_ceiling_text"] += (
                " — nonlinearity contributes less than 10% of the daily signal; "
                "future daily modeling effort is unjustified."
            )
    mat = (corr or {}).get("matrix") or {}
    for arm in arms:
        m = grid.get(arm) or {}
        if not m or m.get("n_dates") in (0, None):
            out["arms"][arm] = {"verdict": "UNAVAILABLE", "live": False, "lead": False}
            continue
        d_ic = _delta(m.get("tail_ic_top"), base.get("tail_ic_top"))
        d_ov = _delta(m.get("overlap"), base.get("overlap"))
        d_ric = _delta(m.get("rankic"), base.get("rankic"))
        is_best = arm == best_arm
        null_tail = bool(is_best and (null or {}).get("passed"))
        null_ric = bool(is_best and ((null or {}).get("rankic") or {}).get("passed"))
        live = bool(
            np.isfinite(d_ic)
            and np.isfinite(d_ov)
            and d_ic >= PHASE8_TAIL_IC_DELTA
            and d_ov >= PHASE8_OVERLAP_DELTA
            and null_tail
        )
        lead = bool((not live) and np.isfinite(d_ric) and d_ric >= PHASE8_RANKIC_DELTA and null_ric)
        if live:
            verdict = "LIVE"
        elif lead:
            verdict = "WHOLE-RANKING LEAD"
        elif not is_best:
            verdict = "NOT LIVE (null not run)"
        else:
            verdict = "NOT LIVE"
        c = None
        try:
            c = (mat.get(arm) or {}).get("frozen_spread")
            if c is None:
                c = (mat.get("frozen_spread") or {}).get(arm)
        except Exception:
            c = None
        ric = m.get("rankic")
        orth = bool(
            c is not None
            and np.isfinite(float(c))
            and float(c) < PHASE8_ORTH_CORR
            and ric is not None
            and np.isfinite(float(ric))
            and float(ric) >= PHASE8_ORTH_RANKIC
        )
        if orth:
            out["orthogonal"].append({"arm": arm, "corr": float(c), "rankic": float(ric)})
        out["arms"][arm] = {
            "verdict": verdict,
            "live": live,
            "lead": lead,
            "delta_tail_ic": d_ic,
            "delta_overlap": d_ov,
            "delta_rankic": d_ric,
            "null_tail_pass": null_tail,
            "null_rankic_pass": null_ric,
            "corr_vs_frozen": None if c is None else float(c),
            "orthogonal": orth,
            "is_best": is_best,
        }
    return out


def pick_best_arm(grid: dict) -> str | None:
    best, best_v = None, -1e18
    for arm in ("cs_attn", "tabpfn", "ridge"):
        m = grid.get(arm) or {}
        v = m.get("rankic")
        if v is None or not np.isfinite(float(v)):
            continue
        if float(v) > best_v:
            best_v = float(v)
            best = arm
    return best


def finish_phase8_null(name: str, cells_by_metric: dict) -> dict:
    real_keys = {
        "tail_ic_top": "real_tail_ic_top",
        "overlap": "real_overlap",
        "monster": "real_monster",
        "rankic": "real_rankic",
    }
    packs = {}
    for metric, cells in cells_by_metric.items():
        v = metric_verdict_e1b_house(cells, real_keys[metric], PHASE8_NULL_K_EXCEED, STOUFFER_Z_MIN)
        packs[metric] = {k: val for k, val in v.items() if k != "cells"}
        packs[f"{metric}_cells"] = cells
    judged = packs.get("tail_ic_top") or {}
    return {
        "name": name,
        "null_design": "vol_matched",
        "passed": bool(judged.get("passed")),
        "judged": "tail_ic_top",
        "bias_min_violations": int(FUTURE_NULL_BIAS_MIN_VIOLATIONS),
        "n_replicates": int(PHASE8_NULL_REPLICATES),
        "k_exceed": int(PHASE8_NULL_K_EXCEED),
        "fold_ids": list(PHASE8_NULL_FOLD_IDS),
        **packs,
    }


def fold_cell(fold: FoldSpec, values: list, real: dict, metric: str) -> dict:
    st = cell_stats_vol_matched(values)
    blob = real.get(fold.fold_id) or {}
    real_v = blob.get(metric, float("nan"))
    try:
        real_v = float(real_v) if real_v is not None else float("nan")
    except (TypeError, ValueError):
        real_v = float("nan")
    key = {
        "tail_ic_top": "real_tail_ic_top",
        "overlap": "real_overlap",
        "monster": "real_monster",
        "rankic": "real_rankic",
    }[metric]
    st.update(
        {
            "fold_id": fold.fold_id,
            "horizon": fold.horizon,
            key: real_v,
            "exceeds_p95": bool(np.isfinite(real_v) and np.isfinite(st["p95"]) and real_v > st["p95"]),
        }
    )
    return st


def real_fold_metrics(preds: pd.DataFrame, folds: list[FoldSpec], labeled, close, btc_id, score_col: str) -> dict:
    out = {}
    pr = preds.copy()
    pr["date"] = _utc(pr["date"])
    for fold in folds:
        if "fold_id" in pr.columns:
            sl = pr[pr["fold_id"].astype(int) == int(fold.fold_id)]
        else:
            sl = pr[(pr["date"] >= fold.val_start) & (pr["date"] <= fold.val_end)]
        out[int(fold.fold_id)] = fold_tail_pack(sl, labeled, close, btc_id, score_col)
    return out


def linspace_dates(dates, n: int = PHASE8_DIAG_DATES) -> list:
    dts = sorted(pd.to_datetime(pd.unique(dates), utc=True))
    if not dts:
        return []
    if len(dts) <= n:
        return [_as_utc(d) for d in dts]
    idx = np.linspace(0, len(dts) - 1, int(n)).round().astype(int)
    seen, out = set(), []
    for i in idx:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(_as_utc(dts[int(i)]))
    return out


def null_shuffle_seeds() -> list[int]:
    return list(NULL_SHUFFLE_SEEDS)[: int(PHASE8_NULL_REPLICATES)]


def score_signal(df: pd.DataFrame, col: str, labeled, close, btc_id, dates=None) -> dict:
    if df is None or df.empty:
        return {"n_dates": 0, "label": col}
    d = restrict_dates(df, dates)
    ev = restrict_eval_frame(d, labeled, close, btc_id, col)
    if ev.empty:
        return {"n_dates": 0, "label": col}
    return per_date_tail_metrics(ev, col)


def run_one_null_cell(
    df: pd.DataFrame,
    fold: FoldSpec,
    arm: str,
    shuffle_seed: int,
    feat_cols: list[str],
    feat_rank_cols: list[str],
    labeled: pd.DataFrame,
    close,
    btc_id: int,
    device: str = "cpu",
) -> dict:
    t0 = time.time()
    pred = pd.DataFrame()
    status = "ok"
    try:
        if arm == "ridge":
            pred, meta = fit_ridge_fold(
                df, fold, feat_rank_cols, shuffle_labels=True, shuffle_seed=int(shuffle_seed)
            )
            status = meta.get("status", "ok")
        elif arm == "cs_attn":
            pred, extra = train_cs_attn_all_folds(
                df,
                [fold],
                feat_cols,
                single_seed=42,
                shuffle_labels=True,
                shuffle_seed=int(shuffle_seed),
            )
            status = "ok" if pred is not None and not pred.empty else "empty"
        elif arm == "tabpfn":
            pred, meta = fit_tabpfn_fold(
                df,
                fold,
                feat_cols,
                device=device,
                shuffle_labels=True,
                shuffle_seed=int(shuffle_seed),
            )
            status = meta.get("status", "ok")
        else:
            raise RuntimeError(f"unknown null arm {arm}")
        sm = fold_tail_pack(pred, labeled, close, btc_id, "signal")
        rec = {k: sm.get(k) for k in ("tail_ic_top", "overlap", "monster", "rankic", "n_dates")}
        rec.update({"status": status, "fold_id": int(fold.fold_id), "shuffle_seed": int(shuffle_seed), "arm": arm})
    except Exception as e:
        rec = {
            "tail_ic_top": None,
            "overlap": None,
            "monster": None,
            "rankic": None,
            "status": f"error:{e}",
            "fold_id": int(fold.fold_id),
            "shuffle_seed": int(shuffle_seed),
            "arm": arm,
        }
    rec["elapsed"] = time.time() - t0
    _log(f"null cell arm={arm} fold={fold.fold_id} seed={shuffle_seed} {rec.get('status')} rankic={rec.get('rankic')}")
    return rec

