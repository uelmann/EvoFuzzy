"""CS-ATTN v0: shared TCN encoder + cross-sectional set-attention + twin tail heads.

ONE frozen config. No architecture search.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.seedutil import seed_everything
from btcb.constants import (
    CSATTN_CONFIG,
    INNER_HOLDOUT_CALENDAR_DAYS,
    PHASE5_ATTN_HEADS,
    PHASE5_ATTN_LAYERS,
    PHASE5_ATTN_WIDTH,
    PHASE5_BATCH_DATES,
    PHASE5_DROPOUT,
    PHASE5_LR,
    PHASE5_MAX_EPOCHS,
    PHASE5_N_CHANNELS,
    PHASE5_PATIENCE,
    PHASE5_SEQ_LEN,
    PHASE5_TCN_BLOCKS,
    PHASE5_TCN_DILATIONS,
    PHASE5_TCN_KERNEL,
    PHASE5_TCN_WIDTH,
    PHASE5_TOP_POS_WEIGHT,
    PHASE5_WEIGHT_DECAY,
    SEED,
)
from btcb.model import FoldSpec, apply_calibrator, fit_isotonic
from btcb.oracle_ladder2 import _half_ic


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC").normalize()
    return t.tz_convert("UTC").normalize()


def _tail_ic_top(score: np.ndarray, excess: np.ndarray) -> float:
    s = pd.Series(np.asarray(score, dtype=float))
    e = pd.Series(np.asarray(excess, dtype=float), index=s.index)
    return float(_half_ic(s, e, "top"))


def per_date_tail_ic_top(dates, scores, excess, min_n: int = 8) -> pd.Series:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates), utc=True),
            "score": np.asarray(scores, dtype=float),
            "excess": np.asarray(excess, dtype=float),
        }
    )
    rows = []
    for dt, g in df.groupby("date", sort=True):
        if len(g) < min_n:
            continue
        v = _tail_ic_top(g["score"].to_numpy(), g["excess"].to_numpy())
        if np.isfinite(v):
            rows.append((_utc(dt), v))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({d: v for d, v in rows}).sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").normalize()
    return s


def make_csattn(n_channels: int = PHASE5_N_CHANNELS):
    import torch
    from torch import nn

    class TCNBlock(nn.Module):
        def __init__(self, width: int, dilation: int, kernel: int, dropout: float):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(width, width, kernel, padding=pad, dilation=dilation)
            self.norm = nn.GroupNorm(8, width)
            self.drop = nn.Dropout(dropout)
            self.act = nn.GELU()
            self.crop = pad

        def forward(self, x):
            y = self.conv(x)
            if self.crop:
                y = y[:, :, : -self.crop]
            y = self.act(self.drop(self.norm(y)))
            return x + y

    class CSATTN(nn.Module):
        def __init__(self):
            super().__init__()
            w = int(PHASE5_TCN_WIDTH)
            self.in_proj = nn.Conv1d(n_channels, w, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    TCNBlock(w, int(d), int(PHASE5_TCN_KERNEL), float(PHASE5_DROPOUT))
                    for d in PHASE5_TCN_DILATIONS[: int(PHASE5_TCN_BLOCKS)]
                ]
            )
            self.out_norm = nn.LayerNorm(w)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=int(PHASE5_ATTN_WIDTH),
                nhead=int(PHASE5_ATTN_HEADS),
                dim_feedforward=int(PHASE5_ATTN_WIDTH) * 4,
                dropout=float(PHASE5_DROPOUT),
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.set_attn = nn.TransformerEncoder(enc_layer, num_layers=int(PHASE5_ATTN_LAYERS))
            self.head_top = nn.Linear(w, 1)
            self.head_bot = nn.Linear(w, 1)

        def encode_series(self, x):
            # x: (B*N, T, C)
            z = x.transpose(1, 2)
            z = self.in_proj(z)
            for blk in self.blocks:
                z = blk(z)
            last = z[:, :, -1]
            return self.out_norm(last)

        def forward(self, x, pad_mask):
            # x: (B, N, T, C)  pad_mask True = PAD
            b, n, t, c = x.shape
            emb = self.encode_series(x.reshape(b * n, t, c)).reshape(b, n, -1)
            valid = ~pad_mask
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(emb.dtype)
            pooled = (emb * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
            tokens = torch.cat([pooled, emb], dim=1)
            cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=pad_mask.device)
            attn_pad = torch.cat([cls_pad, pad_mask], dim=1)
            out = self.set_attn(tokens, src_key_padding_mask=attn_pad)
            coins = out[:, 1:, :] + out[:, :1, :]
            logit_top = self.head_top(coins).squeeze(-1)
            logit_bot = self.head_bot(coins).squeeze(-1)
            return logit_top, logit_bot

    return CSATTN()


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_seq_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    idx = pd.read_parquet(cache_dir / "index.parquet")
    idx["date"] = pd.to_datetime(idx["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    idx["id"] = idx["id"].astype(int)
    X = np.load(cache_dir / "X.npy", mmap_mode="r")
    return idx, X


def _dates_payload(idx: pd.DataFrame, mask: np.ndarray) -> list[dict]:
    sub = idx.loc[mask].copy()
    out = []
    for dt, g in sub.groupby("date", sort=True):
        out.append(
            {
                "date": dt,
                "row_id": g["row_id"].to_numpy(dtype=np.int64),
                "id": g["id"].to_numpy(dtype=np.int64),
                "y_top": g["y_h14"].to_numpy(dtype=np.float32),
                "y_bot": g["y_bot_h14"].to_numpy(dtype=np.float32),
                "excess": g["excess_h14"].to_numpy(dtype=np.float32),
            }
        )
    return out


def _stack_batch(X, items: list[dict], device):
    import torch

    n_max = max(len(it["row_id"]) for it in items)
    b = len(items)
    xb = np.zeros((b, n_max, PHASE5_SEQ_LEN, PHASE5_N_CHANNELS), dtype=np.float32)
    pad = np.ones((b, n_max), dtype=bool)
    y_top = np.full((b, n_max), np.nan, dtype=np.float32)
    y_bot = np.full((b, n_max), np.nan, dtype=np.float32)
    excess = np.full((b, n_max), np.nan, dtype=np.float32)
    ids = np.zeros((b, n_max), dtype=np.int64)
    for i, it in enumerate(items):
        n = len(it["row_id"])
        xb[i, :n] = np.asarray(X[it["row_id"]])
        pad[i, :n] = False
        y_top[i, :n] = it["y_top"]
        y_bot[i, :n] = it["y_bot"]
        excess[i, :n] = it["excess"]
        ids[i, :n] = it["id"]
    return {
        "x": torch.from_numpy(xb).to(device),
        "pad": torch.from_numpy(pad).to(device),
        "y_top": y_top,
        "y_bot": y_bot,
        "excess": excess,
        "ids": ids,
        "dates": [it["date"] for it in items],
        "n": np.array([len(it["row_id"]) for it in items], dtype=np.int32),
    }


def _bce_with_weights(logits, y, pos_weight: float, pad, valid_y):
    import torch
    import torch.nn.functional as F

    # logits, pad: tensors; y numpy
    yt = torch.from_numpy(np.nan_to_num(y, nan=0.0)).to(logits.device)
    m = (~pad) & torch.from_numpy(np.isfinite(y)).to(logits.device)
    if valid_y is not None:
        m = m & torch.from_numpy(valid_y).to(logits.device)
    if int(m.sum()) == 0:
        return logits.sum() * 0.0
    w = torch.where(yt > 0.5, torch.full_like(yt, float(pos_weight)), torch.ones_like(yt))
    loss = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
    return (loss * w)[m].mean()


def _predict_dates(model, X, date_items, device, batch_dates: int):
    import torch

    model.eval()
    recs = []
    use_amp = str(device).startswith("cuda")
    with torch.no_grad():
        for i in range(0, len(date_items), batch_dates):
            chunk = date_items[i : i + batch_dates]
            bat = _stack_batch(X, chunk, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                lt, lb = model(bat["x"], bat["pad"])
            pt = torch.sigmoid(lt.float()).cpu().numpy()
            pb = torch.sigmoid(lb.float()).cpu().numpy()
            for bi, dt in enumerate(bat["dates"]):
                n = int(bat["n"][bi])
                recs.append(
                    pd.DataFrame(
                        {
                            "date": _utc(dt),
                            "id": bat["ids"][bi, :n],
                            "p_top_raw": pt[bi, :n],
                            "p_bot_raw": pb[bi, :n],
                            "y_top": bat["y_top"][bi, :n],
                            "y_bot": bat["y_bot"][bi, :n],
                            "excess_h14": bat["excess"][bi, :n],
                        }
                    )
                )
    if not recs:
        return pd.DataFrame(
            columns=["date", "id", "p_top_raw", "p_bot_raw", "y_top", "y_bot", "excess_h14"]
        )
    return pd.concat(recs, ignore_index=True)


def _mean_tail_ic(pred: pd.DataFrame) -> float:
    if pred is None or pred.empty:
        return float("nan")
    s = per_date_tail_ic_top(pred["date"], pred["p_top_raw"] - pred["p_bot_raw"], pred["excess_h14"])
    return float(s.mean()) if len(s) else float("nan")


def train_csattn_fold(
    cache_dir: Path,
    fold: FoldSpec,
    seed: int = SEED,
    shuffle_labels: bool = False,
    shuffle_seed: int | None = None,
    heartbeat=None,
    budget_ok=None,
) -> tuple[pd.DataFrame, dict]:
    import torch
    from torch.optim import AdamW

    t0 = time.time()
    seed_everything(int(seed) + int(fold.fold_id) + (0 if shuffle_seed is None else int(shuffle_seed)))
    torch.manual_seed(int(seed) + int(fold.fold_id))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + int(fold.fold_id))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    idx, X = load_seq_cache(cache_dir)
    train_mask = (idx["date"] >= _utc(fold.train_start)) & (idx["date"] <= _utc(fold.train_end))
    val_mask = (idx["date"] >= _utc(fold.val_start)) & (idx["date"] <= _utc(fold.val_end))
    train = idx.loc[train_mask].copy()
    valid = idx.loc[val_mask].copy()
    if train.empty or valid.empty:
        return pd.DataFrame(), {
            "fold_id": fold.fold_id,
            "status": "empty",
            "elapsed": time.time() - t0,
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "seed": int(seed),
        }

    cut = _utc(fold.train_end) - pd.Timedelta(days=int(INNER_HOLDOUT_CALENDAR_DAYS))
    inner_tr = train[train["date"] <= cut]
    inner_ho = train[train["date"] > cut]
    if inner_tr.empty or inner_ho.empty:
        dates = sorted(train["date"].unique())
        ncut = max(1, int(len(dates) * 0.85))
        cut_d = dates[ncut - 1]
        inner_tr = train[train["date"] <= cut_d]
        inner_ho = train[train["date"] > cut_d]

    if shuffle_labels:
        ss = int(shuffle_seed) if shuffle_seed is not None else int(seed) + 90_017
        rng = np.random.default_rng(ss)

        def _shuf(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            for _, gix in d.groupby("date", sort=False).groups.items():
                sl = d.loc[gix]
                perm = rng.permutation(len(sl))
                d.loc[gix, "y_h14"] = sl["y_h14"].to_numpy()[perm]
                d.loc[gix, "y_bot_h14"] = sl["y_bot_h14"].to_numpy()[perm]
            return d

        inner_tr = _shuf(inner_tr)
        inner_ho = _shuf(inner_ho)

    def _items(df):
        out = []
        for dt, g in df.groupby("date", sort=True):
            out.append(
                {
                    "date": dt,
                    "row_id": g["row_id"].to_numpy(dtype=np.int64),
                    "id": g["id"].to_numpy(dtype=np.int64),
                    "y_top": g["y_h14"].to_numpy(dtype=np.float32),
                    "y_bot": g["y_bot_h14"].to_numpy(dtype=np.float32),
                    "excess": g["excess_h14"].to_numpy(dtype=np.float32),
                }
            )
        return out

    tr_items, ho_items, va_items = _items(inner_tr), _items(inner_ho), _items(valid)
    device = _device()
    model = make_csattn().to(device)
    opt = AdamW(model.parameters(), lr=float(PHASE5_LR), weight_decay=float(PHASE5_WEIGHT_DECAY))
    n_params = int(sum(p.numel() for p in model.parameters()))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = None
    best_ic = -1e9
    bad = 0
    hist = []
    rng = np.random.default_rng(int(seed) + 17 * int(fold.fold_id))

    for epoch in range(1, int(PHASE5_MAX_EPOCHS) + 1):
        if budget_ok is not None and not budget_ok():
            break
        model.train()
        order = np.arange(len(tr_items))
        rng.shuffle(order)
        ep_loss = []
        bs = int(PHASE5_BATCH_DATES)
        for i in range(0, len(order), bs):
            sel = [tr_items[j] for j in order[i : i + bs]]
            if not sel:
                continue
            bat = _stack_batch(X, sel, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                lt, lb = model(bat["x"], bat["pad"])
                loss = _bce_with_weights(
                    lt, bat["y_top"], float(PHASE5_TOP_POS_WEIGHT), bat["pad"], None
                ) + _bce_with_weights(lb, bat["y_bot"], 1.0, bat["pad"], None)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss.append(float(loss.detach().item()))
            if heartbeat and ((i // bs) % 8 == 0):
                heartbeat.ping(
                    f"fold={fold.fold_id} seed={seed} ep={epoch} batch={i//bs}/{(len(order)-1)//bs}"
                )
        ho_pred = _predict_dates(model, X, ho_items, device, bs)
        ic = _mean_tail_ic(ho_pred)
        hist.append({"epoch": epoch, "loss": float(np.mean(ep_loss) if ep_loss else np.nan), "ho_tail_ic": ic})
        improved = np.isfinite(ic) and ic > best_ic + 1e-6
        if improved:
            best_ic = float(ic)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if heartbeat:
            heartbeat.ping(
                f"fold={fold.fold_id} seed={seed} ep={epoch}/{PHASE5_MAX_EPOCHS} "
                f"loss={hist[-1]['loss']:.4f} ho_tailIC={ic} best={best_ic} bad={bad}"
            )
        if bad >= int(PHASE5_PATIENCE):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    ho_pred = _predict_dates(model, X, ho_items, device, int(PHASE5_BATCH_DATES))
    va_pred = _predict_dates(model, X, va_items, device, int(PHASE5_BATCH_DATES))
    ir_top = None if shuffle_labels else fit_isotonic(ho_pred["p_top_raw"].to_numpy(), ho_pred["y_top"].to_numpy())
    ir_bot = None if shuffle_labels else fit_isotonic(ho_pred["p_bot_raw"].to_numpy(), ho_pred["y_bot"].to_numpy())
    if va_pred.empty:
        pred_df = va_pred
    else:
        pred_df = va_pred.copy()
        pred_df["p_top"] = apply_calibrator(ir_top, pred_df["p_top_raw"].to_numpy())
        pred_df["p_bot"] = apply_calibrator(ir_bot, pred_df["p_bot_raw"].to_numpy())
        pred_df["spread"] = pred_df["p_top"] - pred_df["p_bot"]
        pred_df["spread_raw"] = pred_df["p_top_raw"] - pred_df["p_bot_raw"]
        pred_df["fold_id"] = int(fold.fold_id)
        pred_df["horizon"] = int(fold.horizon)
        pred_df["seed"] = int(seed)
        if "symbol" in valid.columns:
            sm = valid[["date", "id", "symbol"]].drop_duplicates(["date", "id"])
            sm["date"] = pd.to_datetime(sm["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
            pred_df = pred_df.merge(sm, on=["date", "id"], how="left")
    meta = {
        "fold_id": int(fold.fold_id),
        "status": "ok" if not va_pred.empty else "empty_val",
        "elapsed": time.time() - t0,
        "seed": int(seed),
        "n_train_dates": int(len(tr_items)),
        "n_holdout_dates": int(len(ho_items)),
        "n_valid_dates": int(len(va_items)),
        "n_train_rows": int(len(inner_tr)),
        "n_holdout_rows": int(len(inner_ho)),
        "n_valid_rows": int(len(valid)),
        "n_pred": int(len(pred_df)),
        "best_ho_tail_ic": float(best_ic) if np.isfinite(best_ic) and best_ic > -1e8 else float("nan"),
        "n_epochs": int(len(hist)),
        "hist": hist,
        "n_params": n_params,
        "device": str(device),
        "shuffle_labels": bool(shuffle_labels),
        "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
        "calibrated": ir_top is not None and ir_bot is not None,
        "config": dict(CSATTN_CONFIG),
        "train_end": str(pd.Timestamp(fold.train_end).date()),
        "val_start": str(pd.Timestamp(fold.val_start).date()),
        "val_end": str(pd.Timestamp(fold.val_end).date()),
    }
    return pred_df, meta


def merge_seed_ensemble(preds: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [p[["date", "id", "p_top", "p_bot", "spread"]].copy() for p in preds if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame()
    for i, f in enumerate(frames):
        f["date"] = pd.to_datetime(f["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        f["id"] = f["id"].astype(int)
        f = f.rename(columns={"p_top": f"p_top_{i}", "p_bot": f"p_bot_{i}", "spread": f"spread_{i}"})
        frames[i] = f
    m = frames[0]
    for extra in frames[1:]:
        m = m.merge(extra, on=["date", "id"], how="outer")
    tcols = [c for c in m.columns if c.startswith("p_top_")]
    bcols = [c for c in m.columns if c.startswith("p_bot_")]
    scols = [c for c in m.columns if c.startswith("spread_")]
    m["p_top"] = m[tcols].mean(axis=1)
    m["p_bot"] = m[bcols].mean(axis=1)
    m["spread"] = m[scols].mean(axis=1)
    return m[["date", "id", "p_top", "p_bot", "spread"]]
