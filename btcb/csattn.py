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
    PHASE5_SET_N,
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

    class PreLNSetBlock(nn.Module):
        """Pre-LN set-attention + 4× GELU FFN (same as TransformerEncoderLayer).

        Implemented without nn.MultiheadAttention: PyTorch 2.4 fused SDPA pads
        sequence length to a multiple of 8 and then asserts the key_padding_mask
        matches the padded length (smoke: expected (B, 104) got (B, 101)).
        """

        def __init__(self, width: int, nhead: int, dropout: float):
            super().__init__()
            if width % nhead != 0:
                raise ValueError(f"width {width} not divisible by nhead {nhead}")
            self.nhead = int(nhead)
            self.dh = int(width) // int(nhead)
            self.scale = float(self.dh) ** -0.5
            self.norm1 = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, width * 3)
            self.out_proj = nn.Linear(width, width)
            self.drop_attn = nn.Dropout(dropout)
            self.drop1 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(width)
            self.ff = nn.Sequential(
                nn.Linear(width, width * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 4, width),
                nn.Dropout(dropout),
            )

        def forward(self, x, key_padding_mask):
            h = self.norm1(x)
            b, s, d = h.shape
            qkv = self.qkv(h).reshape(b, s, 3, self.nhead, self.dh)
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 1, 3)
            v = qkv[:, :, 2].permute(0, 2, 1, 3)
            # einsum, not matmul — avoid SDPA pattern-matching that pads S to 8.
            attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
            m = key_padding_mask.bool()
            if m.dim() != 2:
                m = m.view(b, -1)
            sk = int(attn.size(-1))
            if m.size(-1) < sk:
                m = torch.nn.functional.pad(m, (0, sk - m.size(-1)), value=True)
            elif m.size(-1) > sk:
                m = m[:, :sk]
            attn = attn.masked_fill(m.view(b, 1, 1, sk), torch.finfo(attn.dtype).min)
            attn = torch.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.drop_attn(attn)
            y = torch.einsum("bhqk,bhkd->bhqd", attn, v).permute(0, 2, 1, 3).contiguous()
            y = y.view(b, s, d)
            y = self.out_proj(y)
            x = x + self.drop1(y)
            x = x + self.ff(self.norm2(x))
            return x

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
            # Manual Pre-LN set-attention. Do NOT use nn.TransformerEncoder:
            # PyTorch 2.4 nested-tensor / _canonical_mask mismatches variable-N
            # key_padding_mask across layers (crash: expected (B, 52) got (B, 37)).
            self.set_layers = nn.ModuleList(
                [PreLNSetBlock(w, int(PHASE5_ATTN_HEADS), float(PHASE5_DROPOUT)) for _ in range(int(PHASE5_ATTN_LAYERS))]
            )
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
            # x: (B, N, T, C)  pad_mask True = PAD. N is PHASE5_SET_N.
            b, n, t, c = x.shape
            emb = self.encode_series(x.reshape(b * n, t, c)).reshape(b, n, -1)
            valid = ~pad_mask
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(emb.dtype)
            pooled = (emb * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
            tokens = torch.cat([pooled, emb], dim=1)
            cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=pad_mask.device)
            attn_pad = torch.cat([cls_pad, pad_mask.bool()], dim=1)
            s_keep = int(tokens.size(1))
            # PyTorch 2.4 CUDA attention/gemm pads S to a multiple of 8.
            s_align = (s_keep + 7) // 8 * 8
            if s_align != s_keep:
                extra = s_align - s_keep
                tokens = torch.nn.functional.pad(tokens, (0, 0, 0, extra))
                attn_pad = torch.nn.functional.pad(attn_pad, (0, extra), value=True)
            # Attention in fp32 — AMP + padding masks is flaky on 2.4.1.
            import contextlib

            ctx = (
                torch.cuda.amp.autocast(enabled=False)
                if torch.is_autocast_enabled()
                else contextlib.nullcontext()
            )
            with ctx:
                out = tokens.float()
                for layer in self.set_layers:
                    out = layer(out, attn_pad)
            out = out.to(dtype=emb.dtype)
            cls = out[:, :1, :]
            coins = out[:, 1 : 1 + n, :] + cls
            logit_top = self.head_top(coins).squeeze(-1)
            logit_bot = self.head_bot(coins).squeeze(-1)
            return logit_top, logit_bot

    return CSATTN()


def smoke_csattn(device) -> dict:
    """Two-batch forward+backward with different valid-N; catches padding-mask crashes."""
    import torch

    model = make_csattn().to(device)
    n = int(PHASE5_SET_N)
    shapes = []
    for n_valid in (36, 51, 100, 1):
        b = 4
        x = torch.zeros(b, n, PHASE5_SEQ_LEN, PHASE5_N_CHANNELS, device=device)
        pad = torch.ones(b, n, dtype=torch.bool, device=device)
        pad[:, : int(n_valid)] = False
        model.train()
        lt, lb = model(x, pad)
        assert lt.shape == (b, n) and lb.shape == (b, n), (lt.shape, lb.shape)
        loss = lt[~pad].mean() + lb[~pad].mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            lt2, _ = model(x, pad)
        assert lt2.shape == (b, n)
        shapes.append({"n_valid": int(n_valid), "out": list(lt.shape)})
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {"ok": True, "cases": shapes}


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_SEQ_CACHE: dict[str, tuple] = {}


def load_seq_cache(cache_dir: Path):
    """Load index + X into process RAM.

    mmap_mode='r' on the Modal volume turns every batch gather into random
    remote reads (~2 s/batch, GPU idle). 2 GB fits in the Stage B container.
    Numerics unchanged: same float32 rows, just DRAM instead of volume mmap.
    """
    cache_dir = Path(cache_dir)
    key = str(cache_dir.resolve())
    hit = _SEQ_CACHE.get(key)
    if hit is not None:
        return hit
    idx = pd.read_parquet(cache_dir / "index.parquet")
    idx["date"] = pd.to_datetime(idx["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    idx["id"] = idx["id"].astype(int)
    t0 = time.time()
    loaded = np.load(cache_dir / "X.npy")
    X = np.array(loaded, dtype=np.float32, copy=True, order="C")
    del loaded
    print(
        f"[csattn] X.npy RAM shape={tuple(X.shape)} nbytes={X.nbytes} elapsed={time.time()-t0:.1f}s",
        flush=True,
    )
    _SEQ_CACHE[key] = (idx, X)
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

    n_pad = int(PHASE5_SET_N)
    b = len(items)
    xb = np.zeros((b, n_pad, PHASE5_SEQ_LEN, PHASE5_N_CHANNELS), dtype=np.float32)
    pad = np.ones((b, n_pad), dtype=bool)
    y_top = np.full((b, n_pad), np.nan, dtype=np.float32)
    y_bot = np.full((b, n_pad), np.nan, dtype=np.float32)
    excess = np.full((b, n_pad), np.nan, dtype=np.float32)
    ids = np.zeros((b, n_pad), dtype=np.int64)
    ns = np.zeros(b, dtype=np.int32)
    for i, it in enumerate(items):
        n = int(len(it["row_id"]))
        if n > n_pad:
            n = n_pad
        if n <= 0:
            continue
        xb[i, :n] = np.asarray(X[it["row_id"][:n]])
        pad[i, :n] = False
        y_top[i, :n] = it["y_top"][:n]
        y_bot[i, :n] = it["y_bot"][:n]
        excess[i, :n] = it["excess"][:n]
        ids[i, :n] = it["id"][:n]
        ns[i] = n
    return {
        "x": torch.from_numpy(xb).to(device),
        "pad": torch.from_numpy(pad).to(device),
        "y_top": y_top,
        "y_bot": y_bot,
        "excess": excess,
        "ids": ids,
        "dates": [it["date"] for it in items],
        "n": ns,
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
    if heartbeat:
        heartbeat.ping(
            f"fold={fold.fold_id} seed={seed} start n_tr={len(tr_items)} n_ho={len(ho_items)} "
            f"n_va={len(va_items)} device={device} params={n_params}"
        )

    for epoch in range(1, int(PHASE5_MAX_EPOCHS) + 1):
        if budget_ok is not None and not budget_ok():
            break
        model.train()
        order = np.arange(len(tr_items))
        rng.shuffle(order)
        ep_loss = []
        bs = int(PHASE5_BATCH_DATES)
        if heartbeat:
            heartbeat.ping(f"fold={fold.fold_id} seed={seed} ep={epoch} train_batches={(len(order)+bs-1)//bs}")
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
            if heartbeat and ((i // bs) % 4 == 0):
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
