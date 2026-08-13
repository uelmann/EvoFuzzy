"""Tiny GRU sequence model + 60×33 sequence cache (GPU train, CPU cache)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.features import FEATURE_COLS
from baseline.seedutil import seed_everything

WINDOW = 60
MIN_WINDOW = 40
N_FEAT = len(FEATURE_COLS)


def build_sequence_cache(feat: pd.DataFrame, out_dir: Path) -> dict:
    """Write index.parquet + X.npy (N, 60, 33) left-padded, last row = date t."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"], utc=True)
    feat["symbol"] = feat["symbol"].astype(str)
    y7 = "y_h7" if "y_h7" in feat.columns else None
    y10 = "y_h10" if "y_h10" in feat.columns else None
    symbols = []
    grouped = {}
    for sym, g in feat.groupby("symbol", sort=True):
        grouped[str(sym)] = g.sort_values("date")
        symbols.append(str(sym))

    def _windows(g: pd.DataFrame):
        arr = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["date"], utc=True).to_numpy()
        yh7 = g[y7].to_numpy(dtype=np.float32) if y7 else np.full(len(g), np.nan, np.float32)
        yh10 = g[y10].to_numpy(dtype=np.float32) if y10 else np.full(len(g), np.nan, np.float32)
        T = len(g)
        items = []
        for t in range(T):
            start = max(0, t - WINDOW + 1)
            sl = arr[start : t + 1]
            if sl.shape[0] < MIN_WINDOW:
                continue
            if not np.isfinite(sl[-1]).any():
                continue
            items.append((t, sl, dates[t], float(yh7[t]) if np.isfinite(yh7[t]) else np.nan, float(yh10[t]) if np.isfinite(yh10[t]) else np.nan))
        return items

    t0 = time.time()
    counts = []
    n_total = 0
    for i, sym in enumerate(symbols, 1):
        g = grouped[sym]
        n = 0 if len(g) < MIN_WINDOW else len(_windows(g))
        counts.append(n)
        n_total += n
        if i % 50 == 0:
            print(f"[HB] seq count {i}/{len(symbols)} n={n_total}", flush=True)
    if n_total == 0:
        raise RuntimeError("sequence cache empty")

    x_path = out_dir / "X.npy"
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n_total, WINDOW, N_FEAT))
    rows = []
    row_id = 0
    for i, sym in enumerate(symbols, 1):
        g = grouped[sym]
        if len(g) < MIN_WINDOW:
            continue
        for t, sl, dt, yh7v, yh10v in _windows(g):
            pad = WINDOW - sl.shape[0]
            win = np.zeros((WINDOW, N_FEAT), dtype=np.float32)
            win[pad:] = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
            X[row_id] = win
            rows.append(
                {
                    "row_id": row_id,
                    "date": dt,
                    "symbol": sym,
                    "seq_len": int(sl.shape[0]),
                    "y_h7": yh7v,
                    "y_h10": yh10v,
                }
            )
            row_id += 1
        if i % 25 == 0 or i == len(symbols):
            print(
                f"[HB] {time.strftime('%H:%M:%S')} seq write {i}/{len(symbols)} "
                f"rows={row_id} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    X.flush()
    del X
    idx = pd.DataFrame(rows)
    idx["date"] = pd.to_datetime(idx["date"], utc=True)
    idx.to_parquet(out_dir / "index.parquet", index=False)
    meta = {
        "n_rows": int(len(idx)),
        "window": WINDOW,
        "n_feat": N_FEAT,
        "feature_cols": list(FEATURE_COLS),
        "x_path": str(x_path),
        "index_path": str(out_dir / "index.parquet"),
        "nbytes": int(n_total * WINDOW * N_FEAT * 4),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[seq] cache n={meta['n_rows']} nbytes={meta['nbytes']/1e9:.2f}GB", flush=True)
    return meta


def _as_utc(date) -> pd.Timestamp:
    t = pd.Timestamp(date)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def window_from_symbol_frame(g: pd.DataFrame, date, cols: list[str] | None = None) -> np.ndarray | None:
    """Left-padded 60×F window ending at date t using rows with date ≤ t. Matches the cache."""
    cols = cols or list(FEATURE_COLS)
    g = g.sort_values("date")
    t = _as_utc(date)
    g = g[pd.to_datetime(g["date"], utc=True) <= t]
    if len(g) < MIN_WINDOW:
        return None
    sl = g.tail(WINDOW)[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(sl[-1]).any():
        return None
    pad = WINDOW - sl.shape[0]
    win = np.zeros((WINDOW, len(cols)), dtype=np.float32)
    win[pad:] = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
    return win


def gru_param_count(hidden: int = 32, n_feat: int = N_FEAT, dropout: float = 0.1) -> int:
    m = _make_gru(n_feat=n_feat, hidden=hidden, dropout=dropout)
    return int(sum(p.numel() for p in m.parameters()))


def _make_gru(n_feat: int = N_FEAT, hidden: int = 32, dropout: float = 0.1):
    import torch
    from torch import nn

    class _TinyGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_feat, hidden, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _out, h = self.gru(x)
            h = self.drop(h[-1])
            return self.head(h).squeeze(-1)

    return _TinyGRU()


def _date_batches(idx: pd.DataFrame, dates, ycol: str) -> list[tuple]:
    """List of (date, row_ids, y) for dates with ≥5 labeled names."""
    sub = idx[idx["date"].isin(pd.DatetimeIndex(dates))].copy()
    sub = sub.dropna(subset=[ycol])
    out = []
    for dt, g in sub.groupby("date", sort=True):
        if len(g) < 5:
            continue
        out.append((pd.Timestamp(dt), g["row_id"].to_numpy(np.int64), g[ycol].to_numpy(np.float32)))
    return out


def _rank_ic_torch(pred: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    if len(pred) < 5:
        return float("nan")
    if np.unique(pred).size < 2 or np.unique(y).size < 2:
        return float("nan")
    r = stats.spearmanr(pred, y)
    c = getattr(r, "correlation", None)
    if c is None:
        c = getattr(r, "statistic", np.nan)
    return float(np.asarray(c, dtype=float).reshape(-1)[0])


def train_gru_fold(
    cache_dir: Path,
    fold,
    horizon: int,
    seed: int,
    inner_holdout_days: int,
    max_epochs: int,
    lr: float = 1e-3,
    hidden: int = 32,
    dropout: float = 0.1,
    patience: int = 10,
    pairwise_rank: bool = False,
    shuffle_labels: bool = False,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    import torch
    from torch import nn

    seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_parquet(cache_dir / "index.parquet")
    idx["date"] = pd.to_datetime(idx["date"], utc=True)
    X_all = np.load(cache_dir / "X.npy", mmap_mode="r")
    ycol = f"y_h{horizon}"

    train_mask = (idx["date"] >= pd.Timestamp(fold.train_start)) & (idx["date"] <= pd.Timestamp(fold.train_end))
    val_mask = (idx["date"] >= pd.Timestamp(fold.val_start)) & (idx["date"] <= pd.Timestamp(fold.val_end))
    train = idx.loc[train_mask]
    valid = idx.loc[val_mask]
    if train.empty or valid.empty:
        return pd.DataFrame(), {"fold_id": fold.fold_id, "seed": seed, "status": "empty", "elapsed": 0.0}

    cut = pd.Timestamp(fold.train_end) - pd.Timedelta(days=inner_holdout_days)
    inner_tr = train[train["date"] <= cut]
    inner_ho = train[train["date"] > cut]
    if inner_tr.empty or inner_ho.empty:
        dates = sorted(train["date"].unique())
        k = max(1, int(len(dates) * 0.85))
        inner_tr = train[train["date"].isin(dates[:k])]
        inner_ho = train[train["date"].isin(dates[k:])]

    tr_batches = _date_batches(inner_tr, inner_tr["date"].unique(), ycol)
    ho_batches = _date_batches(inner_ho, inner_ho["date"].unique(), ycol)
    va_batches = _date_batches(valid, valid["date"].unique(), ycol)
    max_train_date = _as_utc(inner_tr["date"].max()) if not inner_tr.empty else None
    max_ho_date = _as_utc(inner_ho["date"].max()) if not inner_ho.empty else None
    train_end_utc = _as_utc(fold.train_end)
    if max_train_date is not None and max_train_date > train_end_utc:
        raise RuntimeError(f"train dataloader max date {max_train_date.date()} > train_end {train_end_utc.date()}")
    if max_ho_date is not None and max_ho_date > train_end_utc:
        raise RuntimeError(f"inner-holdout max date {max_ho_date.date()} > train_end {train_end_utc.date()}")
    if shuffle_labels:
        rng = np.random.default_rng(int(seed) + 17_389)
        def _shuf(batches):
            out = []
            for dt, rids, y in batches:
                out.append((dt, rids, rng.permutation(np.asarray(y))))
            return out
        tr_batches = _shuf(tr_batches)
        ho_batches = _shuf(ho_batches)
    if not tr_batches or not ho_batches:
        return pd.DataFrame(), {"fold_id": fold.fold_id, "seed": seed, "status": "empty_batches", "elapsed": 0.0}

    model = _make_gru(n_feat=N_FEAT, hidden=hidden, dropout=dropout).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()
    t0 = time.time()
    best_ic = -1e9
    best_state = None
    stale = 0
    history = []

    def _run_batches(batches, train_mode: bool) -> tuple[float, float]:
        ics = []
        losses = []
        if train_mode:
            model.train()
            order = np.random.default_rng(seed + int(time.time() * 1e6) % 100000).permutation(len(batches))
        else:
            model.eval()
            order = np.arange(len(batches))
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for bi in order:
                dt, rids, y = batches[int(bi)]
                xb = torch.from_numpy(np.asarray(X_all[rids])).to(device)
                yb = torch.from_numpy(y).to(device)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = huber(pred, yb)
                if pairwise_rank and train_mode and pred.numel() >= 8:
                    # off by default; pairwise hinge on score order vs label order
                    diff_p = pred.unsqueeze(1) - pred.unsqueeze(0)
                    diff_y = yb.unsqueeze(1) - yb.unsqueeze(0)
                    sign = torch.sign(diff_y)
                    rank_loss = torch.relu(1.0 - sign * diff_p).mean()
                    loss = loss + 0.1 * rank_loss
                if train_mode:
                    loss.backward()
                    opt.step()
                losses.append(float(loss.detach().item()))
                ics.append(_rank_ic_torch(pred.detach().cpu().numpy(), y))
        ics_f = [x for x in ics if np.isfinite(x)]
        return float(np.mean(losses) if losses else np.nan), float(np.mean(ics_f) if ics_f else np.nan)

    for epoch in range(1, max_epochs + 1):
        # deterministic shuffle per epoch
        rng = np.random.default_rng(seed * 1000 + epoch)
        order = rng.permutation(len(tr_batches))
        model.train()
        tr_losses = []
        for bi in order:
            _dt, rids, y = tr_batches[int(bi)]
            xb = torch.from_numpy(np.asarray(X_all[rids])).to(device)
            yb = torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
        ho_loss, ho_ic = _run_batches(ho_batches, train_mode=False)
        history.append({"epoch": epoch, "train_loss": float(np.mean(tr_losses)), "ho_ic": ho_ic, "ho_loss": ho_loss})
        improved = np.isfinite(ho_ic) and ho_ic > best_ic + 1e-6
        if improved:
            best_ic = ho_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale == 0:
            print(
                f"[HB] GRU h={horizon} fold={fold.fold_id} seed={seed} epoch={epoch}/{max_epochs} "
                f"ho_ic={ho_ic:.4f} best={best_ic:.4f} stale={stale} device={device}",
                flush=True,
            )
        if stale >= patience:
            print(f"[GRU] early stop epoch={epoch} best_ic={best_ic:.4f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for dt, rids, y in va_batches:
            xb = torch.from_numpy(np.asarray(X_all[rids])).to(device)
            pred = model(xb).cpu().numpy()
            sub = idx.set_index("row_id").loc[list(rids)]
            for sym, sc, yt in zip(sub["symbol"].tolist(), pred, y):
                rec = {
                    "date": dt,
                    "symbol": str(sym),
                    "score": float(sc),
                    "horizon": int(horizon),
                    "fold_id": int(fold.fold_id),
                    "seed": int(seed),
                    ycol: float(yt),
                }
                rows.append(rec)
    pred_df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    meta = {
        "fold_id": fold.fold_id,
        "seed": seed,
        "horizon": horizon,
        "status": "ok" if not pred_df.empty else "empty_pred",
        "elapsed": elapsed,
        "n_params": n_params,
        "best_ho_ic": float(best_ic) if np.isfinite(best_ic) and best_ic > -1e8 else float("nan"),
        "n_epochs_run": int(len(history)),
        "max_epochs": int(max_epochs),
        "n_train_dates": int(len(tr_batches)),
        "n_ho_dates": int(len(ho_batches)),
        "n_val_dates": int(len(va_batches)),
        "n_pred": int(len(pred_df)),
        "device": str(device),
        "history_tail": history[-5:],
        "pairwise_rank": bool(pairwise_rank),
        "shuffle_labels": bool(shuffle_labels),
        "max_train_date": str(max_train_date.date()) if max_train_date is not None else None,
        "max_ho_date": str(max_ho_date.date()) if max_ho_date is not None else None,
        "fold_train_end": str(pd.Timestamp(fold.train_end).date()),
        "warm_start": False,
    }
    return pred_df, meta


def calibrate_epoch_seconds(cache_dir: Path, fold, horizon: int, seed: int, inner_holdout_days: int) -> dict:
    """Run 1 epoch on one fold and return seconds (used for GPU-hour projection)."""
    t0 = time.time()
    _pred, meta = train_gru_fold(
        cache_dir,
        fold,
        horizon=horizon,
        seed=seed,
        inner_holdout_days=inner_holdout_days,
        max_epochs=1,
        patience=10,
    )
    sec = time.time() - t0
    return {
        "sec_1_epoch": float(sec),
        "n_train_dates": meta.get("n_train_dates"),
        "n_params": meta.get("n_params"),
        "status": meta.get("status"),
        "device": meta.get("device"),
    }


def project_gpu_hours(
    sec_per_epoch: float,
    n_folds: int,
    n_seeds: int,
    n_horizons: int,
    max_epochs: int,
) -> dict:
    total_sec = float(sec_per_epoch) * n_folds * n_seeds * n_horizons * max_epochs
    hours = total_sec / 3600.0
    return {
        "sec_per_epoch": float(sec_per_epoch),
        "n_folds": int(n_folds),
        "n_seeds": int(n_seeds),
        "n_horizons": int(n_horizons),
        "max_epochs": int(max_epochs),
        "gpu_hours": float(hours),
        "gpu_seconds": float(total_sec),
    }
