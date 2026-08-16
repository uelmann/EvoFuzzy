"""Pack Stage-S rows + magnitude labels + regime into date-grouped arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.constants import STAGE_S_COLS
from btcb.phase4b import vol_col_name
from nfn_va.constants import HORIZON, WINSOR_HI, WINSOR_LO


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.normalize()


@dataclass
class PackedPanel:
    z: np.ndarray
    y_rank01: np.ndarray
    y_win: np.ndarray
    y_win_z: np.ndarray
    excess: np.ndarray
    vol: np.ndarray
    ids: np.ndarray
    date_id: np.ndarray
    dates: np.ndarray
    m: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    feat_names: list[str]
    vol_col: str
    excess_col: str


def _cs_magnitude_labels(excess: np.ndarray, date_id: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """Per date: rank-normalize to [0,1] AND winsorize raw magnitude at 1st/99th pct."""
    n = len(excess)
    y_rank = np.full(n, np.nan, dtype=np.float32)
    y_win = np.full(n, np.nan, dtype=np.float32)
    y_z = np.full(n, np.nan, dtype=np.float32)
    for di in range(len(starts)):
        a, b = int(starts[di]), int(ends[di])
        if b <= a:
            continue
        ex = excess[a:b].astype(np.float64)
        m = np.isfinite(ex)
        if int(m.sum()) < 4:
            continue
        v = ex[m]
        lo, hi = np.nanpercentile(v, [100.0 * float(WINSOR_LO), 100.0 * float(WINSOR_HI)])
        if not np.isfinite(lo):
            lo = float(np.nanmin(v))
        if not np.isfinite(hi):
            hi = float(np.nanmax(v))
        if hi < lo:
            lo, hi = hi, lo
        win = np.clip(ex, lo, hi)
        y_win[a:b] = win.astype(np.float32)
        mu = float(np.nanmean(win[m]))
        sd = float(np.nanstd(win[m], ddof=0))
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        z = (win - mu) / sd
        y_z[a:b] = z.astype(np.float32)
        # average ranks of finite names → [0,1]
        r = pd.Series(ex).rank(method="average", na_option="keep")
        k = float(m.sum())
        if k > 1:
            y_rank[a:b] = ((r - 1.0) / (k - 1.0)).to_numpy(dtype=np.float32)
        else:
            y_rank[a:b] = 0.5
    return y_rank, y_win, y_z


def pack_labeled(labeled: pd.DataFrame, regime: pd.DataFrame) -> PackedPanel:
    d = labeled.copy()
    d["date"] = _utc(d["date"])
    d["id"] = d["id"].astype(int)
    ex = f"excess_h{HORIZON}"
    volc = vol_col_name(d)
    feats = list(STAGE_S_COLS)
    need = feats + [ex, volc, "date", "id"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise RuntimeError(f"pack missing cols: {missing}")
    d = d.dropna(subset=feats + [ex]).copy()
    d = d.sort_values(["date", "id"]).reset_index(drop=True)

    reg = regime.copy()
    reg["date"] = _utc(reg["date"])
    mcols = ["m_ew50_btc_20d", "m_cs_disp_14", "m_breadth_pos"]
    d = d.merge(reg[["date"] + mcols], on="date", how="left")
    for c in mcols:
        d[c] = d[c].fillna(0.0)

    uniq = pd.DatetimeIndex(pd.unique(d["date"])).tz_convert("UTC").normalize().sort_values()
    date_to_i = {pd.Timestamp(t): i for i, t in enumerate(uniq)}
    date_id = np.asarray(
        [date_to_i[pd.Timestamp(t).tz_convert("UTC").normalize()] for t in d["date"]],
        dtype=np.int32,
    )

    z = d[feats].to_numpy(dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    excess = d[ex].to_numpy(dtype=np.float32)
    vol = d[volc].to_numpy(dtype=np.float32)
    ids = d["id"].to_numpy(dtype=np.int32)
    m = d[mcols].to_numpy(dtype=np.float32)

    n_dates = int(len(uniq))
    starts = np.zeros(n_dates, dtype=np.int32)
    ends = np.zeros(n_dates, dtype=np.int32)
    i, n = 0, len(date_id)
    while i < n:
        j = i + 1
        while j < n and date_id[j] == date_id[i]:
            j += 1
        di = int(date_id[i])
        starts[di] = i
        ends[di] = j
        i = j

    y_rank, y_win, y_z = _cs_magnitude_labels(excess, date_id, starts, ends)
    print(f"[p7d] packed rows={len(d)} dates={n_dates} feats={len(feats)} mag-labels=rank01+winsor", flush=True)
    return PackedPanel(
        z=z,
        y_rank01=y_rank,
        y_win=y_win,
        y_win_z=y_z,
        excess=excess,
        vol=vol,
        ids=ids,
        date_id=date_id,
        dates=uniq.to_numpy(dtype="datetime64[ns]"),
        m=m,
        starts=starts,
        ends=ends,
        feat_names=feats,
        vol_col=volc,
        excess_col=ex,
    )


def save_pack(pack: PackedPanel, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z=pack.z,
        y_rank01=pack.y_rank01,
        y_win=pack.y_win,
        y_win_z=pack.y_win_z,
        excess=pack.excess,
        vol=pack.vol,
        ids=pack.ids,
        date_id=pack.date_id,
        dates=pack.dates.astype("datetime64[ns]"),
        m=pack.m,
        starts=pack.starts,
        ends=pack.ends,
        feat_names=np.asarray(pack.feat_names),
        vol_col=np.asarray([pack.vol_col]),
        excess_col=np.asarray([pack.excess_col]),
    )


def load_pack(path) -> PackedPanel:
    z = np.load(path, allow_pickle=True)
    return PackedPanel(
        z=z["z"],
        y_rank01=z["y_rank01"],
        y_win=z["y_win"],
        y_win_z=z["y_win_z"],
        excess=z["excess"],
        vol=z["vol"],
        ids=z["ids"],
        date_id=z["date_id"],
        dates=z["dates"],
        m=z["m"],
        starts=z["starts"],
        ends=z["ends"],
        feat_names=[str(x) for x in z["feat_names"].tolist()],
        vol_col=str(z["vol_col"][0]),
        excess_col=str(z["excess_col"][0]),
    )


def date_index_window(pack: PackedPanel, start, end) -> np.ndarray:
    idx = pd.DatetimeIndex(pack.dates)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.normalize()
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    else:
        t1 = t1.tz_convert("UTC")
    t0, t1 = t0.normalize(), t1.normalize()
    mask = (idx >= t0) & (idx <= t1)
    return np.flatnonzero(mask).astype(np.int32)
