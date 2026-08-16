"""Pack Stage-S rows + labels + regime into date-grouped arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btcb.constants import STAGE_S_COLS
from btcb.phase4b import vol_col_name
from nfn.constants import HORIZON


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.normalize()


@dataclass
class PackedPanel:
    z: np.ndarray
    y_top: np.ndarray
    y_bot: np.ndarray
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
    y_top_col: str
    y_bot_col: str
    excess_col: str


def pack_labeled(labeled: pd.DataFrame, regime: pd.DataFrame) -> PackedPanel:
    d = labeled.copy()
    d["date"] = _utc(d["date"])
    d["id"] = d["id"].astype(int)
    y_top = f"y_h{HORIZON}"
    y_bot = f"y_bot_h{HORIZON}"
    ex = f"excess_h{HORIZON}"
    volc = vol_col_name(d)
    feats = list(STAGE_S_COLS)
    need = feats + [y_top, y_bot, ex, volc, "date", "id"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise RuntimeError(f"pack missing cols: {missing}")
    d = d.dropna(subset=feats + [y_top, y_bot, ex]).copy()
    d = d.sort_values(["date", "id"]).reset_index(drop=True)

    reg = regime.copy()
    reg["date"] = _utc(reg["date"])
    mcols = ["m_ew50_btc_20d", "m_cs_disp_14", "m_breadth_pos"]
    d = d.merge(reg[["date"] + mcols], on="date", how="left")
    for c in mcols:
        d[c] = d[c].fillna(0.0)

    dates = d["date"].to_numpy()
    uniq = pd.DatetimeIndex(pd.unique(d["date"])).tz_convert("UTC").normalize().sort_values()
    date_to_i = {pd.Timestamp(t): i for i, t in enumerate(uniq)}
    date_id = np.asarray([date_to_i[pd.Timestamp(t).tz_convert("UTC").normalize()] for t in d["date"]], dtype=np.int32)

    z = d[feats].to_numpy(dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    y_t = d[y_top].to_numpy(dtype=np.float32)
    y_b = d[y_bot].to_numpy(dtype=np.float32)
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

    print(f"[nfn] packed rows={len(d)} dates={n_dates} feats={len(feats)}", flush=True)
    return PackedPanel(
        z=z,
        y_top=y_t,
        y_bot=y_b,
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
        y_top_col=y_top,
        y_bot_col=y_bot,
        excess_col=ex,
    )


def save_pack(pack: PackedPanel, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z=pack.z,
        y_top=pack.y_top,
        y_bot=pack.y_bot,
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
        y_top_col=np.asarray([pack.y_top_col]),
        y_bot_col=np.asarray([pack.y_bot_col]),
        excess_col=np.asarray([pack.excess_col]),
    )


def load_pack(path) -> PackedPanel:
    z = np.load(path, allow_pickle=True)
    return PackedPanel(
        z=z["z"],
        y_top=z["y_top"],
        y_bot=z["y_bot"],
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
        y_top_col=str(z["y_top_col"][0]),
        y_bot_col=str(z["y_bot_col"][0]),
        excess_col=str(z["excess_col"][0]),
    )


def date_index_window(pack: PackedPanel, start, end) -> np.ndarray:
    idx = pd.DatetimeIndex(pack.dates).tz_localize("UTC") if pd.DatetimeIndex(pack.dates).tz is None else pd.DatetimeIndex(pack.dates).tz_convert("UTC")
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
    m = (idx >= t0) & (idx <= t1)
    return np.flatnonzero(m).astype(np.int32)
