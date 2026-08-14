"""Binary excess-vs-BTC labels. y=1 iff h-day forward log-return exceeds BTC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcb.constants import PHASE2_HORIZONS


def add_binary_excess_labels(
    feat: pd.DataFrame,
    panel: pd.DataFrame,
    btc_id: int,
    horizons: tuple[int, ...] = PHASE2_HORIZONS,
) -> pd.DataFrame:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    close = p.pivot(index="date", columns="id", values="close").sort_index()
    if btc_id not in close.columns:
        raise RuntimeError("BTC missing for labels")
    out = feat.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
    logp = np.log(close.clip(lower=1e-18))
    for h in horizons:
        fwd = logp.shift(-h) - logp
        btc_fwd = fwd[btc_id]
        excess = fwd.sub(btc_fwd, axis=0)
        long = excess.stack().rename("excess").reset_index()
        long.columns = ["date", "id", f"excess_h{h}"]
        long["date"] = pd.to_datetime(long["date"], utc=True).dt.tz_convert("UTC").dt.normalize()
        long["id"] = long["id"].astype(int)
        out = out.merge(long, on=["date", "id"], how="left")
        out[f"y_h{h}"] = (out[f"excess_h{h}"] > 0).astype(float)
        out.loc[out[f"excess_h{h}"].isna(), f"y_h{h}"] = np.nan
    return out
