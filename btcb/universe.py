"""PIT investable universes for BTC-BEATER."""

from __future__ import annotations

import pandas as pd

from btcb.constants import PIT_DV_MIN_PERIODS, PIT_DV_WINDOW, STABLE_OR_WRAP


def collapse_duplicate_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one cryptocurrency_id per ticker: highest last observed mcap."""
    last = df.sort_values("date").groupby("id").tail(1)
    last = last.sort_values("mcap", ascending=False).drop_duplicates("symbol", keep="first")
    keep_ids = set(int(x) for x in last["id"])
    return df[df["id"].isin(keep_ids)].copy()


def trailing_rank_frame(df: pd.DataFrame, window: int = PIT_DV_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Wide trailing-30d median dollar volume (fallback mcap). Returns dv_med, mcap_wide, method."""
    panel = df.copy()
    panel = panel[~panel["symbol"].str.upper().isin(STABLE_OR_WRAP)]
    panel = collapse_duplicate_symbols(panel)
    dv = panel.pivot(index="date", columns="symbol", values="dv").sort_index()
    mcap = panel.pivot(index="date", columns="symbol", values="mcap").sort_index()
    dv_med = dv.rolling(window, min_periods=min(PIT_DV_MIN_PERIODS, window)).median()
    vol_cov = float(dv.notna().mean().mean()) if dv.size else 0.0
    if vol_cov < 0.3:
        method = "mcap"
        score = mcap.rolling(window, min_periods=min(PIT_DV_MIN_PERIODS, window)).median()
    else:
        method = "trailing_30d_median_dollar_volume"
        score = dv_med.where(dv_med.notna() & (dv_med > 0), mcap)
        # remaining holes: mcap
        score = score.fillna(mcap)
    return score, mcap, method


def build_pit_topn(score: pd.DataFrame, n: int) -> pd.DataFrame:
    if score.empty:
        return pd.DataFrame(columns=["date", "symbol", "rank", "score"])
    ranked = score.rank(axis=1, ascending=False, method="first")
    mask = (ranked <= int(n)) & score.notna() & (score > 0)
    long = score.where(mask).stack().rename("score").reset_index()
    long.columns = ["date", "symbol", "score"]
    rnk = ranked.where(mask).stack().rename("rank").reset_index()
    rnk.columns = ["date", "symbol", "rank"]
    out = long.merge(rnk, on=["date", "symbol"], how="inner")
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out["rank"] = out["rank"].astype(int)
    return out.sort_values(["date", "rank"]).reset_index(drop=True)
