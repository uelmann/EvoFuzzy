"""PIT top-N universe + weekly rebalance calendar.

Wraps btcb.universe (trailing 30d median dollar volume, mcap fallback).
Volume is the default ranker; mcap is the explicit alternative.
"""

from __future__ import annotations

from typing import Literal

from .constants import REBALANCE_DAYS, UNIVERSE_N

RankMethod = Literal["volume", "mcap"]


def rebalance_dates(dates, every: int = REBALANCE_DAYS):
    """Keep every `every`-th unique date (sorted). First date is always kept."""
    uniq = sorted(set(dates))
    if every <= 1:
        return list(uniq)
    return [d for i, d in enumerate(uniq) if i % int(every) == 0]


def pit_topn(panel, n: int = UNIVERSE_N, method: RankMethod = "volume"):
    """Point-in-time top-N from a CMC-style panel (date, symbol, dv, mcap).

    Requires pandas + btcb.universe. Import is local so fuzzyx.model stays
    importable without pandas.
    """
    import pandas as pd

    from btcb.universe import build_pit_topn, trailing_rank_frame

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    score, mcap, detected = trailing_rank_frame(df)
    if method == "mcap":
        score = mcap
        detected = "mcap"
    uni = build_pit_topn(score, n=int(n))
    uni.attrs["rank_method"] = detected if method == "volume" else "mcap"
    return uni


def hold_from_rebalance(decision_pos, all_dates, rebalance, symbols):
    """Forward-fill weekly decisions onto a daily calendar.

    decision_pos: DataFrame date×symbol of {−1,0,+1} on rebalance dates.
    Returns a daily date×symbol frame (0 before the first decision).
    """
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(all_dates, utc=True)).sort_values().unique()
    dec = decision_pos.reindex(index=pd.DatetimeIndex(pd.to_datetime(rebalance, utc=True)), columns=symbols)
    daily = dec.reindex(idx).ffill().fillna(0.0)
    return daily
