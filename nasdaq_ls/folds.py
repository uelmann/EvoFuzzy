"""Rolling train window on top of A0 make_folds (does not change crypto CV)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from baseline.model import FoldSpec, make_folds


def make_rolling_folds(
    dates: pd.DatetimeIndex,
    horizon: int,
    min_train_days: int,
    val_days: int,
    step_days: int,
    train_max_sessions: int,
) -> list[FoldSpec]:
    folds = make_folds(
        dates,
        horizon=horizon,
        min_train_days=min_train_days,
        val_days=val_days,
        step_days=step_days,
    )
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    out: list[FoldSpec] = []
    for f in folds:
        eligible = idx[(idx >= f.train_start) & (idx <= f.train_end)]
        if len(eligible) > int(train_max_sessions):
            new_start = eligible[-int(train_max_sessions)]
            f = replace(f, train_start=new_start)
        out.append(f)
    return out
