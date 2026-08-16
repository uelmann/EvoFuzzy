"""Rolling train window on top of A0 make_folds (does not change crypto CV)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from baseline.model import FoldSpec, make_folds


def make_session_folds(
    dates: pd.DatetimeIndex,
    horizon: int,
    min_train_days: int = 730,
    val_days: int = 90,
    step_days: int = 90,
) -> list[FoldSpec]:
    """Walk-forward folds with purge/embargo counted in sessions, not calendar days.

    A0 make_folds uses Timedelta(days=h), which equals sessions only on 24/7 crypto.
    For a 126-session equity label that under-purges. Do not change crypto CV.
    """
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    h = int(horizon)
    if len(dates) < min_train_days + val_days + h + 10:
        return []
    start = dates[0]
    folds: list[FoldSpec] = []
    fold_id = 0
    i_train_end = min_train_days - 1
    while True:
        if i_train_end >= len(dates):
            break
        train_end = dates[i_train_end]
        i_purge = max(0, i_train_end - h)
        purge_cut = dates[i_purge]
        i_embargo = min(len(dates) - 1, i_train_end + h + 3)
        embargo_end = dates[i_embargo]
        val_candidates = dates[i_embargo + 1 :]
        if len(val_candidates) < val_days // 2:
            break
        val_start = val_candidates[0]
        val_end = val_candidates[min(val_days - 1, len(val_candidates) - 1)]
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=start,
                train_end=purge_cut,
                purge_end=train_end,
                embargo_end=embargo_end,
                val_start=val_start,
                val_end=val_end,
                horizon=h,
            )
        )
        fold_id += 1
        next_idx = i_train_end + step_days
        if next_idx >= len(dates) - 5:
            break
        i_train_end = next_idx
        last_usable = dates[-(h + 1)] if len(dates) > h + 1 else dates[-1]
        if val_end >= last_usable:
            if len(dates[dates > val_end]) < step_days // 2:
                break
    return folds


def make_rolling_folds(
    dates: pd.DatetimeIndex,
    horizon: int,
    min_train_days: int,
    val_days: int,
    step_days: int,
    train_max_sessions: int,
    session_purge: bool = False,
) -> list[FoldSpec]:
    factory = make_session_folds if session_purge else make_folds
    folds = factory(
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
