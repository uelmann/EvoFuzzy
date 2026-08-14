"""Frozen constants for the symmetry audit (diagnostic label only)."""

from __future__ import annotations

CLASSIFICATION_CRITERION = (
    "The engine is labeled SYMMETRIC if, at h=7 or h=10 on at least two of the "
    "three universes, the full-period TOP spread is positive with NW-t ≥ 2.0 "
    "AND the full-period symmetry ratio ≥ 0.4. It is labeled LONG-SIDE GAP "
    "otherwise. This is a diagnostic label: SYMMETRIC closes the question "
    "(long-leg economics are a raw-material property of the asset class, not a "
    "model defect); LONG-SIDE GAP opens a targeted research question on "
    "winner-side information. Neither label changes the reference book. "
    "No post-hoc adjustment."
)

TOP_NW_T_MIN = 2.0
SYMMETRY_RATIO_MIN = 0.4
N_UNIVERSES_MIN = 2

HORIZONS = (7, 10)
UNIVERSES = ("top20", "top40", "top120")
N_BUCKETS = {"top20": 5, "top40": 5, "top120": 10}

FROZEN_A0_SHA256 = (
    "e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa"
)
PRED_H7_SHA256 = "8d48ea5a2f4ba47df986b57977f0be6ece2376a9277723a60606bafe150cf3a1"
PRED_H10_SHA256 = "74359bff9c68b345a531b96d42876d5b3c492800fab9bbdf9c11cb6f9e51f916"

PRED_H7 = "/data/quant/predictions/lgbm_price_only_h7.parquet"
PRED_H10 = "/data/quant/predictions/lgbm_price_only_h10.parquet"

ALT_SEASON_START = "2023-01-01"
ALT_SEASON_END = "2025-01-01"
ROLLING_DAYS = 90
ANNUALIZATION = 365
