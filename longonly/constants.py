"""Frozen constants for the long-only parallel-product evaluation.

Viability statements are quoted verbatim in the report BEFORE any results.
No post-hoc adjustment. The reference COMBO book is unchanged.
"""

from __future__ import annotations

VIABILITY_CRITERION = (
    "LO-H is VIABLE as a standalone mandate only if its full-period net Sharpe "
    "≥ 0.7 AND trailing-18m net Sharpe ≥ 0.3. LO-U is VIABLE as a standalone "
    "mandate only if its full-period regression alpha vs BTC B&H is positive "
    "with NW-t ≥ 2.0 AND trailing-18m alpha is positive. These are viability "
    "labels for a parallel product; no outcome changes the reference book. "
    "No post-hoc adjustment."
)

LOH_FULL_SHARPE_MIN = 0.7
LOH_TRAIL_SHARPE_MIN = 0.3
LOU_NW_T_MIN = 2.0

P1_H = 7
P1_TAU = 80.0
P1_UNIVERSE = "top20"
P1_COST_BPS = 5.0
P1_SLIP_BPS = 3.0

P2_H = 10
P2_TAU = 70.0
P2_UNIVERSE = "top40"
P2_COST_TIERED = True
P2_LIQ_CAP = 0.005
P2_NOM_USD = 1_000_000.0

COMBO_W = 0.5
TRAIL_DAYS = 547
ANNUALIZATION = 365

FROZEN_A0_SHA256 = (
    "e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa"
)
FROZEN_A0_PATH = "/data/quant/models/lgbm_price_only.txt"
CONFIG_FROZEN = "config_frozen_a0.yaml"

PRED_H7 = "/data/quant/predictions/lgbm_price_only_h7.parquet"
PRED_H10 = "/data/quant/predictions/lgbm_price_only_h10.parquet"
PIT_TOP20 = "/data/quant/universe/top20_pit.parquet"
PIT_TOP40 = "/data/quant/universe/top40_pit.parquet"
PANEL_PATH = "/data/quant/panel/panel.parquet"
