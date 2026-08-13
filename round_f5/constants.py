"""Round F5 pre-registered constants. Do not change after the addendum commit."""

from baseline.features import FEATURE_COLS
from round_f.constants import CTX_COLS, P1_H, P1_TAU, P2_H, P2_TAU

SLEEVE_CRITERION = (
    "A candidate replaces the incumbent P2 sleeve only if its trailing-18m net Sharpe ≥ incumbent + 0.15 "
    "AND its full-period net Sharpe ≥ incumbent − 0.10 AND (for C3 only) its paired ΔRankIC vs plain A0 "
    "on top-40 satisfies the house block criterion at h=10 or h=7 (trail ≥ +0.005, full ≥ 0, ≥60% positive "
    "trailing folds). Among qualifying candidates, the sleeve with the highest trailing-18m net Sharpe is "
    "selected. If none qualify, the incumbent P2 stays. The +0.15 hurdle exists because four candidates "
    "are compared on a 548-day window; no post-hoc adjustment."
)

COMBO_PRIME_CRITERION = (
    "COMBO′ becomes the reference book only if its trailing-18m net Sharpe ≥ COMBO trailing − 0.05 "
    "AND its full-period net Sharpe ≥ COMBO full − 0.05, where COMBO is the Round-F adopted book "
    "(full 1.711, trail 0.997). Otherwise the Round-F COMBO stays the reference."
)

# Frozen Round-F COMBO (verbatim in the COMBO′ rule)
COMBO_F_FULL = 1.711
COMBO_F_TRAIL = 0.997

PRUNED_COLS = [
    "rev_1",
    "rev_3",
    "dv_z_30",
    "dv_trend",
    "ret_28",
    "skew_28",
    "mom_28_skip7",
    "ret_7",
]

P2_PRIME_COLS = [c for c in FEATURE_COLS if c not in PRUNED_COLS] + list(CTX_COLS)
assert len(P2_PRIME_COLS) == 32, len(P2_PRIME_COLS)

LEDGER_C0_FULL = 1.470
LEDGER_C0_TRAIL = 0.723
LEDGER_C1_FULL, LEDGER_C1_TRAIL = 1.257, 1.045
LEDGER_C2_FULL, LEDGER_C2_TRAIL = 1.314, 1.120

__all__ = [
    "SLEEVE_CRITERION",
    "COMBO_PRIME_CRITERION",
    "COMBO_F_FULL",
    "COMBO_F_TRAIL",
    "PRUNED_COLS",
    "P2_PRIME_COLS",
    "CTX_COLS",
    "P1_H",
    "P1_TAU",
    "P2_H",
    "P2_TAU",
]
