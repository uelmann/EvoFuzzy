"""Pre-registered Phase D.2 constants. Do not change after the addendum commit."""

HONESTY_PREAMBLE = (
    "The top-40 hypothesis originates from patterns observed in Phase D results "
    "(micro features helping on the wide universe while failing on top-20), reinforced "
    "by Phases E/E.1b (every surviving signal lives on the wide universe; top-20 IC "
    "decays while pit-120 IC holds). This test is therefore not fully independent. "
    "Protections: the adoption criterion below is pre-registered before running, and "
    "adoption is judged on tradeable net Sharpe with liquidity-tiered costs — a different "
    "object from the pit-120 RankIC that surfaced the pattern."
)

ADOPTION_CRITERION = (
    "Top-40 execution is ADOPTED if P2 or P4 trailing-18m median-τ net Sharpe ≥ P1 + 0.30 "
    "AND its full-period net Sharpe ≥ P1 − 0.20. The micro block is ADOPTED on the chosen "
    "universe if the corresponding paired trailing-18m ΔRankIC on that universe ≥ +0.005 "
    "AND full-period ΔRankIC on that universe ≥ 0 AND its portfolio (P3 or P4 vs its A "
    "counterpart) trailing-18m net Sharpe Δ ≥ 0. Verdicts are mechanical; no post-hoc adjustment."
)

TAU_PCTS = [60.0, 70.0, 80.0, 90.0]
HORIZONS = [7, 10]
NOMINAL_BOOK_USD = 1_000_000.0
LIQ_CAP_ADV_FRAC = 0.005
FEE_BPS_TOP = 5.0
SLIP_BPS_TOP = 3.0
FEE_BPS_NEXT = 10.0
SLIP_BPS_NEXT = 8.0
TRAIL_DAYS = int(365 * 1.5)
