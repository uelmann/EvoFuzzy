"""BTC-BEATER Phase 7 — NEURO-FUZZY NET v0. Frozen a priori. One config."""

from __future__ import annotations

from btcb.constants import DEATH_CONVENTION, STAGE_S_COLS

FIREWALL = (
    "The PI's hand-made formula (gauss-momentum) is quarantined: its rules, ingredients, "
    "and structure are NOT provided to the miner, NOT seeded in the population, NOT added "
    "as features. The primitive set is exactly the 33 frozen house features — no additions. "
    "The only information retained from the falsification run is the reference book's "
    "performance numbers, used as the success bar. If the miner independently rediscovers "
    "a similar rule, that is a finding, not a leak."
)

FIREWALL_PHASE7 = (
    "The PI's hand formula stays quarantined: never seeded, never referenced. "
    "Warm-start rules, if any, come ONLY from the Phase-6 RULE-FORGE output bank "
    "(assert provenance)."
)

PHASE7_CRITERION = (
    "NFN is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 "
    "vs the frozen spread on full OOS; (b) seed dispersion of tail-IC(top) ≤ 0.010; "
    "(c) vol-matched null passes. PARKED otherwise with the failed clause stated. "
    "If PARKED on (a) with (b)(c) clean, the ledger gains: 'learnable fuzzy composition "
    "on daily 33-features does not exceed the frozen spread — the daily ceiling holds "
    "for this family too; hourly features (Phase 5 panel) are the designated retry "
    "substrate.' Nothing adopted; production use requires a fresh phase. Mechanical, "
    "no post-hoc adjustment."
)

PHASE7_CEILING = (
    "learnable fuzzy composition on daily 33-features does not exceed the frozen spread "
    "— the daily ceiling holds for this family too; hourly features (Phase 5 panel) are "
    "the designated retry substrate."
)

PHASE7_NULL_REGISTRATION = (
    "VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with "
    "Modal .map (concurrency 40), seed 42 config; house bias tolerance; skill = ≥3/4 above "
    "p95 OR Stouffer ≥ 3."
)

# Architecture (ONE config, zero search)
N_FEATURES = 33
assert len(STAGE_S_COLS) == N_FEATURES
N_MEMBERSHIPS = 3
MEMBERSHIP_C_INIT = (-0.67, 0.0, 0.67)
MEMBERSHIP_S_INIT = 1.0
MEMBERSHIP_S_MIN = 0.2
N_PRIMITIVES = N_FEATURES * N_MEMBERSHIPS * 2  # μ and (1−μ)
N_RULES = 24
N_INIT_PRIMITIVES = 3
LOG_EPS = 1e-6
L1_LAMBDA = 1e-3
FILM_HIDDEN = 8
FILM_M_DIM = 3
POS_WEIGHT_TOP = 3.0
LISTWISE_COEF = 0.3
LISTWISE_K = 10
WARMSTART_RULES = 8

# Training
ADAMW_LR = 3e-4
ADAMW_WD = 0.0
MAX_EPOCHS = 30
ES_FLOOR_EPOCH = 8
ES_PATIENCE = 6
UNDERTRAINED_BEST_LT = 10
HORIZON = 14
SEEDS = (42, 43, 44)
NULL_FOLD_IDS = (5, 15, 21, 24)
NULL_REPLICATES = 15
NULL_K_EXCEED = 3
NULL_STOUFFER_Z = 3.0
NULL_MAP_CONCURRENCY = 40
NULL_TRAIN_SEED = 42
CACHE_VER = "p7v1"

TAIL_IC_DELTA = 0.010
OVERLAP_DELTA = 0.015
SEED_DISP_MAX = 0.010

REGIME_EW_DAYS = 20
REGIME_RET_DAYS = 14

FORBIDDEN_TOKENS = (
    "gauss-momentum",
    "gaussmom",
    "gauss_momentum",
    "scores_at",
    "MANUEL2_FORMULA",
    "gauss(ret_14d)",
    "gauss(std_63d)",
    "std_63d",
    "ret_28d",
)

assert N_PRIMITIVES == 198
assert N_RULES == 24

__all__ = [
    "DEATH_CONVENTION",
    "FIREWALL",
    "FIREWALL_PHASE7",
    "PHASE7_CEILING",
    "PHASE7_CRITERION",
    "PHASE7_NULL_REGISTRATION",
]
