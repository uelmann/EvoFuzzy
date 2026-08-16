"""BTC-BEATER Phase 7.d — NFN VARIANT A. Frozen a priori. One config."""

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

PHASE7D_CRITERION = (
    "VARIANT-A is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 "
    "vs the frozen spread; (b) seed dispersion of tail-IC(top) ≤ 0.010; (c) vol-matched null passes. "
    "MAGNITUDE-GAIN (separate, informational label) if the mean realized excess return of its top-10 "
    "picks exceeds the frozen spread's by ≥ 20% relative, regardless of the LIVE verdict — this isolates "
    "whether magnitude labels buy bigger winners even when rank metrics do not move. If PARKED, state "
    "which clause failed. Nothing adopted; production use requires a fresh phase. Mechanical, "
    "no post-hoc adjustment."
)

PHASE7D_FIREWALL = FIREWALL_PHASE7

PHASE7_NULL_REGISTRATION = (
    "VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with "
    "Modal .map (concurrency 40), seed 42 config; house bias tolerance; skill = ≥3/4 above "
    "p95 OR Stouffer ≥ 3."
)

# Architecture — BYTE-IDENTICAL to Phase 7 except head (single scalar).
N_FEATURES = 33
assert len(STAGE_S_COLS) == N_FEATURES
N_MEMBERSHIPS = 3
MEMBERSHIP_C_INIT = (-0.67, 0.0, 0.67)
MEMBERSHIP_S_INIT = 1.0
MEMBERSHIP_S_MIN = 0.2
N_PRIMITIVES = N_FEATURES * N_MEMBERSHIPS * 2
N_RULES = 24
N_INIT_PRIMITIVES = 3
LOG_EPS = 1e-6
L1_LAMBDA = 1e-3
FILM_HIDDEN = 8
FILM_M_DIM = 3
WARMSTART_RULES = 8
N_HEADS_PHASE7 = 2
N_HEADS = 1  # the single change
PHASE7_N_PARAMS = 5488
VARIANT_A_N_PARAMS = PHASE7_N_PARAMS - 25  # drop head_bot Linear(24,1)

PHASE7_ARCH = {
    "n_features": N_FEATURES,
    "n_memberships": N_MEMBERSHIPS,
    "membership_c_init": MEMBERSHIP_C_INIT,
    "membership_s_init": MEMBERSHIP_S_INIT,
    "membership_s_min": MEMBERSHIP_S_MIN,
    "n_primitives": N_PRIMITIVES,
    "n_rules": N_RULES,
    "n_init_primitives": N_INIT_PRIMITIVES,
    "log_eps": LOG_EPS,
    "l1_lambda": L1_LAMBDA,
    "film_hidden": FILM_HIDDEN,
    "film_m_dim": FILM_M_DIM,
    "warmstart_rules": WARMSTART_RULES,
}

# Variant A loss / labels (the allowed diff)
TAU = 1.0
RANK_COEF = 1.0
MAG_COEF = 0.5
HUBER_DELTA = 1.0
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Training craft (Phase 7.c corrections; frozen)
ADAMW_LR = 1e-4
ADAMW_LR_MIN = 1e-5
ADAMW_WD = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 40
ES_FLOOR_EPOCH = 10
ES_PATIENCE = 8
TRAIL_WINDOW = 3
SWA_K = 3
N_INITS = 5
INNER_HOLDOUT_DATES = 120
UNDERTRAINED_BEST_LT = 10
HORIZON = 14
SEEDS = (42, 43, 44)
NULL_FOLD_IDS = (5, 15, 21, 24)
NULL_REPLICATES = 15
NULL_K_EXCEED = 3
NULL_STOUFFER_Z = 3.0
NULL_MAP_CONCURRENCY = 40
NULL_TRAIN_SEED = 42
CACHE_VER = "p7dva1"

TAIL_IC_DELTA = 0.010
OVERLAP_DELTA = 0.015
SEED_DISP_MAX = 0.010
MAG_GAIN_REL = 0.20
TOP_K_PICKS = 10

REGIME_EW_DAYS = 20
REGIME_RET_DAYS = 14

# Read-only Phase 7 NFN v0 numbers (PR 15 PARKED). Overridden if volume report exists.
NFN_V0_READONLY = {
    "source": "Phase 7 NFN v0 report (PARKED; read-only)",
    "label": "nfn_v0",
    "verdict": "PARKED",
    "n_params": 5488,
    "tail_ic_top": 0.06322926171542223,
    "tail_ic_bot": 0.14553782618524191,
    "overlap": 0.0746956013838343,
    "monster": 0.04677180212966707,
    "rankic": 0.1554503632916237,
    "n_dates": 2473,
}

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
assert VARIANT_A_N_PARAMS == 5463

ALLOWED_DIFFS = frozenset(
    {
        "n_heads",
        "isotonic",
        "label",
        "loss",
        "craft",
    }
)
