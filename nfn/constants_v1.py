"""BTC-BEATER Phase 7.c — NFN v1 craft constants. Architecture imported from Phase 7."""

from __future__ import annotations

from nfn.constants import (
    FILM_HIDDEN,
    FILM_M_DIM,
    HORIZON,
    L1_LAMBDA,
    LISTWISE_COEF,
    LISTWISE_K,
    LOG_EPS,
    MEMBERSHIP_C_INIT,
    MEMBERSHIP_S_INIT,
    MEMBERSHIP_S_MIN,
    N_FEATURES,
    N_INIT_PRIMITIVES,
    N_MEMBERSHIPS,
    N_PRIMITIVES,
    N_RULES,
    NULL_FOLD_IDS,
    NULL_K_EXCEED,
    NULL_MAP_CONCURRENCY,
    NULL_REPLICATES,
    NULL_STOUFFER_Z,
    NULL_TRAIN_SEED,
    OVERLAP_DELTA,
    POS_WEIGHT_TOP,
    SEED_DISP_MAX,
    SEEDS,
    TAIL_IC_DELTA,
)

PHASE7C_CRITERION = (
    "NFN-v1 is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 "
    "vs the frozen spread; (b) seed dispersion of tail-IC(top) ≤ 0.010; (c) vol-matched null "
    "passes. CRAFT-CONFIRMED (separate, informational label) if dispersion improves to ≤ 0.010 "
    "and the mean selected-epoch window is ≥ 8 — i.e., the training pathology is fixed regardless "
    "of the performance verdict. If CRAFT-CONFIRMED but not LIVE, the ledger records: 'the neuro-fuzzy "
    "family was trained correctly and still does not exceed the frozen spread on daily price/volume "
    "features — training craft is excluded as an explanation; the daily ceiling is "
    "architectural-independent. Hourly features remain the designated retry substrate for this "
    "identical pipeline.' If neither, state which clause failed. Nothing adopted. Mechanical, "
    "no post-hoc adjustment."
)

PHASE7C_CRAFT_LEDGER = (
    "the neuro-fuzzy family was trained correctly and still does not exceed the frozen spread "
    "on daily price/volume features — training craft is excluded as an explanation; the daily "
    "ceiling is architectural-independent. Hourly features remain the designated retry substrate "
    "for this identical pipeline."
)

PHASE7C_NULL_REGISTRATION = (
    "VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with "
    "Modal .map (concurrency 40), seed-42 5-init bagged v1 craft (cold per fold); house bias "
    "tolerance; skill = ≥3/4 above p95 OR Stouffer ≥ 3."
)

# Byte-identical architecture snapshot (asserted at import / run start)
ARCH_FROZEN = {
    "N_FEATURES": 33,
    "N_MEMBERSHIPS": 3,
    "N_PRIMITIVES": 198,
    "N_RULES": 24,
    "N_INIT_PRIMITIVES": 3,
    "LOG_EPS": 1e-6,
    "L1_LAMBDA": 1e-3,
    "FILM_HIDDEN": 8,
    "FILM_M_DIM": 3,
    "POS_WEIGHT_TOP": 3.0,
    "LISTWISE_COEF": 0.3,
    "LISTWISE_K": 10,
    "MEMBERSHIP_C_INIT": (-0.67, 0.0, 0.67),
    "MEMBERSHIP_S_INIT": 1.0,
    "MEMBERSHIP_S_MIN": 0.2,
    "HORIZON": 14,
    "SEEDS": (42, 43, 44),
    "NULL_FOLD_IDS": (5, 15, 21, 24),
    "NULL_REPLICATES": 15,
    "TAIL_IC_DELTA": 0.010,
    "OVERLAP_DELTA": 0.015,
    "SEED_DISP_MAX": 0.010,
}

# Six craft items only
ADAMW_LR = 1e-4
ADAMW_LR_MIN = 1e-5
ADAMW_WD = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 40
ES_FLOOR_EPOCH = 10
ES_PATIENCE = 8
INNER_HOLDOUT_DATES = 120
PURGE_EXTRA = 3  # purged by h+3
N_BAG = 5
TRAILING_K = 3
SWA_K = 3
CACHE_VER = "p7cv1"
COLD_CONTROL_FOLDS = (9, 18, 27)
CRAFT_MEAN_EPOCH_MIN = 8.0
N_PARAMS_EXPECTED = 5488
V0_SEED_DISPERSION = 0.023081138810684305  # Phase 7 recorded; informational

CRAFT_KEYS = (
    "MODEL_SELECTION",
    "INTRA_FOLD_BAGGING",
    "WARMSTART_ACROSS_FOLDS",
    "OPTIMIZER",
    "INNER_HOLDOUT",
    "SEEDS_BAGGING_ENSEMBLE",
)


def assert_architecture_unchanged() -> dict:
    live = {
        "N_FEATURES": int(N_FEATURES),
        "N_MEMBERSHIPS": int(N_MEMBERSHIPS),
        "N_PRIMITIVES": int(N_PRIMITIVES),
        "N_RULES": int(N_RULES),
        "N_INIT_PRIMITIVES": int(N_INIT_PRIMITIVES),
        "LOG_EPS": float(LOG_EPS),
        "L1_LAMBDA": float(L1_LAMBDA),
        "FILM_HIDDEN": int(FILM_HIDDEN),
        "FILM_M_DIM": int(FILM_M_DIM),
        "POS_WEIGHT_TOP": float(POS_WEIGHT_TOP),
        "LISTWISE_COEF": float(LISTWISE_COEF),
        "LISTWISE_K": int(LISTWISE_K),
        "MEMBERSHIP_C_INIT": tuple(float(x) for x in MEMBERSHIP_C_INIT),
        "MEMBERSHIP_S_INIT": float(MEMBERSHIP_S_INIT),
        "MEMBERSHIP_S_MIN": float(MEMBERSHIP_S_MIN),
        "HORIZON": int(HORIZON),
        "SEEDS": tuple(int(s) for s in SEEDS),
        "NULL_FOLD_IDS": tuple(int(x) for x in NULL_FOLD_IDS),
        "NULL_REPLICATES": int(NULL_REPLICATES),
        "TAIL_IC_DELTA": float(TAIL_IC_DELTA),
        "OVERLAP_DELTA": float(OVERLAP_DELTA),
        "SEED_DISP_MAX": float(SEED_DISP_MAX),
    }
    mismatches = []
    for k, exp in ARCH_FROZEN.items():
        got = live.get(k)
        if got != exp:
            mismatches.append({"key": k, "expected": exp, "got": got})
    if mismatches:
        raise RuntimeError(f"architecture drifted from Phase 7: {mismatches}")
    return {"passed": True, "n_keys": len(ARCH_FROZEN), "craft_keys": list(CRAFT_KEYS)}


assert_architecture_unchanged()

__all__ = [
    "PHASE7C_CRAFT_LEDGER",
    "PHASE7C_CRITERION",
    "PHASE7C_NULL_REGISTRATION",
    "assert_architecture_unchanged",
]
