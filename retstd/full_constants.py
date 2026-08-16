"""Frozen constants for RETSTD-FULL (3000 trees, no early stop)."""

from __future__ import annotations

IMPROVE_CRITERION = (
    "RETSTD-FULL IMPROVES on A0-LO10 only if ALL of: "
    "(a) pooled OOS RankIC of RETSTD-FULL predicted probability vs the continuous "
    "h=10 USDT-return / forward-path-std ratio exceeds A0's RankIC vs the same ratio; "
    "(b) RETSTD-FULL top-decile minus universe 10-day USDT simple return exceeds A0-LO10's; "
    "(c) RETSTD-FULL long-only net Sharpe exceeds A0-LO10 net Sharpe; "
    "(d) the RETSTD-FULL label-shuffle null is GREEN. "
    "This is an A/B target test, not a replacement for COMBO. No post-hoc adjustment."
)

VIABILITY_CRITERION = (
    "RETSTD-FULL is VIABLE as a standalone long-only mandate only if ALL of: "
    "(a) full-period net Sharpe ≥ 0.50; (b) trailing-18m net Sharpe ≥ 0.00; "
    "(c) full-period total return > 0; (d) average deployed gross ≥ 0.15; "
    "(e) the RETSTD-FULL label-shuffle null is GREEN. "
    "It does not replace COMBO, SPREAD-LS, or LONG-TIDE. No post-hoc adjustment."
)

NULL_GATE = (
    "Bias: every judged fold's null mean RankIC must satisfy |mean| ≤ 2·(SD / √R). "
    "Skill passes if the real RETSTD-FULL OOS RankIC exceeds the null 95th percentile on "
    "**both** judged folds. Failure = PARKED (CONTAMINATED if bias fails, "
    "PARKED-NO-SKILL if bias passes and skill fails). No override, no retest with different folds."
)

TRAIN_RULE = (
    "Every fold trains exactly 3000 LightGBM trees. No early stopping. "
    "No median-best_iteration fallback. This matches the configured n_estimators cap "
    "(lr=0.03). The judged book uses those 3000-tree models."
)

FULL_TREES = 3000
OUT_ROOT = "/data/quant/retstd_full"
