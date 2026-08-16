# BTC-BEATER — Phase 7.b FUZZY-STACK freeze addendum

**Status:** FROZEN before results. Learned fuzzy rules as LightGBM features + fixed-membership product library. Two arms: (A) product library of fixed CDF memberships (unconditional); (B) stack of learned rule activations from RULE-FORGE / NFN (conditional).
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Zero GPU. One shot on Modal.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned panel, floored PIT top-100 DV, 2.c folds, h=14, canonical pricing = Binance (3.e). Frozen 33 Stage-S features. 2.c spread cache reused as the baseline, not retrained.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

## Why this phase (on the record)

Phases 6/7 train memberships/rules by gradient or evolution where those methods are strong. COMPOSITION is done by LightGBM, where it is strong. This phase is that composition: fixed-CDF pairwise products (Arm A) and, when the rule banks exist and pass their own verdicts, the stack of those rule activations (Arm B).

## Firewall (verbatim)

> The PI's hand formula stays quarantined. Rule features come ONLY from RULE-FORGE/NFN outputs (assert provenance).

Operationalisation (frozen with the registration, not a result):

- The MANUEL-2 score `gauss(ret14)·gauss(ret28)/gauss(std63)` is **not** a feature, not a parent, not a baseline arm.
- Product-library primitives are `μ_j = Φ(z_cs(feature_j))` and `(1−μ_j)` over the frozen 33 Stage-S columns only. `ret_28` and `std_63` are not in that set.
- Arm B columns must carry provenance `ruleforge` or `nfn`. Any name matching the hand-formula needles is rejected.

## Training-hygiene guards (every LightGBM in this phase)

ES floor 200 iterations (patience 100, cap 3000). Log `best_iteration` and train/holdout curves every 50 iters. Flag UNDERTRAINED if `best_iteration < 250` and count. Same expanding 2.c folds / purge / embargo / seed. Twin-head binary classifiers, `per_date_auc` early-stop metric, isotonic per fold, spread = `p_top − p_bot`. Deterministic LightGBM (`deterministic=True`, `force_row_wise=True`). Zero GPU.

## Arm A — Product library (unconditional)

- Primitives: `μ_j = Φ(z_cs(feature_j))` and `(1−μ_j)` for the 33 features → 66. Library = all unordered pairwise products (`C(66,2) = 2,145` features), plus the 33 originals. One prune, no iteration.
- Two-stage (pre-registered): train the twin heads once on `[33 + library]`; keep the top-150 library features by **total gain** (TOP head) ∪ top-150 (BOTTOM head); retrain on `[33 + kept]`; isotonic per fold; spread as usual.
- Report the kept products (printed formulas, top-30 by gain).

## Arm B — Stack (conditional)

- Precondition per source: RULE-FORGE bank exists AND its report verdict ≥ VIABLE → its ≤8 rule activations become features. NFN report exists AND verdict = LIVE → its 24 ensemble rule activations become features. If neither passes, print `STACK-SKIPPED` with reasons and run Arm A only.
- Retrain twin heads on `[33 + available rule features]`; same folds/guards; spread as usual.
- ARM-A+B (only if both arms run): retrain on `[33 + kept products + rule features]`.

## Judgment (tail primary, house battery)

Grid: frozen spread (baseline) · ARM-A spread · ARM-B spread (if run) · ARM-A+B (if both). Columns: tail-IC top/bottom (NW-t), top-decile overlap, monster capture, whole-list RankIC, vol-rank-corr — full OOS / trailing-18m / per-cycle.

VOL-MATCHED NULL (house standard, folds `{0,5,9,15,21,24}` × 25) on the **best arm's** tail-IC(top) and overlap. Best arm = highest full-OOS tail-IC(top-half) among the fuzzy arms that ran.

Crude-14d book per signal (information only). Feature-importance dump: share of total gain captured by library/rule features vs originals.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Pre-registered criteria (verbatim, before results)

> An arm EXTRACTS if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread with the vol-matched null passing. COMPOSITION-WINS if additionally the arm beats its rule source's own standalone signal on tail-IC(top) (the stack must add over both parents). Whole-list note: if an arm fails tail thresholds but improves whole-list RankIC by ≥ +0.010 with null passing, it is recorded as a WHOLE-RANKING LEAD for a fresh production-book phase (the 4.c precedent), not adopted here. If all arms fail everything, the ledger gains: 'fuzzy-GBM composition on daily 33-features does not exceed the frozen spread; the daily composition question is closed.' Mechanical, no post-hoc adjustment.

Verdicts mechanical. Nothing adopted. Frozen products untouched.
