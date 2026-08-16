# BTC-BEATER — Phase 8 MODEL-ZOO freeze addendum

**Status:** FROZEN before results. Three non-GBM model classes on the daily cross-section: (A) CS-ATTN-DAILY, (B) TabPFN v2, (C) RIDGE ON RANKS.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. One config per arm, zero architecture search. Independent of Phases 7.c / 7.d (those phases are not in this repo; Arm A training craft is frozen here from the Phase 8 brief).
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**. Frozen 2.c / 4.b / 4v2 / FORGE caches are not mutated.
Cleaned panel, floored PIT top-100 DV, 2.c fold schedule (purge/embargo h+3), h=14, canonical pricing = Binance-hybrid (3.e). Features = frozen 33 (`STAGE_S_COLS`).

GPU allowed **ONLY** for Arm B if TabPFN requires it. Hard cap **$20**, logged as wall-time × published A10G rate. Arms A and C are CPU.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

## Firewall (verbatim)

> The PI's hand-made formulas (including MANUEL-2) are quarantined from this phase: not imported, not seeded, not used as features or targets.

Pricing helpers (`hybrid_close_wide`) are allowed. `MANUEL2_FORMULA` is not imported.

## PI data-perimeter (verbatim; still binding)

> Catalyst and attention data families (unlocks, listing announcements, search volume) are OUT OF SCOPE by PI decision; the data perimeter is price/volume plus derivatives data already retrievable (funding, open interest, basis, taker flows).

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Pre-registered criteria (verbatim, before results)

> An arm is LIVE if Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread with the vol-matched null passing. An arm is a WHOLE-RANKING LEAD if instead ΔRankIC ≥ +0.010 with the null passing (recorded for a fresh production-book phase, not adopted here). LINEAR-CEILING (informational, from Arm C): if Arm C's whole-list RankIC ≥ 0.90 × the frozen spread's, the ledger records that nonlinearity contributes less than 10% of the daily signal and future daily modeling effort is unjustified. If any arm's signal has correlation < 0.6 with the frozen spread while reaching RankIC ≥ 0.10, it is recorded as an ORTHOGONAL SIGNAL candidate for a fresh blending phase. Nothing adopted here. Mechanical, no post-hoc adjustment.

Mechanical operationalisation (frozen with the criteria, not a result):

- LIVE / LEAD deltas are vs the frozen 2.c spread on the **judgment date set** (full OOS, or the pre-declared 1-in-3 subsample if TabPFN cannot finish full OOS inside the GPU cap).
- LIVE is checked first. WHOLE-RANKING LEAD fires only if LIVE does not (`instead`).
- Vol-matched null is run on the **single** zoo arm with the highest whole-list RankIC on the judgment date set. LIVE uses the tail-IC(top-half) gate; LEAD uses the RankIC gate on the same shuffles. Other arms are not nulled and therefore cannot be LIVE or LEAD.
- LINEAR-CEILING and ORTHOGONAL SIGNAL do not require the null. Correlation = mean per-date Spearman of the two signals on the judgment date set.
- Nothing adopted.

## TabPFN caveat (verbatim, before results)

> TabPFN assumes i.i.d.-like structure from its synthetic prior; financial non-stationarity violates it. This arm tests whether that matters in practice.

## Date subsample (verbatim, before results)

> Primary comparison = full OOS for every arm that completes. If TabPFN cannot finish full OOS inside the $20 GPU cap, ALL arms (and the frozen spread) are judged on the pre-declared 1-in-3 OOS date subsample: sorted unique OOS dates, keep i % 3 == 0 (0-indexed). Arms A/C still report full-OOS metrics as informational. The 1-in-3 rule is frozen before results.

## Vol-matched null (verbatim, before results)

> VOL-MATCHED NULL on the single BEST zoo arm by whole-list RankIC: folds {5,15,21,24} × 15 within-vol-quintile shuffles (first 15 of NULL_SHUFFLE_SEEDS), Modal .map fan-out. Skill = real exceeds vol-matched null p95 on ≥3/4 folds OR Stouffer z ≥ 3.0. Bias = 2·SE band around the fold's own null mean; CONTAMINATED iff ≥2 fold violations. LIVE uses the tail-IC(top-half) gate; WHOLE-RANKING LEAD uses the RankIC gate on the same shuffles. Only the best arm is nulled; other arms cannot be LIVE or LEAD. CS-ATTN null cells are cold-start, primary seed 42 only (not 3-init bag; null folds are non-contiguous so warm-start does not apply). Ridge and TabPFN nulls rerun the frozen procedure. RankIC null band is recorded for the chart.

Shuffle = house `vol_matched` (within-date `yz_vol_30` quintiles; missing vol = own bucket). Joint across the arm's label columns.

## Shared perimeter (frozen)

- Features: `STAGE_S_COLS` (33). Context columns are not used. CS-z re-applied (ddof=0, clip 5) so Arm A's input contract is explicit even though `feat_s` is already Stage-S z-scored.
- Labels: `add_twin_quintile_labels` at h=14 (`y_h14`, `y_bot_h14`, `excess_h14`). BTC rows dropped.
- Folds: `make_expanding_folds` (purge h, embargo h+3), same 2.c schedule.
- Inner holdout (Arms A and C): last **120 train dates**, purged by h (inner-train dates ≤ inner-ho start − h). Not the LGBM 90-calendar-day cut.
- Eval: floored PIT top-100 labeled CS vs next-14d excess, restricted to Binance-listed names at t (house `restrict_eval_frame`). Canonical close = Binance-hybrid.
- Frozen spread: 2.c pred cache, `collapse_fold_preds`. Not retrained.
- Judgment grid: tail-IC top/bottom (NW-t, lag=14), top-decile overlap, monster top-3 capture, whole-list RankIC — full OOS / trailing-18m / per-cycle; seed dispersion for Arm A.
- Crude-14d books: Ladder-1 EW top-decile, 10% cap, 10 bps/side. Information only.
- Correlation matrix of four signals (frozen spread + A + B + C): mean per-date Spearman. Orthogonality is itself a finding.

## Arm A — CS-ATTN-DAILY (one config, no search)

- Input per date: `[n_coins × 33]` CS-z features. **No temporal encoder.** The set is unordered: **no positional encoding**.
- Per-coin embed: `Linear(33 → 64)`. Then 2 × (pre-norm multi-head self-attention over coins of that date, 4 heads, width 64, `dim_feedforward=128`, dropout 0, residual) → per-coin twin linear heads (top-quintile / bottom-quintile). Signal = calibrated `p_top − p_bottom`. Target ~60k params; actual count is dumped.
- Loss: mean BCE on both heads, labeled coins only. Unlabeled coins may still sit in the set (attention context) but do not enter the loss.
- Training craft (frozen; 7.c is not in-repo): 3-epoch trailing-mean selection of inner-ho mean per-date RankIC of the raw spread vs `excess_h14`; SWA over the 3 epochs in the best trailing window; ES floor 10, patience 8, cap 40; 3-init intra-fold bagging seeds `{42,43,44}`; warm-start weights across folds (fold 0 cold; optimizer state is **not** carried); AdamW 1e-4 cosine → 1e-5 over 40 epochs, wd 1e-4, clip 1.0; date-batch 64.
- Bagging: average the three seeds' `p_top` / `p_bot`, then per-fold isotonic of each head on inner-ho, then spread.
- Diagnostic: mean attention entropy per date (normalized by `log n_coins`; mean over queries, heads, layers). Top-5 attended peers of the highest-scored coin on 10 linspace OOS dates, using that date's fold model at seed 42. Question on the record: does it use the cross-section, or collapse to self-attention?
- CPU only.

## Arm B — TabPFN v2 (in-context, no gradient training)

- Per fold: context = stratified subsample of the train window, **capped at 10,000 rows** (TabPFN v2 documented sample limit), sampled evenly across train dates, seed 42. The same context is reused for every query date in the fold (the train window does not change within a fold). Features = the 33. Two heads: binary top-quintile-within-date and binary bottom-quintile-within-date. Signal = `p_top − p_bottom`.
- No fine-tuning, no gradient steps. `n_estimators=8` (package default), one config.
- Inference is **batched per fold** (all val-date query rows in one `predict_proba` per head) because per-date forwards would multiply GPU time by ~n_val_dates and blow the cap. Wall-time per date = fold_predict_seconds / n_val_dates. Total = sum over folds and heads. Declared before results.
- GPU (A10G) allowed for this arm only. If import/GPU fails, the arm is UNAVAILABLE (not silently swapped). If projected cost exceeds $20, remaining query dates switch to the 1-in-3 subsample (and the comparison table follows).
- Package: `tabpfn>=2.0,<3`. Installed version is dumped. Context cap remains 10,000 even if a newer package allows more.

## Arm C — RIDGE ON RANKS (the linear floor)

- Per date: features = CS percentile ranks of the 33. Target = CS percentile rank of 14d excess return. `sklearn.linear_model.Ridge`, `fit_intercept=True`.
- Alpha chosen **once per fold** by inner-holdout RankIC on the declared grid `{1e-2, 1e-1, 1, 10, 100}` only. Ties → larger alpha. Then refit on the full purged train window. Signal = predicted rank.
- Zero nonlinearity, zero interactions. This is the diagnostic floor.

## What this freeze does not do

- No schedules, no live components, no product file changes
- No architecture search, no extra TabPFN / attention / ridge configs beyond the declared ridge alpha grid
- No mutation of the 2.c pred cache, Phase 4 / FORGE caches, or the CMC panel
- No import or seeding of the PI hand formula
- No GPU for Arms A or C
- Nothing adopted
