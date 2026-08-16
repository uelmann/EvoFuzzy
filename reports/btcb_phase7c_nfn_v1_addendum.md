# BTC-BEATER — Phase 7.c NFN v1 freeze addendum (training craft)

**Status:** FROZEN before results. Architecture byte-identical to Phase 7 NFN v0. The only changes are the six training-craft items listed below. Zero architecture search. Zero GPU.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Modal `.map` fan-out for vol-matched null (concurrency 40) and for the 15 seed×bag walk-forward chains.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned panel, floored PIT top-100 DV (primary), 2.c fold schedule (purge/embargo h+3), h=14, canonical hybrid pricing. Frozen 33 Stage-S features. Caches sha256-verified. 2.c spread cache reused as the frozen-spread baseline, not retrained.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

Purpose: Phase 7 NFN PARKED on all three clauses, but hygiene showed `best_epoch` = 1–2 in most folds and seed dispersion 0.023. This phase removes "our training was wrong" as an explanation, permanently, in either direction.

## Firewall (verbatim, first)

> The PI's hand-made formula (gauss-momentum) is quarantined: its rules, ingredients, and structure are NOT provided to the miner, NOT seeded in the population, NOT added as features. The primitive set is exactly the 33 frozen house features — no additions. The only information retained from the falsification run is the reference book's performance numbers, used as the success bar. If the miner independently rediscovers a similar rule, that is a finding, not a leak.

Phase-7 restatement (verbatim):

> The PI's hand formula stays quarantined: never seeded, never referenced. Warm-start rules, if any, come ONLY from the Phase-6 RULE-FORGE output bank (assert provenance).

Operationalisation (unchanged from Phase 7):

- Miner / net source is grepped for hand-formula tokens. Assert none.
- `btcb.manuel2` is not imported anywhere under `nfn/`.
- Primitive set = 33 frozen `STAGE_S_COLS`. Regime vector `m_t` is date-level FiLM context, not an added feature column.
- Phase-6 RULE-FORGE bank remains absent / not VIABLE on this tree. Fold-0 inits are random. Craft item 3 (fold-to-fold weight continuity) is the v1 warm-start; it does not seed the PI formula.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Architecture (byte-identical to Phase 7; asserted)

Memberships, log-space rule layer with L1 exponents, FiLM regime gate, twin tail-aware heads, 5,488 params. Frozen constants: 33×3×2 = 198 primitives, 24 rules, 3 init primitives at e=1, L1 λ=1e−3, FiLM hidden 8, BCE pos_weight=3, listwise 0.3× top-10, membership c init {−0.67, 0, +0.67}, s init 1.0 clamp s≥0.2. Config equality is asserted at run start except the six craft items.

## Training-craft corrections (the only changes; all frozen a priori)

1. MODEL SELECTION: no argmax on a single epoch. Selection metric = 3-epoch trailing mean of inner-holdout per-date tail-IC(top); the chosen checkpoint is the SWA average of the weights of the best 3 epochs by that criterion. ES floor 10 epochs, patience 8, cap 40.
2. INTRA-FOLD BAGGING: 5 independent inits per fold; fold signal = mean of their calibrated spreads. (This is the variance killer; report per-fold spread of the 5.)
3. WARM-START ACROSS FOLDS: each fold initializes from the previous fold's final weights (expanding window continuity), except fold 0. Report a control: 3 folds re-run cold-start, to quantify the warm-start effect.
4. OPTIMIZER: AdamW lr 1e-4 with cosine schedule to 1e-5, weight decay 1e-4, gradient clip 1.0.
5. INNER HOLDOUT: last 120 dates of the train window (was 90), purged by h+3.
6. SEEDS: {42,43,44} at the bagging-ensemble level (so 3 × 5 = 15 inits per fold); ensemble = mean of the three seed signals.

Cold-start control folds (frozen): {9, 18, 27}, seed 42, all 5 bags.

## Vol-matched null (verbatim)

> VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with Modal .map (concurrency 40), seed-42 5-init bagged v1 craft (cold per fold); house bias tolerance; skill = ≥3/4 above p95 OR Stouffer ≥ 3.

House bias tolerance (Phase 3.c): CONTAMINATED iff ≥2 fold-level 2·SE violations. Judged skill metric = tail-IC(top-half). Overlap reported on the same design. Retrain (not a cheap permute of frozen scores).

## Judgment grid (identical to Phase 7, house battery)

Vs frozen 2.c spread baseline: tail-IC top/bottom (NW-t), top-decile overlap, monster capture, whole-list RankIC — full OOS / trailing-18m / per-cycle; per-seed and ensemble; seed dispersion.

Hygiene table v2: per fold and seed — selected epoch window, SWA span, 5-init spread, warm-start vs cold-start control deltas, elapsed.

Crude-14d book (Ladder construction, DV-100, information only). Interpretability dump as in Phase 7 (top rules, membership movement).

## Pre-registered criteria (verbatim, before results)

> NFN-v1 is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread; (b) seed dispersion of tail-IC(top) ≤ 0.010; (c) vol-matched null passes. CRAFT-CONFIRMED (separate, informational label) if dispersion improves to ≤ 0.010 and the mean selected-epoch window is ≥ 8 — i.e., the training pathology is fixed regardless of the performance verdict. If CRAFT-CONFIRMED but not LIVE, the ledger records: 'the neuro-fuzzy family was trained correctly and still does not exceed the frozen spread on daily price/volume features — training craft is excluded as an explanation; the daily ceiling is architectural-independent. Hourly features remain the designated retry substrate for this identical pipeline.' If neither, state which clause failed. Nothing adopted. Mechanical, no post-hoc adjustment.

## Config dump (frozen)

- Architecture: Phase 7 NFN v0, 5488 params, CPU, `CUDA_VISIBLE_DEVICES=""`.
- Seeds {42,43,44} × 5 bags. Cache ver `p7cv1`.
- Walk-forward `make_expanding_folds` h=14, last-fold-wins.
- Null folds {5,15,21,24} × 15, Modal `.map` concurrency 40.

Nothing adopted. Mechanical, no post-hoc adjustment.
