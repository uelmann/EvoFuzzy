# BTC-BEATER — Phase 7.d NFN VARIANT A freeze addendum

**Status:** FROZEN before results. One architectural change vs Phase 7 NFN v0: magnitude labels + ranking loss + single head. Independent of Phase 7.c — do not wait for it.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. CPU only. Zero GPU. One shot on Modal.
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned panel, floored PIT top-100 DV (primary), 2.c fold schedule (purge/embargo h+3), h=14, canonical hybrid pricing. Frozen 33 Stage-S features. Caches sha256-verified. 2.c spread cache reused as the frozen-spread baseline, not retrained. Phase 7 NFN report, if present, is read-only.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

## Firewall (verbatim, first)

> The PI's hand-made formula (gauss-momentum) is quarantined: its rules, ingredients, and structure are NOT provided to the miner, NOT seeded in the population, NOT added as features. The primitive set is exactly the 33 frozen house features — no additions. The only information retained from the falsification run is the reference book's performance numbers, used as the success bar. If the miner independently rediscovers a similar rule, that is a finding, not a leak.

Phase-7 restatement (verbatim):

> The PI's hand formula stays quarantined: never seeded, never referenced. Warm-start rules, if any, come ONLY from the Phase-6 RULE-FORGE output bank (assert provenance).

Operationalisation (frozen with this registration, not a result):

- Miner / net source is grepped for hand-formula tokens (`gauss-momentum`, `scores_at`, `std_63d`, `ret_28d`, `MANUEL2_FORMULA`). Assert none.
- `btcb.manuel2` is not imported anywhere under `nfn_va/`.
- Fold 0 is cold: if a Phase-6 RULE-FORGE bank exists **and** its mechanical verdict is VIABLE, copy at most 8 of 24 rules onto init-0 of fold 0 (provenance path + sha logged). Else random init. Subsequent folds warm-start from the previous fold's SWA weights (7.c craft). The path that ran is stated in the report.
- The 33 frozen `STAGE_S_COLS` are the only name-level inputs. Regime vector `m_t` is date-level FiLM context, not an added feature column.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## The single change (frozen)

Everything else BYTE-IDENTICAL to Phase 7: membership layer, log-space rule layer with L1 exponents, FiLM regime gate, 5,488-param scale (minus the second head). Assert config equality on that frozen architecture block at runtime.

- LABEL: y(i,t) = 14-day log excess return vs BTC, then per date: cross-sectional rank-normalize to [0,1] AND winsorize the raw magnitude at the date's 1st/99th percentile. Both forms are used (see loss).
- LOSS (replaces the twin BCE + listwise term):
  L = 1.0 · L_rank + 0.5 · L_mag + L1(e)
  - L_rank = ListNet-style listwise cross-entropy per date between softmax(model scores / τ) and softmax(winsorized magnitudes / τ), τ = 1.0 — magnitude-weighted ranking: bigger winners pull harder.
  - L_mag = Huber loss (δ = 1.0) on the standardized winsorized excess return — keeps the head calibrated to size, not just order.
- HEAD: single scalar output per (coin, date) — no twin heads, no isotonic (nothing to calibrate; the score is the signal). Report the score's cross-sectional distribution stability per year.

## Architecture (Phase 7 frozen; unchanged except head)

Membership layer: per feature `j`, three learnable sigmoidal memberships

`μ_{j,k}(x) = σ((z_cs(x) − c_{j,k}) / s_{j,k})`

- `c` init at cross-sectional z ∈ {−0.67, 0, +0.67}
- `s` init 1.0 (learnable, clamped `s ≥ 0.2`)
- Complements via paired primitive `(1−μ)`, not negative exponents.
- Primitive count = 33 × 3 × 2 = 198.

Rule layer (log-space): 24 rule nodes

`r_k = exp(Σ_p e_{k,p} · log(primitive_p + 1e−6))`

- exponents `e ≥ 0` (softplus-parameterized)
- init: 3 random primitives per rule at `e=1`, rest 0
- L1 penalty on `e` (λ = 1e−3) — differentiable feature selection

Regime FiLM gate: market vector

`m_t = [EW top-50/BTC 20d return, cross-sectional dispersion of 14d returns, breadth>0]`

small MLP(8) → `γ_t`, `β_t` modulating rule weights: `w̃_k = γ_t · w_k + β_t`.

Head: one linear scalar on `h = r ⊙ w̃`. Signal = raw score. No isotonic.

Expected n_params = 5488 − 25 = 5463 (drop `head_bot`). Assert at build.

## Training craft (same corrections as Phase 7.c; frozen)

3-epoch trailing-mean selection + SWA over the best 3 epochs; ES floor 10, patience 8, cap 40; 5-init intra-fold bagging (mean score); warm-start from the previous fold (fold 0 cold); AdamW lr 1e-4 cosine → 1e-5, wd 1e-4, clip 1.0; inner holdout = last 120 train dates, purged; seeds {42,43,44}, ensemble = mean of seed signals. Hygiene table mandatory (selected-epoch window, 5-init spread, UNDERTRAINED count).

UNDERTRAINED if the selected-epoch window centre < 10.

## Vol-matched null (verbatim)

> VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with Modal .map (concurrency 40), seed 42 config; house bias tolerance; skill = ≥3/4 above p95 OR Stouffer ≥ 3.

House bias tolerance (Phase 3.c): CONTAMINATED iff ≥2 fold-level 2·SE violations. Judged skill metric = tail-IC(top-half). Overlap reported on the same design. Retrain (not a cheap permute of frozen scores).

## Judgment grid (tail primary)

Vs frozen 2.c spread AND vs the Phase-7 NFN baseline (read-only, if its report exists): tail-IC top/bottom (NW-t), top-decile overlap, monster top-3 capture, whole-list RankIC — full OOS / trailing-18m / per-cycle; per-seed + ensemble; seed dispersion.

Magnitude-specific diagnostics: mean realized excess return of the top-10 picks per date (vs the frozen spread's), and decile-mean-return curve.

Crude-14d book (Ladder construction, DV-100, information only). Interpretability dump: top rules and membership movement, as in Phase 7.

## Pre-registered criteria (verbatim, before results)

> VARIANT-A is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread; (b) seed dispersion of tail-IC(top) ≤ 0.010; (c) vol-matched null passes. MAGNITUDE-GAIN (separate, informational label) if the mean realized excess return of its top-10 picks exceeds the frozen spread's by ≥ 20% relative, regardless of the LIVE verdict — this isolates whether magnitude labels buy bigger winners even when rank metrics do not move. If PARKED, state which clause failed. Nothing adopted; production use requires a fresh phase. Mechanical, no post-hoc adjustment.

## PHASE7D_CRITERION

VARIANT-A is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread; (b) seed dispersion of tail-IC(top) ≤ 0.010; (c) vol-matched null passes. MAGNITUDE-GAIN (separate, informational label) if the mean realized excess return of its top-10 picks exceeds the frozen spread's by ≥ 20% relative, regardless of the LIVE verdict — this isolates whether magnitude labels buy bigger winners even when rank metrics do not move. If PARKED, state which clause failed. Nothing adopted; production use requires a fresh phase. Mechanical, no post-hoc adjustment.

## PHASE7D_FIREWALL

The PI's hand formula stays quarantined: never seeded, never referenced. Warm-start rules, if any, come ONLY from the Phase-6 RULE-FORGE output bank (assert provenance).

## PHASE7_NULL_REGISTRATION

VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with Modal .map (concurrency 40), seed 42 config; house bias tolerance; skill = ≥3/4 above p95 OR Stouffer ≥ 3.

## DEATH_CONVENTION

A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.
