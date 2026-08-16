# BTC-BEATER — Phase 7 NEURO-FUZZY NET v0 freeze addendum

**Status:** FROZEN before results. One architecture config. Zero search.
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no product changes. Master only. CPU only. Zero GPU. Modal `.map` fan-out for the vol-matched null (concurrency 40).
Frozen COMBO `v2.0-combo-final`, SPREAD-LS BOOK-HYBRID, LONG-TIDE (official long), and BTC-BEATER v1 (record-only) are **untouched**.
Cleaned panel, floored PIT top-100 DV (primary), 2.c fold schedule (purge/embargo h+3), h=14, canonical hybrid pricing. Frozen 33 Stage-S features. Caches sha256-verified. 2.c spread cache reused as the frozen-spread baseline, not retrained.

This phase produces a **record**, not a product. Nothing is adopted. Any production change requires a fresh pre-registered phase.

## Firewall (verbatim, first)

> The PI's hand-made formula (gauss-momentum) is quarantined: its rules, ingredients, and structure are NOT provided to the miner, NOT seeded in the population, NOT added as features. The primitive set is exactly the 33 frozen house features — no additions. The only information retained from the falsification run is the reference book's performance numbers, used as the success bar. If the miner independently rediscovers a similar rule, that is a finding, not a leak.

Phase-7 restatement (verbatim):

> The PI's hand formula stays quarantined: never seeded, never referenced. Warm-start rules, if any, come ONLY from the Phase-6 RULE-FORGE output bank (assert provenance).

Operationalisation (frozen with this registration, not a result):

- Miner / net source is grepped for hand-formula tokens (`gauss-momentum`, `scores_at`, `std_63d`, `ret_28d`, `MANUEL2_FORMULA`). Assert none.
- `btcb.manuel2` is not imported anywhere under `nfn/`.
- Warm-start is **conditional**: if a Phase-6 RULE-FORGE bank exists **and** its mechanical verdict is VIABLE, copy at most 8 of 24 rules from that bank (provenance path + sha logged). Else random init throughout. The path that ran is stated in the report.
- The 33 frozen `STAGE_S_COLS` are the only name-level inputs. Regime vector `m_t` is date-level FiLM context, not an added feature column.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Architecture (frozen config; one shot)

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

Heads: twin logits (top-quintile, bottom-quintile within date) on `Σ w̃_k r_k` representations; signal = `p_top − p_bottom` after per-fold isotonic.

Loss: class-weighted BCE (weight 3 on true-top errors, TOP head) + 0.3 × listwise softmax cross-entropy on the date's top-10 (tail emphasis) + L1(`e`).

~4k–5.5k parameters. Three orders below the failed sequence models. **One config. Zero architecture search.**

## Training (hygiene guards ON)

- AdamW 3e−4, batch = full date groups, max 30 epochs/fold.
- Early stop on inner-holdout per-date tail-IC(top) with **ES FLOOR: no stop before epoch 8**.
- Log per fold: `best_epoch`, train and holdout curves every epoch.
- Flag **UNDERTRAINED** if `best_epoch < 10`; count in report.
- Walk-forward on the 2.c schedule; last-fold-wins concatenation.
- Seeds {42, 43, 44}, full runs each; ensemble = mean calibrated spread.
- Warm start as above.

## Vol-matched null (verbatim)

> VOL-MATCHED NULL: folds {5,15,21,24} × 15 within-vol-quintile shuffles, fan-out with Modal .map (concurrency 40), seed 42 config; house bias tolerance; skill = ≥3/4 above p95 OR Stouffer ≥ 3.

House bias tolerance (Phase 3.c): CONTAMINATED iff ≥2 fold-level 2·SE violations. Judged skill metric = tail-IC(top-half). Overlap reported on the same design. Retrain (not a cheap permute of frozen scores).

## Judgment grid (tail primary)

Vs frozen 2.c spread baseline: tail-IC top/bottom (NW-t), top-decile overlap, monster capture, whole-list RankIC (secondary) — full OOS / trailing-18m / per-cycle; per-seed + ensemble; seed dispersion.

Crude-14d book (Ladder construction, DV-100, information only).

Interpretability dump: learned membership curves (`c`, `s` per feature, top-20 by movement from init), printed top rules with exponents, FiLM `γ`/`β` trajectories vs regime.

## Pre-registered criteria (verbatim, before results)

> NFN is LIVE if ALL of: (a) ensemble Δtail-IC(top-half) ≥ +0.010 AND Δoverlap ≥ +0.015 vs the frozen spread on full OOS; (b) seed dispersion of tail-IC(top) ≤ 0.010; (c) vol-matched null passes. PARKED otherwise with the failed clause stated. If PARKED on (a) with (b)(c) clean, the ledger gains: 'learnable fuzzy composition on daily 33-features does not exceed the frozen spread — the daily ceiling holds for this family too; hourly features (Phase 5 panel) are the designated retry substrate.' Nothing adopted; production use requires a fresh phase. Mechanical, no post-hoc adjustment.

Verdicts mechanical. Nothing adopted.
