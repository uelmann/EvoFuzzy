# FuzzyX-v1b addendum — pay-to-play loss (no occupancy floors)

**Written and frozen before any v1b walk-forward number is observed.** Separate test. Does not replace COMBO / A0 / LightGBM. Does not retune v1 knobs in place: v1 stays CONTAMINATED as recorded in `reports/fuzzyx_v1_report.md`.

v1 failure mode this shot addresses: occupancy floors (`L≥0.20`, `S≥0.30`, traded `≥0.25`) plus the `core/1e5` nuke forced the policy to stay in the market (~99.6% ±1). Default is now **flat**. Being long or short must pay for itself.

How default-flat is enforced (all of these, together):

1. Drop the occupancy nuke and the occupancy-up hinge. The objective is no longer illegal unless the book is 20/30/25% long/short/traded.
2. Add pay-to-play: `loss += λ_active · mean σ((|pos|−0.05)/0.05)` with `λ_active=0.20`. Every name that is not flat costs the objective; occupancy is no longer a floor to hit.
3. Initialize the FLAT logit at `+1.5` so the untrained policy is mostly `0`.
4. Soft position is `P_L² − P_S²`, so uncertain names collapse to 0 instead of a coin-flip long vs short.
5. Book weights are `w = p / max(Σ|p|, 1)`. Tiny leftover positions stay tiny; they are not levered up to a 100% book.

---

## What changed vs v1 (frozen)

| item | v1 | v1b |
|---|---|---|
| occupancy nuke `core/1e5` | on | **off** |
| occupancy hinge pushing traded/L/S **up** | `λ_occ=8` | **off** |
| pay-to-play | none | `λ_active=0.20` on mean `σ((\|pos\|−0.05)/0.05)` |
| FLAT init | 0 | output bias FLAT `+1.5` |
| soft position | `P(L)−P(S)` | `(P(L)−P(S)) · (P(L)+P(S))` so uncertain ≈ 0 |
| book scale | `w = p / Σ\|p\|` (tiny pos → 100% book) | **`w = p / max(Σ\|p\|, 1)` — no lever-up of dust** |
| shuffle-bias statistic | path-loss core | **mean weekly net PnL** (dollar-neutral null) |
| shuffle model | last fold on both | **that fold’s own OOS weights** |

Everything else identical to v1: A0 33 features, DeepSets, weekly, PIT top-30 volume, seed 42, Adam `1e-3`, 80 epochs, patience 12, costs 5+3 bps, `λ_turn=0.05`, `λ_bias=0.05`.

---

## Pre-registered keep rule (verbatim)

> FuzzyX-v1b is VIABLE only if: (i) leakage gates pass (feature_lookahead, universe_lookahead_top30, seed_determinism); (ii) BIAS: on the first and last OOS folds, using that fold’s trained weights, 10 within-date shuffles of the 7-day forward simple return (seeds 101–110), the null mean of **weekly net PnL** satisfies \|mean\| ≤ 2·(SD/√R). Either fold fail → CONTAMINATED; stop. (iii) hard weekly book, costs on, lag-0, full-OOS net Sharpe ≥ 0. (iv) vs A0 Sleeve A identical-days: SKIP if A0 preds missing; else ΔSharpe ≥ −0.10. Otherwise PARK. No retune. LOCAL-RESTRICTED cannot be official VIABLE vs A0.

Reported but non-binding: traded_frac (expect well below v1’s 0.996), MaxDD, turnover, long/short split.

---

## Non-goals

No occupancy-floor sweep, no λ_active sweep, no xsec, no Kronos, no replacing A0.
