# FuzzyX-v1f addendum — dollar-neutral book, `−mean(net PnL)`

**Written and frozen before any v1f walk-forward number is observed.** Separate test. Does not replace COMBO / A0 / LightGBM. Does not retune v1–v1e in place.

v1c–v1e path losses (Pearson of returns, Pearson of wealth, Pearson × last cumret) were CONTAMINATED for the same reason: leftover net exposure. Within-date shuffle keeps the CS mean, so `E[w·π(r)] = mean(r)·Σw`. Changing the path scalar does not zero `Σw`.

This shot does one construction change and one objective change:

```text
p ← p − masked_mean(p)     # Σp = 0 on the investable set
w ← p / Σ|p|               # unit gross if any CS dispersion; else 0
loss = −mean(net weekly PnL)   # costs 5+3 bps already in PnL
```

If every name has the same position, demean → 0 and the book is flat. No Pearson, no occupancy nuke, no pay-to-play. Same FuzzyX stack as v1e (A0 33, DeepSets, weekly, PIT top-30, seed 42).

---

## What changed vs v1e (frozen)

| item | v1e | v1f |
|---|---|---|
| weights | `w = p / Σ\|p\|` (net exposure allowed) | **`p ← p − mean(p)`, then unit-gross** |
| train core | `corr(wealth, t)·(1+cumRet[-1])` | **`mean(net weekly PnL)`** |

Everything else identical to v1e: no DD in the scalar, `λ_*=0`, FLAT init 0, `P(L)−P(S)`, costs 5+3 bps, Adam `1e-3`, 80 epochs, patience 12.

---

## Pre-registered keep rule (verbatim)

> FuzzyX-v1f is VIABLE only if: (i) leakage gates pass (feature_lookahead, universe_lookahead_top30, seed_determinism); (ii) BIAS: on the first and last OOS folds, using that fold’s trained weights, 10 within-date shuffles of the 7-day forward simple return (seeds 101–110), the null mean of **weekly net PnL** satisfies \|mean\| ≤ 2·(SD/√R). Either fold fail → CONTAMINATED; stop. (iii) hard weekly book, costs on, lag-0, full-OOS net Sharpe ≥ 0. (iv) vs A0 Sleeve A identical-days: SKIP if A0 preds missing; else ΔSharpe ≥ −0.10. Otherwise PARK. No retune. LOCAL-RESTRICTED cannot be official VIABLE vs A0.

Reported but non-binding: mean `|Σw|` (expect ~0), traded_frac, MaxDD, long/short split.

---

## Non-goals

No occupancy floors, no corr/path scalar, no λ sweep, no replacing A0. If this shot fails shuffle or Sharpe, FuzzyX path-policy is PARK on this stack — no fifth loss tweak.
