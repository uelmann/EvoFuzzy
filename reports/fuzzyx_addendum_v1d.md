# FuzzyX-v1d addendum — `corr(cumprod(1+st_r), t)` (wealth path)

**Written and frozen before any v1d walk-forward number is observed.** Separate test. Does not replace COMBO / A0 / LightGBM. Does not retune v1 / v1b / v1c in place.

v1c used `corr(st_r, arange(T))` on **period returns**. Pearson is scale-invariant and invariant to adding a constant, so that scalar is the signed R of `st_r ~ t`: it scores whether later weeks beat earlier weeks, not whether the book made money. `corr(1 + st_r, t)` is **identical** to `corr(st_r, t)` and is not a new shot.

The series that actually carries return into the path is compounded wealth:

```python
st_w = np.cumprod(1.0 + st_r)
np.corrcoef(st_w, np.arange(st_w.shape[0]))[1, 0]
```

Train loss is `−corr`. No DD multipliers, no occupancy nuke, no pay-to-play. Same book construction as v1c (unit-gross, `P(L)−P(S)`).

Pearson of a smooth uptrend still saturates near +1 regardless of +1%/week vs +10%/week. This shot only switches the left argument from `st_r` to `cumprod(1+st_r)`. A slope / `equity[-1]` shot is a later non-goal.

---

## What changed vs v1c (frozen)

| item | v1c | v1d |
|---|---|---|
| train core | `corr(st_r, arange)` | **`corr(cumprod(1+st_r), arange)`** |
| `corr(1+st_r, t)` | not used (≡ v1c) | not used (≡ v1c) |

Everything else identical to v1c: A0 33 features, DeepSets, weekly, PIT top-30, seed 42, Adam `1e-3`, 80 epochs, patience 12, costs 5+3 bps, `λ_*=0`, FLAT init 0, lever-up on.

---

## Pre-registered keep rule (verbatim)

> FuzzyX-v1d is VIABLE only if: (i) leakage gates pass (feature_lookahead, universe_lookahead_top30, seed_determinism); (ii) BIAS: on the first and last OOS folds, using that fold’s trained weights, 10 within-date shuffles of the 7-day forward simple return (seeds 101–110), the null mean of **weekly net PnL** satisfies \|mean\| ≤ 2·(SD/√R). Either fold fail → CONTAMINATED; stop. (iii) hard weekly book, costs on, lag-0, full-OOS net Sharpe ≥ 0. (iv) vs A0 Sleeve A identical-days: SKIP if A0 preds missing; else ΔSharpe ≥ −0.10. Otherwise PARK. No retune. LOCAL-RESTRICTED cannot be official VIABLE vs A0.

Reported but non-binding: OOS `corr(wealth, t)`, `corr(st_r, t)`, traded_frac, MaxDD, long/short split.

---

## Non-goals

No occupancy floors, no λ sweep, no replacing Pearson with OLS slope in this shot, no replacing A0.
