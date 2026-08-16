# FuzzyX-v1e addendum — `corr(wealth, t) · (1 + cumRet[-1])`

**Written and frozen before any v1e walk-forward number is observed.** Separate test. Does not replace COMBO / A0 / LightGBM. Does not retune v1–v1d in place.

v1c/v1d Pearson is scale-invariant (signed R of a path-on-time regression). The intended scalar was the **product** with ending wealth, not `corr(1+st_r, t)` (that equals v1c) and not Pearson of `cumprod` alone (v1d).

```python
st_w = np.cumprod(1.0 + st_r)
cor = np.corrcoef(st_w, np.arange(st_w.shape[0]))[1, 0]
cumret_last = st_w[-1] - 1.0          # last cumulative simple return
core = cor * (1.0 + cumret_last)      # = cor * st_w[-1]
loss = -core
```

A smooth +10%/week path can now outscore a smooth +1%/week path with the same corr≈1. No DD multipliers, no occupancy nuke, no pay-to-play. Same book as v1d (unit-gross, `P(L)−P(S)`).

---

## What changed vs v1d (frozen)

| item | v1d | v1e |
|---|---|---|
| train core | `corr(cumprod(1+st_r), t)` | **`corr(cumprod(1+st_r), t) · (1 + cumRet[-1])`** |

Everything else identical to v1d.

---

## Pre-registered keep rule (verbatim)

> FuzzyX-v1e is VIABLE only if: (i) leakage gates pass (feature_lookahead, universe_lookahead_top30, seed_determinism); (ii) BIAS: on the first and last OOS folds, using that fold’s trained weights, 10 within-date shuffles of the 7-day forward simple return (seeds 101–110), the null mean of **weekly net PnL** satisfies \|mean\| ≤ 2·(SD/√R). Either fold fail → CONTAMINATED; stop. (iii) hard weekly book, costs on, lag-0, full-OOS net Sharpe ≥ 0. (iv) vs A0 Sleeve A identical-days: SKIP if A0 preds missing; else ΔSharpe ≥ −0.10. Otherwise PARK. No retune. LOCAL-RESTRICTED cannot be official VIABLE vs A0.

Reported but non-binding: OOS corr, `cumRet[-1]`, traded_frac, MaxDD, long/short split.

---

## Non-goals

No occupancy floors, no λ sweep, no OLS slope instead of this product, no replacing A0.
