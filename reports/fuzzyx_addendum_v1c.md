# FuzzyX-v1c addendum — `corr(st_r, t)` as the only train loss

**Written and frozen before any v1c walk-forward number is observed.** Separate test. Does not replace COMBO / A0 / LightGBM. Does not retune v1 or v1b in place: those reports stay as recorded.

v1 occupancy floors forced always-in-market. v1b pay-to-play + FLAT prior collapsed the hard book to `traded_frac = 0`. This shot isolates the notebook line:

```python
np.corrcoef(st_r, np.arange(0, st_r.shape[0]))[1, 0]
```

`st_r` is the **net weekly portfolio return** (costs 5+3 bps already subtracted). Train loss is `−corr`. No `(1−maxDD)`, no DD-duration, no occupancy nuke, no pay-to-play, no turnover/bias hinges.

Pearson is affine-invariant in the time index, so `arange(T)` equals `linspace(0,1,T)` as used in v1's equity trend. The change is the **left** argument: period returns, not `cumprod` equity.

v1b knobs that force FLAT are off for this shot (otherwise the book is identically zero and the corr is undefined/0). Unit-gross `w = p / Σ|p|` is restored so a non-zero preference is a real path. Soft pos is `P(L)−P(S)` again. FLAT remains a legal eval action; it is not subsidized.

---

## What changed vs v1b (frozen)

| item | v1b | v1c |
|---|---|---|
| train core | `corr(equity, t)·(1−maxDD)·(1−ddur)` | **`corr(st_r, arange(T))`** |
| occupancy nuke / hinge | off | off |
| pay-to-play `λ_active` | 0.20 | **0** |
| `λ_turn`, `λ_bias` | 0.05 | **0** |
| FLAT init | +1.5 | **0** |
| soft position | `P_L² − P_S²` | **`P(L)−P(S)`** |
| book scale | `p / max(Σ\|p\|, 1)` | **`p / Σ\|p\|`** (unit gross if any pos ≠ 0) |

Everything else identical: A0 33 features, DeepSets, weekly, PIT top-30 volume, seed 42, Adam `1e-3`, 80 epochs, patience 12, costs 5+3 bps.

---

## Pre-registered keep rule (verbatim)

> FuzzyX-v1c is VIABLE only if: (i) leakage gates pass (feature_lookahead, universe_lookahead_top30, seed_determinism); (ii) BIAS: on the first and last OOS folds, using that fold’s trained weights, 10 within-date shuffles of the 7-day forward simple return (seeds 101–110), the null mean of **weekly net PnL** satisfies \|mean\| ≤ 2·(SD/√R). Either fold fail → CONTAMINATED; stop. (iii) hard weekly book, costs on, lag-0, full-OOS net Sharpe ≥ 0. (iv) vs A0 Sleeve A identical-days: SKIP if A0 preds missing; else ΔSharpe ≥ −0.10. Otherwise PARK. No retune. LOCAL-RESTRICTED cannot be official VIABLE vs A0.

Reported but non-binding: `corr(st_r, t)` OOS, traded_frac, MaxDD, long/short split.

---

## Non-goals

No occupancy-floor return, no λ sweep, no switching the left argument to `st_cum` in this shot, no replacing A0.
