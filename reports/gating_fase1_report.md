# FASE 1 — A0 baseline harness

**BACKTEST ONLY.** No Stage A–D. COMBO / A0 not replaced.

**Verdict: RED.** 10/11 anti-leak tests PASS. `test_shifted_target_degrades` FAIL (pre-registered, not loosened). `results/baseline.json` is still the ladder numerator.

Source data: Binance Vision UM 1d reconstructed at `/data/quant` (this pod had no pre-mounted volume). 831 listed → 695 with history; PIT train top-120 / exec 40 and 20. Span 2020-01-01 → 2026-07-31, 2404 bars, 18 expanding folds, h=7, lag 0, seed 42.

## RankIC (harness is the numerator)

Round F cited 0.0923 on n_days=875. This run has **n_days=1620**, so §D uses the harness number only (`tolerance.mode=harness_numerator_only`).

| universe | mean IC | n_days | NW-t (lag=7) | ICIR | trail-18m IC | trail NW-t |
|---|---|---|---|---|---|---|
| top-20 | **0.0976** | 1620 | 7.44 | 5.45 | 0.0822 | 3.60 |
| top-40 | **0.0803** | 1620 | 7.80 | 5.90 | 0.0925 | 4.73 |

Zero negative RankIC folds on both universes (18/18).

## Portfolio (costs in the criterion, Huber unchanged)

Top-40 h=7 tranche, τ causal `fold_train`, **median τ of grid = 70**, funding on, tiered 5+3 / 10+8, ADV cap 0.5%, book $1e6.

| book | net Sharpe full | trail-18m | MaxDD | DD days | ann TO |
|---|---|---|---|---|---|
| lag0 1x (headline) | **1.822** | **1.097** | −0.268 | 322 | 24.5 |
| lag1 stress | 1.835 | 1.212 | −0.222 | 259 | 24.5 |
| lag0 2x bps | 1.721 | 1.012 | −0.268 | 332 | 24.5 |
| lag0 3x bps | 1.620 | 0.927 | −0.269 | 372 | 24.5 |
| top-20 informational τ=70 | 1.799 | 0.572 | — | — | — |

Sharpe by τ 1x lag0: 60→1.04, **70→1.82**, 80→0.90, 90→0.73.

## Anti-leak

| test | result |
|---|---|
| `test_no_lookahead_filters` | PASS (0 hits in baseline/gating_ladder/pipeline) |
| `test_scaler_fold_isolation` | PASS N/A (A0 CS-z per bar) |
| `feature_lookahead` | PASS |
| `universe_lookahead` top-20/40/120 | PASS |
| `test_gate_identity_leakage` | PASS N/A (no gate) |
| `label_shuffle` | PASS (mean IC 0.0005, \|·\|<0.005, 25 shuffles) |
| `seed_determinism` | PASS (max score diff 0) |
| `test_shifted_target_degrades` (+10) | **FAIL** |
| `test_axis_slicing` | PASS shape (1620, 239) on top-40 |

### Why shifted-target failed (not a threshold tweak)

On top-40, score_t vs y_{t→t+7} has mean RankIC 0.0803. Same scores vs y shifted +10 bars (y_{t+10→t+17}, **zero overlap** with a 7-day forward) still have mean IC **0.0585** (73% of unshifted) and NW-t **6.74**. Pre-reg FAIL if shifted IC > 50% of unshifted **or** shifted NW-t ≥ 2. Both fired.

That is static/slow cross-sectional structure (name persistence: vol, ADV, beta), not 7-day timing. The test did its job. Threshold stays frozen. FASE 1 is not green; Stage A does not start on this result.

## Files

- `results/baseline.json`
- `results/fase1_suite.json`
- `results/LADDER.md`
