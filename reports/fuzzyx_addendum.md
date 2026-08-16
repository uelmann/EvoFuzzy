# FuzzyX-v1 addendum — keep rule frozen before results

**Written and frozen before any walk-forward number is observed.** Backtest/analysis only. No schedules or live components. Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`. COMBO / A0 are untouched.

This addendum is the pre-registration for the first FuzzyX shot. Design: `reports/fuzzyx_architecture.md`. Config: `config_fuzzyx.yaml`. Prototype: `fuzzyx/`.

---

## Scope (one shot)

| knob | frozen value |
|---|---|
| inputs | A0 `FEATURE_COLS` only (33 CS-z, clip ±5). No Kronos, no micro, no signatures |
| universe | PIT top-30 by trailing-30d median dollar volume (`baseline.data.build_pit_topn`). Stables/wraps excluded. BTC always kept in the feature panel |
| rank sensitivity | mcap ranking is **not** run in this shot |
| rebalance | every 7 sessions, positions ffilled. Daily is **not** run |
| encoder | `deepsets` only. `xsec` attention is **not** run |
| rules | 24, 3 Gaussians, IGNORE/AND/NAND |
| d_model | 32 |
| seed | 42 |
| horizon | 7 (matches weekly hold) |
| CV | expanding, `min_train_days=730`, `val_days=90`, `step_days=90`, purge 7, embargo 7+3, inner holdout 90d |
| optimiser | Adam, `lr=1e-3`, `weight_decay=1e-4`, `max_epochs=80`, patience 12 on inner-holdout path loss |
| costs | 5 bps fee + 3 bps slip, lag-0 close, gross 1.0, no leverage |
| loss | `−corr(cum,t)·(1−maxDD)·(1−DDdur)` + occupancy nuke + `λ_turn=0.05` + `λ_bias=0.05` |
| output | train soft `P(long)−P(short)`; eval argmax `{+1,0,−1}` |

If the A0 Vision panel is absent locally, a restricted Binance Vision download may be used to exercise the pipeline. That run is labelled **LOCAL-RESTRICTED** and cannot produce a VIABLE verdict against A0. The official shot is the full PIT top-30 panel (Modal volume `/data/quant` or equivalent).

CMC historical snapshots are **not** required for this shot. Volume rank on the A0 Vision panel is the same PIT machinery as Sleeve A/B. A CMC-mcap sensitivity is a later addendum.

---

## Pre-registered keep rule (verbatim)

> FuzzyX-v1 (DeepSets, weekly, PIT top-30 volume, seed 42) is VIABLE only if all of the following hold. (i) LEAKAGE: `feature_lookahead`, `universe_lookahead_top30`, and `seed_determinism` pass. (ii) BIAS: on folds {first OOS fold, last OOS fold}, 10 within-date shuffles of the 7-day forward simple return (shuffle seeds 101–110), the null mean of the path-loss **core** satisfies \|mean\| ≤ 2·(SD/√R). If either fold violates, verdict = CONTAMINATED; stop. (iii) SKILL/BOOK: the hard weekly book, costs on, lag-0, has full-OOS net Sharpe ≥ 0. (iv) vs A0: if Sleeve A (h=7, PIT top-20, median-τ) identical-days net Sharpe is available, the FuzzyX book restricted to those days and to names in that top-20 must not lose more than 0.10 Sharpe vs Sleeve A. If A0 predictions are missing, (iv) is SKIP and does not block (iii). Otherwise PARK. No retune. The xsec encoder is not part of this shot.

Mechanical. No post-hoc knob change. Occupancy, MaxDD, turnover, and per-year Sharpe are reported but do not override (i)–(iv).

---

## Gates (detail)

- **feature_lookahead:** A0 `baseline.gates.gate_feature_lookahead` on the panel used to build features. Pass = max abs diff < 1e-10.
- **universe_lookahead_top30:** `baseline.gates.gate_universe_lookahead` with n=30. Membership at t invariant to future rows.
- **seed_determinism:** two independent trains, seed 42, max abs soft-position diff on a fixed fold < 1e-6 (eval mode, no dropout).
- **label-shuffle bias:** retrain is **not** required. Apply the frozen model to the fold’s features; shuffle the **forward returns** within date (the quantity that enters path loss), recompute core. This tests whether a non-zero core is an artifact of occupancy/bias regularizers rather than ranking skill. R=10.

---

## Non-goals (this shot)

- xsec transformer, Kronos, daily rebalance, mcap universe, vs-COMBO replacement, live trading, hyperparameter search.

---

## Decision table

| (i) | (ii) | (iii) | (iv) | verdict |
|---|---|---|---|---|
| fail | — | — | — | PARK (leakage) |
| pass | fail | — | — | CONTAMINATED |
| pass | pass | fail | — | PARK |
| pass | pass | pass | fail (A0 present) | PARK |
| pass | pass | pass | SKIP or pass | VIABLE candidate (does not replace COMBO) |
