# BTC-BEATER — ORACLE LADDER 2 freeze addendum

**Status:** FROZEN before results. Decompose the Ladder-1 BELOW-CURVE gap into tail-blindness vs translation slack.
**Scope:** ANALYSIS ONLY. No retraining, no product changes, no schedules, no live components. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final, SPREAD-LS, LONG-TIDE, and BTC-BEATER v1 are **untouched**. The 2.c spread cache is reused byte-identical (sha256 verified). CMC raw data is read-only. Pricing follows the 3.e canonical convention (Binance).

This phase produces a **split of the bill**, not a product. Nothing is adopted.

Known methodology bias (recorded a priori): white-noise-degraded oracles are TAIL-AWARE (errors uniform across the ranking); our model's discrimination is bottom-heavy (symmetry audit; B10<B9). Part of the BELOW-CURVE gap is therefore information missing in the right tail, not recoverable by translation.

## Pre-registered reading (verbatim, before results)

> The gap decomposition is: TAIL-INFORMATION share = the part explained by overlap/tail-IC deficits vs the equal-IC ladder; CONSTRUCTION share = the best of V1–V3 minus the crude base, plus the production-construction delta measured on the ladder signal. No variant is adopted here; any adoption requires a fresh pre-registered phase with the house criteria. If the best translation variant improves CAGR by ≥ +10pp over the crude base at comparable MaxDD, translation work is declared the next priority; otherwise the right-tail information hunt (catalysts/attention data) is declared the next priority. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical formulae (frozen)

Window, universe, and crude 14d book = Ladder 1 (floored PIT top-100, Binance-listed at t, BTC excluded, EW top decile, 10% name cap, residual cash idle, 10 bps/side, h=14 full rebalance).

Equal-IC ladder = noisy oracle with per-date RankIC target **0.116** (Ladder-1 model RankIC), 5 seeds (101–105). Reference ladder-0.16 uses the same seeds.

**TAIL-INFORMATION pp** = `CAGR(ladder-0.116, crude) − CAGR*` where `CAGR*` is linear interpolation on the `(mean top-decile overlap → crude CAGR)` curve of `{ladder-0.116, ladder-0.16, oracle}` evaluated at the model's mean top-decile overlap. Below/above the overlap range, linearly extrapolate from the nearest two points.

**CONSTRUCTION pp** = `(best of V1–V3 CAGR − crude-model CAGR) + (ladder-0.116 production CAGR − ladder-0.116 crude CAGR)`.
Production construction = LONG-TIDE mechanics **without** the Stage-T gate: h=14 overlapping tranches, decile-enter / quintile-stay hysteresis (`k_enter=10`, `k_stay=20`), `n_hold=10`, 10% name cap, anti-blowoff `ret_7>50%`, BTC parking of residual, 10 bps/side alts and 2 bps BTC. Not LONG-TIDE-the-product.

**UNEXPLAINED pp** = `(CAGR(ladder-0.116, crude) − crude-model CAGR) − TAIL − CONSTRUCTION`.

**Comparable MaxDD:** variant MaxDD ≥ crude-model MaxDD − 10pp.
**PRIORITY = TRANSLATION** if the best V1–V3 with comparable MaxDD has CAGR − crude ≥ +10pp; else **TAIL-INFORMATION**.

The brief's "33.6% CAGR reference" is **not** a gate. The mechanical crude base is the recomputed naked 14d EW top-decile book on this window (Ladder 1 printed 13.9%).

## Variants on OUR signal (crude 14d, costs on, idle cash)

- V1 score-weighted: top decile; weights ∝ rank-percentile of spread within the decile; cap 15%.
- V2 concentrated: top-5 by spread, EW, 20% cap.
- V3 tail-threshold: enter only names whose spread exceeds the 95th cross-sectional percentile; variable count; cap 10%.

No sweeps. No adoption.

## What this freeze does not do

- Does not recompute signals, retrain, or change any frozen product.
- Does not adopt V1/V2/V3 or a production-construction change.
- Does not use GPU.
