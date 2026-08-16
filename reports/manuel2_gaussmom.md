# BTC-BEATER MANUEL-2 — gauss-momentum falsification

**BACKTEST AND ANALYSIS ONLY.** Nothing adopted. No schedules, no live components, no product changes. CPU only, zero GPU. Frozen products untouched. Pricing = Binance-hybrid where listed (3.e canonical). Master only.

## Formula (verbatim, frozen before results)

> Score = gauss(ret_14d) × gauss(ret_28d) / gauss(std_63d), gauss(x) = Φ(z_cs(x)), z_cs = cross-sectional z-score per date across that day's universe. All inputs use data ≤ t.

## Spec completions (verbatim, frozen before results)

> STABLECOIN EXCLUSION: pegged assets (stablecoins, wrapped/pegged tokens) are excluded from the universe. Rationale on record: the formula divides by gauss(std63); std≈0 names get near-infinite scores — the literal formula would buy Tether. Exclusion list built from the panel's pegged-asset tags + |90d total return| < 2% heuristic, logged.

> BTC VARIANTS: run BOTH btc-in and btc-ex universes (BTC's low std makes the denominator favor it; the PI's thesis is a non-BTC-like curve, so both are shown).

> gauss(x) = Φ(z_cs(x)): normal CDF of the CROSS-SECTIONAL z-score, computed per date across the day's universe. All inputs use data ≤ t.

## Best-book rule (verbatim, frozen)

> Best book = highest full-window total return among the four pre-declared books (daily/weekly × btc-in/btc-ex). Verdicts apply to that book. All four are reported. This is the disclosed 4-way look. No other selection.

## MaxDD convention (verbatim, frozen)

> MaxDD is the house signed quantity (negative). STRONG clause 'MaxDD ≤ 1.10 × BTC's' is |book MaxDD| ≤ 1.10 × |BTC MaxDD|, i.e. book_maxdd >= 1.10 * btc_maxdd when both ≤ 0.

## Pre-registered verdicts (verbatim, before results)

> MANUEL-2 is CONFIRMED if the best pre-declared book (daily or weekly, btc-in or btc-ex — 4 books, disclosed as a 4-way look) on Binance-hybrid pricing has: total ≥ BTC B&H AND daily correlation with BTC ≤ 0.70. It is STRONG if additionally relative-line Sharpe ≥ 0.50 and MaxDD ≤ 1.10 × BTC's. It is PARTIAL if exactly one of the two claim clauses passes (state which). REFUTED if neither. Per-cycle honesty table mandatory; no single cycle overrides. If CONFIRMED, a fresh pre-registered phase evaluates adoption. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Identity

- Window 2019-10-20 → 2026-08-08 n_days=2485
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- GPU used = `False`
- Hybrid name-days Binance share = `0.699`

## Pegged-asset exclusion (logged)

- Tagged ids = `62` (STABLE_OR_WRAP ∪ extra stables ∪ USD-suffix ∪ name/slug needles)
- Heuristic `|90d total return| < 2%`: median flagged/date = `43.0` mean = `55.3`
- Sample tagged: AEUR, AUSD, BFUSD, BTCB, BUSD, CBBTC, CRVUSD, CUSD, DAI, DEUSD, DUSD, EURC, EURS, EURt, EZETH, FDUSD, FRAX, GHO, GUSD, GUSD, HBTC, HUSD, LUSD, METH

Rationale: `/gauss(std63)` would otherwise buy near-zero-vol pegs.

## 1 — Four books (4-way look)

Ladder construction: EW top 5% (`ceil` → 3 on a full 50), no name cap, always invested, 10 bps/side, death-in-position. Daily = headline; weekly Mondays = secondary.

| book | total | CAGR | USD Sharpe | trail-18m | MaxDD | rel Sharpe | rel total | corr vs BTC | avg n | ann TO | forced |
|------|-------|------|------------|-----------|-------|------------|-----------|-------------|-------|--------|--------|
| daily btc-in | 5939.3% | 82.6% | 1.106 | -0.672 | -75.5% | 0.755 | 639.5% | 0.414 | 3.00 | 90.8 | 0 |
| daily btc-ex | 5522.8% | 80.7% | 1.092 | -0.700 | -76.2% | 0.744 | 588.5% | 0.398 | 3.00 | 93.2 | 0 |
| weekly btc-in | 251.3% | 20.3% | 0.636 | -0.836 | -85.2% | 0.210 | -57.0% | 0.507 | 3.00 | 25.9 | 1 |
| weekly btc-ex | 796.1% | 38.0% | 0.799 | -0.632 | -79.0% | 0.398 | 9.7% | 0.493 | 3.00 | 26.1 | 1 |

Best book (highest total) = **daily_btc_in**.

Informational DV row (floored PIT top-100, daily, btc-ex, same formula):

| book | total | CAGR | USD Sharpe | trail-18m | MaxDD | rel Sharpe | rel total | corr vs BTC | avg n | ann TO | forced |
|------|-------|------|------------|-----------|-------|------------|-----------|-------------|-------|--------|--------|
| daily DV100 btc-ex | 28898.0% | 130.0% | 1.395 | -0.717 | -91.5% | 1.071 | 3451.0% | 0.528 | 5.00 | 91.1 | 0 |

## 2 — Benchmarks (identical window)

| book | total | CAGR | USD Sharpe | trail-18m | MaxDD | rel Sharpe | rel total | corr vs BTC | avg n | ann TO | forced |
|------|-------|------|------------|-----------|-------|------------|-----------|-------------|-------|--------|--------|
| BTC B&H | 689.3% | 35.5% | 0.809 | -0.393 | -76.6% | 0.000 | 0.0% | 1.000 | 1.00 | 0.0 | 0 |
| EW mcap-50 ex-pegged | 105.4% | 11.2% | 0.525 | -0.587 | -86.9% | -0.270 | -74.8% | 0.828 | 50.00 | 16.1 | 1 |
| frozen-spread crude (ref) | 127.4% | 12.9% | 0.515 | -0.664 | -80.7% | -0.239 | -71.2% | 0.790 | 9.00 | 13.0 | 0 |

## 3 — Claim metrics (best book)

- Best = `daily_btc_in`
- total = `5939.3%` vs BTC `689.3%` (clause 1 pass=True)
- daily PnL corr vs BTC = `0.414` n=2485 (clause 2 pass=True; need ≤ 0.70)
- relative-line Sharpe = `0.755` total `639.5%` (STRONG need ≥ 0.50; pass=True)
- MaxDD = `-75.5%` vs BTC `-76.6%` (STRONG |DD| ≤ 1.10×BTC; pass=True)

## 4 — Score-vs-vol diagnostic

Mean per-date cross-sectional Spearman of score vs std_63d (what `/gauss(std)` tilts toward). Negative = low-vol tilt.

- daily btc-in: `-0.442` n=2485
- daily btc-ex: `-0.431` n=2485
- weekly btc-in: `-0.441` n=356
- weekly btc-ex: `-0.431` n=356
- daily DV100 btc-ex: `-0.449` n=2485

## 5 — Top-5 name PnL concentration (best book)

| rank | id | symbol | contrib | share |
|------|----|--------|---------|-------|
| 1 | 52 | XRP | 0.6909 | 8.74% |
| 2 | 1 | BTC | 0.6269 | 7.93% |
| 3 | 4172 | LUNC | 0.5955 | 7.54% |
| 4 | 3635 | CRO | 0.4813 | 6.09% |
| 5 | 2416 | THETA | 0.4397 | 5.57% |

Top-5 abs share of Σ|contrib| = `16.6%`.

## 6 — Per-cycle honesty (total / USD Sharpe)

| cycle | daily btc-in | daily btc-ex | weekly btc-in | weekly btc-ex | BTC |
|-------|------|------|------|------|------|
| 2019-20 | 470.5% / 1.950 | 301.2% / 1.655 | 251.0% / 1.549 | 258.2% / 1.556 | 263.9% |
| 2021 | 1019.8% / 2.386 | 986.8% / 2.380 | 209.5% / 1.515 | 246.7% / 1.609 | 59.8% |
| 2022 | -51.3% / -0.854 | -33.5% / -0.331 | -58.5% / -1.047 | -51.4% / -0.745 | -64.2% |
| 2023-24 | 424.0% / 1.477 | 500.4% / 1.532 | 204.6% / 1.210 | 396.1% / 1.570 | 465.7% |
| 2025-26 | -62.9% / -0.603 | -67.7% / -0.731 | -74.4% / -0.748 | -70.1% / -0.721 | -30.6% |

No single cycle overrides.

## 7 — Mechanical verdict

- **MANUEL-2 STRONG**
- Best book = `daily_btc_in`
- Clause 1 (total ≥ BTC) = `True` (`5939.3%` vs `689.3%`)
- Clause 2 (corr ≤ 0.70) = `True` (`0.414`)
- STRONG extras: rel Sharpe `0.755` pass=True; MaxDD pass=True

Mechanical, no post-hoc adjustment. Nothing adopted.

## Plain language

Best book daily_btc_in: total 59.39273490080397 vs BTC 6.892730650871092 (clause1=True); daily corr vs BTC 0.41445744898283027 (clause2=True). Rel-line Sharpe 0.7554201699668939; MaxDD -0.7545584327122357 vs BTC -0.7662925431645938. Score-vol tilt (Spearman score vs std63) = -0.4419882390985122. Verdict MANUEL-2 STRONG. Nothing adopted.

## Notes

- Four books are a disclosed look; best = highest total. No parameter search.
- EW basket is the no-selection line on the same mcap top-50 ex-pegged (btc-in) universe.
- Frozen-spread crude book is Ladder-1 (h=14) from the 2.c cache; reference only.
- Chart: `charts/manuel2_equity.png`. Elapsed s=`107.1`. GPU=`False`.

COMBO, SPREAD-LS BOOK-HYBRID, LONG-TIDE, and BTC-BEATER v1 untouched.

