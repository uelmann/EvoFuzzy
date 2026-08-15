# BTC-BEATER — Phase 3.c freeze addendum

**Status:** FROZEN before results. Binance replay of the production SPREAD-LS book (β-matched, h=14, floored PIT top-100 DV). Real exchange prices + native funding. Replaces Phase 3.b (MASTER dropped by PI).
**Scope:** BACKTEST AND ANALYSIS ONLY. No schedules, no live components, no retraining, no signal changes, no MASTER/combination books. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final and BTC-BEATER v1 are **untouched**. The 2.c spread cache is reused byte-identical (sha256 verified). Book config is unchanged: β-matched, h=14, floored PIT top-100 dollar-volume.

Signals are **not** recomputed. The 2.c spread cache drives positions identically; only pricing and funding change.

## 00 — Addenda (verbatim, frozen before results)

### 1. β-match designation with post-observation disclosure (verbatim from the 3.b spec, unchanged)

> The Phase 3 freeze designated dollar-neutral as the judged headline and β-matched as reported-not-judged. After seeing results (DN β=−0.122, β-matched β=0.025), β-matched is designated the production SPREAD-LS book. This is disclosed, not hidden. Phase 3 mechanical verdicts (VIABLE, not SLEEVE-GRADE, not replacement) remain those of the DN headline and are not retroactively rewritten. All subsequent work (funding-on, MASTER) uses the β-matched book.

### 2. House-rule correction (record only)

> The bias clause of future null gates reverts to the original E.1b tolerance: CONTAMINATED requires ≥2 fold-level violations of the 2·SE bound, not 1. The 'every fold' variant has ≈25% false-alarm probability with 6 folds. No past verdicts change.

This is a record-only freeze. Existing gates and past verdicts (including the horizon-sweep h=3 CONTAMINATED on a single fold) are not rewritten and are not re-run.

### 3. MASTER removed from scope

> MASTER (COMBO+SPREAD-LS combination) removed from scope by PI decision; the 0.157 correlation remains on the ledger for allocation purposes.

No MASTER book is built in this phase. No combination weights. The 0.157 COMBO overlap correlation stays on the ledger for allocation.

## Pre-registered validation and adoption (verbatim, before results)

> PRICES ARE VALIDATED if, on the replayable subset, BOOK-BINANCE-ONLY daily PnL correlation with the same-days CMC-priced book is ≥ 0.95 AND its net Sharpe ≥ (same-days CMC Sharpe − 0.15). If validated, BOOK-HYBRID (funding-on) becomes the OFFICIAL SPREAD-LS record; funding-off CMC numbers are deprecated with a ledger footnote. If NOT validated, the discrepancy is quantified per year and per name-tier, the official record is suspended, and no improvement work proceeds until the pricing gap is understood. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Three books, identical positions

Positions are taken from one β-matched h=14 run on the 2.c spread (the 3.x production book). The position log is shared. Only the price source and funding differ.

- **BOOK-CMC:** existing 3.x book (reference, funding=0) — reproduced byte-identical as sanity.
- **BOOK-HYBRID:** Binance prices + native funding where replayable, CMC prices elsewhere (flagged share). Becomes the OFFICIAL book if validation passes.
- **BOOK-BINANCE-ONLY:** restricted to fully replayable name-days (both entry and exit on Binance) — the pure-exchange subset, for the validation metric.

Longs price on Binance **spot** USDT 1d klines. Shorts price on Binance **USDT-M perpetual** 1d klines. Funding accrues at each 8h event on short positions from real `fundingRate` files; funding PnL is its own column. Longs are spot: no funding.

A name-date is **REPLAYABLE** only if the relevant Binance market was live that day (PIT listing/delisting). Names never listed on Binance stay at CMC prices in the hybrid book and are flagged.

## Coverage (fixed)

- % of long name-days and % of short name-days replayable, per year.
- Names never listed on Binance: kept at CMC in the hybrid book, flagged.

## Data (fixed)

- Old-project caches reused: USDT-M perp klines (`/data/quant/raw/klines`) and funding (`/data/quant/raw/funding`).
- New downloads only for missing **spot** klines, logged. Destination: `/data/quant/raw/spot_klines`.
- 2.c pred cache sha256 must equal `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (112 files). Abort if the hash mutates.

## What this freeze does not do

- Does not recompute signals, retrain heads, or change the 2.c spread.
- Does not change β-matched sizing, h=14, decile/quintile hysteresis, anti-blowoff, death convention, or costs.
- Does not build a MASTER / combination book.
- Does not introduce schedules or live components.
- Does not touch COMBO, the system card, frozen A0 scores, or BTC-BEATER v1.
- Does not use GPU.
- Does not rewrite past null-gate verdicts.
