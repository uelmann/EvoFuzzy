# BTC-BEATER — LONG-TIDE freeze addendum

**Status:** FROZEN before results. Backtest and analysis only.
**Scope:** Full-size long leg + frozen Stage-T regime gate, BTC parking. No schedules, no live components, no retraining, no signal changes. Master only. CPU only. Zero GPU.
Frozen COMBO v2.0-combo-final, SPREAD-LS as a product, and BTC-BEATER v1 as a product are **untouched**. The 2.c spread cache is reused byte-identical (sha256 verified). CMC raw data is read-only. Pricing follows the 3.e canonical convention (Binance).

## Precondition (mechanical)

EXECUTE ONLY IF the Phase 3.e forensics verdict is SIGNAL-CONFIRMED. Otherwise print `BLOCKED-BY-SUSPENSION: 3.e verdict is <verdict>` and STOP. The 3.c suspension clause governs.

## Pre-registered criteria (verbatim, before results)

> LONG-TIDE is VIABLE if: (a) total return ≥ BTC B&H; (b) relative-line (book/BTC) Sharpe > 0; (c) MaxDD ≤ BTC B&H MaxDD. It SUPERSEDES BTC-BEATER v1 as the official long product only if additionally: (d) relative-line Sharpe ≥ v1's + 0.15 on the common window; (e) average alt deployment ≥ 15%; (f) no cycle with relative-line Sharpe < −0.30. If (a–c) pass but (d–f) do not, LONG-TIDE is recorded as a parallel long variant and v1 stays official. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Book (fixed)

- Selection: top decile by 2.c spread within the floored PIT top-100 dollar-volume universe. Longs restricted to Binance-spot-listed names at date t (primary). Unrestricted-CMC column is reference only.
- EW, 10% name cap, K ≤ 10 per entry day, quintile-exit hysteresis, h=14 tranches, anti-blowoff (7d raw > +50%), death convention. No shorts, no funding.
- Timing: frozen Stage-T regime gate, parameters unchanged (`REGIME_BREADTH=0.50`, `REGIME_OFF_HYSTERESIS=5`). ON when EW top-50/BTC ratio > its 90d SMA AND breadth top-100 > 0.5. OFF requires both false 5 consecutive days. Gate ON → long book at gross 1.0 (deployment floats with qualifiers). Gate OFF → 100% BTC (h-tranche unwind).
- Costs: 10 bps/side spot alts, 2 bps BTC parking moves.
- NAKED LONG LEG control: same selection, no gate, cash idle (not BTC).

## What this freeze does not do

- Does not recompute signals, retrain heads, or change the 2.c spread.
- Does not change SPREAD-LS, COMBO, or BTC-BEATER v1 as products (v1 is replayed read-only).
- Does not change Stage-T gate parameters.
- Does not schedule, go live, or use GPU.
