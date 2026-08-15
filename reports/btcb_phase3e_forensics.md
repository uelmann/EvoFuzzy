# BTC-BEATER Phase 3.e — pricing-gap forensics

**ANALYSIS ONLY.** Same 3.c positions. No signal or book changes. CPU only, zero GPU. COMBO untouched. 3.c artifacts reused byte-identical.

## Pre-registered outcomes (verbatim, frozen before results)

> SIGNAL-CONFIRMED if RankIC(spread vs Binance returns) ≥ RankIC(same-names CMC) − 0.02 on both full and trailing windows: the gap is then an execution/pricing-level effect; the official record RESUMES as BOOK-HYBRID funding-on, with the suspension footnote replaced by the forensic decomposition, and Binance-priced numbers become canonical for all future phases. SIGNAL-PARTLY-ARTIFACT if RankIC drops > 0.02 on either window: the record stays suspended, the artifact share is quantified, and the next phase MUST re-derive the book on Binance-only pricing before anything else. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical verdict

- **SIGNAL-CONFIRMED**
- RankIC Binance full `0.1517` vs same-names CMC `0.1542` (Δ `-0.0025`; need ≥ −0.02; pass=True)
- RankIC Binance trail-18m `0.2292` vs CMC `0.2307` (Δ `-0.0014`; pass=True)
- Official SPREAD-LS record **RESUMES as BOOK-HYBRID (funding-on)**. Binance-priced numbers are canonical for all future phases. The 3.c suspension footnote is replaced by this forensic decomposition.

Mechanical, no post-hoc adjustment.

## Identity

- 2.c pred cache sha256 = `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78` (expected `28b0719167fe567d1a32e56bbb7bac77c597affb44b968dd40994ac985843f78`)
- Position-log sha256 = `f47f7ece40d6cee536b2a07c25961d1d69284f92ddf716447a52b5f57fcc232b`
- CMC panel sha256 = `c8062ed5d524584c1369e2dab1a075e51c1e6b7c2ad90982bf810ee76eb11249` (read-only assert True)
- BOOK-BINANCE-ONLY Sharpe ON `1.299` / OFF `1.495` / same-days CMC `1.563`
- GPU used = `False`

## (a) Funding vs repricing

ΔSharpe(funding) = `-0.196` (ON `1.299` − OFF `1.495`).
ΔSharpe(repricing) = `-0.067` (OFF `1.495` − CMC `1.563`).

| year | n | Sharpe ON | Sharpe OFF | Sharpe CMC | ΔSh funding | ΔSh repricing | funding PnL | repricing PnL |
|------|---|-----------|------------|------------|-------------|---------------|-------------|---------------|
| 2019 | 74 | -1.701 | -1.701 | -1.786 | 0.000 | 0.084 | 0.0000 | 0.0006 |
| 2020 | 366 | 1.998 | 1.782 | 1.824 | 0.216 | -0.041 | 0.0615 | -0.0054 |
| 2021 | 365 | 1.002 | 0.506 | 0.460 | 0.496 | 0.046 | 0.1529 | 0.0136 |
| 2022 | 365 | 2.564 | 2.818 | 2.882 | -0.254 | -0.063 | -0.0465 | -0.0170 |
| 2023 | 365 | 0.465 | 0.912 | 1.070 | -0.447 | -0.158 | -0.0680 | -0.0254 |
| 2024 | 366 | 0.906 | 0.633 | 0.848 | 0.273 | -0.215 | 0.0446 | -0.0327 |
| 2025 | 365 | 1.780 | 2.349 | 2.297 | -0.569 | 0.052 | -0.1104 | 0.0011 |
| 2026 | 212 | 1.237 | 2.638 | 2.820 | -1.401 | -0.182 | -0.3831 | -0.0300 |

## (b) By side (repricing PnL = Σ w·Δr on replayable name-days)

| side | name-days | PnL diff sum | share of gap |
|------|-----------|--------------|--------------|
| long | 49234 | 0.0325 | -34.1% |
| short | 46197 | -0.1279 | 134.1% |

## (c) By PIT rank tier

| tier | name-days | PnL diff sum | share of gap |
|------|-----------|--------------|--------------|
| 1-30 | 37003 | -0.0101 | 12.5% |
| 31-60 | 24670 | -0.0402 | 49.6% |
| 61-100 | 28798 | -0.0307 | 37.9% |

## (d) Concentration (top-30 name-days by |w·Δr|)

Top-30 share of signed gap = `9.7%`; share of |contrib| = `4.6%`.

| date | id | symbol | side | w | r_cmc | r_bn | Δr | w·Δr | class | rank |
|------|----|--------|------|---|-------|------|----|------|-------|------|
| 2026-07-24 | 7326 | DEXE | short | -0.0712 | 1.2318 | 1.5954 | 0.3637 | -0.02590 | LEVEL-DIFF | 49 |
| 2026-06-11 | 36922 | H | short | -0.0135 | -0.0310 | 1.2824 | 1.3134 | -0.01768 | OTHER | 42 |
| 2025-08-22 | 8911 | STRK | short | -0.0059 | 2.3439 | 0.1012 | -2.2427 | 0.01312 | LEVEL-DIFF | 87 |
| 2021-11-25 | 1966 | MANA | short | -0.0485 | 0.2540 | 0.0059 | -0.2481 | 0.01204 | OTHER | 5 |
| 2021-11-12 | 1808 | OMG | short | -0.0411 | -0.2601 | 0.0206 | 0.2807 | -0.01154 | OTHER | 15 |
| 2026-06-16 | 36922 | H | short | -0.0267 | -0.1613 | 0.2702 | 0.4315 | -0.01153 | OTHER | 34 |
| 2021-11-25 | 6210 | SAND | short | -0.0311 | 0.3264 | -0.0385 | -0.3649 | 0.01135 | OTHER | 13 |
| 2021-11-24 | 1966 | MANA | short | -0.0500 | 0.0866 | 0.2832 | 0.1966 | -0.00983 | LEVEL-DIFF | 8 |
| 2022-12-09 | 4195 | FTT | short | -0.0381 | 0.2468 | 0.0000 | -0.2468 | 0.00940 | OTHER | 70 |
| 2022-09-09 | 8891 | BTCST | short | -0.0387 | 0.2411 | 0.0000 | -0.2411 | 0.00933 | OTHER | None |
| 2026-06-14 | 36922 | H | short | -0.0182 | 0.7071 | 0.2121 | -0.4950 | 0.00900 | LEVEL-DIFF | 41 |
| 2021-11-25 | 8766 | ALICE | short | -0.0290 | 0.1582 | -0.1310 | -0.2892 | 0.00838 | OTHER | 69 |
| 2021-11-25 | 3513 | FTM | short | -0.0485 | 0.1270 | -0.0447 | -0.1717 | 0.00833 | STALE | 23 |
| 2026-06-09 | 36922 | H | short | -0.0123 | 1.0164 | 0.3537 | -0.6626 | 0.00816 | LEVEL-DIFF | 50 |
| 2026-06-17 | 36922 | H | short | -0.0316 | -0.0429 | -0.2784 | -0.2354 | 0.00743 | LEVEL-DIFF | 35 |
| 2021-11-25 | 1934 | LRC | short | -0.0485 | 0.0905 | -0.0625 | -0.1530 | 0.00742 | OTHER | 19 |
| 2022-07-28 | 6538 | CRV | short | -0.0431 | 0.0000 | 0.1662 | 0.1662 | -0.00717 | STALE | 29 |
| 2021-11-11 | 1808 | OMG | short | -0.0363 | 0.1570 | -0.0392 | -0.1962 | 0.00712 | OTHER | 15 |
| 2021-11-24 | 3513 | FTM | short | -0.0500 | -0.0207 | 0.1186 | 0.1393 | -0.00696 | OTHER | 23 |
| 2026-07-02 | 33223 | LAB | short | -0.1000 | 0.3134 | 0.2444 | -0.0689 | 0.00689 | LEVEL-DIFF | 64 |
| 2021-11-24 | 6210 | SAND | short | -0.0317 | 0.1809 | 0.3949 | 0.2140 | -0.00678 | LEVEL-DIFF | 13 |
| 2024-08-23 | 28111 | MEME | short | -0.0343 | -0.0175 | 0.1783 | 0.1958 | -0.00672 | OTHER | 85 |
| 2020-11-16 | 1104 | REP | long | 0.0264 | 0.2517 | 0.0164 | -0.2353 | -0.00621 | OTHER | 92 |
| 2020-03-13 | 2011 | XTZ | short | -0.1000 | 0.2840 | 0.3436 | 0.0596 | -0.00596 | LEVEL-DIFF | 22 |
| 2026-07-06 | 33223 | LAB | short | -0.0913 | -0.0859 | -0.1500 | -0.0641 | 0.00585 | LEVEL-DIFF | 66 |
| 2022-12-16 | 4195 | FTT | short | -0.0362 | -0.1576 | 0.0000 | 0.1576 | -0.00571 | OTHER | None |
| 2020-03-13 | 1321 | ETC | short | -0.0618 | 0.2226 | 0.3147 | 0.0921 | -0.00569 | LEVEL-DIFF | 8 |
| 2020-10-27 | 2280 | FIL | short | -0.0308 | 0.0452 | 0.2183 | 0.1731 | -0.00533 | LEVEL-DIFF | 64 |
| 2021-06-14 | 8104 | 1INCH | short | -0.0236 | 0.0003 | 0.2257 | 0.2254 | -0.00531 | STALE | 62 |
| 2026-07-25 | 7326 | DEXE | short | -0.0810 | 0.2376 | 0.1724 | -0.0652 | 0.00529 | LEVEL-DIFF | 41 |

## Stale-price classification

STALE share of signed repricing gap = `-19.4%`.

| class | n | PnL diff sum | share of gap | share of |contrib| |
|-------|---|--------------|--------------|------------------|
| STALE | 1046 | 0.0185 | -19.4% | 2.6% |
| LEVEL-DIFF | 56841 | 0.0126 | -13.2% | 58.1% |
| OTHER | 37544 | -0.1264 | 132.6% | 39.3% |

## Gap waterfall (PnL: BN_on − CMC_sub)

- funding `-0.3490` (`78.5%`)
- stale repricing `0.0185` (`-4.2%`)
- diffuse repricing `-0.1138` (`25.6%`)
- total `-0.4443`

## RankIC of frozen spread vs h=14 excess (replayable names, same-names CMC)

| window | RankIC BN | RankIC CMC | Δ | n dates |
|--------|-----------|------------|---|---------|
| full | 0.1517 | 0.1542 | -0.0025 | 2390 |
| trail-18m | 0.2292 | 0.2307 | -0.0014 | 548 |

Quintile bucket curve (mean h=14 excess):

| Q | n | mean excess BN | mean excess CMC |
|---|---|----------------|-----------------|
| 1 | 38841 | -0.0544 | -0.0567 |
| 2 | 36861 | -0.0330 | -0.0336 |
| 3 | 37079 | -0.0219 | -0.0220 |
| 4 | 34976 | -0.0181 | -0.0181 |
| 5 | 33776 | -0.0100 | -0.0103 |

## Structural funding

Held-short funding `-4.15` bps/day vs shortable-universe `1.82` (Δ `-5.98`). Total funding PnL `-0.3490`.

| year | n name-days | funding PnL | mean rate |
|------|-------------|-------------|-----------|
| 2020 | 2054 | 0.0615 | 0.000435 |
| 2021 | 7366 | 0.1529 | 0.001103 |
| 2022 | 6080 | -0.0465 | -0.000300 |
| 2023 | 7480 | -0.0680 | -0.000245 |
| 2024 | 9025 | 0.0446 | 0.000401 |
| 2025 | 9160 | -0.1104 | -0.000829 |
| 2026 | 5032 | -0.3831 | -0.002386 |

## Never-listed longs (CMC book contribution)

96 names, 14716 name-days, CMC PnL `0.4043` = `12.5%` of CMC-book name-day gross.

Charts: `charts/btcb_phase3e_gap_waterfall.png`, `charts/btcb_phase3e_rankic.png`.

## Ledger

Official SPREAD-LS record **RESUMES as BOOK-HYBRID (funding-on)**. Binance-priced numbers are canonical for all future phases. The 3.c suspension footnote is replaced by this forensic decomposition.
