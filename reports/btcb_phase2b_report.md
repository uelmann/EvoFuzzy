# BTC-BEATER Phase 2.b — hygiene, naive v4, MODEL-V2

**BACKTEST ONLY.** Hygiene before any backtest. Stage S is within-date only (no context). Stage T is a frozen regime gate, not learned. CPU only, zero GPU. COMBO untouched.

## Pre-registered criteria (verbatim, before results)

> STAGE-S has SELECTION SKILL if, at h=14, full-OOS mean per-date AUC ≥ 0.52 with the empirical-null gates passing. MODEL-V2 is VIABLE if, on the full OOS window: (a) relative-line Sharpe > 0; (b) total return ≥ BTC B&H; (c) MaxDD ≤ BTC B&H MaxDD. It REPLACES the naive v4 floor if additionally relative-line Sharpe ≥ naive v4 + 0.15. Per-cycle honesty table mandatory; no single cycle overrides. Mechanical, no post-hoc adjustment.

## Naive v4 label (verbatim)

> NAIVE-ROTATION is a LIVE BENCHMARK if its full-window relative-line Sharpe (book/BTC) > 0 and its total return ≥ BTC B&H. Whatever the label, its numbers become the floor every ML phase of this project must beat net of costs.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## 1. Naive v3 autopsy (the +9.98M% book)

Window 2019-10-20 → 2026-08-13; book total 9982249.6%; rel Sharpe 0.589. Additive PnL contributions (top 10):

| rank | id | symbol | contrib | share | max daily ret | max |abs| daily ret | BTC? |
|------|----|--------|---------|-------|---------------|----------------------|------|
| 1 | 3367 | CRD | 212.0302 | 98.8% | 1144.16 | 1144.16 | False |
| 2 | 1 | BTC | 0.4182 | 0.2% | 0.25 | 0.37 | True |
| 3 | 4162 | STO | 0.3692 | 0.2% | 22.90 | 22.90 | False |
| 4 | 74 | DOGE | 0.2875 | 0.1% | 3.56 | 3.56 | False |
| 5 | 2010 | ADA | 0.2775 | 0.1% | 1.37 | 1.37 | False |
| 6 | 5426 | SOL | 0.2743 | 0.1% | 0.47 | 0.47 | False |
| 7 | 5168 | BXC | 0.2578 | 0.1% | 38573.20 | 38573.20 | False |
| 8 | 5805 | AVAX | 0.2340 | 0.1% | 0.75 | 0.75 | False |
| 9 | 1106 | SHND | 0.2152 | 0.1% | 45.02 | 45.02 | False |
| 10 | 6758 | SUSHI | 0.1956 | 0.1% | 1.65 | 1.65 | False |

Top alt share of additive PnL = 98.8%. Flag >25%: **True**.

## 2. Redenom/split cleaning

- Jump threshold |daily ret| > 5. Ids touched=610; splice events=773; truncate events=357.
- Rows 4446095 → 4042032.
- Remaining |daily ret|>5 after clean: 15 (BTC-exempt jumps ignored in cleaner).
- Full per-id splice/truncate log is in `reports/btcb_phase2b_report.json` (`splice_log`); table below is the first 120 of 610 ids.

| id | symbol | n_jumps | truncated | actions |
|----|--------|---------|-----------|---------|
| 4 | TRC | 3 | True | truncate@2016-09-13 |
| 6 | NVC | 1 | True | truncate@2020-10-04 |
| 10 | FRC | 3 | True | truncate@2016-03-11 |
| 13 | IXC | 14 | True | truncate@2016-05-09 |
| 16 | WDC | 6 | True | splice@2023-03-25,splice@2024-10-22,truncate@2024-10-31 |
| 18 | DGC | 3 | False | splice@2019-12-20,splice@2020-01-14,splice@2025-04-11 |
| 35 | PXC | 1 | False | splice@2021-08-28 |
| 43 | ANC | 2 | True | truncate@2017-05-24 |
| 45 | CSC | 1 | False | splice@2021-03-24 |
| 53 | QRK | 6 | True | truncate@2020-02-05 |
| 56 | ZET | 1 | False | splice@2016-07-10 |
| 66 | NXT | 1 | True | truncate@2024-07-03 |
| 67 | UNO | 2 | True | splice@2022-01-03,truncate@2022-01-05 |
| 72 | DEM | 2 | False | splice@2019-02-07,splice@2021-03-10 |
| 77 | DMD | 1 | False | splice@2024-05-30 |
| 80 | ORB | 2 | True | truncate@2023-03-12 |
| 83 | OMNI | 2 | False | splice@2023-12-22,splice@2024-04-12 |
| 87 | TIPS | 10 | True | truncate@2017-08-28 |
| 90 | DIME | 33 | True | truncate@2016-02-16 |
| 93 | 42 | 2 | False | splice@2017-01-16,splice@2017-04-11 |
| 118 | RDD | 26 | True | splice@2023-06-11,splice@2023-12-09,splice@2023-12-15,splice@2023-12-28 |
| 120 | KAT | 3 | True | splice@2023-01-28,splice@2023-03-04,truncate@2023-04-04 |
| 122 | POT | 7 | True | splice@2024-02-21,truncate@2024-03-05 |
| 128 | MAX | 7 | True | splice@2017-02-03,truncate@2017-09-18 |
| 141 | MINT | 3 | True | splice@2021-01-05,truncate@2021-05-01 |
| 145 | DOPE | 2 | True | splice@2014-07-23,truncate@2014-10-14 |
| 168 | UFO | 1 | True | truncate@2020-11-19 |
| 184 | NOTE | 5 | False | splice@2018-05-03,splice@2020-05-20,splice@2020-10-04,splice@2021-01-09 |
| 212 | ECC | 16 | True | splice@2014-08-04,splice@2014-08-08,splice@2014-10-11,splice@2015-10-16 |
| 217 | BELA | 3 | True | truncate@2014-05-14 |
| 234 | EFL | 9 | True | splice@2021-02-18,splice@2021-02-20,splice@2021-02-22,splice@2021-02-25 |
| 258 | GRS | 1 | True | truncate@2026-02-10 |
| 260 | XPD | 5 | True | splice@2014-05-16,splice@2014-11-19,truncate@2016-07-11 |
| 268 | XWC | 1 | False | splice@2020-03-21 |
| 276 | BITS | 1 | True | truncate@2024-11-29 |
| 290 | BLU | 4 | True | splice@2016-05-16,truncate@2016-05-18 |
| 293 | XBC | 4 | True | splice@2017-03-27,splice@2024-12-30,truncate@2024-12-31 |
| 298 | NYC | 41 | True | truncate@2014-07-25 |
| 341 | SUPER | 27 | True | splice@2016-07-31,truncate@2016-08-18 |
| 362 | CLOAK | 2 | True | truncate@2023-12-27 |
| 366 | BSD | 3 | False | splice@2015-02-16,splice@2015-09-29,splice@2016-11-22 |
| 372 | BCN | 1 | False | splice@2026-05-27 |
| 377 | NAV | 1 | False | splice@2016-07-05 |
| 416 | THC | 3 | False | splice@2019-06-08,splice@2024-06-02,splice@2024-10-22 |
| 460 | CLAM | 2 | False | splice@2023-02-01,splice@2023-03-13 |
| 470 | VIA | 1 | False | splice@2023-12-05 |
| 501 | XCN | 1 | False | splice@2017-03-18 |
| 502 | CARBON | 17 | True | splice@2014-12-05,truncate@2014-12-22 |
| 506 | CANN | 1 | False | splice@2019-10-19 |
| 551 | DONU | 1 | True | truncate@2022-08-16 |
| 558 | EMC | 1 | True | truncate@2024-03-27 |
| 576 | GAME | 3 | True | truncate@2014-09-12 |
| 584 | N8V | 2 | True | truncate@2021-03-21 |
| 624 | BITCNY | 2 | True | splice@2015-11-08,truncate@2016-02-01 |
| 633 | EXCL | 1 | False | splice@2016-07-11 |
| 638 | TROLL | 3 | True | truncate@2015-12-02 |
| 644 | BSTY | 3 | False | splice@2017-01-10,splice@2017-01-24,splice@2020-09-12 |
| 656 | PXI | 1 | False | splice@2015-03-04 |
| 659 | BITS | 2 | True | truncate@2023-12-10 |
| 693 | XVG | 1 | True | truncate@2015-02-19 |
| 702 | SPR | 1 | False | splice@2016-08-02 |
| 703 | RBT | 4 | True | splice@2015-01-06,splice@2015-01-20,truncate@2015-03-19 |
| 707 | BLOCK | 2 | False | splice@2025-07-16,splice@2026-02-27 |
| 720 | CRW | 3 | False | splice@2015-03-06,splice@2015-03-22,splice@2016-02-21 |
| 730 | GCN | 7 | True | splice@2015-10-27,splice@2017-09-28,splice@2017-10-03,truncate@2017-10-18 |
| 760 | OK | 5 | True | truncate@2015-12-15 |
| 764 | XPY | 1 | False | splice@2017-09-08 |
| 788 | COVAL | 1 | True | truncate@2017-04-01 |
| 799 | SMLY | 4 | True | splice@2017-05-24,truncate@2017-05-29 |
| 815 | KOBO | 4 | False | splice@2016-01-01,splice@2016-02-02,splice@2016-07-28,splice@2017-01-28 |
| 819 | BITB | 14 | True | splice@2022-04-02,splice@2023-12-28,splice@2024-02-05,splice@2024-03-19 |
| 837 | XCO | 2 | False | splice@2015-08-06,splice@2016-06-29 |
| 857 | SONG | 2 | True | truncate@2019-08-19 |
| 859 | LOG | 1 | True | truncate@2025-02-28 |
| 894 | NTRN | 5 | True | splice@2017-04-04,truncate@2017-07-13 |
| 911 | AIB | 8 | True | truncate@2017-05-07 |
| 921 | UNIT | 7 | True | splice@2017-05-16,truncate@2017-05-22 |
| 934 | PKB | 1 | False | splice@2015-05-21 |
| 938 | ARB | 7 | True | splice@2015-05-29,truncate@2015-07-11 |
| 945 | BTA | 148 | True | splice@2019-07-14,splice@2020-01-01,splice@2020-10-20,splice@2020-11-29 |
| 948 | ADC | 3 | True | truncate@2021-11-08 |
| 977 | GXX | 1 | False | splice@2021-03-23 |
| 986 | CREVA | 1 | False | splice@2015-08-16 |
| 990 | ZNY | 1 | False | splice@2020-01-17 |
| 1004 | HNC | 6 | True | splice@2015-07-23,splice@2015-08-19,truncate@2020-01-02 |
| 1019 | MANNA | 35 | True | splice@2017-05-27,splice@2020-10-15,splice@2020-11-26,truncate@2023-06-03 |
| 1032 | TX | 11 | True | splice@2020-10-19,truncate@2020-10-22 |
| 1033 | GCC | 6 | True | splice@2015-09-01,truncate@2015-12-28 |
| 1044 | KWD | 8 | True | splice@2023-05-15,splice@2023-05-22,splice@2023-07-16,splice@2023-07-18 |
| 1053 | BOLI | 3 | False | splice@2015-09-15,splice@2016-06-17,splice@2024-10-22 |
| 1066 | PAK | 7 | True | splice@2015-09-28,truncate@2015-10-05 |
| 1070 | EXP | 10 | True | splice@2016-02-11,truncate@2019-04-04 |
| 1082 | SIB | 1 | True | truncate@2022-04-06 |
| 1106 | SHND | 44 | True | splice@2019-10-09,truncate@2019-10-27 |
| 1107 | PAC | 10 | True | truncate@2017-10-16 |
| 1120 | DFT | 20 | True | truncate@2023-01-31 |
| 1156 | YOC | 7 | True | splice@2016-02-09,truncate@2016-06-08 |
| 1159 | SLS | 6 | False | splice@2022-04-14,splice@2022-04-29,splice@2022-05-16,splice@2022-05-21 |
| 1164 | FRN | 1 | True | truncate@2017-04-11 |
| 1165 | EVIL | 5 | True | splice@2016-03-14,splice@2016-04-24,splice@2016-05-09,truncate@2016-07-28 |
| 1172 | SFT | 3 | True | truncate@2016-12-05 |
| 1175 | RBIES | 1 | False | splice@2018-06-18 |
| 1185 | FREED | 6 | True | splice@2021-03-18,splice@2024-11-26,splice@2025-04-21,splice@2025-04-28 |
| 1191 | MEME | 1 | False | splice@2021-01-29 |
| 1200 | NEVA | 28 | True | truncate@2017-11-03 |
| 1212 | MOJO | 7 | True | splice@2017-03-07,splice@2024-08-21,splice@2024-10-21,splice@2024-12-06 |
| 1216 | EDRC | 1 | False | splice@2016-06-01 |
| 1223 | BERN | 1 | False | splice@2016-04-14 |
| 1247 | ARCO | 4 | True | splice@2016-12-03,truncate@2020-05-27 |
| 1250 | ZUR | 27 | True | splice@2017-08-05,splice@2021-04-15,truncate@2021-05-21 |
| 1254 | XPTX | 4 | False | splice@2016-06-03,splice@2020-08-02,splice@2020-08-13,splice@2020-10-12 |
| 1266 | MXT | 2 | False | splice@2017-06-22,splice@2019-10-23 |
| 1281 | ION | 13 | True | splice@2016-12-07,splice@2021-08-17,splice@2021-08-21,splice@2021-09-10 |
| 1282 | HVCO | 4 | True | splice@2016-06-15,truncate@2016-08-13 |
| 1285 | GB | 1 | True | truncate@2025-05-23 |
| 1294 | RISE | 2 | True | truncate@2021-12-16 |
| 1297 | CHESS | 2 | True | truncate@2017-07-01 |
| 1306 | CJ | 1 | True | truncate@2017-07-14 |
| 1308 | HEAT | 4 | True | splice@2020-03-20,splice@2020-07-31,splice@2020-08-02,truncate@2022-07-26 |
| 1381 | BTDX | 1 | True | truncate@2019-06-17 |

## 3. Investability floor (PIT rebuild)

Floor: 30d median DV ≥ $2e6, price ≥ 1e-6, ≥60 prior sessions, no |ret|>200% in 30d.
- top-50 floored: eligible ids=1716 med eligible/day=326.0 pit rows=167223 vs unfloored rows=224944 ids=981 med/day=50.0
- top-100 floored: eligible ids=1716 med eligible/day=326.0 pit rows=321542 vs unfloored rows=441507 ids=1437 med/day=100.0

## 4. Naive v4 (cleaned + floored)

- **NAIVE-ROTATION v4 is NOT A LIVE BENCHMARK**
- rel Sharpe=-0.670; book -94.7% vs BTC 1395.9%
- MaxDD -98.9%; %BTC 6.0%; forced=7
- Top alt contrib share 23.5%; flag >25%: **False**

| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC | ann TO | forced |
|------|-------|------|------------|------------|-------|------------|-------|--------|--------|
| naive v4 | -94.7% | -28.3% | 0.113 | -0.670 | -98.9% | 12.47 | 6.0% | 13.10 | 7 |
| BTC B&H | 1395.9% | 35.6% | 0.795 | 0.000 | -83.4% | 0.00 | 100.0% | 0.00 | 0 |

| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | % BTC |
|-------|---|----------|---------|------------|------------|-------|-------|
| 2018-19 | 730 | -92.3% | -49.2% | -0.898 | -1.422 | -96.7% | 12.2% |
| 2020-21 | 731 | 663.3% | 543.7% | 1.507 | 0.462 | -64.5% | 5.1% |
| 2022 | 365 | -90.2% | -64.3% | -1.903 | -1.907 | -90.8% | 1.6% |
| 2023-24 | 731 | 60.6% | 464.6% | 0.695 | -0.816 | -60.9% | 2.8% |
| 2025-26 | 590 | -82.8% | -30.5% | -1.095 | -1.358 | -86.6% | 4.6% |

Naive v4 top-10 contributors:

| rank | id | symbol | share | max daily ret |
|------|----|--------|-------|---------------|
| 1 | 1 | BTC | 57.6% | 0.25 |
| 2 | 5426 | SOL | 23.5% | 0.47 |
| 3 | 1958 | TRX | 21.2% | 1.20 |
| 4 | 5805 | AVAX | 19.8% | 0.75 |
| 5 | 74 | DOGE | 19.5% | 3.56 |
| 6 | 2010 | ADA | 19.1% | 1.37 |
| 7 | 6758 | SUSHI | 17.4% | 1.65 |
| 8 | 20947 | SUI | 16.5% | 0.39 |
| 9 | 7963 | vETH | 16.3% | 2.86 |
| 10 | 512 | XLM | 16.2% | 1.06 |

## 5. Stage-S gates

Gates official: **FAIL**.

| gate | passed | detail |
|------|--------|--------|
| feature_lookahead | True | `{'max_abs_diff': 0.0, 'id': 8, 'date': '2020-06-16'}` |
| universe_lookahead_top50_floor | True | `{'n': 50, 'date': '2020-06-16', 'base_n': 50, 'symmetric_diff': 0, 'floored': True}` |
| universe_lookahead_top100_floor | True | `{'n': 100, 'date': '2020-06-16', 'base_n': 100, 'symmetric_diff': 0, 'floored': True}` |
| no_context_in_stage_s | True | `{'leaked': [], 'n_feat': 33}` |
| seed_determinism | True | `{'max_score_diff': 0.0, 'best_iteration': 320, 'fold_id': 0, 'horizon': 14}` |

Label-shuffle null (mean per-date AUC, 2 folds × 25): **PARKED-NO-SKILL** bias_pass=True skill_pass=False. Bias: |null mean − 0.5| ≤ 2·(SD/√R).

| fold | n | null mean | SD | 95th | real pdauc | bias_ok | exceeds_p95 |
|------|---|-----------|----|------|------------|---------|-------------|
| 0 | 25 | 0.5024 | 0.0212 | 0.5329 | 0.5349 | True | True |
| 9 | 25 | 0.5002 | 0.0220 | 0.5355 | 0.5334 | True | False |

## Stage-S per-date AUC by fold

| h | fold | val_start | val_end | n_valid | pdauc | rankIC | best_iter |
|---|------|-----------|---------|---------|-------|--------|-----------|
| 14 | 0 | 2019-10-18 | 2020-01-16 | 8807 | 0.5348 | 0.0319 | 320 |
| 14 | 1 | 2020-01-16 | 2020-04-15 | 8899 | 0.5466 | -0.0292 | 59 |
| 14 | 2 | 2020-04-15 | 2020-07-14 | 8890 | 0.5265 | -0.0658 | 63 |
| 14 | 3 | 2020-07-14 | 2020-10-12 | 8773 | 0.5379 | 0.0372 | 1 |
| 14 | 4 | 2020-10-12 | 2021-01-10 | 8595 | 0.5477 | 0.0085 | 101 |
| 14 | 5 | 2021-01-10 | 2021-04-10 | 8832 | 0.5552 | -0.0173 | 12 |
| 14 | 6 | 2021-04-10 | 2021-07-09 | 8954 | 0.5441 | 0.0332 | 5 |
| 14 | 7 | 2021-07-09 | 2021-10-07 | 8908 | 0.5593 | -0.0739 | 44 |
| 14 | 8 | 2021-10-07 | 2022-01-05 | 8912 | 0.5710 | -0.0013 | 93 |
| 14 | 9 | 2022-01-05 | 2022-04-05 | 8916 | 0.5312 | -0.0568 | 9 |
| 14 | 10 | 2022-04-05 | 2022-07-04 | 8862 | 0.5139 | -0.0749 | 23 |
| 14 | 11 | 2022-07-04 | 2022-10-02 | 8919 | 0.5651 | 0.0390 | 51 |
| 14 | 12 | 2022-10-02 | 2022-12-31 | 8965 | 0.5703 | 0.0392 | 90 |
| 14 | 13 | 2022-12-31 | 2023-03-31 | 8945 | 0.5168 | -0.0531 | 111 |
| 14 | 14 | 2023-03-31 | 2023-06-29 | 8899 | 0.5687 | -0.0278 | 18 |
| 14 | 15 | 2023-06-29 | 2023-09-27 | 8895 | 0.5635 | -0.0313 | 76 |
| 14 | 16 | 2023-09-27 | 2023-12-26 | 8863 | 0.5191 | -0.0862 | 23 |
| 14 | 17 | 2023-12-26 | 2024-03-25 | 8636 | 0.5446 | -0.0207 | 4 |
| 14 | 18 | 2024-03-25 | 2024-06-23 | 8543 | 0.6156 | 0.0503 | 173 |
| 14 | 19 | 2024-06-23 | 2024-09-21 | 8755 | 0.5202 | -0.0759 | 156 |
| 14 | 20 | 2024-09-21 | 2024-12-20 | 8753 | 0.5513 | -0.0181 | 3 |
| 14 | 21 | 2024-12-20 | 2025-03-20 | 8571 | 0.6067 | 0.0061 | 50 |
| 14 | 22 | 2025-03-20 | 2025-06-18 | 8705 | 0.6126 | 0.0253 | 71 |
| 14 | 23 | 2025-06-18 | 2025-09-16 | 8832 | 0.5865 | -0.0077 | 146 |
| 14 | 24 | 2025-09-16 | 2025-12-15 | 8725 | 0.6930 | 0.1567 | 56 |
| 14 | 25 | 2025-12-15 | 2026-03-15 | 8787 | 0.5675 | -0.0348 | 113 |
| 14 | 26 | 2026-03-15 | 2026-06-13 | 8958 | 0.5506 | 0.0190 | 203 |
| 14 | 27 | 2026-06-13 | 2026-08-13 | 6005 | 0.5396 | 0.0202 | 792 |
| 30 | 0 | 2019-11-03 | 2020-02-01 | 8823 | 0.5242 | 0.0178 | 6 |
| 30 | 1 | 2020-02-01 | 2020-05-01 | 8873 | 0.5313 | -0.0004 | 268 |
| 30 | 2 | 2020-05-01 | 2020-07-30 | 8914 | 0.4712 | -0.0805 | 28 |
| 30 | 3 | 2020-07-30 | 2020-10-28 | 8701 | 0.5621 | 0.1832 | 2 |
| 30 | 4 | 2020-10-28 | 2021-01-26 | 8667 | 0.5055 | -0.0393 | 34 |
| 30 | 5 | 2021-01-26 | 2021-04-26 | 8832 | 0.5568 | 0.0747 | 49 |
| 30 | 6 | 2021-04-26 | 2021-07-25 | 8948 | 0.5556 | 0.0594 | 2 |
| 30 | 7 | 2021-07-25 | 2021-10-23 | 8885 | 0.5347 | 0.0446 | 10 |
| 30 | 8 | 2021-10-23 | 2022-01-21 | 8940 | 0.5606 | 0.1414 | 96 |
| 30 | 9 | 2022-01-21 | 2022-04-21 | 8903 | 0.5449 | 0.0315 | 11 |
| 30 | 10 | 2022-04-21 | 2022-07-20 | 8875 | 0.5061 | -0.0513 | 58 |
| 30 | 11 | 2022-07-20 | 2022-10-18 | 8923 | 0.6346 | 0.1930 | 8 |
| 30 | 12 | 2022-10-18 | 2023-01-16 | 8949 | 0.5746 | 0.0819 | 7 |
| 30 | 13 | 2023-01-16 | 2023-04-16 | 8961 | 0.5730 | 0.1193 | 10 |
| 30 | 14 | 2023-04-16 | 2023-07-15 | 8868 | 0.5490 | 0.0083 | 13 |
| 30 | 15 | 2023-07-15 | 2023-10-13 | 8884 | 0.5174 | 0.0329 | 8 |
| 30 | 16 | 2023-10-13 | 2024-01-11 | 8839 | 0.5153 | 0.0676 | 94 |
| 30 | 17 | 2024-01-11 | 2024-04-10 | 8618 | 0.5178 | -0.0097 | 3 |
| 30 | 18 | 2024-04-10 | 2024-07-09 | 8591 | 0.5821 | 0.0512 | 12 |
| 30 | 19 | 2024-07-09 | 2024-10-07 | 8775 | 0.4015 | -0.1248 | 308 |
| 30 | 20 | 2024-10-07 | 2025-01-05 | 8705 | 0.5348 | -0.0364 | 63 |
| 30 | 21 | 2025-01-05 | 2025-04-05 | 8584 | 0.5856 | 0.0271 | 4 |
| 30 | 22 | 2025-04-05 | 2025-07-04 | 8688 | 0.5862 | 0.0236 | 21 |
| 30 | 23 | 2025-07-04 | 2025-10-02 | 8892 | 0.6460 | 0.1559 | 194 |
| 30 | 24 | 2025-10-02 | 2025-12-31 | 8701 | 0.6785 | 0.2101 | 152 |
| 30 | 25 | 2025-12-31 | 2026-03-31 | 8815 | 0.5631 | 0.0610 | 45 |
| 30 | 26 | 2026-03-31 | 2026-06-29 | 8960 | 0.5353 | 0.0993 | 241 |
| 30 | 27 | 2026-06-29 | 2026-08-13 | 4436 | 0.3871 | -0.0387 | 647 |

## Mechanical verdicts

- **STAGE-S has SELECTION SKILL: False** (h=14 mean per-date AUC=0.5596, need ≥ 0.52; mean per-date RankIC=-0.0083; null=PARKED-NO-SKILL; gates_ok=False)
- **MODEL-V2 is NOT VIABLE**
- **REPLACES naive v4 floor: False**
- median p_enter=0.6; OOS 2019-10-19 → 2026-08-13 n=2491
- (a) rel Sharpe 0.419 > 0 → True
- (b) book 738.2% vs BTC 714.0% → True
- (c) MaxDD -76.6% vs BTC -76.6% → True
- replace need ≥ -0.426 (naive v4 same-window -0.576 + 0.15)
- % time in BTC 99.3%; gate ON frac 19.1%; forced=0

A verdict is not overridden by any single cycle.

## MODEL-V2 book vs naive v4 vs BTC (same OOS window)

| book | total | CAGR | USD Sharpe | rel Sharpe | MaxDD | avg #names | % BTC | ann TO | forced |
|------|-------|------|------------|------------|-------|------------|-------|--------|--------|
| MODEL-V2 h=14 p=0.6 | 738.2% | 36.6% | 0.825 | 0.419 | -76.6% | 0.12 | 99.3% | 0.24 | 0 |
| naive v4 (same window) | -80.8% | -21.5% | 0.194 | -0.576 | -98.1% | 12.76 | 3.8% | 13.01 | 7 |
| BTC B&H | 714.0% | 36.0% | 0.818 | 0.000 | -76.6% | 0.00 | 100.0% | 0.00 | 0 |

## p_enter grid (h=14)

| p_enter | rel Sharpe | total | MaxDD | % BTC | gate ON |
|---------|------------|-------|-------|-------|---------|
| 0.55 | 0.513 | 746.0% | -76.6% | 99.3% | 19.1% |
| 0.6 ← median | 0.419 | 738.2% | -76.6% | 99.3% | 19.1% |
| 0.65 | 0.374 | 734.9% | -76.6% | 99.5% | 19.1% |

## Per-cycle honesty (headline h=14)

| cycle | n | book tot | BTC tot | USD Sharpe | rel Sharpe | MaxDD | % BTC |
|-------|---|----------|---------|------------|------------|-------|-------|
| 2019-20 | 440 | 267.6% | 263.7% | 1.913 | 0.839 | -51.9% | 99.7% |
| 2021 | 365 | 59.7% | 59.7% | 0.983 | 0.000 | -53.1% | 100.0% |
| 2022 | 365 | -64.3% | -64.3% | -1.298 | 0.000 | -66.9% | 100.0% |
| 2023-24 | 731 | 466.5% | 464.6% | 2.020 | 0.243 | -26.3% | 99.2% |
| 2025-26 | 590 | -29.5% | -30.5% | -0.284 | 0.547 | -53.1% | 98.3% |

## Feature importances (mean gain, h=14 Stage S — should be per-coin)

| rank | feature | mean gain |
|------|---------|-----------|
| 1 | `dist_high_90` | 8986.71 |
| 2 | `range_pos_28` | 7343.42 |
| 3 | `dist_ath` | 7312.88 |
| 4 | `log_age` | 7149.44 |
| 5 | `log_mcap` | 6934.78 |
| 6 | `beta_btc_60` | 5642.77 |
| 7 | `yz_vol_60` | 4722.74 |
| 8 | `idio_vol_60` | 4423.15 |
| 9 | `skew_60` | 4187.79 |
| 10 | `amihud_14` | 4026.91 |
| 11 | `corr_btc_28` | 3999.79 |
| 12 | `dist_low_90` | 3484.20 |
| 13 | `sma20_sma50` | 3386.45 |
| 14 | `yz_vol_30` | 3374.28 |
| 15 | `mcap_rank` | 3353.54 |

Elapsed s=3880.5. GPU=False. n_features=33.

COMBO untouched (v2.0-combo-final).

