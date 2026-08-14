# BTC-BEATER Phase 0.c — full-map rebuild + re-audit

**DATA + ANALYSIS ONLY.** Frozen COMBO v2.0-combo-final is untouched. The KuCoin-filtered 828-coin archive and its benchmarks are discarded unread.

## Pre-registered gate v2 (verbatim)

> The dataset is USABLE-FROM-YYYY-MM at the first quarterly CMC historical snapshot D whose true-top-100 coverage is ≥ 85% and remains ≥ 85% at every later snapshot, measured against the external snapshot lists. If that first D is after 2023-01, the project is BLOCKED pending a different data source. Mechanical, no post-hoc adjustment.

## Death-in-position convention (verbatim)

> A held coin whose data ends is force-exited at its last available close (no better information assumed). The count and PnL impact of such forced exits is reported in every backtest of this project.

## Mechanical verdict

- **USABLE-FROM-2017-09**
- BLOCKED: False. Usable start: `2017-09-30`.
- Ended-count: **554** / 2447 (histories ending before 2026-01-01). ended>0 (graveyard retained).

## Download provenance + credit guard

- Plan: public-data-api (existing download_cmc_kucoin.py; credit_count=0; no plan meter)
- Credits projected=0 available=None observed_credit_count=0
- HTTP projected remaining=0 used=1 hard_stop=False
- Target ids=3159 (snapshot union + 828). Cached before OHLCV=3159.
- Map: n=36469 active=8062 inactive=28407 untracked=0

## Schema

- rows=4446095 ids=2447 symbols=2332
- date range 2014-04-19 00:00:00+00:00 → 2026-08-13 00:00:00+00:00
- extra columns: listing_status, last_available_date. Primary key=cryptocurrency_id.

## Named graveyard (must be present with terminal dates)

| query | present | with_terminal | id | symbol | slug | first | last | n | terminal | status | event |
|-------|---------|---------------|----|--------|------|-------|------|---|----------|--------|-------|
| SRM | True | True | 6187 | SRM | serum | 2020-08-11 | 2026-08-13 | 2190 | SURVIVOR | active | Serum / FTX complex 2022 |
| CEL | True | True | 2700 | CEL | celsius | 2018-05-03 | 2026-08-13 | 2874 | SURVIVOR | active | Celsius bankruptcy 2022 |
| UST | True | True | 7129 | USTC | terrausd | 2020-11-25 | 2026-08-08 | 2083 | SURVIVOR | active | TerraUSD collapse May 2022 |
| ANC | True | True | 8857 | ANC | anchor-protocol | 2021-03-17 | 2025-06-26 | 1563 | ENDED | inactive | Anchor Protocol / Terra 2022 |
| SAFEMOON | True | True | 8757 | SAFEMOON | safemoon | 2021-03-13 | 2024-01-25 | 983 | ENDED | inactive | SafeMoon collapse / delist |
| BCC | False | False | None | None | None | None | None | 0 | MISSING | None | BitConnect era 2017–18 |
| XEM | True | True | 873 | XEM | nem | 2015-04-01 | 2026-08-13 | 4153 | SURVIVOR | active | XEM secular decline |
| FTT | True | True | 4195 | FTT | ftx-token | 2019-08-01 | 2026-08-08 | 2565 | SURVIVOR | active | FTX collapse Nov 2022 (continuity) |
| LUNC | True | True | 4172 | LUNC | terra-luna | 2019-07-26 | 2026-08-08 | 2571 | SURVIVOR | active | Terra Classic after May 2022 (continuity) |

Graveyard one-liner input: **8/9 present-with-terminal**.

## Coverage vs external snapshots (top-50/100/200)

Threshold 85% on true-top-100, sustained at all later snapshots. PIT method: **trailing_30d_median_dollar_volume**.

| D | used | top50 | top100 | top200 | pass100 |
|---|------|-------|--------|--------|---------|
| 2017-03-31 | 2017-03-31 | 82.0% | 71.0% | 61.5% | False |
| 2017-06-30 | 2017-06-30 | 86.0% | 77.0% | 64.0% | False |
| 2017-09-30 | 2017-09-30 | 94.0% | 87.0% | 72.5% | True |
| 2017-12-31 | 2017-12-31 | 96.0% | 91.0% | 82.5% | True |
| 2018-03-31 | 2018-03-31 | 98.0% | 97.0% | 88.5% | True |
| 2018-06-30 | 2018-06-30 | 98.0% | 96.0% | 91.5% | True |
| 2018-09-30 | 2018-09-30 | 100.0% | 95.0% | 89.5% | True |
| 2018-12-31 | 2018-12-31 | 100.0% | 91.0% | 89.0% | True |
| 2019-03-31 | 2019-03-31 | 98.0% | 94.0% | 93.0% | True |
| 2019-06-30 | 2019-06-30 | 100.0% | 94.0% | 92.0% | True |
| 2019-09-30 | 2019-09-30 | 96.0% | 87.0% | 87.5% | True |
| 2019-12-31 | 2019-12-31 | 94.0% | 89.0% | 87.0% | True |
| 2020-03-31 | 2020-03-31 | 94.0% | 89.0% | 87.0% | True |
| 2020-06-30 | 2020-06-30 | 98.0% | 97.0% | 93.5% | True |
| 2020-09-30 | 2020-09-30 | 100.0% | 100.0% | 98.0% | True |
| 2020-12-31 | 2020-12-31 | 100.0% | 100.0% | 99.0% | True |
| 2021-03-31 | 2021-03-31 | 100.0% | 100.0% | 99.5% | True |
| 2021-06-30 | 2021-06-30 | 100.0% | 100.0% | 99.5% | True |
| 2021-09-30 | 2021-09-30 | 100.0% | 100.0% | 100.0% | True |
| 2021-12-31 | 2021-12-31 | 100.0% | 100.0% | 100.0% | True |
| 2022-03-31 | 2022-03-31 | 100.0% | 100.0% | 100.0% | True |
| 2022-06-30 | 2022-06-30 | 100.0% | 100.0% | 100.0% | True |
| 2022-09-30 | 2022-09-30 | 100.0% | 100.0% | 100.0% | True |
| 2022-12-31 | 2022-12-31 | 100.0% | 100.0% | 100.0% | True |
| 2023-03-31 | 2023-03-31 | 100.0% | 100.0% | 100.0% | True |
| 2023-06-30 | 2023-06-30 | 100.0% | 100.0% | 100.0% | True |
| 2023-09-30 | 2023-09-30 | 100.0% | 100.0% | 100.0% | True |
| 2023-12-31 | 2023-12-31 | 98.0% | 99.0% | 99.5% | True |
| 2024-03-31 | 2024-03-31 | 100.0% | 99.0% | 99.5% | True |
| 2024-06-30 | 2024-06-30 | 100.0% | 100.0% | 99.5% | True |
| 2024-09-30 | 2024-09-30 | 100.0% | 100.0% | 99.5% | True |
| 2024-12-31 | 2024-12-31 | 100.0% | 100.0% | 100.0% | True |
| 2025-03-31 | 2025-03-31 | 100.0% | 100.0% | 100.0% | True |
| 2025-06-30 | 2025-06-30 | 100.0% | 100.0% | 100.0% | True |
| 2025-09-30 | 2025-09-30 | 100.0% | 100.0% | 100.0% | True |
| 2025-12-31 | 2025-12-31 | 100.0% | 100.0% | 100.0% | True |

PIT files: `universe/btcb_top50_pit.parquet`, `universe/btcb_top100_pit.parquet`.
Elapsed s=93.4. GPU=false. COMBO untouched.

