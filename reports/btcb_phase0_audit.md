# BTC-BEATER Phase 0 — dataset audit

**BACKTEST AND ANALYSIS ONLY.** New project. Frozen COMBO v2.0-combo-final is untouched.

## Pre-registered gate

> The dataset is USABLE-FROM-YYYY-MM if, from that date onward, ≥80% of the historical top-200 sample coins are present with correct terminal histories and a PIT universe is reconstructable. The earliest such date is the project's backtest start. If no date before 2021-01 qualifies, the 2018–2020 era is declared FICTION and excluded; if no date before 2023-01 qualifies, the project is BLOCKED pending a different data source. Mechanical, no post-hoc adjustment.

## Mechanical verdict

- **USABLE-FROM-2018-01**
- Sample OK need ≥ 80.0% of n=30 (80% of the 30-coin historical top-200 sample).
- 2018–2020 FICTION: False. BLOCKED: False.
- Usable start: `2018-01-01`.

Source file: `/data/kronos/historical_data_full.csv`. Method for PIT score: **trailing_30d_median_dollar_volume**.

## Schema

- rows=1026057 ids=828 symbols=828 slugs=828
- date range 2016-01-01 00:00:00+00:00 → 2026-08-08 00:00:00+00:00
- columns: `['cryptocurrency_id', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'marketCap', 'currency_name', 'currency_symbol', 'currency_slug']`
- null fractions: `{'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0, 'mcap': 0.0}`
- symbols mapping to >1 id: 0; ids with >1 symbol: 0
- terminal histories: survivors=828 ended=0 (an archive with ended=0 does not retain delisted names — survivorship bias).

## Graveyard — named dead/collapsed assets

| query | present | symbol | slug | first | last | n | terminal | event | crash window |
|-------|---------|--------|------|-------|------|---|----------|-------|--------------|
| LUNA | True | LUNC | terra-luna | 2019-07-26 | 2026-08-08 | 2571 | SURVIVOR | Terra collapse May 2022 / LUNA 2.0 | window 2022-05 max=116.409 min=4.81247e-05 min/max=0.0000 |
| LUNA | True | LUNA | terra-luna-v2 | 2022-05-28 | 2026-08-08 | 1534 | SURVIVOR | Terra collapse May 2022 / LUNA 2.0 | window 2022-05 max=10.7141 min=2.17852 min/max=0.2033 |
| LUNC | True | LUNC | terra-luna | 2019-07-26 | 2026-08-08 | 2571 | SURVIVOR | Terra Classic (ex-LUNA) after May 2022 | window 2022-05 max=116.409 min=4.81247e-05 min/max=0.0000 |
| FTT | True | FTT | ftx-token | 2019-08-01 | 2026-08-08 | 2565 | SURVIVOR | FTX collapse Nov 2022 | window 2022-11 max=26.1094 min=1.2558 min/max=0.0481 |
| BCC | False | None | None | None | None | 0 | MISSING | BCC ticker / BCH-forks era 2017–18 | None |
| BCH | True | BCH | bitcoin-cash | 2017-07-23 | 2026-08-08 | 3304 | SURVIVOR | Bitcoin Cash (BCHABC/BCHSV forks 2018) | window 2018-11 max=628.507 min=102.073 min/max=0.1624 |
| BSV | True | BCHSV | bitcoin-sv | 2018-11-09 | 2026-08-08 | 2830 | SURVIVOR | BCHSV fork Nov 2018 | window 2018-11 max=208.402 min=42.7513 min/max=0.2051 |
| SRM | False | None | None | None | None | 0 | MISSING | Serum / FTX complex 2022 | None |
| CEL | False | None | None | None | None | 0 | MISSING | Celsius bankruptcy 2022 | None |
| BTT | True | BTT | bittorrent-new | 2022-01-11 | 2026-08-08 | 1671 | SURVIVOR | BTT redenomination ~Jan 2022 | window 2022-01 max=2.86557e-06 min=2.05203e-06 min/max=0.7161 |
| XEM | False | None | None | None | None | 0 | MISSING | XEM secular decline | None |

## Historical top-200 sample (n=30, seed=42)

Drawn from the union of year-end mcap top-200 in 2018/2019/2020 (BTC, stables, wrapped excluded). In-file year-end pool sizes (survivors only): `{'2018': 78, '2019': 116, '2020': 185}`. A 2018–2020 top-200 that never appears in this archive cannot enter the sample — the 80% present test is nearly tautological; named-list misses are the real graveyard check.

| id | symbol | name | slug | first | last | n | gap_frac | terminal | in_years |
|----|--------|------|------|-------|------|---|----------|----------|----------|
| 1168 | DCR | Decred | decred | 2016-02-10 | 2026-08-08 | 3833 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 1437 | ZEC | Zcash | zcash | 2016-10-29 | 2026-08-08 | 3571 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 1698 | ZEN | Horizen | horizen | 2017-06-01 | 2026-08-08 | 3356 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 1785 | GAS | Gas | gas | 2017-07-06 | 2026-08-08 | 3321 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 1839 | BNB | BNB | bnb | 2017-07-25 | 2026-08-08 | 3302 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2011 | XTZ | Tezos | tezos | 2018-07-01 | 2026-08-08 | 2961 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2137 | ETN | Electroneum | electroneum | 2017-11-02 | 2026-08-08 | 3202 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2394 | TEL | Telcoin | telcoin | 2018-01-15 | 2026-08-08 | 3128 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2467 | TRAC | OriginTrail | origintrail | 2018-01-25 | 2026-08-08 | 3118 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2469 | ZIL | Zilliqa | zilliqa | 2018-01-25 | 2026-08-08 | 3118 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2572 | BAX | BABB | babb | 2018-03-09 | 2026-08-08 | 3075 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2606 | WAN | Wanchain | wanchain | 2018-03-23 | 2026-08-08 | 3061 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2868 | DAG | Constellation | constellation | 2018-06-20 | 2026-08-08 | 2972 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 2916 | NIM | Nimiq | nimiq | 2018-07-26 | 2026-08-08 | 2936 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 3077 | VET | VeChain | vechain | 2018-08-03 | 2026-08-08 | 2928 | 0.000 | SURVIVOR | 2018,2019,2020 |
| 4006 | AWE | AWE | awe-network | 2019-06-12 | 2026-08-08 | 2615 | 0.000 | SURVIVOR | 2019,2020 |
| 4039 | ARPA | ARPA | arpa-chain | 2019-07-15 | 2026-08-08 | 2582 | 0.000 | SURVIVOR | 2019,2020 |
| 4166 | RIO | Realio Network | realio-network | 2020-06-25 | 2026-08-08 | 2236 | 0.000 | SURVIVOR | 2020 |
| 4172 | LUNC | Terra Classic | terra-luna | 2019-07-26 | 2026-08-08 | 2571 | 0.000 | SURVIVOR | 2019,2020 |
| 4197 | SHR | ShareToken | sharetoken | 2019-11-26 | 2026-08-08 | 2448 | 0.000 | SURVIVOR | 2019,2020 |
| 4948 | CKB | Nervos Network | nervos-network | 2019-11-19 | 2026-08-08 | 2455 | 0.000 | SURVIVOR | 2019,2020 |
| 5266 | MLK | MiL.k | milk-alliance | 2020-08-05 | 2026-08-08 | 2195 | 0.000 | SURVIVOR | 2020 |
| 5444 | CTSI | Cartesi | cartesi | 2020-04-23 | 2026-08-08 | 2299 | 0.000 | SURVIVOR | 2020 |
| 5552 | HTR | Hathor | hathor | 2020-09-25 | 2026-08-08 | 2144 | 0.000 | SURVIVOR | 2020 |
| 5964 | TWT | Trust Wallet Token | trust-wallet-token | 2020-07-30 | 2026-08-08 | 2201 | 0.000 | SURVIVOR | 2020 |
| 6535 | NEAR | NEAR Protocol | near-protocol | 2020-10-14 | 2026-08-08 | 2125 | 0.000 | SURVIVOR | 2020 |
| 6783 | AXS | Axie Infinity | axie-infinity | 2020-11-04 | 2026-08-08 | 2104 | 0.000 | SURVIVOR | 2020 |
| 6892 | EGLD | MultiversX | multiversx-egld | 2020-09-04 | 2026-08-08 | 2165 | 0.000 | SURVIVOR | 2020 |
| 6958 | ACH | Alchemy Pay | alchemy-pay | 2020-09-09 | 2026-08-08 | 2160 | 0.000 | SURVIVOR | 2020 |
| 7083 | UNI | Uniswap | uniswap | 2020-09-17 | 2026-08-08 | 2152 | 0.000 | SURVIVOR | 2020 |

Graveyard one-liner input: **30/30 present** in the drawn sample (survivors=30, ended=0). Named-list misses are in the table above.

## PIT reconstruction

Trailing 30d median dollar volume, fallback mcap. Stables/wrapped excluded. Files: `universe/btcb_top50_pit.parquet`, `universe/btcb_top100_pit.parquet`.
PIT method: **trailing_30d_median_dollar_volume**. Dates with ≥50 ranked names: 3126 / 3873.
BTC is retained in the PIT file (typically rank 1) and excluded from alt picks; undeployed capital parks in BTC. Stables/wrapped excluded from ranking.

## Data quality

- Stables in panel: ['USDP']
- Wrapped in panel: ['WBTC']
- Duplicate tickers: []
- Redenomination suspects (|daily ret| > 5): n=37
- Gap fraction p50=0.000 p90=0.000 n(gap>5%)=0

### Agreement vs Binance daily closes (overlapping liquid coins)

Median return correlation = **0.9959** (flag suspect if < 0.99: **False**). Compared n=20.

| symbol | n | corr | max\|Δr\| | mean\|Δr\| |
|--------|---|------|---------|----------|
| BTC | 2403 | 0.9960 | 0.0498 | 0.00130 |
| ETH | 2403 | 0.9933 | 0.1115 | 0.00181 |
| BNB | 2363 | 0.9971 | 0.0528 | 0.00163 |
| XRP | 2393 | 0.9967 | 0.0613 | 0.00218 |
| ADA | 2373 | 0.9969 | 0.0591 | 0.00225 |
| DOGE | 2212 | 0.9979 | 0.3423 | 0.00224 |
| SOL | 2141 | 0.9927 | 0.2372 | 0.00230 |
| DOT | 2169 | 0.9926 | 0.1888 | 0.00202 |
| LTC | 2390 | 0.9959 | 0.0590 | 0.00216 |
| LINK | 2386 | 0.9959 | 0.1118 | 0.00243 |
| BCH | 2403 | 0.9950 | 0.1046 | 0.00228 |
| ATOM | 2366 | 0.9964 | 0.0860 | 0.00241 |
| AVAX | 2137 | 0.9953 | 0.2042 | 0.00223 |
| ETC | 2388 | 0.9965 | 0.0921 | 0.00217 |
| FIL | 2109 | 0.9881 | 0.1914 | 0.00243 |
| NEAR | 2110 | 0.9962 | 0.1292 | 0.00239 |
| UNI | 2142 | 0.9932 | 0.1025 | 0.00266 |
| AAVE | 2114 | 0.9956 | 0.0694 | 0.00261 |
| TRX | 2384 | 0.9962 | 0.0618 | 0.00185 |
| XLM | 2379 | 0.9961 | 0.0633 | 0.00254 |

## Usable-start scan (month starts)

| D | sample_ok/n | sample_frac | PIT frac | pass |
|---|-------------|-------------|----------|------|
| 2018-01 | 30/30 | 100.0% | 99.5% | True |
| 2018-02 | 30/30 | 100.0% | 100.0% | True |
| 2018-03 | 30/30 | 100.0% | 100.0% | True |
| 2018-04 | 30/30 | 100.0% | 100.0% | True |
| 2018-05 | 30/30 | 100.0% | 100.0% | True |
| 2018-06 | 30/30 | 100.0% | 100.0% | True |
| 2018-07 | 30/30 | 100.0% | 100.0% | True |
| 2018-08 | 30/30 | 100.0% | 100.0% | True |
| 2018-09 | 30/30 | 100.0% | 100.0% | True |
| 2018-10 | 30/30 | 100.0% | 100.0% | True |
| 2018-11 | 30/30 | 100.0% | 100.0% | True |
| 2018-12 | 30/30 | 100.0% | 100.0% | True |
| 2019-01 | 30/30 | 100.0% | 100.0% | True |
| 2019-02 | 30/30 | 100.0% | 100.0% | True |
| 2019-03 | 30/30 | 100.0% | 100.0% | True |
| 2019-04 | 30/30 | 100.0% | 100.0% | True |
| 2019-05 | 30/30 | 100.0% | 100.0% | True |
| 2019-06 | 30/30 | 100.0% | 100.0% | True |
| 2019-07 | 30/30 | 100.0% | 100.0% | True |
| 2019-08 | 30/30 | 100.0% | 100.0% | True |
| 2019-09 | 30/30 | 100.0% | 100.0% | True |
| 2019-10 | 30/30 | 100.0% | 100.0% | True |
| 2019-11 | 30/30 | 100.0% | 100.0% | True |
| 2019-12 | 30/30 | 100.0% | 100.0% | True |
| 2020-01 | 30/30 | 100.0% | 100.0% | True |
| 2020-02 | 30/30 | 100.0% | 100.0% | True |
| 2020-03 | 30/30 | 100.0% | 100.0% | True |
| 2020-04 | 30/30 | 100.0% | 100.0% | True |
| 2020-05 | 30/30 | 100.0% | 100.0% | True |
| 2020-06 | 30/30 | 100.0% | 100.0% | True |
| 2020-07 | 30/30 | 100.0% | 100.0% | True |
| 2020-08 | 30/30 | 100.0% | 100.0% | True |
| 2020-09 | 30/30 | 100.0% | 100.0% | True |
| 2020-10 | 30/30 | 100.0% | 100.0% | True |
| 2020-11 | 30/30 | 100.0% | 100.0% | True |
| 2020-12 | 30/30 | 100.0% | 100.0% | True |
| 2021-01 | 30/30 | 100.0% | 100.0% | True |
| 2021-02 | 30/30 | 100.0% | 100.0% | True |
| 2021-03 | 30/30 | 100.0% | 100.0% | True |
| 2021-04 | 30/30 | 100.0% | 100.0% | True |
| 2021-05 | 30/30 | 100.0% | 100.0% | True |
| 2021-06 | 30/30 | 100.0% | 100.0% | True |
| 2021-07 | 30/30 | 100.0% | 100.0% | True |
| 2021-08 | 30/30 | 100.0% | 100.0% | True |
| 2021-09 | 30/30 | 100.0% | 100.0% | True |
| 2021-10 | 30/30 | 100.0% | 100.0% | True |
| 2021-11 | 30/30 | 100.0% | 100.0% | True |
| 2021-12 | 30/30 | 100.0% | 100.0% | True |
| 2022-01 | 30/30 | 100.0% | 100.0% | True |
| 2022-02 | 30/30 | 100.0% | 100.0% | True |
| 2022-03 | 30/30 | 100.0% | 100.0% | True |
| 2022-04 | 30/30 | 100.0% | 100.0% | True |
| 2022-05 | 30/30 | 100.0% | 100.0% | True |
| 2022-06 | 30/30 | 100.0% | 100.0% | True |
| 2022-07 | 30/30 | 100.0% | 100.0% | True |
| 2022-08 | 30/30 | 100.0% | 100.0% | True |
| 2022-09 | 30/30 | 100.0% | 100.0% | True |
| 2022-10 | 30/30 | 100.0% | 100.0% | True |
| 2022-11 | 30/30 | 100.0% | 100.0% | True |
| 2022-12 | 30/30 | 100.0% | 100.0% | True |
| 2023-01 | 30/30 | 100.0% | 100.0% | True |

Elapsed s=26.6. GPU=false. COMBO untouched.

