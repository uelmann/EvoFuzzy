# Round F — context, complexity, pruning, two-sleeve combo

- Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
- Scope: backtest only; zero GPU; causal (training-window) τ house standard.
- Ledger: `reports/numbers_ledger.md`
- Addendum (criteria frozen before results): `reports/roundF_addendum.md`
- Hurst: single-scale R/S, H = log(R/S)/log(n) on the 90d residual window.

**Mechanical verdicts:** F1 (context) KILL top-20 / **KEEP top-40**. F2 (complexity) KILL both universes. F3 (both) KILL both universes. F4 (pruning) KILL top-20 / **KEEP top-40**. **COMBO ADOPTED** as the reference book. Ledger: causal τ only.

## Pre-registered KEEP criterion (verbatim, before results)

> Block X is KEPT on universe U only if trailing-18m ΔRankIC on U ≥ +0.005 at h=7 or h=10 AND full-OOS ΔRankIC on U ≥ 0 AND Δ positive in ≥60% of trailing-18m folds on U AND the corresponding portfolio trailing-18m net Sharpe Δ on U ≥ 0. F4 (pruning) uses the same criterion with thresholds ΔRankIC ≥ 0 (trailing) and ≥ −0.002 (full): pruning is KEPT if it does not hurt. Verdicts per-universe, mechanical, no post-hoc adjustment.

## Pre-registered COMBO criterion (verbatim, before results)

> COMBO is ADOPTED as the reference book only if its trailing-18m net Sharpe ≥ max(P1, P2 trailing) − 0.10 AND its full-period net Sharpe ≥ max(P1, P2 full) − 0.10. Otherwise the adopted book remains P2 with P1 as reference.

## Gates

- `label_shuffle`: **PASS**
- `feature_lookahead`: **PASS**
- `universe_lookahead_top20`: **PASS**
- `universe_lookahead_top40`: **PASS**
- `universe_lookahead_top120`: **PASS**
- `seed_determinism`: **PASS**

## F4 pruned features (8 lowest mean A0 gain)

A0 metas used: 80

| rank | feature | mean gain |
|------|---------|-----------|
| 1 | `rev_1` | 3.74 |
| 2 | `rev_3` | 8.40 |
| 3 | `dv_z_30` | 10.38 |
| 4 | `dv_trend` | 12.73 |
| 5 | `ret_28` | 13.98 |
| 6 | `skew_28` | 14.29 |
| 7 | `mom_28_skip7` | 14.37 |
| 8 | `ret_7` | 14.98 |

All A0 features by rising mean gain:

`rev_1` (3.7), `rev_3` (8.4), `dv_z_30` (10.4), `dv_trend` (12.7), `ret_28` (14.0), `skew_28` (14.3), `mom_28_skip7` (14.4), `ret_7` (15.0), `ret_14` (15.0), `ret_90` (15.4), `sma20_sma50` (16.2), `yz_vol_14` (16.7), `close_sma50` (18.1), `ret_56` (18.1), `yz_vol_60` (18.6), `ema12_ema26` (19.1), `vol_ratio` (19.8), `idio_vol_60` (20.7), `dist_high_90` (20.8), `min_ret_14` (21.5), `close_sma100` (21.6), `yz_vol_30` (21.9), `amihud_14` (21.9), `max_ret_14` (21.9), `skew_60` (22.4), `pk_vol_14` (23.7), `mom_90_skip14` (23.7), `close_sma20` (28.7), `vol_of_vol_30` (30.1), `beta_btc_60` (31.4), `dist_low_90` (32.7), `corr_btc_28` (34.7), `range_pos_28` (46.9)

## Ablation RankIC vs A0

| block | h | universe | window | A0 IC | F IC | ΔIC | n_days |
|-------|---|----------|--------|-------|------|-----|--------|
| F1 | 7 | top20 | full | 0.0923 | 0.1297 | 0.0374 | 875 |
| F1 | 7 | top20 | trail18m | 0.0814 | 0.1722 | 0.0907 | 238 |
| F1 | 7 | top20 | y2022 | 0.0696 | -0.0066 | -0.0763 | 103 |
| F1 | 7 | top20 | y2023 | 0.0946 | 0.0965 | 0.0019 | 350 |
| F1 | 7 | top20 | y2024 | 0.1236 | 0.2013 | 0.0777 | 190 |
| F1 | 7 | top20 | y2025 | 0.1028 | 0.2270 | 0.1242 | 157 |
| F1 | 7 | top20 | y2026 | 0.0454 | 0.0864 | 0.0410 | 75 |
| F1 | 10 | top20 | full | 0.1010 | 0.1351 | 0.0342 | 1620 |
| F1 | 10 | top20 | trail18m | 0.0656 | 0.1237 | 0.0582 | 548 |
| F1 | 10 | top20 | y2022 | 0.1343 | 0.1079 | -0.0264 | 344 |
| F1 | 10 | top20 | y2023 | 0.0949 | 0.1474 | 0.0525 | 365 |
| F1 | 10 | top20 | y2024 | 0.1304 | 0.1651 | 0.0347 | 366 |
| F1 | 10 | top20 | y2025 | 0.0934 | 0.1907 | 0.0973 | 365 |
| F1 | 10 | top20 | y2026 | 0.0050 | -0.0111 | -0.0162 | 180 |
| F1 | 7 | top40 | full | 0.0792 | 0.1128 | 0.0336 | 898 |
| F1 | 7 | top40 | trail18m | 0.0921 | 0.1692 | 0.0772 | 259 |
| F1 | 7 | top40 | y2022 | 0.0590 | -0.0475 | -0.1064 | 104 |
| F1 | 7 | top40 | y2023 | 0.0739 | 0.0783 | 0.0044 | 351 |
| F1 | 7 | top40 | y2024 | 0.0798 | 0.1850 | 0.1052 | 190 |
| F1 | 7 | top40 | y2025 | 0.0964 | 0.1776 | 0.0812 | 178 |
| F1 | 7 | top40 | y2026 | 0.0932 | 0.1597 | 0.0665 | 75 |
| F1 | 10 | top40 | full | 0.0811 | 0.1127 | 0.0316 | 1620 |
| F1 | 10 | top40 | trail18m | 0.0943 | 0.1260 | 0.0317 | 548 |
| F1 | 10 | top40 | y2022 | 0.0956 | 0.0707 | -0.0249 | 344 |
| F1 | 10 | top40 | y2023 | 0.0598 | 0.1140 | 0.0542 | 365 |
| F1 | 10 | top40 | y2024 | 0.0686 | 0.1307 | 0.0621 | 366 |
| F1 | 10 | top40 | y2025 | 0.1064 | 0.1629 | 0.0566 | 365 |
| F1 | 10 | top40 | y2026 | 0.0707 | 0.0517 | -0.0190 | 180 |
| F2 | 7 | top20 | full | 0.0923 | 0.0534 | -0.0388 | 1620 |
| F2 | 7 | top20 | trail18m | 0.0814 | 0.0924 | 0.0110 | 548 |
| F2 | 7 | top20 | y2022 | 0.0696 | 0.0308 | -0.0389 | 347 |
| F2 | 7 | top20 | y2023 | 0.0946 | 0.0453 | -0.0494 | 365 |
| F2 | 7 | top20 | y2024 | 0.1236 | 0.0208 | -0.1028 | 366 |
| F2 | 7 | top20 | y2025 | 0.1028 | 0.1318 | 0.0290 | 365 |
| F2 | 7 | top20 | y2026 | 0.0454 | 0.0207 | -0.0247 | 177 |
| F2 | 10 | top20 | full | 0.1010 | 0.0918 | -0.0091 | 1620 |
| F2 | 10 | top20 | trail18m | 0.0656 | 0.0767 | 0.0112 | 548 |
| F2 | 10 | top20 | y2022 | 0.1343 | 0.0946 | -0.0397 | 344 |
| F2 | 10 | top20 | y2023 | 0.0949 | 0.0814 | -0.0135 | 365 |
| F2 | 10 | top20 | y2024 | 0.1304 | 0.1229 | -0.0074 | 366 |
| F2 | 10 | top20 | y2025 | 0.0934 | 0.1023 | 0.0089 | 365 |
| F2 | 10 | top20 | y2026 | 0.0050 | 0.0233 | 0.0182 | 180 |
| F2 | 7 | top40 | full | 0.0792 | 0.0598 | -0.0194 | 1620 |
| F2 | 7 | top40 | trail18m | 0.0921 | 0.1056 | 0.0135 | 548 |
| F2 | 7 | top40 | y2022 | 0.0590 | 0.0519 | -0.0071 | 347 |
| F2 | 7 | top40 | y2023 | 0.0739 | 0.0218 | -0.0521 | 365 |
| F2 | 7 | top40 | y2024 | 0.0798 | 0.0356 | -0.0442 | 366 |
| F2 | 7 | top40 | y2025 | 0.0964 | 0.1203 | 0.0240 | 365 |
| F2 | 7 | top40 | y2026 | 0.0932 | 0.0790 | -0.0141 | 177 |
| F2 | 10 | top40 | full | 0.0811 | 0.0770 | -0.0041 | 1620 |
| F2 | 10 | top40 | trail18m | 0.0943 | 0.0905 | -0.0037 | 548 |
| F2 | 10 | top40 | y2022 | 0.0956 | 0.0734 | -0.0222 | 344 |
| F2 | 10 | top40 | y2023 | 0.0598 | 0.0676 | 0.0078 | 365 |
| F2 | 10 | top40 | y2024 | 0.0686 | 0.0704 | 0.0018 | 366 |
| F2 | 10 | top40 | y2025 | 0.1064 | 0.1004 | -0.0060 | 365 |
| F2 | 10 | top40 | y2026 | 0.0707 | 0.0686 | -0.0021 | 180 |
| F3 | 7 | top20 | full | 0.0923 | 0.0978 | 0.0055 | 1322 |
| F3 | 7 | top20 | trail18m | 0.0814 | 0.1090 | 0.0276 | 526 |
| F3 | 7 | top20 | y2022 | 0.0696 | 0.0474 | -0.0222 | 174 |
| F3 | 7 | top20 | y2023 | 0.0946 | 0.1140 | 0.0193 | 352 |
| F3 | 7 | top20 | y2024 | 0.1236 | 0.0804 | -0.0432 | 276 |
| F3 | 7 | top20 | y2025 | 0.1028 | 0.1450 | 0.0422 | 346 |
| F3 | 7 | top20 | y2026 | 0.0454 | 0.0490 | 0.0036 | 174 |
| F3 | 10 | top20 | full | 0.1010 | 0.1182 | 0.0173 | 1620 |
| F3 | 10 | top20 | trail18m | 0.0656 | 0.1206 | 0.0550 | 548 |
| F3 | 10 | top20 | y2022 | 0.1343 | 0.0907 | -0.0435 | 344 |
| F3 | 10 | top20 | y2023 | 0.0949 | 0.1036 | 0.0087 | 365 |
| F3 | 10 | top20 | y2024 | 0.1304 | 0.1553 | 0.0249 | 366 |
| F3 | 10 | top20 | y2025 | 0.0934 | 0.1667 | 0.0733 | 365 |
| F3 | 10 | top20 | y2026 | 0.0050 | 0.0267 | 0.0217 | 180 |
| F3 | 7 | top40 | full | 0.0792 | 0.0924 | 0.0132 | 1322 |
| F3 | 7 | top40 | trail18m | 0.0921 | 0.1121 | 0.0200 | 526 |
| F3 | 7 | top40 | y2022 | 0.0590 | 0.0213 | -0.0377 | 174 |
| F3 | 7 | top40 | y2023 | 0.0739 | 0.0992 | 0.0253 | 352 |
| F3 | 7 | top40 | y2024 | 0.0798 | 0.0916 | 0.0118 | 276 |
| F3 | 7 | top40 | y2025 | 0.0964 | 0.1258 | 0.0294 | 346 |
| F3 | 7 | top40 | y2026 | 0.0932 | 0.0845 | -0.0087 | 174 |
| F3 | 10 | top40 | full | 0.0811 | 0.1030 | 0.0219 | 1620 |
| F3 | 10 | top40 | trail18m | 0.0943 | 0.1213 | 0.0270 | 548 |
| F3 | 10 | top40 | y2022 | 0.0956 | 0.0674 | -0.0282 | 344 |
| F3 | 10 | top40 | y2023 | 0.0598 | 0.0932 | 0.0334 | 365 |
| F3 | 10 | top40 | y2024 | 0.0686 | 0.1201 | 0.0515 | 366 |
| F3 | 10 | top40 | y2025 | 0.1064 | 0.1429 | 0.0365 | 365 |
| F3 | 10 | top40 | y2026 | 0.0707 | 0.0756 | 0.0049 | 180 |
| F4 | 7 | top20 | full | 0.0923 | 0.0915 | -0.0008 | 1620 |
| F4 | 7 | top20 | trail18m | 0.0814 | 0.1028 | 0.0214 | 548 |
| F4 | 7 | top20 | y2022 | 0.0696 | 0.0647 | -0.0049 | 347 |
| F4 | 7 | top20 | y2023 | 0.0946 | 0.0704 | -0.0242 | 365 |
| F4 | 7 | top20 | y2024 | 0.1236 | 0.1185 | -0.0051 | 366 |
| F4 | 7 | top20 | y2025 | 0.1028 | 0.1338 | 0.0310 | 365 |
| F4 | 7 | top20 | y2026 | 0.0454 | 0.0445 | -0.0009 | 177 |
| F4 | 10 | top20 | full | 0.1010 | 0.0960 | -0.0049 | 1620 |
| F4 | 10 | top20 | trail18m | 0.0656 | 0.0808 | 0.0153 | 548 |
| F4 | 10 | top20 | y2022 | 0.1343 | 0.1024 | -0.0319 | 344 |
| F4 | 10 | top20 | y2023 | 0.0949 | 0.0967 | 0.0018 | 365 |
| F4 | 10 | top20 | y2024 | 0.1304 | 0.1134 | -0.0170 | 366 |
| F4 | 10 | top20 | y2025 | 0.0934 | 0.1115 | 0.0181 | 365 |
| F4 | 10 | top20 | y2026 | 0.0050 | 0.0158 | 0.0108 | 180 |
| F4 | 7 | top40 | full | 0.0792 | 0.0853 | 0.0061 | 1620 |
| F4 | 7 | top40 | trail18m | 0.0921 | 0.1096 | 0.0175 | 548 |
| F4 | 7 | top40 | y2022 | 0.0590 | 0.0624 | 0.0035 | 347 |
| F4 | 7 | top40 | y2023 | 0.0739 | 0.0703 | -0.0036 | 365 |
| F4 | 7 | top40 | y2024 | 0.0798 | 0.0824 | 0.0027 | 366 |
| F4 | 7 | top40 | y2025 | 0.0964 | 0.1228 | 0.0264 | 365 |
| F4 | 7 | top40 | y2026 | 0.0932 | 0.0898 | -0.0034 | 177 |
| F4 | 10 | top40 | full | 0.0811 | 0.0777 | -0.0034 | 1620 |
| F4 | 10 | top40 | trail18m | 0.0943 | 0.1006 | 0.0063 | 548 |
| F4 | 10 | top40 | y2022 | 0.0956 | 0.0737 | -0.0219 | 344 |
| F4 | 10 | top40 | y2023 | 0.0598 | 0.0603 | 0.0005 | 365 |
| F4 | 10 | top40 | y2024 | 0.0686 | 0.0646 | -0.0041 | 366 |
| F4 | 10 | top40 | y2025 | 0.1064 | 0.1145 | 0.0082 | 365 |
| F4 | 10 | top40 | y2026 | 0.0707 | 0.0732 | 0.0024 | 180 |

### Paired NW t and fold fraction

| block | h | universe | window | mean ΔIC | NW-t | n | frac+ trail18m folds |
|-------|---|----------|--------|----------|------|---|----------------------|
| F1 | 7 | top20 | full | 0.0140 | 0.64 | 875 | nan |
| F1 | 7 | top20 | trail18m | 0.0867 | 2.11 | 238 | 0.833 |
| F1 | 10 | top20 | full | 0.0342 | 2.55 | 1620 | nan |
| F1 | 10 | top20 | trail18m | 0.0582 | 2.21 | 548 | 0.714 |
| F1 | 7 | top40 | full | 0.0177 | 1.11 | 898 | nan |
| F1 | 7 | top40 | trail18m | 0.0670 | 2.38 | 259 | 0.833 |
| F1 | 10 | top40 | full | 0.0316 | 3.27 | 1620 | nan |
| F1 | 10 | top40 | trail18m | 0.0317 | 1.85 | 548 | 0.714 |
| F2 | 7 | top20 | full | -0.0388 | -2.74 | 1620 | nan |
| F2 | 7 | top20 | trail18m | 0.0110 | 0.42 | 548 | 0.571 |
| F2 | 10 | top20 | full | -0.0091 | -0.64 | 1620 | nan |
| F2 | 10 | top20 | trail18m | 0.0112 | 0.41 | 548 | 0.429 |
| F2 | 7 | top40 | full | -0.0194 | -1.93 | 1620 | nan |
| F2 | 7 | top40 | trail18m | 0.0135 | 0.68 | 548 | 0.571 |
| F2 | 10 | top40 | full | -0.0041 | -0.42 | 1620 | nan |
| F2 | 10 | top40 | trail18m | -0.0037 | -0.21 | 548 | 0.571 |
| F3 | 7 | top20 | full | 0.0030 | 0.17 | 1322 | nan |
| F3 | 7 | top20 | trail18m | 0.0269 | 0.97 | 526 | 0.714 |
| F3 | 10 | top20 | full | 0.0173 | 1.14 | 1620 | nan |
| F3 | 10 | top20 | trail18m | 0.0550 | 2.10 | 548 | 0.714 |
| F3 | 7 | top40 | full | 0.0051 | 0.41 | 1322 | nan |
| F3 | 7 | top40 | trail18m | 0.0155 | 0.80 | 526 | 0.857 |
| F3 | 10 | top40 | full | 0.0219 | 1.98 | 1620 | nan |
| F3 | 10 | top40 | trail18m | 0.0270 | 1.61 | 548 | 0.714 |
| F4 | 7 | top20 | full | -0.0008 | -0.08 | 1620 | nan |
| F4 | 7 | top20 | trail18m | 0.0214 | 1.30 | 548 | 0.571 |
| F4 | 10 | top20 | full | -0.0049 | -0.82 | 1620 | nan |
| F4 | 10 | top20 | trail18m | 0.0153 | 1.48 | 548 | 0.571 |
| F4 | 7 | top40 | full | 0.0061 | 0.94 | 1620 | nan |
| F4 | 7 | top40 | trail18m | 0.0175 | 1.28 | 548 | 0.857 |
| F4 | 10 | top40 | full | -0.0034 | -0.82 | 1620 | nan |
| F4 | 10 | top40 | trail18m | 0.0063 | 0.88 | 548 | 0.429 |

## Portfolio Δ (causal median-τ) on adopted books

P1 book = A0 top-20 h=7; P2 book = A0 top-40 h=10 (tiered costs + ADV cap).

| block | book | τ_A0 | τ_F | A0 Sharpe full | F full | A0 trail18m | F trail18m | Δ trail18m |
|-------|------|------|-----|----------------|--------|-------------|------------|------------|
| F1 | P1 top20 h=7 | 80.0 | 70.0 | 1.207 | 0.558 | 1.009 | 0.689 | -0.320 |
| F1 | P2 top40 h=10 | 70.0 | 70.0 | 1.470 | 1.257 | 0.723 | 1.045 | 0.322 |
| F2 | P1 top20 h=7 | 80.0 | 90.0 | 1.207 | 0.195 | 1.009 | -0.047 | -1.057 |
| F2 | P2 top40 h=10 | 70.0 | 60.0 | 1.470 | 1.267 | 0.723 | 0.703 | -0.020 |
| F3 | P1 top20 h=7 | 80.0 | 70.0 | 1.207 | 0.384 | 1.009 | -0.173 | -1.182 |
| F3 | P2 top40 h=10 | 70.0 | 70.0 | 1.470 | 0.469 | 0.723 | 0.023 | -0.701 |
| F4 | P1 top20 h=7 | 80.0 | 80.0 | 1.207 | 1.343 | 1.009 | 0.642 | -0.368 |
| F4 | P2 top40 h=10 | 70.0 | 60.0 | 1.470 | 1.314 | 0.723 | 1.120 | 0.396 |

Per-year net Sharpe Δ (F − A0) on identical days:

| block | book | 2022 | 2023 | 2024 | 2025 | 2026 |
|-------|------|------|------|------|------|------|
| F1 | P1 top20 h=7 | 0.871 | -3.230 | -0.098 | -1.339 | 1.247 |
| F1 | P2 top40 h=10 | -1.256 | 0.044 | -1.040 | -0.074 | 1.049 |
| F2 | P1 top20 h=7 | 1.284 | -1.848 | -1.500 | -1.154 | -1.825 |
| F2 | P2 top40 h=10 | -0.910 | 0.513 | -1.537 | 0.492 | -0.531 |
| F3 | P1 top20 h=7 | -0.388 | -2.350 | 0.556 | -1.233 | -1.361 |
| F3 | P2 top40 h=10 | -1.421 | -1.369 | -0.959 | -0.756 | -0.480 |
| F4 | P1 top20 h=7 | 2.703 | -0.674 | 0.067 | -0.774 | 0.422 |
| F4 | P2 top40 h=10 | -1.709 | 0.160 | -0.916 | 0.773 | 0.057 |

## Mechanical KEEP verdicts

> Block X is KEPT on universe U only if trailing-18m ΔRankIC on U ≥ +0.005 at h=7 or h=10 AND full-OOS ΔRankIC on U ≥ 0 AND Δ positive in ≥60% of trailing-18m folds on U AND the corresponding portfolio trailing-18m net Sharpe Δ on U ≥ 0. F4 (pruning) uses the same criterion with thresholds ΔRankIC ≥ 0 (trailing) and ≥ −0.002 (full): pruning is KEPT if it does not hurt. Verdicts per-universe, mechanical, no post-hoc adjustment.

### F1

- **top20: KILL** (IC any-h=True, port Δtrail18m=-0.320, port_ok=False)
- **top40: KEEP** (IC any-h=True, port Δtrail18m=0.322, port_ok=True)

| universe | h | ΔIC trail18m | ΔIC full | frac+ folds | IC pass |
|----------|---|--------------|----------|-------------|---------|
| top20 | 7 | 0.0907 | 0.0374 | 0.833 | True |
| top20 | 10 | 0.0582 | 0.0342 | 0.714 | True |
| top40 | 7 | 0.0772 | 0.0336 | 0.833 | True |
| top40 | 10 | 0.0317 | 0.0316 | 0.714 | True |

### F2

- **top20: KILL** (IC any-h=False, port Δtrail18m=-1.057, port_ok=False)
- **top40: KILL** (IC any-h=False, port Δtrail18m=-0.020, port_ok=False)

| universe | h | ΔIC trail18m | ΔIC full | frac+ folds | IC pass |
|----------|---|--------------|----------|-------------|---------|
| top20 | 7 | 0.0110 | -0.0388 | 0.571 | False |
| top20 | 10 | 0.0112 | -0.0091 | 0.429 | False |
| top40 | 7 | 0.0135 | -0.0194 | 0.571 | False |
| top40 | 10 | -0.0037 | -0.0041 | 0.571 | False |

### F3

- **top20: KILL** (IC any-h=True, port Δtrail18m=-1.182, port_ok=False)
- **top40: KILL** (IC any-h=True, port Δtrail18m=-0.701, port_ok=False)

| universe | h | ΔIC trail18m | ΔIC full | frac+ folds | IC pass |
|----------|---|--------------|----------|-------------|---------|
| top20 | 7 | 0.0276 | 0.0055 | 0.714 | True |
| top20 | 10 | 0.0550 | 0.0173 | 0.714 | True |
| top40 | 7 | 0.0200 | 0.0132 | 0.857 | True |
| top40 | 10 | 0.0270 | 0.0219 | 0.714 | True |

### F4

- **top20: KILL** (IC any-h=False, port Δtrail18m=-0.368, port_ok=False)
- **top40: KEEP** (IC any-h=True, port Δtrail18m=0.396, port_ok=True)

| universe | h | ΔIC trail18m | ΔIC full | frac+ folds | IC pass |
|----------|---|--------------|----------|-------------|---------|
| top20 | 7 | 0.0214 | -0.0008 | 0.571 | False |
| top20 | 10 | 0.0153 | -0.0049 | 0.571 | False |
| top40 | 7 | 0.0175 | 0.0061 | 0.857 | True |
| top40 | 10 | 0.0063 | -0.0034 | 0.429 | False |

## COMBO 50/50 P1+P2

> COMBO is ADOPTED as the reference book only if its trailing-18m net Sharpe ≥ max(P1, P2 trailing) − 0.10 AND its full-period net Sharpe ≥ max(P1, P2 full) − 0.10. Otherwise the adopted book remains P2 with P1 as reference.

**COMBO verdict: ADOPTED**

need trail18m ≥ 0.909 (max P=1.009−0.10); need full ≥ 1.377 (max P=1.477−0.10).

| book | full | trail18m | 2022 | 2023 | 2024 | 2025 | 2026 | MaxDD | corr sleeves | ann to |
|------|------|----------|------|------|------|------|------|-------|--------------|--------|
| COMBO | 1.711 | 0.997 | 1.097 | 2.544 | 2.843 | 1.444 | 0.441 | -0.332 | 0.254 | 26.48 |
| P1 | 1.216 | 1.009 | -0.329 | 2.755 | 1.391 | 1.148 | 0.721 |  |  |  |
| P2 | 1.477 | 0.733 | 2.317 | 1.547 | 3.445 | 1.216 | 0.268 |  |  |  |

Ledger confirmation: all Round F portfolios used `tau_mode=fold_train` (causal).

## Operational notes

- Hurst is **single-scale R/S** (`H = log(R/S) / log(n)`), not DFA. catch22 + extras use trailing 90d residual log-returns, min 60 observations else NaN; then CS-z per date, clip ±5.
- Volume caches: `/data/quant/round_f/{context,residuals,complexity,features_round_f}.parquet` plus `cx_sym/*.parquet` and F1–F4 prediction parquets.
- F1/F3 h=7 RankIC `n_days` is the overlap of days with a finite daily RankIC in both A0 and F (875 / 1322 vs 1620). h=10 is complete (1620). F1 top-40 KEEP also holds on the complete h=10 series (trail ΔIC +0.0317, full +0.0316, frac+ 0.714).
- Zero GPU. No schedules, no live components. Frozen A0 hash verified before any ablation number.

