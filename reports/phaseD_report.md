# Phase D — decay diagnostic + microstructure ablation

- Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`
- Scope: backtest/analysis only; no schedules or live components.

## 1. Decay diagnostic (frozen A0, tranche h=7, τ=60, funding on, no gate)

Gross alpha proxy formula: `PROXY_Y = mean_daily_top20_RankIC_Y * mean_daily_CS_std(y_h7)_Y * avg_n_positions_full`

| year | RankIC | dispersion | score_disp | avg_npos | %nonempty | proxy | gross | cost | cost_share | funding | fund_share | net | Sharpe |
|------|--------|------------|------------|----------|-----------|-------|-------|------|------------|---------|------------|-----|--------|
| 2022 | 0.0696 | 0.0797 | 0.0211 | 9.62 | 0.77 | 0.05334 | -0.0424 | 0.0212 | 0.500 | 0.0067 | 0.158 | -0.0028 | -0.019 |
| 2023 | 0.0946 | 0.0785 | 0.0067 | 9.62 | 0.63 | 0.07150 | 0.4216 | 0.0160 | 0.038 | -0.0863 | 0.205 | 0.3982 | 2.145 |
| 2024 | 0.1236 | 0.0923 | 0.0036 | 9.62 | 0.76 | 0.10978 | 0.5274 | 0.0187 | 0.035 | -0.0101 | 0.019 | 0.7688 | 2.355 |
| 2025 | 0.1028 | 0.0845 | 0.0030 | 9.62 | 1.00 | 0.08355 | 0.4364 | 0.0153 | 0.035 | -0.0423 | 0.097 | 0.3298 | 1.319 |
| 2026 | 0.0454 | 0.1310 | 0.0050 | 9.62 | 1.00 | 0.05722 | 0.1772 | 0.0042 | 0.024 | -0.0178 | 0.100 | -0.0580 | -0.815 |

Proxy↔gross corr (years): **0.858**

**Diagnostic verdict: IC_DECAY**

Selected IC_DECAY: ic_drop=0.32, disp_drop=-0.26, fric_rise=-0.02; IC 0.109→0.074, disp 0.0854→0.1078, Sharpe 2.25→0.25, proxy↔gross corr=0.8581977263206795.

## 2. Microstructure coverage

| source/field | n_sym | cov_sym | min_date | max_date | n_rows | note |
|--------------|-------|---------|----------|----------|--------|------|
| funding_rate | 665 | 1.000 | 2020-01-01 | 2026-07-31 | 562436 |  |
| premium_close | 662 | 0.995 | 2020-01-02 | 2026-07-31 | 555886 |  |
| sum_open_interest | 662 | 0.995 | 2020-09-01 | 2026-07-31 | 204223 |  |
| count_long_short_ratio | 662 | 0.995 | 2020-09-01 | 2026-07-31 | 201730 |  |
| sum_taker_long_short_vol_ratio | 662 | 0.995 | 2020-09-01 | 2026-07-31 | 188964 |  |
| liquidationSnapshot_um | None | nan | None | None | None | data/futures/um/*/liquidationSnapshot/ is empty on data.binance.vision; CM liquidations are not used. liq_imb_1/liq_imb_7 remain NaN. |

NaN handling: unavailable fields left as NaN (no zero-imputation); LightGBM native NaN.

## 3. Microstructure feature block (12 features)

Per (symbol, date), data ≤ close of t only, then cross-sectional z-score per date, clip ±5:

- `funding_now`, `funding_z_30`, `funding_cum_7`, `funding_cs_rank`
- `basis_z_30`
- `oi_chg_1`, `oi_chg_7`, `oi_turnover`
- `liq_imb_1`, `liq_imb_7` (always NaN — UM liquidationSnapshot absent on Vision)
- `taker_imb_z`, `ls_ratio_z`

## 4. Ablation criterion (pre-registered)

> The microstructure block is KEPT only if trailing-18-month top-20 ΔRankIC ≥ +0.005 at h=7 or h=10 AND full-OOS ΔRankIC ≥ 0 AND Δ is positive in ≥60% of trailing-18-month folds. Otherwise verdict = KILL.

**Ablation verdict: KILL**

Details: `{'h7': {'delta_trail18m': -0.005185616639392668, 'delta_full': -0.048218920515006924, 'frac_pos_folds_trail18m': 0.2857142857142857, 'passes': False}, 'h10': {'delta_trail18m': 0.008083274916542704, 'delta_full': -0.03663645471383399, 'frac_pos_folds_trail18m': 0.7142857142857143, 'passes': False}}`

## Ablation tables (A = A0, D = A0+micro)

| h | universe | window | A IC | D IC | ΔIC | n_days |
|---|----------|--------|------|------|-----|--------|
| 7 | top20 | full | 0.0923 | 0.0441 | -0.0482 | 1620 |
| 7 | top20 | trail18m | 0.0814 | 0.0762 | -0.0052 | 548 |
| 7 | top20 | y2022 | 0.0696 | 0.0377 | -0.0319 | 347 |
| 7 | top20 | y2023 | 0.0946 | -0.0206 | -0.1153 | 365 |
| 7 | top20 | y2024 | 0.1236 | 0.0608 | -0.0628 | 366 |
| 7 | top20 | y2025 | 0.1028 | 0.1023 | -0.0004 | 365 |
| 7 | top20 | y2026 | 0.0454 | 0.0351 | -0.0103 | 177 |
| 7 | pit120 | full | 0.0491 | 0.0600 | 0.0110 | 1620 |
| 7 | pit120 | trail18m | 0.0646 | 0.1168 | 0.0522 | 548 |
| 7 | pit120 | y2022 | 0.0187 | 0.0222 | 0.0035 | 347 |
| 7 | pit120 | y2023 | 0.0498 | 0.0145 | -0.0353 | 365 |
| 7 | pit120 | y2024 | 0.0526 | 0.0569 | 0.0043 | 366 |
| 7 | pit120 | y2025 | 0.0654 | 0.1270 | 0.0616 | 365 |
| 7 | pit120 | y2026 | 0.0660 | 0.0967 | 0.0307 | 177 |
| 10 | top20 | full | 0.1010 | 0.0643 | -0.0366 | 1620 |
| 10 | top20 | trail18m | 0.0656 | 0.0736 | 0.0081 | 548 |
| 10 | top20 | y2022 | 0.1343 | 0.1025 | -0.0318 | 344 |
| 10 | top20 | y2023 | 0.0949 | 0.0164 | -0.0785 | 365 |
| 10 | top20 | y2024 | 0.1304 | 0.0638 | -0.0666 | 366 |
| 10 | top20 | y2025 | 0.0934 | 0.1062 | 0.0127 | 365 |
| 10 | top20 | y2026 | 0.0050 | 0.0047 | -0.0004 | 180 |
| 10 | pit120 | full | 0.0452 | 0.0546 | 0.0094 | 1620 |
| 10 | pit120 | trail18m | 0.0533 | 0.0871 | 0.0338 | 548 |
| 10 | pit120 | y2022 | 0.0446 | 0.0398 | -0.0048 | 344 |
| 10 | pit120 | y2023 | 0.0341 | 0.0353 | 0.0012 | 365 |
| 10 | pit120 | y2024 | 0.0444 | 0.0399 | -0.0045 | 366 |
| 10 | pit120 | y2025 | 0.0607 | 0.0889 | 0.0283 | 365 |
| 10 | pit120 | y2026 | 0.0397 | 0.0827 | 0.0430 | 180 |

### Paired NW t on daily ΔIC

| h | window | mean ΔIC | NW-t | n_days |
|---|--------|----------|------|--------|
| 7 | full | -0.0482 | -3.26 | 1620 |
| 7 | trail18m | -0.0052 | -0.22 | 548 |
| 10 | full | -0.0366 | -3.65 | 1620 |
| 10 | trail18m | 0.0081 | 0.54 | 548 |

### Coverage-conditional ΔIC (≥80% book micro coverage)

| h | window | mean ΔIC | NW-t | n_days |
|---|--------|----------|------|--------|
| 7 | full | -0.0482 | -3.26 | 1620 |
| 7 | trail18m | -0.0052 | -0.22 | 548 |
| 10 | full | -0.0366 | -3.65 | 1620 |
| 10 | trail18m | 0.0081 | 0.54 | 548 |

### Δ median-τ net Sharpe (tranche, funding on, paired days)

| h | A Sharpe | D Sharpe | Δ |
|---|----------|----------|---|
| 7 | 1.401 | 0.577 | -0.824 |
| 10 | 1.023 | 0.757 | -0.266 |

### Microstructure LightGBM gain importances

| h | feature | mean_gain | median_gain |
|---|---------|-----------|-------------|
| 7 | funding_now | 182.67 | 206.30 |
| 7 | funding_cs_rank | 172.57 | 189.98 |
| 7 | oi_turnover | 81.78 | 90.86 |
| 7 | funding_cum_7 | 79.29 | 68.65 |
| 7 | taker_imb_z | 59.61 | 45.74 |
| 7 | oi_chg_7 | 52.67 | 57.81 |
| 7 | oi_chg_1 | 31.48 | 27.27 |
| 7 | ls_ratio_z | 24.79 | 28.59 |
| 7 | funding_z_30 | 20.44 | 9.64 |
| 7 | basis_z_30 | 6.13 | 3.51 |
| 7 | liq_imb_1 | 0.00 | 0.00 |
| 7 | liq_imb_7 | 0.00 | 0.00 |
| 10 | funding_now | 573.28 | 572.20 |
| 10 | funding_cs_rank | 543.60 | 521.54 |
| 10 | funding_cum_7 | 483.07 | 491.86 |
| 10 | oi_turnover | 442.36 | 467.73 |
| 10 | oi_chg_7 | 283.17 | 293.38 |
| 10 | taker_imb_z | 254.97 | 248.33 |
| 10 | oi_chg_1 | 212.00 | 224.58 |
| 10 | ls_ratio_z | 150.91 | 165.45 |
| 10 | funding_z_30 | 127.19 | 127.53 |
| 10 | basis_z_30 | 40.76 | 41.19 |
| 10 | liq_imb_1 | 0.00 | 0.00 |
| 10 | liq_imb_7 | 0.00 | 0.00 |
