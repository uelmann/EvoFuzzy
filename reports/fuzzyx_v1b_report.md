# FuzzyX-v1b report

**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. Does not replace COMBO / A0. Addendum: `reports/fuzzyx_addendum_v1b.md`.

**Mode:** `LOCAL-RESTRICTED`
**Verdict:** **CONTAMINATED**
**Params:** 17668

## Keep rule (verbatim)

> See `reports/fuzzyx_addendum_v1b.md`. VIABLE only if leakage, shuffle-bias on mean weekly net PnL (per-fold weights), full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs A0 Sleeve A when A0 preds exist. LOCAL-RESTRICTED cannot be official VIABLE.

## Gates

| clause | result |
|---|---|
| (i) leakage | PASS |
| (ii) shuffle-bias | FAIL |
| (iii) Sharpe ≥ 0 | PASS (0.000 weekly) |
| (iv) vs A0 | SKIP |

- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top30`: **PASS** `{'name': 'universe_lookahead_top30', 'passed': True, 'n': 30, 'date': '2023-04-17', 'base_n': 30, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0}`

## Shuffle-bias folds

- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.0023932626994792373, 'sd': 0.0010251018618943472, 'se': 0.00032416567172655053, 'threshold': 0.0006483313434531011, 'n': 10, 'pnls': [0.002441529417410493, 0.002098567085340619, 0.002391611924394965, 0.0018326030112802982, 0.0035564142744988203, 0.0029173477087169886, 0.0004879872431047261, 0.0032216995023190975, 0.003735483856871724, 0.00124938297085464], 'fold_id': 0}`
- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.0007697376888245345, 'sd': 0.00014936065678252823, 'se': 4.723198682514657e-05, 'threshold': 9.446397365029314e-05, 'n': 10, 'pnls': [0.0009161102352663875, 0.0009865817846730351, 0.0005400332156568766, 0.0005635304842144251, 0.0008055285434238613, 0.0007645546575076878, 0.0008314470178447664, 0.0007132517639547586, 0.0009066170314326882, 0.0006697221542708576], 'fold_id': 17}`

## Book

```json
{
  "net_sharpe_weekly": 0.0,
  "mean_weekly_pnl": 0.0,
  "hard_loss": {
    "loss": 0.0,
    "core": 0.0,
    "trend": 0.0,
    "maxdd": -0.0,
    "ddur": 0.0,
    "long_frac": 0.0,
    "short_frac": 0.0,
    "traded_frac": 0.0,
    "turnover": 0.0,
    "bias": 0.0,
    "ann_mean": 0.0,
    "mean_pnl": 0.0
  },
  "soft_loss": {
    "loss": 0.008899982261067928,
    "core": -0.003119151726615745,
    "trend": -0.5464002533333869,
    "maxdd": 0.33780850561454245,
    "ddur": 0.9913793103448276,
    "long_frac": 0.0,
    "short_frac": 0.0011545677586953383,
    "traded_frac": 0.0011545677586953383,
    "turnover": 0.11136832235088379,
    "bias": 0.00424828833815987,
    "ann_mean": 0.08302907264229895,
    "mean_pnl": 0.00022747691134876422
  },
  "long_frac": 0.0,
  "short_frac": 0.0,
  "traded_frac": 0.0,
  "n_reb": 232,
  "n_symbols": 40
}
```

## vs A0

```json
{
  "skipped": true,
  "reason": "A0 predictions not on disk"
}
```

## Folds

- fold 0 2022-01-10→2022-04-09 status=ok best_epoch=3 hold_loss=0.11279967427253723 n_reb=13
- fold 1 2022-04-10→2022-07-08 status=ok best_epoch=68 hold_loss=-0.30463889241218567 n_reb=13
- fold 2 2022-07-09→2022-10-06 status=ok best_epoch=1 hold_loss=-0.3897794187068939 n_reb=13
- fold 3 2022-10-07→2023-01-04 status=ok best_epoch=7 hold_loss=0.12053827941417694 n_reb=13
- fold 4 2023-01-05→2023-04-04 status=ok best_epoch=1 hold_loss=-0.19519466161727905 n_reb=12
- fold 5 2023-04-05→2023-07-03 status=ok best_epoch=7 hold_loss=0.09948328137397766 n_reb=13
- fold 6 2023-07-04→2023-10-01 status=ok best_epoch=1 hold_loss=-0.38959574699401855 n_reb=13
- fold 7 2023-10-02→2023-12-30 status=ok best_epoch=15 hold_loss=-0.2752322256565094 n_reb=13
- fold 8 2023-12-31→2024-03-29 status=ok best_epoch=5 hold_loss=0.18722862005233765 n_reb=13
- fold 9 2024-03-30→2024-06-27 status=ok best_epoch=1 hold_loss=-0.27680349349975586 n_reb=13
- fold 10 2024-06-28→2024-09-25 status=ok best_epoch=16 hold_loss=-0.23975852131843567 n_reb=13
- fold 11 2024-09-26→2024-12-24 status=ok best_epoch=1 hold_loss=-0.30952030420303345 n_reb=12
- fold 12 2024-12-25→2025-03-24 status=ok best_epoch=0 hold_loss=0.12846067547798157 n_reb=13
- fold 13 2025-03-25→2025-06-22 status=ok best_epoch=16 hold_loss=-0.36782166361808777 n_reb=13
- fold 14 2025-06-23→2025-09-20 status=ok best_epoch=6 hold_loss=0.21817409992218018 n_reb=13
- fold 15 2025-09-21→2025-12-19 status=ok best_epoch=0 hold_loss=0.1614336222410202 n_reb=13
- fold 16 2025-12-20→2026-03-19 status=ok best_epoch=18 hold_loss=-0.3995843529701233 n_reb=13
- fold 17 2026-03-20→2026-06-17 status=ok best_epoch=19 hold_loss=-0.36522579193115234 n_reb=13

## Sample rules (eval argmax literals)

- `R00 SHORT: NOT dv_trend IS HIGH AND ret_14 IS LOW AND NOT skew_60 IS LOW AND NOT close_sma100 IS LOW`
- `R01 FLAT: dist_low_90 IS LOW AND NOT corr_btc_28 IS LOW AND NOT yz_vol_30 IS HIGH AND ema12_ema26 IS MID`
- `R02 LONG: NOT dv_trend IS HIGH AND NOT dist_high_90 IS MID AND yz_vol_60 IS HIGH AND NOT beta_btc_60 IS LOW`
- `R03 FLAT: NOT dv_trend IS MID AND NOT max_ret_14 IS MID AND range_pos_28 IS MID AND NOT close_sma100 IS MID`
- `R04 LONG: NOT dist_high_90 IS HIGH AND NOT rev_1 IS HIGH AND NOT vol_of_vol_30 IS HIGH AND NOT ret_28 IS LOW`
- `R05 FLAT: NOT min_ret_14 IS MID AND pk_vol_14 IS MID AND amihud_14 IS LOW AND NOT vol_ratio IS LOW`
- `R06 SHORT: NOT ema12_ema26 IS LOW AND NOT beta_btc_60 IS LOW AND dv_trend IS LOW AND range_pos_28 IS LOW`
- `R07 SHORT: NOT dv_trend IS MID AND NOT ret_28 IS MID AND NOT range_pos_28 IS HIGH AND NOT ret_7 IS MID`
- `R08 LONG: dv_z_30 IS HIGH AND sma20_sma50 IS LOW AND vol_of_vol_30 IS HIGH AND NOT mom_28_skip7 IS LOW`
- `R09 SHORT: NOT ret_56 IS MID AND close_sma50 IS HIGH AND NOT ema12_ema26 IS MID AND beta_btc_60 IS LOW`
- `R10 SHORT: NOT vol_of_vol_30 IS MID AND beta_btc_60 IS MID AND sma20_sma50 IS HIGH AND skew_28 IS HIGH`
- `R11 SHORT: idio_vol_60 IS LOW AND ret_90 IS LOW AND yz_vol_30 IS HIGH AND NOT min_ret_14 IS HIGH`
- `R12 SHORT: pk_vol_14 IS HIGH AND NOT rev_1 IS LOW AND sma20_sma50 IS HIGH AND NOT rev_3 IS MID`
- `R13 SHORT: NOT close_sma20 IS LOW AND NOT corr_btc_28 IS HIGH AND amihud_14 IS HIGH AND NOT max_ret_14 IS HIGH`
- `R14 SHORT: max_ret_14 IS LOW AND ret_28 IS HIGH AND ret_56 IS LOW AND NOT ret_90 IS MID`
- `R15 LONG: sma20_sma50 IS MID AND NOT rev_1 IS MID AND NOT vol_ratio IS HIGH AND NOT ret_90 IS HIGH`

## Notes

- Clause (iv) SKIP: A0 h=7 preds missing.
- Official VIABLE is disabled for LOCAL-RESTRICTED even if Sharpe≥0.

## Reading (not a retune)

The occupancy push is gone. Hard `{+1,0,−1}` occupancy is **0.0 / 0.0 / 0.0** over 232 weekly rebalances (v1 was 0.50 / 0.50 / 0.996). Eval never leaves FLAT, so weekly net Sharpe is reported as 0.000 because the return series is identically zero (`std=0` → Sharpe defined as 0 in `eval._sharpe`).

Pay-to-play + FLAT init overshot: the discrete book does not trade. Soft occupancy is 0.12% traded; shuffle-bias is still run on **soft** positions (same as v1) and fails because those leftovers keep a net-long residual. Within-date return shuffle preserves the date’s cross-section mean, so a net-long soft book earns crypto drift under the null. Hard-book shuffle would be trivially centered (all zeros); that is not the pre-registered statistic.

No λ_active sweep. v1 report untouched. COMBO / A0 not replaced.
