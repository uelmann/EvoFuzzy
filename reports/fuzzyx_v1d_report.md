# FuzzyX-v1d report

**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. Does not replace COMBO / A0. Addendum: `reports/fuzzyx_addendum_v1d.md`.

**Mode:** `LOCAL-RESTRICTED`
**Verdict:** **CONTAMINATED**
**Params:** 17668

## Keep rule (verbatim)

> See `reports/fuzzyx_addendum_v1d.md`. VIABLE only if leakage, shuffle-bias on mean weekly net PnL (per-fold weights), full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs A0 Sleeve A when A0 preds exist. LOCAL-RESTRICTED cannot be official VIABLE.

## Gates

| clause | result |
|---|---|
| (i) leakage | PASS |
| (ii) shuffle-bias | FAIL |
| (iii) Sharpe ≥ 0 | PASS (1.257 weekly) |
| (iv) vs A0 | SKIP |

- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top30`: **PASS** `{'name': 'universe_lookahead_top30', 'passed': True, 'n': 30, 'date': '2023-04-17', 'base_n': 30, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0}`

## Shuffle-bias folds

- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.011108070425689221, 'sd': 0.003059697903419046, 'se': 0.0009675614326846076, 'threshold': 0.0019351228653692151, 'n': 10, 'pnls': [0.013810361735522747, 0.0071248505264520645, 0.010395506396889687, 0.007810300216078758, 0.01638791896402836, 0.011378014460206032, 0.0077978624030947685, 0.009987258352339268, 0.01384088397026062, 0.012547747232019901], 'mean_corr_st_r': -0.5451418370008468, 'corrs': [-0.5093086957931519, -0.6374911665916443, -0.5715349316596985, -0.6419315934181213, -0.40412697196006775, -0.5383322238922119, -0.5922717452049255, -0.549673318862915, -0.4965035319328308, -0.5102441906929016], 'fold_id': 0}`
- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.012713110097683967, 'sd': 0.007255425491340968, 'se': 0.002294366994628373, 'threshold': 0.004588733989256746, 'n': 10, 'pnls': [0.02430262230336666, 0.022007359191775322, 0.005212294403463602, 0.008854961954057217, 0.01647389680147171, 0.006956559140235186, 0.012620540335774422, 0.003726719180122018, 0.01862018182873726, 0.008355965837836266], 'mean_corr_st_r': 0.6161470204591751, 'corrs': [0.8611975908279419, 0.837374746799469, 0.18504783511161804, 0.5102438926696777, 0.8501626253128052, 0.5389245748519897, 0.5138412117958069, 0.45918768644332886, 0.7712490558624268, 0.634240984916687], 'fold_id': 17}`

## Book

```json
{
  "net_sharpe_weekly": 1.2571932622136717,
  "mean_weekly_pnl": 0.005027559533165222,
  "hard_loss": {
    "loss": -0.22563148257175758,
    "core": 0.22563148257175758,
    "trend": 0.22563148257175758,
    "trend_equity": 0.22563148257175758,
    "trend_returns": 0.07080572976732391,
    "maxdd": 0.5936881762867684,
    "ddur": 0.9008620689655172,
    "long_frac": 0.05888295569346226,
    "short_frac": 0.514937220378121,
    "traded_frac": 0.5738201760715832,
    "turnover": 0.47343843398478425,
    "bias": 0.01628691288164757,
    "ann_mean": 1.835059229605306,
    "mean_pnl": 0.005027559533165222
  },
  "soft_loss": {
    "loss": 0.8274909528249115,
    "core": -0.8274909528249115,
    "trend": -0.8274909528249115,
    "trend_equity": -0.8274909528249115,
    "trend_returns": 0.057466380726996426,
    "maxdd": 0.9442888906431925,
    "ddur": 0.9224137931034483,
    "long_frac": 0.06075912830134218,
    "short_frac": 0.5436570933756675,
    "traded_frac": 0.6044162216770097,
    "turnover": 0.7440599122977496,
    "bias": 0.01823236457038098,
    "ann_mean": -0.08125054512234325,
    "mean_pnl": -0.0002226042332118993
  },
  "long_frac": 0.05888295569346226,
  "short_frac": 0.514937220378121,
  "traded_frac": 0.5738201760715832,
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

- fold 0 2022-01-10→2022-04-09 status=ok best_epoch=1 hold_loss=0.655208170413971 n_reb=13
- fold 1 2022-04-10→2022-07-08 status=ok best_epoch=35 hold_loss=-0.460771381855011 n_reb=13
- fold 2 2022-07-09→2022-10-06 status=ok best_epoch=0 hold_loss=-0.9484782814979553 n_reb=13
- fold 3 2022-10-07→2023-01-04 status=ok best_epoch=2 hold_loss=0.5686196684837341 n_reb=13
- fold 4 2023-01-05→2023-04-04 status=ok best_epoch=1 hold_loss=-0.5316262245178223 n_reb=12
- fold 5 2023-04-05→2023-07-03 status=ok best_epoch=13 hold_loss=0.7190939784049988 n_reb=13
- fold 6 2023-07-04→2023-10-01 status=ok best_epoch=4 hold_loss=-0.9097530245780945 n_reb=13
- fold 7 2023-10-02→2023-12-30 status=ok best_epoch=30 hold_loss=-0.9536799192428589 n_reb=13
- fold 8 2023-12-31→2024-03-29 status=ok best_epoch=50 hold_loss=0.7953047156333923 n_reb=13
- fold 9 2024-03-30→2024-06-27 status=ok best_epoch=0 hold_loss=0.8081049919128418 n_reb=13
- fold 10 2024-06-28→2024-09-25 status=ok best_epoch=31 hold_loss=-0.773326575756073 n_reb=13
- fold 11 2024-09-26→2024-12-24 status=ok best_epoch=4 hold_loss=-0.8025060892105103 n_reb=12
- fold 12 2024-12-25→2025-03-24 status=ok best_epoch=1 hold_loss=0.9352104067802429 n_reb=13
- fold 13 2025-03-25→2025-06-22 status=ok best_epoch=1 hold_loss=-0.8980311155319214 n_reb=13
- fold 14 2025-06-23→2025-09-20 status=ok best_epoch=37 hold_loss=-0.10599377751350403 n_reb=13
- fold 15 2025-09-21→2025-12-19 status=ok best_epoch=0 hold_loss=0.8795050382614136 n_reb=13
- fold 16 2025-12-20→2026-03-19 status=ok best_epoch=10 hold_loss=-0.9766103029251099 n_reb=13
- fold 17 2026-03-20→2026-06-17 status=ok best_epoch=14 hold_loss=-0.8519500494003296 n_reb=13

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
- `R10 SHORT: NOT vol_of_vol_30 IS MID AND beta_btc_60 IS MID AND sma20_sma50 IS HIGH AND skew_28 IS MID`
- `R11 SHORT: idio_vol_60 IS LOW AND ret_90 IS LOW AND yz_vol_30 IS HIGH AND NOT min_ret_14 IS HIGH`
- `R12 SHORT: pk_vol_14 IS HIGH AND NOT rev_1 IS LOW AND sma20_sma50 IS HIGH AND NOT rev_3 IS MID`
- `R13 SHORT: NOT close_sma20 IS LOW AND NOT corr_btc_28 IS HIGH AND amihud_14 IS HIGH AND NOT max_ret_14 IS HIGH`
- `R14 SHORT: max_ret_14 IS LOW AND ret_28 IS HIGH AND ret_56 IS LOW AND NOT ret_90 IS MID`
- `R15 LONG: sma20_sma50 IS MID AND NOT rev_1 IS MID AND NOT ret_90 IS HIGH AND NOT vol_ratio IS HIGH`

## Notes

- Clause (iv) SKIP: A0 h=7 preds missing.
- Official VIABLE is disabled for LOCAL-RESTRICTED even if Sharpe≥0.
