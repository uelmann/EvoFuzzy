# FuzzyX-v1e report

**BACKTEST ONLY.** One shot. DeepSets, weekly, PIT top-30 volume, seed 42. Does not replace COMBO / A0. Addendum: `reports/fuzzyx_addendum_v1e.md`.

**Mode:** `LOCAL-RESTRICTED`
**Verdict:** **CONTAMINATED**
**Params:** 17668

## Keep rule (verbatim)

> See `reports/fuzzyx_addendum_v1e.md`. VIABLE only if leakage, shuffle-bias on mean weekly net PnL (per-fold weights), full-OOS net Sharpe ≥ 0, and ≤ 0.10 Sharpe vs A0 Sleeve A when A0 preds exist. LOCAL-RESTRICTED cannot be official VIABLE.

## Gates

| clause | result |
|---|---|
| (i) leakage | PASS |
| (ii) shuffle-bias | FAIL |
| (iii) Sharpe ≥ 0 | FAIL (-0.882 weekly) |
| (iv) vs A0 | SKIP |

- `feature_lookahead`: **PASS** `{'name': 'feature_lookahead', 'passed': True, 'max_abs_diff': 0.0, 'symbol': 'BTCUSDT', 'date': '2023-04-17'}`
- `universe_lookahead_top30`: **PASS** `{'name': 'universe_lookahead_top30', 'passed': True, 'n': 30, 'date': '2023-04-17', 'base_n': 30, 'symmetric_diff': 0}`
- `seed_determinism`: **PASS** `{'name': 'seed_determinism', 'passed': True, 'max_score_diff': 0.0}`

## Shuffle-bias folds

- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.010532888304442167, 'sd': 0.001000435959460053, 'se': 0.00031636562850296434, 'threshold': 0.0006327312570059287, 'n': 10, 'pnls': [0.010901822708547115, 0.009481756016612053, 0.00999183114618063, 0.008977928198873997, 0.012102197855710983, 0.010949118062853813, 0.009896771050989628, 0.01011983398348093, 0.011726551689207554, 0.01118107233196497], 'mean_corr_st_r': -0.5759396433830262, 'corrs': [-0.580601155757904, -0.5908040404319763, -0.5909457802772522, -0.6095395088195801, -0.5490750670433044, -0.5680779218673706, -0.573646068572998, -0.575516939163208, -0.5600485801696777, -0.5611413717269897], 'fold_id': 0}`
- `{'name': 'label_shuffle_bias', 'statistic': 'mean_weekly_net_pnl', 'passed': False, 'mean_pnl': 0.009568751354527194, 'sd': 0.009720018578743168, 'se': 0.0030737397607981123, 'threshold': 0.0061474795215962245, 'n': 10, 'pnls': [0.023010138422250748, 0.025209205225110054, -0.00017972748901229352, 0.006524461787194014, 0.016343750059604645, 0.0019933416042476892, 0.009722964838147163, -0.0030601592734456062, 0.013227571733295918, 0.00289596663787961], 'mean_corr_st_r': 0.1535012185573578, 'corrs': [0.9288052320480347, 0.8786620497703552, -0.5603887438774109, -0.16296082735061646, 0.7409171462059021, -0.14237818121910095, -0.09471683204174042, -0.4464275538921356, 0.43804165720939636, -0.04454176127910614], 'fold_id': 17}`

## Book

```json
{
  "net_sharpe_weekly": -0.881923311737392,
  "mean_weekly_pnl": -0.004675724020889638,
  "hard_loss": {
    "loss": 0.08430621791836235,
    "core": -0.08430621791836235,
    "trend": -0.8788511222014826,
    "trend_equity": -0.8788511222014826,
    "trend_returns": 0.024491960799985606,
    "equity_end": 0.09592775817043853,
    "cumret_last": -0.9040722418295615,
    "maxdd": 0.9585298474968633,
    "ddur": 0.9224137931034483,
    "long_frac": 0.039399624765478425,
    "short_frac": 0.720883244335402,
    "traded_frac": 0.7602828691008804,
    "turnover": 0.39386158947287475,
    "bias": 0.022810146650812525,
    "ann_mean": -1.706639267624718,
    "mean_pnl": -0.004675724020889638
  },
  "soft_loss": {
    "loss": 0.2680340794181739,
    "core": -0.2680340794181739,
    "trend": -0.8595867336006561,
    "trend_equity": -0.8595867336006561,
    "trend_returns": 0.015679393429851956,
    "equity_end": 0.31181737565379447,
    "cumret_last": -0.6881826243462055,
    "maxdd": 0.9075813568257164,
    "ddur": 0.7844827586206896,
    "long_frac": 0.04748159907634579,
    "short_frac": 0.780054841968538,
    "traded_frac": 0.8275364410448838,
    "turnover": 0.444578857956888,
    "bias": 0.023158331231782382,
    "ann_mean": -0.2146034111455557,
    "mean_pnl": -0.0005879545510837143
  },
  "long_frac": 0.039399624765478425,
  "short_frac": 0.720883244335402,
  "traded_frac": 0.7602828691008804,
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

- fold 0 2022-01-10→2022-04-09 status=ok best_epoch=5 hold_loss=0.3242994546890259 n_reb=13
- fold 1 2022-04-10→2022-07-08 status=ok best_epoch=79 hold_loss=-0.5212590098381042 n_reb=13
- fold 2 2022-07-09→2022-10-06 status=ok best_epoch=1 hold_loss=-2.1875171661376953 n_reb=13
- fold 3 2022-10-07→2023-01-04 status=ok best_epoch=5 hold_loss=0.45946064591407776 n_reb=13
- fold 4 2023-01-05→2023-04-04 status=ok best_epoch=3 hold_loss=-0.6892364621162415 n_reb=12
- fold 5 2023-04-05→2023-07-03 status=ok best_epoch=1 hold_loss=0.38200613856315613 n_reb=13
- fold 6 2023-07-04→2023-10-01 status=ok best_epoch=0 hold_loss=-1.0621110200881958 n_reb=13
- fold 7 2023-10-02→2023-12-30 status=ok best_epoch=20 hold_loss=-1.130102515220642 n_reb=13
- fold 8 2023-12-31→2024-03-29 status=ok best_epoch=40 hold_loss=0.27993959188461304 n_reb=13
- fold 9 2024-03-30→2024-06-27 status=ok best_epoch=3 hold_loss=0.4038371741771698 n_reb=13
- fold 10 2024-06-28→2024-09-25 status=ok best_epoch=26 hold_loss=-0.8202095031738281 n_reb=13
- fold 11 2024-09-26→2024-12-24 status=ok best_epoch=0 hold_loss=-0.999782145023346 n_reb=12
- fold 12 2024-12-25→2025-03-24 status=ok best_epoch=1 hold_loss=0.3261740207672119 n_reb=13
- fold 13 2025-03-25→2025-06-22 status=ok best_epoch=3 hold_loss=-1.6393911838531494 n_reb=13
- fold 14 2025-06-23→2025-09-20 status=ok best_epoch=4 hold_loss=0.5663655996322632 n_reb=13
- fold 15 2025-09-21→2025-12-19 status=ok best_epoch=2 hold_loss=0.6471707224845886 n_reb=13
- fold 16 2025-12-20→2026-03-19 status=ok best_epoch=1 hold_loss=-1.4996236562728882 n_reb=13
- fold 17 2026-03-20→2026-06-17 status=ok best_epoch=57 hold_loss=-1.5641289949417114 n_reb=13

## Sample rules (eval argmax literals)

- `R00 SHORT: NOT dv_trend IS HIGH AND NOT close_sma100 IS LOW AND dv_z_30 IS LOW AND NOT skew_60 IS LOW`
- `R01 FLAT: dist_low_90 IS LOW AND NOT yz_vol_30 IS HIGH AND NOT corr_btc_28 IS LOW AND ema12_ema26 IS MID`
- `R02 LONG: NOT dv_trend IS HIGH AND yz_vol_60 IS HIGH AND NOT dist_high_90 IS MID AND NOT beta_btc_60 IS LOW`
- `R03 FLAT: NOT dv_trend IS HIGH AND NOT max_ret_14 IS MID AND range_pos_28 IS MID AND dist_low_90 IS HIGH`
- `R04 LONG: NOT dist_high_90 IS HIGH AND NOT rev_1 IS HIGH AND NOT vol_of_vol_30 IS HIGH AND NOT ret_28 IS LOW`
- `R05 FLAT: NOT min_ret_14 IS MID AND NOT vol_ratio IS LOW AND pk_vol_14 IS MID AND amihud_14 IS LOW`
- `R06 SHORT: NOT beta_btc_60 IS LOW AND NOT ema12_ema26 IS LOW AND dv_trend IS MID AND range_pos_28 IS LOW`
- `R07 SHORT: NOT dv_trend IS MID AND NOT ret_28 IS MID AND NOT range_pos_28 IS HIGH AND NOT ret_7 IS MID`
- `R08 LONG: dv_z_30 IS HIGH AND sma20_sma50 IS LOW AND vol_of_vol_30 IS HIGH AND NOT mom_28_skip7 IS LOW`
- `R09 SHORT: NOT ret_56 IS MID AND close_sma50 IS HIGH AND NOT ema12_ema26 IS MID AND beta_btc_60 IS LOW`
- `R10 SHORT: NOT vol_of_vol_30 IS MID AND beta_btc_60 IS MID AND sma20_sma50 IS HIGH AND skew_28 IS MID`
- `R11 SHORT: idio_vol_60 IS LOW AND ret_90 IS LOW AND yz_vol_30 IS LOW AND NOT min_ret_14 IS HIGH`
- `R12 SHORT: pk_vol_14 IS HIGH AND NOT rev_1 IS LOW AND sma20_sma50 IS HIGH AND NOT rev_3 IS MID`
- `R13 SHORT: NOT corr_btc_28 IS HIGH AND NOT close_sma20 IS LOW AND amihud_14 IS HIGH AND NOT max_ret_14 IS HIGH`
- `R14 SHORT: max_ret_14 IS LOW AND ret_28 IS HIGH AND ret_56 IS LOW AND NOT ret_90 IS MID`
- `R15 LONG: sma20_sma50 IS MID AND NOT rev_1 IS LOW AND NOT vol_ratio IS HIGH AND NOT ret_90 IS HIGH`

## Notes

- Clause (iv) SKIP: A0 h=7 preds missing.
- Official VIABLE is disabled for LOCAL-RESTRICTED even if Sharpe≥0.

## Reading (not a retune)

This is the product you meant: `corr(wealth, t) * (1 + cumRet[-1])` with `cumRet[-1] = wealth[-1] − 1`. Not `corr(1+r, t)`.

Hard book: wealth-corr **−0.879**, last cumret **−90.4%** (`equity_end = 0.096`), Sharpe **−0.882**, traded_frac 0.76, almost all short (L 0.04 / S 0.72). Multiplying by ending wealth did inject return — the OOS path is a large loss, so the product is negative. Shuffle still FAIL (~**+1.05%/week** fold 0): the product does not remove net market exposure on 13-week expanding folds.

No retune. v1–v1d untouched. COMBO / A0 not replaced.
