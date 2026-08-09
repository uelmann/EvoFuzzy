# Kronos BTC daily → LONG / HOLD / SHORT

Zero-shot **Kronos-base** on **BTCUSDT daily** bars, converted into a trading stance.

## Data source
- **BTC signal (existing):** Binance public klines via `data-api.binance.vision` (fallbacks: `api.binance.us`, `api.binance.com`), `BTCUSDT` / `1d`
- **Cross-asset crypto panel:** CoinMarketCap data-api (same logic as the KuCoin BT notebook) — KuCoin spot universe, daily OHLCV + marketCap

```bash
# top-60 by mcap among KuCoin-listed (skips stables); writes kronos_signal/data/historical_data.csv
python -m kronos_signal.download_cmc_kucoin --max-coins 60
# full KuCoin universe (~800 bases): --max-coins 0
```

## Official Kronos recipe (match their FT protocol)

Mirrored from `Kronos/finetune/` (tokenizer → predictor → TopkDropout **long-only**):

| Step | Their A-share demo | Our run |
|------|--------------------|---------|
| Data | Qlib CSI300 | Crypto panel (same pickle schema) |
| Lookback / predict | **90 / 10** | **90 / 10** |
| Start checkpoint | **Kronos-small** | **small** (`--predictor-size base` optional) |
| Epochs / batch / LR | 30 / 50 / 2e-4 & 4e-5 | same |
| Backtest | TopkDropout hold 50 / drop 5 | scaled **hold 5 / drop 1** on PIT top-30 |
| Benchmark | CSI300 | BTC |

```bash
python -m kronos_signal.test_official
modal run kronos_signal/modal_app.py --mode official --predictor-size small --official-epochs 30
# smoke
modal run kronos_signal/modal_app.py --mode official --official-epochs 2
```

Literal China CSI300 still needs Qlib `cn_data`; this mirrors their **code path** on crypto.

### Experimental fork (not official): Top3 long / Worst3 short
`cross_asset_bt.py` — dollar-neutral L/S. **Not** their README backtest.

## Defaults (v1 BTC timing)

| Knob | Value |
|------|-------|
| Symbol / interval | BTCUSDT `1d` |
| Model | `NeoQuasar/Kronos-base` |
| Lookback | 400 days |
| Horizon | 5 days |
| Monte Carlo paths | 30 |
| LONG | `p_up ≥ 60%` and `mean_r ≥ +0.5%` |
| SHORT | `p_up ≤ 40%` and `mean_r ≤ -0.5%` |
| HOLD | otherwise |

`p_up` = share of paths with predicted horizon close above the last close.  
`mean_r` = mean horizon return across paths.

## Run on Modal (GPU)

```bash
# live signal
modal run kronos_signal/modal_app.py
modal run kronos_signal/modal_app.py --n-paths 5

# walk-forward backtest (non-overlapping 5d steps)
modal run kronos_signal/modal_app.py --mode backtest --n-paths 10 --max-steps 40
```

Writes `kronos_signal/last_signal.json` or `kronos_signal/last_backtest.json`.

### Backtest design
- Non-overlapping: decide every `pred_len` days, hold stance for that horizon
- LONG PnL = realized return; SHORT = −realized; HOLD = 0
- Metrics: hit-rate (active trades), total return, buy&hold, max drawdown

### Improve pipeline
```bash
# v1: logistic meta + AR fine-tune
modal run kronos_signal/modal_app.py --mode improve --n-paths 10 --max-steps 150

# v2: LightGBM meta (purge/embargo) + supervised direction head
modal run kronos_signal/modal_app.py --mode improve_v2

# Long history from 2021 + annual meta retrain/test (2022–2026), 5d horizon
modal run kronos_signal/modal_app.py --mode long_annual --n-paths 10 --start-asof 2021-01-01

# Next-day prediction + daily rebalancing (Kronos-base, pred_len=1, step=1)
modal run kronos_signal/modal_app.py --mode long_annual --pred-len 1 --n-paths 10 --start-asof 2021-01-01
python -m kronos_signal.plot_annual
```

v2 compares raw rule vs LightGBM meta vs market-only ablation vs supervised head vs meta+sup.

## Local unit tests (no GPU)

```bash
python -m kronos_signal.test_signals
python -m kronos_signal.test_backtest
python -m kronos_signal.test_meta_model
```
