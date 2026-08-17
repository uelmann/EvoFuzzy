# FASE 1 — decomposizione statico/dinamico + test y alternative

Nessuno Stage A. Nessun emendamento. Pre-reg non toccata.

μ_i = media dello score del modello del fold sulla sola finestra TRAIN `[train_start, train_end]` (purged). f_it = s_it − μ_i. Re-inferenza A0 (stesso seed/HP); max |Δ| score val vs parquet OOS = **0**.

JSON: `results/fase1_static_dyn.json`.

## Esclusi (nessuna storia train nel fold)

OOS top-40: 56331 righe. Keep: 53470. Drop: **2861** righe, **105** coppie (fold, symbol), **97** simboli.

| fold | n coppie | simboli |
|---:|---:|---|
| 0 | 1 | PEOPLEUSDT |
| 1 | 2 | APEUSDT, GMTUSDT |
| 2 | 1 | OPUSDT |
| 3 | 2 | 1000LUNCUSDT, LDOUSDT |
| 4 | 1 | APTUSDT |
| 5 | 8 | AGIXUSDT, ARBUSDT, CFXUSDT, COCOSUSDT, HIGHUSDT, IDUSDT, RNDRUSDT, STXUSDT |
| 6 | 5 | 1000PEPEUSDT, AMBUSDT, LEVERUSDT, PERPUSDT, SUIUSDT |
| 7 | 2 | SEIUSDT, WLDUSDT |
| 8 | 10 | 1000BONKUSDT, BIGTIMEUSDT, CKBUSDT, MEMEUSDT, ORDIUSDT, PENDLEUSDT, POLYXUSDT, PYTHUSDT, TIAUSDT, UMAUSDT |
| 9 | 9 | AEVOUSDT, BOMEUSDT, ENAUSDT, ETHFIUSDT, ONDOUSDT, ONGUSDT, POLYXUSDT, TONUSDT, WIFUSDT |
| 10 | 6 | ENAUSDT, IOUSDT, NOTUSDT, SAGAUSDT, TAOUSDT, ZROUSDT |
| 11 | 5 | 1MBABYDOGEUSDT, NEIROUSDT, POPCATUSDT, UXLINKUSDT, ZROUSDT |
| 12 | 5 | ACTUSDT, FARTCOINUSDT, GOATUSDT, PNUTUSDT, UXLINKUSDT |
| 13 | 7 | ALCHUSDT, FARTCOINUSDT, KAITOUSDT, LAYERUSDT, ORCAUSDT, PENGUUSDT, TRUMPUSDT |
| 14 | 6 | BANANAS31USDT, FUNUSDT, HYPEUSDT, IPUSDT, MYXUSDT, PUMPUSDT |
| 15 | 7 | ASTERUSDT, LIGHTUSDT, MYXUSDT, PIPPINUSDT, RESOLVUSDT, WLFIUSDT, XPLUSDT |
| 16 | 13 | ASTERUSDT, ENSOUSDT, KITEUSDT, LIGHTUSDT, LYNUSDT, POWERUSDT, RIVERUSDT, SAHARAUSDT, SENTUSDT, SIRENUSDT, VVVUSDT, XAUUSDT, 币安人生USDT |
| 17 | 15 | ARIAUSDT, BSBUSDT, CRCLUSDT, ESPORTSUSDT, EWYUSDT, INTCUSDT, LABUSDT, MSTRUSDT, NVDAUSDT, PLAYUSDT, RAVEUSDT, SKYAIUSDT, TSLAUSDT, UBUSDT, XAGUSDT |

## 1. RankIC vs y_h7 (top-40, solo keep)

| score | mean IC | NW-t | n_days |
|---|---:|---:|---:|
| s (OOS, keep) | 0.0742 | 7.08 | 1620 |
| μ only | 0.1151 | 7.53 | 1620 |
| f only | 0.0166 | 1.71 | 1620 |

## 2. Libri (top-40, τ=70, lag 0, 1x, funding on, stesso motore)

| libro | net Sharpe full | trail-18m | TO annuo |
|---|---:|---:|---:|
| baseline | 1.822 | 1.097 | 24.5 |
| MU-ONLY | 0.562 | 1.346 | 8.22 |
| F-ONLY | 0.702 | −0.646 | 21.3 |

n_days libro = 1619; trail n = 549.

## 3. Cos’è μ — Spearman(μ_i, media train del fattore)

Per fold, CS su i; poi media sui 18 fold. Pooled = tutti i (fold, symbol).

| fattore | stored | Spearman pooled | Spearman media fold | n coppie |
|---|---|---:|---:|---:|
| yz_vol_30_raw | raw | −0.119 | −0.298 | 4332 |
| amihud_14 | CS-z (no raw in feat) | −0.133 | −0.025 | 4332 |
| dollar_volume | raw | −0.020 | −0.128 | 4332 |
| beta_btc_60_raw | raw | −0.253 | −0.274 | 4332 |
| idio_vol_60 | CS-z (no raw in feat) | −0.326 | −0.493 | 4332 |

## 4. IC_k score OOS originale vs y alternative (top-40, non filtrato μ)

| y | k=0 IC | k=0 NW-t | k=10 IC | k=10 NW-t | k=60 IC | k=60 NW-t |
|---|---:|---:|---:|---:|---:|---:|
| y_h7 (residuo β, headline) | 0.0803 | 7.80 | 0.0585 | 6.74 | 0.0583 | 5.52 |
| y_raw = log-ret 7d, no β | 0.0765 | 7.02 | 0.0502 | 5.63 | 0.0539 | 5.34 |
| y_dm = y_raw − media CS | 0.0765 | 7.02 | 0.0497 | 5.34 | 0.0598 | 5.64 |
| y_volrank = rank y_h7 in terzili yz_vol_30_raw | 0.0392 | 4.84 | 0.0309 | 4.05 | 0.0320 | 3.53 |

n_days: 1620 / 1610 / 1560.

## 5. Dip intorno a k=20 (y_h7, score OOS originale)

| k | IC_k | NW-t | n_days |
|---:|---:|---:|---:|
| 12 | 0.0506 | 5.83 | 1608 |
| 15 | 0.0417 | 4.63 | 1605 |
| 20 | 0.0483 | 5.01 | 1600 |
| 25 | 0.0472 | 4.56 | 1595 |
| 30 | 0.0449 | 4.28 | 1590 |
