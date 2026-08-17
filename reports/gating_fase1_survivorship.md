# FASE 1 referto — survivorship + persistenza score

**Data UTC:** 2026-08-16. **Nessun Stage A. Nessun modello nuovo.** Pre-reg e soglia di `test_shifted_target_degrades` **non toccate**.

Numeri in `results/fase1_survivorship.json` e `results/fase1_survivorship_persist.json`.

---

## 1. Survivorship (bloccante)

### Listing degli 831

Gli 831 **non** sono uno snapshot `exchangeInfo`.

`baseline.data.list_um_symbols` fa `ListObjectsV2` su Binance Vision S3:

- prefix `data/futures/um/monthly/klines/`
- delimiter `/`
- filtra i CommonPrefixes `*USDT`, poi `should_exclude`

Le directory Vision **restano** per i contratti delisted finché i dump monthly sono in archivio. È enumerazione storica, non il book live.

FASE 1 ha scaricato quel set in `/data/quant/raw/klines/{SYMBOL}.parquet`. On disk: **831** parquet, tutti non vuoti. Panel FASE 1 = `n ≥ 100` barre → **695**. I 136 droppati sono `min_history_days=100`, non un filtro da sopravvissuti.

**Non ricostruire il universe.** Non è survivorship per costruzione.

### Ultima barra &lt; 2026-07

Cutoff: `last_kline < 2026-07-01` UTC (nessuna barra a luglio 2026). Max last del panel: **2026-07-31**.

| set | n | last &lt; 2026-07-01 | % |
|---|---:|---:|---:|
| Vision listed (parquet) | 831 | **31** | 3.7% |
| Panel `n≥100` | **695** | **28** | **4.0%** |
| Drop short history | 136 | 3 | — |

**28 ≠ ~0.** Il panel **non** è solo sopravvissuti. `baseline.json` **non** è invalido per survivorship.

### Delisted panel, per anno di last bar

| year | n | symbols |
|---:|---:|---|
| 2020 | 1 | LENDUSDT |
| 2021 | 1 | BZRXUSDT |
| 2022 | 7 | AKROUSDT, BTTUSDT, DODOUSDT, KEEPUSDT, LUNAUSDT, NUUSDT, YFIIUSDT |
| 2023 | 0 | — |
| 2024 | 14 | ANTUSDT, AUDIOUSDT, BLUEBIRDUSDT, BTSUSDT, COCOSUSDT, FOOTBALLUSDT, FRONTUSDT, GALUSDT, HNTUSDT, MATICUSDT, MBLUSDT, RNDRUSDT, SRMUSDT, TOMOUSDT |
| 2025 | 1 | EOSUSDT |
| 2026 (fino a giugno) | 4 | AERGOUSDT, BDXNUSDT, BTCSTUSDT, SXPUSDT |
| **totale panel** | **28** | |

Listed ma fuori panel (n&lt;100), last &lt; 2026-07: DOTECOUSDT (2021), 1000BTTCUSDT, ANCUSDT (2022).

2020–2026 sul panel **non** è ~0. I delisted **entrano nel PIT**: 18/28 sono stati in top-40, 10/28 in top-20 (LUNAUSDT 284 giorni top-20 2021-03-20→2022-06-02; MATICUSDT 940; EOSUSDT 802). Rotazione fuori dal top-120 ≠ delisting: chi ha ancora kline a luglio 2026 non è in questa lista.

---

## 2. Persistenza (panel valido, headline top-40)

Definizioni, lag **10**, stesso OOS delle predizioni A0 (`n_days=1620`, score vs `y_h7`):

- **ρ** = media su t della correlazione **cross-section** `corr_i(score_{i,t}, score_{i,t+10})` sui nomi presenti a t e t+10.
- **Score ortogonale** = residuo OLS **entro barra** `score_t = a + b·score_{t-10} + e`. Poi mean RankIC di `e` vs `y_{t→t+7}`, NW-t con lag=h=7.

Riferimento shifted-target già in `baseline.json` (soglia **non** mossa): unshifted RankIC **0.0803**, shifted +10 **0.0585**, ratio **0.728**.

| universo | ρ Pearson | ρ Spearman | unshifted RankIC | ortho RankIC | ortho NW-t | ortho / unshifted |
|---|---:|---:|---:|---:|---:|---:|
| **top-40 (headline)** | 0.344 | **0.295** | 0.0803 | **0.0609** | **6.40** | 0.759 |
| top-20 | 0.362 | 0.320 | 0.0976 | 0.0753 | 6.18 | 0.772 |
| OOS all names | 0.308 | 0.257 | 0.0528 | 0.0383 | 6.55 | 0.724 |

### Ipotesi da falsificare

> Lo shifted/unshifted IC (0.73) è spiegato **interamente** da ρ. Se ρ ~ 0.73 il test non ha trovato leak, ha trovato lentezza.

**Falsificata.** ρ Spearman top-40 = **0.295**, non 0.73 (gap −0.43). Se lo shifted IC fosse solo stickiness CS dello score, il ratio atteso sarebbe ≈ ρ ≈ 0.30, non 0.73. Pearson 0.344 non chiude il buco.

Il residuo entro-barra tiene ancora RankIC **0.0609** (NW-t **6.40**, n=1610): l’innovazione rispetto a `score_{t-10}` continua a rankare `y_h7`. Non è “tutto nome lento”.

Cosa **non** conclude questo referto: non riassegna FAIL/PASS a `test_shifted_target_degrades`. La soglia pre-reg resta. I numeri sopra sono il materiale per l’emendamento che **tu** proponi (sostituire quel test con l’IC del residuo, con timestamp e motivo). Qui non è stato scritto.

---

## Extra — ICIR 5.90 è annualizzato

Confermato in codice (`baseline.evaluate.summarize_ic`):

```text
ICIR = mean(daily RankIC) / std(daily RankIC, ddof=1) * sqrt(252)
```

Headline top-40: **5.9009**. Scritto esplicito in `results/baseline.json` come `icir_convention` / `rank_ic.top40.icir_annualized=true`.

Con N≈40 il SE di uno Spearman CS daily è ~1/√40 ≈ **0.16**. Un ICIR **daily** (senza √252) pari a 5.9 richiederebbe mean/std ≈ 5.9: impossibile su quel rumore. mean/std daily = 5.90/√252 ≈ **0.37**.

---

## Cosa non è stato fatto

- Nessuna modifica a `reports/gating_ladder_preregistration.md`.
- Nessun ritocco alla soglia di `test_shifted_target_degrades`.
- Nessuno Stage A, nessun retraining.
