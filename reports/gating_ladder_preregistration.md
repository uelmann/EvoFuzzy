# Gating contestuale + attenzione CS sul ranker A0

**BACKTEST NOT RUN.** This file exports the frozen FASE 0 follow-up: what FuzzyX-v1f PARK is, and the complete pre-registration for the gating ladder. Written before any FASE 1+ number. Commit on a parked export branch so it is readable on GitHub; implementation runs on **main locally** with `/data/quant` mounted.

Does not replace COMBO / A0 / LightGBM.

---

# 1. Cosa contiene FuzzyX-v1f PARK

**Oggetto:** policy congiunta `{+1,0,−1}` su un universo PIT, **non** un ranker LightGBM. Non sostituisce COMBO/A0. Shot unico, seed 42. Verdict meccanico **PARK**. Mode **`LOCAL-RESTRICTED`** (40 perps Binance Vision scaricati sul pod cloud; `/data/quant` assente). VIABLE ufficiale disabilitato anche se Sharpe≥0.

Source: `reports/fuzzyx_v1f_report.md`, `reports/fuzzyx_addendum_v1f.md`, `reports/fuzzyx_architecture.md`, `config_fuzzyx.yaml`.

## Architettura (stack v1, invariato da v1e; cambia solo costruzione pesi + loss)

```
33 feature A0 (CS-z, clip ±5)  →  identiche a FEATURE_COLS, nessuna extra
        │
L1  3 Gaussiane per feature (Low/Mid/High), μ init −1/0/+1, σ≈0.85
        │
L2  MarketGate MASTER/TFT: token di mercato (aggregati CS) → softmax su F
    α condiviso su tutti i nomi del giorno, τ=0.5
        │
L3  24 regole sparse IGNORE/AND/NAND, t-norma prodotto, teste LONG/SHORT/FLAT
        │
L4  DeepSets / residuo CS (~17.7k param). Attenzione 1-layer = ablation, non default
        │
train: p = P(L)−P(S)
eval:  argmax → {+1, 0, −1}
```

**v1f construction (unica differenza vs v1e):**

```
p ← p − mean_mascherata(p)     # dollar-neutral, Σp = 0
w ← p / Σ|p|                   # unit gross
loss = −mean(net weekly PnL)   # costi 5+3 già nel PnL
```

Niente Pearson, niente occupancy nuke, `λ_*=0`. Rebalance **7 giorni**, hold ffill. Universo di design: PIT top-30 volume; run locale: **40 simboli**. WF A0: 730/90/90, purge 7, embargo +3. Adam 1e-3, 80 epoche, patience 12. ~17 668 parametri.

v1–v1e restano congelati (CONTAMINATED: path-corr × esposizione netta residua; lo shuffle entro data teneva la media CS).

## Feature

Le stesse 33 di A0. Nessun input nuovo. Il “fuzzy” è **lift** Low/Mid/High + regole, non un blocco feature.

## Risultati misurati (hard weekly book, lag 0, costi on)

| voce | valore |
|---|---|
| leakage (`feature_lookahead`, `universe_lookahead_top30`, `seed_determinism`) | **PASS** |
| shuffle-bias mean weekly net PnL (fold 0 e 17, 10 shuffle) | **PASS** (`net_expo` ~ 0) |
| Sharpe netto weekly full-OOS | **−2.000** |
| mean weekly PnL | **−0.24%** |
| equity_end | 0.54 (−46%) |
| MaxDD | 0.46 |
| long / short / traded | 0.4% / 99.5% / ~100% |
| vs A0 Sleeve A | **SKIP** (pred A0 non su disco) |
| n_reb / n_symbols | 232 / 40 |

Dopo demean il buco `E[w·π(r)] = mean(r)·Σw` è chiuso. Lo skill CS residuo non c’è: collasso almost-all-short; dopo demean è “chi non è short”, e perde. Soft stessa forma, più mite (−0.05%/settimana).

**Perché PARK:** keep rule pre-registrata: leakage PASS, shuffle PASS, Sharpe < 0 → PARK. Pre-registrato: **niente quinto tweak di loss su questo stack.**

## Stage A duplica FuzzyX?

**No come libro, sì come cugini di layer.** Non è un rerun e non va mixato.

| pezzo FuzzyX | ladder A–D | stesso oggetto? |
|---|---|---|
| L2 MarketGate: **un** `α_f` per giorno, softmax, condiviso sui nomi | Stage A: MLP(`c`) → hard-concrete L0 **per (asset, barra)**, `c` senza identità ticker | **No.** Stessa tesi (“accendi feature in regime”), meccanica diversa |
| L1 3 Gaussiane su CS-z | Stage B RBF K=4–6, μ = quantili **di fold** | **Quasi sì** se B parte dopo A. Non copiare i μ FuzzyX (−1/0/+1) |
| L4 DeepSets / attenzione set | Stage C set transformer sull’asse asset | **Stesso asse.** C è il test pulito su ranker A0; L4 FuzzyX è una testa su regole weekly |
| L3 rule bank + argmax `{+1,0,−1}` weekly + path/PnL | nessuno stage A–D | **No.** Ranker daily, y residuo h=7, Huber/listwise, τ-book |
| universo 30 weekly LOCAL-RESTRICTED | top-40 daily, volume `/data/quant` | **No** |

Conclusione operativa: v1f **non** è un prior empirico su Stage A. È un prior negativo su “policy discreta + DeepSets + loss di path/PnL sulle stesse 33”. Stage A si valuta sul ranker A0, da zero. Non importare pesi FuzzyX, non accendere L1/L3 “perché ci sono già”. Se B o C passeranno, la sovrapposizione con L1/L4 va dichiarata nel report di quello stage, non usata come scusa per tenere FuzzyX.

---

# 2. Pre-registrazione — gating contestuale + attenzione CS sul ranker

**Scritta prima di qualsiasi numero di FASE 1+.** Da committare su **main in locale** prima dei run. Non si modifica a posteriori. Se una soglia deve cambiare: timestamp + motivo nel file, poi si ri-esegue tutto. Backtest only. COMBO/A0 non si toccano finché un KEEP meccanico non li sostituisce (questo programma **non** adotta un nuovo libro di default; adotta solo stage del ranker).

**Blocco ambiente:** FASE 1 **non parte** se `/data/quant` non è montato. Panel LOCAL-RESTRICTED / 40 perps **non** è baseline ufficiale.

**Git:** niente commit di implementazione sul branch FuzzyX parcheggiato (`cursor/fuzzyx-architecture-2f45`).

---

## A. Oggetto e non-oggetti

**Oggetto:** quattro estensioni **sopra** le 33 feature A0 già calcolate. Nessuna feature nuova in `x`. Cambia solo il learner / il gating / l’attenzione.

**Non-oggetti:** FuzzyX (qualsiasi shot), Kronos, GRU, path signatures, microstructure block, LangChain/etc., fee live, book L2, y rank-YZ come headline, rolling WF, lag≥1 come contratto.

**Baseline da battere:** A0 LightGBM congelato, hash `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`, seed 42, y residuo BTC winsorizzato h=7, RankIC top-20 h=7 full **0.0923** (Round F tabella). Ledger: P1 1.207/1.009, P2 1.470/0.723, COMBO 1.711/0.997 (τ causale). Il numero **di questo ladder** è A0 **nello stesso harness** salvato in `results/baseline.json` (seed, hash, config, fold). Se l’harness non riproduce A0 a tolleranza dichiarata sotto, FASE 1 è rossa e gli stage non partono.

---

## B. Contratti fissati (FASE 0 Q1–Q5)

| item | valore congelato |
|---|---|
| y headline | `y_h7` = log-ret 7d − β_BTC_60_raw · BTC_fwd, winsor 1/99 per data. **Invariato.** Vol YZ resta solo al sizing (`yz_vol_30_raw`). |
| y secondario | rank CS di (forward residuo / yz_vol_30_raw), `[0,1]` per barra. **Un solo run, Stage A, top-40, braccio “re-label”. Mai headline. Mai criterio KEEP.** |
| orizzonte | **7** per il ladder. h=10 non è selection. |
| lag | **0** per tutti. Lag 1 = riga stress in ogni report, non KEEP. `test_execution_lag` **cancellato**. |
| WF | `make_folds` **invariato**: expanding, min 730 / val 90 / step 90 / purge h / embargo h+3 / inner holdout 90d sul train. Val fold mai per early stop / HP / feature select. |
| costi nominale | top-20: 5+3 bps. top-40 ranks 21–40: 10+8 bps + ADV cap 0.5% book $1e6. Funding Vision come A0. |
| costi nel KEEP | sì, criterio su Sharpe **netto**. Non solo report. |
| sensitività costi | moltiplicatore sui bps assunti ∈ {1x, 2x, 3x}. Funding non scalato. Se KEEP solo a 1x → stage **KEEP-FRAGILE**, non entra negli stage successivi come base accettata (si taglia per lo stack; resta riga in LADDER). |
| scaler A0 | CS-z entro barra: `test_scaler_fold_isolation` **vacuo** su A0. |
| scaler A/B | centri RBF, statistiche di `c`, eventuali quantili/standardizzatori di `c`: **fit solo su indici < inizio test del fold** (train dopo purge). Inner holdout ⊂ train. |

---

## C. Universi e ordine di valutazione

- PIT top-N, mediana 30d DV ≤ t, come `build_pit_topn`. Train sempre top-120.
- **Stage A si valuta prima su exec top-40.** Se FAIL KEEP su top-40 → **taglio**. Top-20 non è un secondo tentativo di salvataggio.
- Se KEEP su top-40 → allora si misura top-20 come **informational** (stesso modello, stesso fold). KEEP su top-20 **non** è richiesto per accettare A. Un pass top-20 con fail top-40 è **illegale** come headline (è il cuneo inverso di F1; F1 uccise top-20 e tenne top-40: qui non si inverte a posteriori).
- Stage B/C/D: solo se lo stage precedente è KEEP (non KEEP-FRAGILE). Universo di selezione = **top-40**.
- Stage C: bonus obbligatorio top-30/50/100 **dopo** KEEP su top-40, stesso modello addestrato una volta. Non è un keep rule extra; è sensitività pre-dichiarata. Fallimento 30/50/100 si riporta; non ritocca C.

---

## D. Harness (FASE 1 deve essere verde prima di ogni modello nuovo)

Watchdog su ogni test/train/benchmark: `PYTHONUNBUFFERED=1 … 2>&1 | tee logs/<nome>_$(date +%s).log`; poll 90s; kill se 180s senza nuove righe; report fold/stage/stato. Nessun fallback silenzioso; fold che non converge → run FAIL (niente NaN, niente skip). Niente `asyncio.wait_for` sopra retry senza log dei retry.

**Metriche (unico set):** per fold e aggregate:

- Rank IC medio CS per barra; t Newey–West lag = 7
- IC IR (media/std IC)
- spread decile lordo e netto
- turnover medio giornaliero
- Sharpe netto, maxDD, durata DD
- lag-1 Sharpe (stress, non KEEP)

Aggregata buona con due fold negativi sul criterio primario = **fallimento**.

**Anti-leak, ogni stage (bloccanti):**

1. **`label_shuffle`** (`baseline/gates.py`, non reinventare): 25 permutation entro data; \|mean RankIC\| < **0.005**. IC significativo su y shufflata = leak = FAIL run. (Il nome `test_permutation` del prompt originale **punta a questo test**.)
2. **`feature_lookahead`**: già esistente; invariato.
3. **`test_no_lookahead_filters`**: grep `fft`, `filtfilt`, `savgol`, `center=True`, `.interpolate()` senza `limit_direction='forward'`. Ogni hit: giustificazione in pre-reg o rimozione. Nuovo hit dopo freeze = FAIL.
4. **`test_scaler_fold_isolation`**: skip/N/A su baseline A0; assert su A/B (e C/D se stateful). Fit solo su date < test start del fold.
5. **`test_shifted_target_degrades`**: y spostato +5 barre → RankIC e Sharpe trail devono crollare sotto le soglie sotto. Se non crollano = struttura statica = FAIL.
6. **`test_axis_slicing`**: assert shape prima di ogni slice su tensori batched.
7. Universe lookahead top-20/40/120 e seed determinism: come A0, devono PASS.

**Cancellati:** `test_execution_lag` (lag≥1).

**Baseline FASE 1:** A0 invariato in questo harness → `results/baseline.json`. Tolleranza di riproduzione (pre-dichiarata): \|Δ mean RankIC\| vs 0.0923 su top-20 h=7 full ≤ **0.003** **oppure**, se il campione giorni differisce dal Round F (n_days≠875 per quella cella), si registra n_days e si usa **solo** il `baseline.json` di questo harness come numeratore, senza citare 0.0923 come KEEP. Non si “aggiusta” A0.

---

## E. Stage A — spec (primo da implementare; stop e report)

Gate **mai** sull’identità asset. Nessun embedding ticker, nessuna maschera per simbolo.

**Vettore `c` (8–15 dim, per asset-barra), causale, poi CS-standardizzato entro barra:**

1. vol realizzata percentile CS (`yz_vol_30_raw`)
2. ADV / liquidità percentile (`dollar_volume` o mediana 30d)
3. beta rolling BTC (`beta_btc_60_raw`)
4. funding z-score (Vision; missing = 0 **esplicito**, non silenzioso; flag di copertura in report)
5. proxy spread/depth: **Amihud già in A0** (`amihud_14` raw pre-CS se disponibile, altrimenti CS-z). Non L2.
6. **giorni dal listing**, con left-censor: se first-bar del simbolo ≤ 2019-09-01 (inizio panel) → bucket/flag **`listed_before_panel=1`** e days **non** interpretati come età (BTC non risulta “giovane”). Simboli nati dopo: days = (t − first_close).days, poi percentile CS tra i non-censurati; i censurati hanno un canale binario dedicato.
7. corr rolling al fattore medio universo
8. dispersione CS dei ritorni della barra (scalare di regime, uguale per tutti i nomi della barra)
9. momentum e reversal brevi in percentile (`ret_7` / `rev_1` o `rev_3` raw → percentile CS)

`c` è **contesto**, non si concatena in `FEATURE_COLS` come nuove feature del GBM baseline. Il GBM/MLP di testa vede `x_gated` sulle 33.

**Gating:** MLP condivisa `c → log_α ∈ R^{33}`. Hard-concrete L0 (Louizos 2018): stretch (−0.1, 1.1), β=2/3, penalità L0 attesa `Σ sigmoid(log_α − β·log(−γ/ζ))`. Eval: gate deterministico. `x_gated = x ⊙ g`. Testa: MLP 2 layer, ≤128 unità, sul y headline (residuo h=7). **Non** si tocca LightGBM in A se l’esperimento è “stesso learner”:

Precisazione KEEP-attribution: A è “unico stage che non tocca il learner”. Quindi **A headline = LightGBM Huber identico ad A0 su `x_gated`**, stesso early-stop RankIC, stessi HP. La MLP testa è **ablation**, non headline. Se si mixa learner nuovo, Δ non è attribuibile al gate. La MLP due-teste (rank + dispersione) è **fuori da A headline**; se si vuole, un braccio diagnostic dopo KEEP di A, non nel criterio.

Aspettativa pre-dichiarata (non è keep): A e C passano, D muore, B incerto. F1 history: context IC↑ Sharpe↓ su top-20 → qui A deve sopravvivere su **top-40 netto**.

Output obbligatorio A: `results/stage_a_gates.parquet` + heatmap gate × regime. Se i gate sono densi: **debug causa radice**, non clip/soglia.

**Re-label arm:** un run A su y rank-YZ, top-40, stesso seed. Tabella a parte. Non KEEP.

---

## F. Stage B / C / D (solo se il precedente è KEEP, non FRAGILE)

**B:** K∈{4,5,6} scelto **a priori: K=5**. μ = quantili della feature sul **train del fold**; σ apprendibile; `exp(−0.5((x−μ)/σ)²)`. **Dopo** il gate; il gate opera sulla feature originale. Non anticipare B durante A.

**C:** set transformer sull’asse **asset**, 1–2 layer, poche teste, dim piccola, no PE, mask assenti. N effettivo = n. barre. Poi eval top-30/50/100.

**D:** bilineare low-rank 8–16 (rango **12** a priori). Complex-valued **vietato** finché D non KEEP. Aspettativa: D fallisce; si gira comunque.

Taglio: se uno stage non KEEP, **non entra** nei successivi. Niente “quasi”.

---

## G. Criterio KEEP (verbatim, meccanico)

Sia S lo stage sotto test, P lo stage accettato precedente (P = A0 harness per Stage A). Universo di selezione U = **PIT top-40**. Horizon **h=7**. Lag **0**. τ causale `fold_train`, griglia {60,70,80,90}; **median-τ del candidato** vs A0 ledger τ=70 su P2 per il confronto libro; per il ladder interno, ΔSharpe su **giorni identici**, stesso τ-mode. Costi 1x = nominale sopra. Funding on.

> **S è KEEP su top-40 solo se TUTTI i punti valgono. Altrimenti CUT.**
>
> 1. Suite anti-leak PASS, incluso `label_shuffle` \|mean RankIC\| < 0.005.
> 2. **Screen IC (necessario, non sufficiente):** Δ mean RankIC trail-18m (S−P) ≥ **+0.005** AND Δ mean RankIC full-OOS ≥ **0** AND Newey–West t-stat del RankIC di S (lag=7) **> 2** AND nessun fold OOS con Δ RankIC < **−0.003**.
> 3. **Primario — Sharpe netto trailing:** Δ net Sharpe trail-18m (S−P) ≥ **0** sul libro top-40 h=7, costi 1x, lag 0, funding on. **Se Δ RankIC ≥ soglia e Δ Sharpe trail < 0 → KILL immediato** (cuneo F1). Non si “tiene per l’IC”.
> 4. Full-OOS net Sharpe: Δ ≥ **−0.05** (non può bruciare il sample intero per un trail rumoroso). Non sostituisce (3).
> 5. Decile spread **netto** trail-18m: Δ ≥ 0 (stesso segno del primario; se IC/Sharpe passano e il netto decile crolla, CUT: i costi di turnover stanno mangiando il rank).
> 6. Sensitività 2x e 3x: si riportano sempre. **KEEP** richiede (3) anche a **2x**. Se (3) vale solo a 1x → **KEEP-FRAGILE** = **CUT per lo stack** (non è P per B/C/D). 3x è informativo; fail 3x da solo non taglia.
> 7. Nessun fold con Δ net Sharpe trail-window del fold < **−0.10** (soglia più larga dell’IC perché Sharpe di 90d è rumoroso; due fold con Sharpe netto full-fold < 0 → CUT comunque, come da harness).
>
> KEEP-FRAGILE e CUT si scrivono in `results/LADDER.md` con una frase di meccanismo, non solo il numero.

**Top-20:** mai KEEP-rule di questo ladder. Solo riga informational dopo KEEP top-40.

**Non si adotta un nuovo COMBO** in questo programma. Un KEEP di C non rimpiazza Sleeve B finché non esiste un addendum libro separato. Qui si accetta solo “stage nel ranker”.

---

## H. Deliverable per stage

1. Codice su **main locale**, commit atomici, messaggio inglese.
2. `results/stage_<x>.json`: metriche per fold + aggregate, seed, commit hash, config, wall time, costi 1x/2x/3x, lag0/lag1.
3. Curva equity netta vs baseline (lag 0, 1x).
4. Esito suite anti-leak.
5. Riga `results/LADDER.md`: stage, ΔIC, NW t, ΔSharpe trail, 1x/2x, KEEP/CUT/FRAGILE, motivo.
6. Se CUT: una frase sul **perché** (meccanismo).
7. Stage A: `results/stage_a_gates.parquet` + heatmap.

Ordine: FASE 1 verde → FASE 2 (y headline già A0; re-label solo braccio A) → A (stop+report) → B → C → D. Non anticipare stage. Non ottimizzare HP sul fold di test.

---

## I. Checklist FASE 1 rossa (non si gira A)

- `/data/quant` assente
- `make_folds` modificato
- y headline ≠ residuo h=7
- baseline.json mancante o non riproduce A0 entro la regola (B)
- watchdog assente
- soglie di questa pre-reg cambiate dopo un run

---

## J. FASE 0 map (accepted; no code)

Ranker = frozen A0 LightGBM. Features: 33 CS-z in `baseline/features.py`. Target: residual BTC forward log-return, winsor 1/99, h∈{7,10}, primary 7. Expanding WF. Costs in backtest only today (5+3 / 10+8 assumed, not live venue). Production lag 0. PIT top-120 train / 20 or 40 exec. Round F F1 context: KILL top-20 (IC up, P1 trail Sharpe Δ −0.320). This Stage A is a different gate (no ticker identity, hard-concrete L0 from context vector) but the same thesis; KEEP is net Sharpe trailing on top-40 first.
