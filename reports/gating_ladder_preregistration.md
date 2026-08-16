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

**Emendamenti obbligatori (2026-08-16, prima di qualsiasi run):** (1) procedura gate+MLP poi GBM congelato; attribuzione riformulata. (2) ricentraggio post-gate + log Cov(x,g). (3) `test_gate_identity_leakage`. (4) shift target ≥ 10. (5) 5 seed, KEEP sulla mediana. (6) ΔSharpe≥0 è guard; CI block-bootstrap obbligatorio. (7) Stage C: N uguale train/eval in headline + riga transfer 120→40. (8) funding: niente fill-zero; canale escluso pre-copertura.

---

## A. Oggetto e non-oggetti

**Oggetto:** quattro estensioni **sopra** le 33 feature A0 già calcolate. Nessuna feature nuova in `x`. Cambia il **pre-processing apprendibile** (gate, RBF, attenzione, interferenza) a monte del ranker; il GBM A0 resta l’headliner di A con HP identici, ma **non** è vero che il learner-stack è intoccato (vedi §E).

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
| scaler A/B | centri RBF, statistiche di `c`, k-means del leakage test, eventuali quantili/standardizzatori di `c`: **fit solo su indici < inizio test del fold** (train dopo purge). Inner holdout ⊂ train. |
| seed NN | **5 seed** `{42, 43, 44, 45, 46}` per ogni componente NN (gate A, RBF/σ B, attenzione C, bilineare D). KEEP sulla **mediana** dei 5. Spread (min, max, IQR) obbligatorio in `LADDER.md`. GBM A0 resta `seed 42 + fold_id` come produzione (non si moltiplica il GBM). Baseline A0 = shot unico seed 42. Non ripetere FuzzyX shot-unico. |
| ricentraggio A | **obbligatorio** dopo il gate, per feature per barra (vedi §E). Unit-std post-gate = braccio separato, non headline. |

---

## C. Universi e ordine di valutazione

- PIT top-N, mediana 30d DV ≤ t, come `build_pit_topn`. Train sempre top-120.
- **Stage A si valuta prima su exec top-40.** Se FAIL KEEP su top-40 → **taglio**. Top-20 non è un secondo tentativo di salvataggio.
- Se KEEP su top-40 → allora si misura top-20 come **informational** (stesso modello, stesso fold). KEEP su top-20 **non** è richiesto per accettare A. Un pass top-20 con fail top-40 è **illegale** come headline (è il cuneo inverso di F1; F1 uccise top-20 e tenne top-40: qui non si inverte a posteriori).
- Stage B/C/D: solo se lo stage precedente è KEEP (non KEEP-FRAGILE). Universo di selezione = **top-40**.
- **Stage C headline:** stessa cardinalità in train e eval = **PIT top-40** (softmax sull’insieme presente quella barra, N≤40). Train-120 / eval-40 **non** è il KEEP: la softmax dipende dal set.
- **Stage C transfer (riga dedicata, non KEEP):** un train a PIT top-120, eval top-40 con softmax **rinormalizzata sui 40**. Riga `C-transfer-120-40` in LADDER. Obbligatoria se C headline gira.
- Stage C cardinalità 30/50/100: **dopo** KEEP headline, un train a PIT top-100, eval 30/50/100 con softmax rinormalizzata sul set di eval. Non KEEP. Non ritoccare C. N=30 dal modello KEEP-40 (subset + rinormalizza) è informational extra, non sostituisce il train-100.

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
5. **`test_shifted_target_degrades`**: y spostato di **+10 barre** (non +5). h=7 a lag +5 lascia 2/7 di overlap (corr attesa ~0.5); a **+10 = embargo (h+3)** l’overlap è zero. FAIL se mean RankIC shifato > **0.5 ×** mean RankIC non shifato (stesso universo, stesso seed mediano) **oppure** se NW t dello IC shifato ≥ 2. Se non crolla = struttura statica = FAIL.
6. **`test_axis_slicing`**: assert shape prima di ogni slice su tensori batched.
7. Universe lookahead top-20/40/120 e seed determinism: come A0, devono PASS.
8. **`test_gate_identity_leakage`** (NUOVO, bloccante su A/B/C; N/A su baseline A0): vedi soglia congelata in §E. Mutual information tra gate (o token attenzione) e identità simbolo sul fold di test. FAIL su un fold → stage FAIL.

**Cancellati:** `test_execution_lag` (lag≥1).

**Baseline FASE 1:** A0 invariato in questo harness → `results/baseline.json`. Tolleranza di riproduzione (pre-dichiarata): \|Δ mean RankIC\| vs 0.0923 su top-20 h=7 full ≤ **0.003** **oppure**, se il campione giorni differisce dal Round F (n_days≠875 per quella cella), si registra n_days e si usa **solo** il `baseline.json` di questo harness come numeratore, senza citare 0.0923 come KEEP. Non si “aggiusta” A0.

---

## E. Stage A — spec (primo da implementare; stop e report)

Gate **mai** sull’identità asset. Nessun embedding ticker, nessuna maschera per simbolo. Days dal listing **non** in forma continua: solo bucket (sotto). Verifica: `test_gate_identity_leakage`.

### E.1 Procedura di training del gate (bloccante — LightGBM non dà gradienti)

Hard-concrete L0 ha bisogno di gradienti. Il GBM A0 non ne produce. Procedura **esatta**, dentro ogni fold, train dopo purge, inner holdout = ultimi 90d del train (come A0):

```
per seed s ∈ {42, 43, 44, 45, 46}:
  1. Fit statistiche di c (percentili CS, flag) solo su date ≤ train_end (post-purge).
  2. Addestra MLP_gate(c) → log_α ∈ R^{33} + hard-concrete L0
     + testa MLP throwaway (2 layer, ≤128) sul y headline,
     early-stop RankIC sull'inner holdout. Unici gradienti del gate.
  3. Congela il gate. Eval: concrete deterministico, g ∈ [0,1]^{33}.
  4. x_hat = x ⊙ g
  5. Ricentra per feature per barra:
       x_bar_{i,f,t} = x_hat_{i,f,t} − mean_{j ∈ barra t}(x_hat_{j,f,t})
     Headline: NESSUNA ri-scalatura a std 1. Braccio opzionale unit-std = tabella a parte.
  6. Addestra LightGBM A0 identico (Huber, HP, early-stop RankIC, seed 42+fold_id)
     su x_bar. Non sull'x grezzo, non su x_hat non ricentrato.
  7. OOS: gate congelato + GBM congelato. Nessun refit su val.
```

La testa MLP throwaway **non** è il ranker headline. Un ranker MLP al posto del GBM resta ablation, non KEEP. Due teste (rank + dispersione) fuori da A headline.

**Attribuzione (riformulata):** «A non tocca il learner» è **falso** in senso stretto. Una MLP entra a monte, addestrata con una testa usa-e-getta, poi congelata. Il GBM di A ha obiettivo/HP/early-stop identici ad A0; la **legge delle feature** no. Δ vs A0 è attribuibile a **(gate addestrato + ricentraggio + shift della misura di x)**, non a «solo un mask sulle stesse 33 con learner intatto». Non si può isolare il gate dal ricentraggio: il ricentraggio è obbligatorio proprio perché senza di esso il gate re-inietta le esposizioni CS (vedi E.2). Non si confronta A senza ricentraggio come headline.

### E.2 Ricentraggio post-gate (bloccante)

Le 33 sono CS-z: mean_i(x_f)=0 entro barra. Dopo il gate:

```
mean_i(x_f · g_f) = Cov_i(x_f, g_f)   (perché mean x = 0)
```

g dipende da c (vol, ADV, beta, …) quindi il gating **re-inietta** le esposizioni che la CS-z toglieva. Meccanismo candidato del cuneo F1 (IC su / Sharpe giù): il GBM vede di nuovo un fattore vol/ADV/beta mascherato da “feature gated”.

- Obbligatorio: ricentrare `x_gated` per feature per barra **prima** del GBM (passo 5).
- Obbligatorio: loggare `Cov(x_f, g_f)` per feature per barra in `results/stage_a_gates.parquet` (cov campionaria sui nomi presenti quella barra).
- Unit-std dopo il ricentraggio: opzionale, braccio separato, mai headline, mai KEEP.

Se i gate sono densi: debug causa radice (L0, temperatura, collasso della testa), non clip.

### E.3 Vettore `c` (8–15 dim, per asset-barra)

Causale, poi CS-standardizzato entro barra **solo sui canali definiti quella barra**. `c` non entra in `FEATURE_COLS`.

1. vol realizzata percentile CS (`yz_vol_30_raw`)
2. ADV / liquidità percentile (`dollar_volume` o mediana 30d)
3. beta rolling BTC (`beta_btc_60_raw`)
4. **funding:** Vision. Copertura di fatto da **2020-09-01** (congelato; se i dump partono un altro giorno si timestampa e si ri-esegue). **Prima di coverage_start il canale z-score NON esiste** (non si concatena). Flag dedicato `funding_available ∈ {0,1}` **non** CS-z. **Vietato** riempire di zeri: CS-z di un vettore costante è 0/0 e il confine 2019-09→2020-09 diventa un indicatore d’epoca dentro lo z. Dopo coverage_start: z-score **solo** tra i nomi con rate non-missing quella barra; missing post-copertura → `funding_missing=1` su quella riga, **niente** 0 imputato nello z. Se una barra post-copertura non ha nessun rate, quella barra non ha il canale z (solo i flag).
5. proxy spread/depth: Amihud (`amihud_14` raw se c’è, altrimenti CS-z). Non L2.
6. **listing (coarsened, mai days continui):**
   - `listed_before_panel=1` se first-bar ≤ 2019-09-01 (BTC non è “giovane”).
   - altrimenti bucket **`<90d` / `90–365d` / `>365d`** (one-hot; 2 dummy libere + il flag panel).
   - **Vietato** `days = (t−first).days` continuo o percentile fine: è quasi-costante per simbolo (δ=1/giorno) e fingerprint dei listing post-2019. `listed_before_panel` sui top-40 identifica i major: per questo il leakage test è bloccante.
7. corr rolling al fattore medio universo
8. dispersione CS dei ritorni della barra (scalare di regime, uguale per tutti i nomi della barra)
9. momentum e reversal brevi in percentile (`ret_7` / `rev_1` o `rev_3` raw → percentile CS)

**Gating:** MLP condivisa `c → log_α ∈ R^{33}`. Hard-concrete L0 (Louizos 2018): stretch (−0.1, 1.1), β=2/3, penalità L0 attesa `Σ sigmoid(log_α − β·log(−γ/ζ))`. Eval deterministico. Poi E.1 passi 4–7.

### E.4 `test_gate_identity_leakage` — soglia congelata

Il vincolo «niente identità ticker» è architetturale e va **misurato**. I(g; S | c)=0 è tautologico se g=f(c). Test non vacuo: il gate (e i canali listing) non devono identificare il simbolo oltre il contesto **senza listing**.

**Stimatore (ogni fold OOS, seed mediano + report min/max seed):**

- g = gate eval 33-d sulle righe test.
- Fit **k-means k=8** sui g del **train** del fold; assegna il test. Stesso k-means protocollo su `c_no_listing` = canali {1,2,3,5,7,8,9} (niente listing, niente funding z; i flag funding/listing esclusi).
- I_MM = mutual information plugin **Miller–Madow**, in **bit**, tra cluster e `symbol`.
- H(S) = entropia empirica di `symbol` sul fold test.

**KEEP/PASS del test (tutti i fold OOS, mediana seed):**

> **FAIL se** `I_MM(kmeans_8(g); S) / H(S) > 0.30`
> **oppure se** `I_MM(g; S) − I_MM(kmeans_8(c_no_listing); S) > 0.20 bit`.

**Motivazione (congelata):** H(S)≤log2(40)≈5.32 bit. k=8 dà I max = 3 bit. Un partizione legittima tipo tertile ADV × vol è ~3–9 celle, NMI fino ~0.5–0.6. **0.30** (~1.6 bit) è una split ~3-way: i tre bucket età + major vs resto, non un lookup a 40. Il secondo taglio (**+0.20 bit** vs contesto senza listing) è ~2–4 SE di un plugin MI su n≈90×40 con k=8; listing può marcare la coorte 2019 ma non comprare un’identità extra. `listed_before_panel` sui top-40 è identità mascherata dei major: se alza I di più di 0.20 bit rispetto a vol/ADV/beta, FAIL.

Stesso test su B (g resta il gate) e su C (sostituire g con il pooling vector per-asset in input all’ultimo layer, 33-d o d_model che sia; stesso k=8, stesse soglie).

Aspettativa pre-dichiarata (non è keep): A e C passano, D muore, B incerto. F1: context IC↑ Sharpe↓ su top-20 → A deve sopravvivere su **top-40** al **guard** Sharpe (G.3), non “tenere per l’IC”.

Output obbligatorio A: `results/stage_a_gates.parquet` (g, regime, **Cov(x,g) per feature per barra**, seed) + heatmap gate × regime.

**Re-label arm:** y rank-YZ, top-40, **5 seed**, mediana in tabella a parte. Non KEEP.

---

## F. Stage B / C / D (solo se il precedente è KEEP, non FRAGILE)

**B:** K∈{4,5,6} scelto **a priori: K=5**. μ = quantili della feature sul **train del fold**; σ apprendibile; `exp(−0.5((x−μ)/σ)²)`. **Dopo** il gate; il gate opera sulla feature originale. Non anticipare B durante A.

**C:** set transformer sull’asse **asset**, 1–2 layer, poche teste, dim piccola, no PE, mask assenti. N effettivo = n. barre. **Headline KEEP: train=eval=PIT top-40** (softmax su quel set). Softmax su 120 ≠ softmax su 40: non si addestra a 120 e si valuta a 40 silenziosamente. Transfer 120→40 e cardinalità 30/50/100: §C, righe dedicate, non KEEP. `test_gate_identity_leakage` sul token per-asset.

**D:** bilineare low-rank 8–16 (rango **12** a priori). Complex-valued **vietato** finché D non KEEP. Aspettativa: D fallisce; si gira comunque.

Taglio: se uno stage non KEEP, **non entra** nei successivi. Niente “quasi”.

---

## G. Criterio KEEP (verbatim, meccanico)

Sia S lo stage sotto test, P lo stage accettato precedente (P = A0 harness per Stage A). Universo di selezione U = **PIT top-40**. Horizon **h=7**. Lag **0**. τ causale `fold_train`, griglia {60,70,80,90}; **median-τ del candidato** vs A0 ledger τ=70 su P2 per il confronto libro; per il ladder interno, ΔSharpe su **giorni identici**, stesso τ-mode. Costi 1x = nominale sopra. Funding on. Statistiche KEEP = **mediana sui 5 seed NN** (A0: un seed).

> **S è KEEP su top-40 solo se TUTTI i punti valgono. Altrimenti CUT.**
>
> 1. Suite anti-leak PASS, incluso `label_shuffle` \|mean RankIC\| < 0.005 e `test_gate_identity_leakage` su A/B/C.
> 2. **Screen IC = evidenza (necessario, non sufficiente):** Δ mean RankIC trail-18m (S−P) ≥ **+0.005** AND Δ mean RankIC full-OOS ≥ **0** AND Newey–West t-stat del RankIC di S (lag=7) **> 2** AND nessun fold OOS con Δ RankIC < **−0.003**. Tutto sulla mediana seed.
> 3. **Guard Sharpe, non evidenza:** Δ net Sharpe trail-18m (S−P) ≥ **0** sul libro top-40 h=7, costi 1x, lag 0, funding on. **Non si alza questa soglia.** SE(ΔSharpe) su 18m top-40 è ~0.2–0.4: Δ≥0 è un coinflip e **non** conta come prova di miglioramento. Serve solo a bloccare il cuneo F1: **se lo screen IC passa e Δ Sharpe trail < 0 → KILL immediato**. Non si “tiene per l’IC”.
> 3b. **CI obbligatorio (disclosure, non KEEP):** block bootstrap **stazionario** (Politis–Romano) sui net returns **giornalieri** di S e P, giorni identici, trail-18m. Lunghezza media blocco **ℓ=10** ( = embargo h+3; overlap-zero vs y a 7d). B=**2000**. Report in `LADDER.md` il ΔSharpe punto (mediana seed) e il **CI 90%** (p5–p95) di ΔSharpe. Non usare il CI per alzare G.3. Se il punto ≥0 e il CI è largo e attraversa zero, KEEP può comunque scattare (guard ok, evidenza = G.2); la colonna CI è l’onestà.
> 4. Full-OOS net Sharpe: Δ ≥ **−0.05** (non può bruciare il sample intero per un trail rumoroso). Non sostituisce il guard (3).
> 5. Decile spread **netto** trail-18m: Δ ≥ 0 (stesso segno del guard; se IC passa e il netto decile crolla, CUT: i costi di turnover stanno mangiando il rank).
> 6. Sensitività 2x e 3x: si riportano sempre. **KEEP** richiede il **guard (3)** anche a **2x**. Se (3) vale solo a 1x → **KEEP-FRAGILE** = **CUT per lo stack** (non è P per B/C/D). 3x è informativo; fail 3x da solo non taglia. CI bootstrap anche a 2x in LADDER.
> 7. Nessun fold con Δ net Sharpe trail-window del fold < **−0.10** (soglia più larga dell’IC perché Sharpe di 90d è rumoroso; due fold con Sharpe netto full-fold < 0 → CUT comunque, come da harness).
>
> KEEP-FRAGILE e CUT si scrivono in `results/LADDER.md` con una frase di meccanismo, non solo il numero. Colonne minime LADDER: stage, seed median+spread, ΔIC, NW t, ΔSharpe trail, CI90 ΔSharpe, 1x/2x, KEEP/CUT/FRAGILE, motivo.

**Top-20:** mai KEEP-rule di questo ladder. Solo riga informational dopo KEEP top-40.

**Non si adotta un nuovo COMBO** in questo programma. Un KEEP di C non rimpiazza Sleeve B finché non esiste un addendum libro separato. Qui si accetta solo “stage nel ranker”.

---

## H. Deliverable per stage

1. Codice su **main locale**, commit atomici, messaggio inglese.
2. `results/stage_<x>.json`: metriche per fold + aggregate, **5 seed + mediana**, commit hash, config, wall time, costi 1x/2x/3x, lag0/lag1, CI90 ΔSharpe.
3. Curva equity netta vs baseline (lag 0, 1x, seed mediano).
4. Esito suite anti-leak incluso `test_gate_identity_leakage` (A/B/C) e shifted-target **+10**.
5. Riga `results/LADDER.md`: stage, mediana+spread seed, ΔIC, NW t, ΔSharpe trail, **CI90 block-bootstrap**, 1x/2x, KEEP/CUT/FRAGILE, motivo.
6. Se CUT: una frase sul **perché** (meccanismo).
7. Stage A: `results/stage_a_gates.parquet` (**Cov(x,g) per feature per barra**) + heatmap.
8. Stage C: riga `C-transfer-120-40` se C headline gira.

Ordine: FASE 1 verde → FASE 2 (y headline già A0; re-label solo braccio A) → A (stop+report) → B → C → D. Non anticipare stage. Non ottimizzare HP sul fold di test.

---

## I. Checklist FASE 1 rossa (non si gira A)

- `/data/quant` assente
- `make_folds` modificato
- y headline ≠ residuo h=7
- baseline.json mancante o non riproduce A0 entro la regola (B)
- watchdog assente
- soglie di questa pre-reg cambiate dopo un run
- gate addestrato “attraverso” LightGBM (nessun gradiente) o GBM su `x_gated` **non** ricentrato
- funding z riempito di 0 pre-2020-09
- days_since_listing continuo in `c`
- Stage C headline con N_train ≠ N_eval
- KEEP dichiarato su un solo seed NN

---

## J. FASE 0 map (accepted; no code)

Ranker = frozen A0 LightGBM. Features: 33 CS-z in `baseline/features.py`. Target: residual BTC forward log-return, winsor 1/99, h∈{7,10}, primary 7. Expanding WF. Costs in backtest only today (5+3 / 10+8 assumed, not live venue). Production lag 0. PIT top-120 train / 20 or 40 exec. Round F F1 context: KILL top-20 (IC up, P1 trail Sharpe Δ −0.320). Stage A: MLP gate + ricentraggio + GBM A0; KEEP = screen IC (evidenza) + guard Sharpe trail ≥0 sul top-40 (non evidenza). ΔSharpe CI block-bootstrap obbligatorio.
