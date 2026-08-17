# FASE 1 — curva di decadimento OOS (no training)

**FALSIFICATO.** Emendamento su `test_shifted_target_degrades` **non scritto**. Nessuno Stage A.

Top-40, score OOS esistenti. `IC_k` = mean RankIC(`score_t`, `y_{t+k→t+k+7}`), NW-t lag=7. `rho_k` = mean Spearman CS(`score_t`, `score_{t+k}`). `c` = `rho_k / (IC_k/IC_0)`.

| k | IC_k | NW-t | n | rho_k | IC_k/IC_0 | c |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0803 | 7.80 | 1620 | 1.000 | 1.000 | 1.000 |
| 3 | 0.0705 | 7.16 | 1617 | 0.586 | 0.878 | 0.668 |
| 5 | 0.0660 | 6.99 | 1615 | 0.467 | 0.822 | **0.568** |
| 10 | 0.0585 | 6.74 | 1610 | 0.295 | 0.728 | 0.406 |
| 20 | 0.0483 | 5.01 | 1600 | 0.167 | 0.601 | **0.278** |
| 40 | 0.0534 | 5.12 | 1580 | 0.116 | 0.665 | **0.175** |
| 60 | 0.0583 | 5.52 | 1560 | 0.112 | 0.726 | 0.154 |

Conferma richiesta: `c ∈ [0.32, 0.48]` per k∈{5,10,20,40}. **No** (solo k=10). `c` su quei k è **monotona decrescente** (0.568 → 0.406 → 0.278 → 0.175). `IC_k` resta alto (floor ~0.05, a k=60 ancora 73% di IC_0, NW-t 5.52) mentre `rho_k` collassa a ~0.11.

JSON: `results/fase1_decay_curve.json`.

## Indagine riaperta (niente training)

Spearman CS di `y_h7` sullo stesso top-40: k=5 → 0.246 (overlap 2/7); k=10 → **0.026**; k=20 → 0.005; k=60 → 0.025. Dopo k=10 l’overlap del forward a 7d è zero e **y non è persistente**. Quindi il floor di IC_k non è AR(1) su y e non è SNR×ρ_θ a fattore unico: `score_t` continua a rankare residuali 7d che partono 40–60 barre dopo, mentre il vettore score si rimescola.

Modello “stato persistente + rumore ⇒ c costante ~0.40” è falso. Pre-reg invariata.
