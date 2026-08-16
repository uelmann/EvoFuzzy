# FuzzyX — architecture (design, not a book)

**Status:** design + numpy prototype. Not trained. Does not replace COMBO / A0.
**Prototype:** `fuzzyx/`, config `config_fuzzyx.yaml`, smoke `python -m fuzzyx.smoke`.
**Inputs:** A0 LightGBM features (`baseline/features.py`, 33 CS-z columns).
**Universe:** CMC PIT top-30, volume first (mcap fallback), rebalance every 7 days.
**Output:** `{+1, 0, −1}` per asset, joint over the cross-section.

The notebook `WRKS_L_S_NNET_Nov_22_` is used only as a prompt: keep the fuzzy lift, the hard feature mask idea, the path loss, and the discrete actions. Do not keep the complex-128 MLP + PyGAD search as the production backbone.

---

## Verdict

The idea is right in four places and wrong in two.

Keep:

1. **Fuzzy memberships as a lift**, not as the whole model. Raw CS-z features become Low / Mid / High in `[0, 1]`. That is the notebook CFS *amplitude*.
2. **A learned on/off over inputs.** The notebook `x_null` binary mask is the right instinct. Make it *soft and market-conditioned*, not a GA bitstring.
3. **Explicit AND/OR trading rules.** Readable conjunctions, then a disjunction into long / short / flat.
4. **A path loss on the book**, not a pointwise log-loss. The notebook core
   `corr(cum, t) · (1 − maxDD) · (1 − DD duration)`
   plus occupancy floors (`L ≥ 0.20`, `S ≥ 0.30`, traded `≥ 0.25`) is the right *family*.

Change:

1. **Do not put LightGBM in the joint-reasoning seat.** Trees cannot see all 30 names in one forward pass, cannot backprop a path-DD loss, and cannot condition a feature gate on today's market token. Keep A0 as the baseline to beat, and optionally as a teacher. The joint reasoner is a *tiny* cross-section transformer.
2. **Do not train a complex-valued net with a genetic algorithm over all weights.** That is what the notebook does (`Net` / `Net_X` + `pygad.torchga`). It does not scale to a 30-asset date batch and it is not SOTA. Adam on a differentiable graph; evolutionary search only if you later want to tune membership centers as a small outer loop (the original EvoFuzzy role).

The Phase E GRU already failed the skill gate on this book (`reports/phaseE1_report.md`, `reports/phaseE1b_report.md`). Path signatures were killed. The complexity block was killed. So this design is *not* "another sequence model." Tokens are **assets on a decision date**, not a 60-bar history of one coin.

---

## What SOTA actually says (2024–2026)

| piece you asked for | SOTA analogue | takeaway for us |
|---|---|---|
| Accendi/spegni input | **MASTER** market-guided gating (AAAI 2024, [arXiv:2312.15235](https://arxiv.org/abs/2312.15235)); TFT Variable Selection Networks (Lim et al. 2021) | Softmax over features, conditioned on a market vector. Temperature `< 1` sharpens toward on/off. |
| Ragiona su tutti gli asset | **MASTER** inter-stock attention; Portfolio Transformer (Kisiel & Gorse 2022); SC-Transformer self-clustering (2026) | One token per name + one market token. 1 layer, `d=32`, 4 heads. SC-Transformer used **17k** params. |
| Regole AND/OR | **KANFIS** (2026) — ANFIS without exponential rule blow-up; **ANDRE** attention AND/OR (2026); **NEURULES** soft predicates → crisp lists (2025); differentiable fuzzy neurons / t-norms | A bank of `R=16–32` sparse rules. Product t-norm for AND, probabilistic sum for OR. Linear in `F`, not `K^F`. |
| Loss di portafoglio | Notebook path; **DiffQuant** differentiable simulator (Sharpe + turnover + DD + occupancy/bias); SIT trains **CVaR** end-to-end (2025); PT trains Sharpe + costs | Differentiable mark-to-market. Soft positions in train, `{+1,0,−1}` at eval. Hybrid regularizers or the policy collapses. |
| Fuzzy + deep | IFDNN / FASA-PM / LSTM-TSK (2024–25) | Fuzzy is the *front-end and the rule layer*, not a complex-valued MLP. |
| Discrete latent (optional later) | PRISM-VQ, VQ-PTE (2025–26) | Codebook over regimes. Not in v1. |

MASTER is the single closest published object to "a model that turns features on/off from the market and then attends across names." KANFIS is the single closest object to "ANFIS that does not explode at 33 inputs." DiffQuant is the single closest object to "backprop through the book, not through a proxy label."

---

## Structure (v1)

```
CMC daily panel
    │
    ├─ PIT top-30 by trailing-30d median dollar volume
    │    (mcap fallback if DV coverage < 30%, same as btcb.universe)
    │    stables / wraps excluded
    │
    ├─ A0 features, CS-z per date, clip ±5
    │
    └─ decision dates = every 7th session (daily = ablation)
              │
              ▼
     X_t ∈ R^{N×33}     N ≤ 30
     m_t = [CS mean(X_t), CS std(X_t)]     market token
              │
     L1  Gaussian memberships (3 per feature: Low / Mid / High)
              │  M ∈ [0,1]^{N×33×3}
     L2  Market gate  α = 33 · softmax(W m_t / τ)
              │  M̃_{n,f,k} = α_f · M_{n,f,k}
     L3  Rule bank (24 rules)
              │  per rule, per feature: softmax{IGNORE, AND, NAND}
              │  AND = product t-norm
              │  LONG/SHORT/FLAT = probabilistic-OR of rule heads
              │  → firings A ∈ [0,1]^{N×24}, scores S ∈ R^{N×3}
     L4  Cross-section encoder (1 layer, d=32, 4 heads)
              │  token_n = [x_n ; flatten(M̃_n) ; A_n ; S_n]
              │  + market token
              │  logits_n ∈ R^3
              │
     train:  pos = P(long) − P(short)     soft, ∈ [−1, +1]
     eval:   argmax → {+1, 0, −1}
              │
     w = pos / ‖pos‖_1 · gross 1.0
     hold 7 days (ffill)
     cost only on rebalance (5 + 3 bps)
              │
     L = − corr(cum, t)·(1−maxDD)·(1−DDdur)
         + λ_turn · turnover
         + λ_bias · |mean w|
         occupancy fail → core /= 1e5     (notebook)
```

Param budget of the numpy skeleton (untrained, seed 42): **35,394**. Target band is **15–50k**. If a later torch port exceeds ~50k, cut rules or `d_model`, do not add layers.

---

## Layer notes

### L1 — memberships

Three Gaussians per feature, centers initialized at `−1, 0, +1` (CS-z space), `σ ≈ 0.85`. This is the notebook

```
rs = exp(−0.5 · ((x − μ) / σ)²)
```

without the complex phase `exp(j · ω)`. The CFS phase (`λ` in the notebook) is a registered ablation, default **off**. Real-valued MFs are enough to get AND/OR semantics and they train with ordinary Adam.

Do **not** build a full ANFIS grid. 33 features × 3 MFs is `3^33` rules. That is the failure mode KANFIS exists to avoid. The repo `anfis.py` stays as the Mackey-Glass toy.

### L2 — dynamic on/off

MASTER: market vector → linear → softmax → scale by `F`. TFT does the same idea with a GRN. Temperature `τ = 0.5` is sharper than uniform; `τ → 0` recovers the notebook bit mask.

This is **not** LightGBM `feature_fraction`. That is random dropout, not a learned market switch.

α is shared across names on a given day (regime), not per name. Per-name gating is a later ablation; it overfits faster.

### L3 — rules

Each rule `r` picks, for each feature, one of `{IGNORE, AND, NAND}` and one of `{LOW, MID, HIGH}`. Firing is a product t-norm. Heads are a 3-way softmax (long / short / flat). The OR across rules is a probabilistic sum, then the encoder is allowed to *revise* those scores after seeing the other names.

At eval, `RuleBank.describe` prints lines like

```
R07 LONG: mom_28_skip7 IS HIGH AND NOT yz_vol_30 IS HIGH AND dv_trend IS HIGH
```

Sparsity is structural (IGNORE is biased at init). Add L1 on `1 − P(IGNORE)` only if the printed sheet is still dense after a first train.

### L4 — why a transformer at all

Without L4 the model is a per-name fuzzy classifier. You asked for "riceve gli input per ogni asset insieme." The only cheap way to do that is attention over the 30 names. One layer is enough to express "this name is strong *relative to the other 29*." That is the cross-section.

Do not add a temporal transformer on 60 bars. That is Phase E. It failed the skill gate.

### Head and discreteness

Train soft (`P(L) − P(S)`). Eval hard (`argmax`). This is the DiffQuant `tanh(d/τ) · σ(g/τ)` idea, simplified to a 3-way simplex so flat is a first-class action, not a leftover.

A straight-through estimator is the torch-port detail, not a v1 requirement. The numpy skeleton only needs both tensors for the loss.

---

## Loss

Notebook (train path, torch version):

```
loss = corr(st_cum, linspace) * (1 - min(maxdd, 0.99)) * (1 - max_ddur)
```

plus the occupancy nuke (`loss / 1e5` if the book is not two-sided and not active enough).

That core is kept. Two DiffQuant terms are added because a pure Calmar-like objective has known failure modes on a long-biased crypto sample:

- **turnover** — otherwise the soft policy chatters every day
- **|mean w|** — otherwise the book goes permanently long

Weekly rebalance already cuts turnover. Daily is the ablation that will tell you whether the extra decisions pay for the extra cost.

Do **not** train on residual RankIC as the primary objective. That is A0's job. This model is a *policy*. If you want a teacher, distill A0 scores into the rule heads as an auxiliary loss, never as the only loss.

---

## Universe and rebalance

Use the existing PIT machinery (`btcb/universe.py`):

- rank = trailing 30d median dollar volume
- fallback = mcap when DV coverage is thin (the function already does this)
- `n = 30`
- stables / wraps excluded (`STABLE_OR_WRAP`)
- death-in-position: force-exit at last close (house convention)

**Volume, not mcap, is the default.** Mcap is the sensitivity. Top-30 by mcap in crypto is a BTC/ETH/stable-adjacent club; volume is closer to what you can actually trade. A0 already trains on PIT top-120 and executes top-20 / top-40. Top-30 sits between the two sleeves on purpose.

**7-day rebalance first.** Reasons:

- matches the A0 `h=7` label
- 7× fewer decisions → less overfit of a path loss
- costs only hit on the decision day
- the notebook loss is a *path* statistic; weekly paths are less noisy

Daily is the registered ablation, same costs, same universe. Do not tune `every ∈ {1,2,3,5,7,10}` in the first run.

Positions are decided on rebalance dates and ffilled. The model still *sees* daily features on the decision day only (no leakage from future membership). Between decisions the book is constant.

---

## What LightGBM is still for

| role | keep? |
|---|---|
| Official A0 / COMBO book | yes, untouched |
| Feature list | yes, frozen 33 |
| Joint reasoner over 30 names | no |
| Path-DD optimizer | no |
| Teacher: RankIC scores as aux target | optional, later |
| Init: use A0 gain to bias which features start as AND vs IGNORE | optional, later |

If you force LightGBM into the second seat you get a two-step "forecast then threshold" pipeline. That is exactly the objective mismatch SIT / Portfolio Transformer / DiffQuant argue against.

---

## Training protocol (when this leaves the skeleton)

Same house CV as A0. No new degrees of freedom here.

- expanding walk-forward, `min_train_days=730`, `val_days=90`, `step_days=90`
- purge last 7 train days, embargo 7+3 before val
- inner holdout 90d for early stop on the **path loss**, not on RankIC
- seed 42
- date-batched forward: one step = all names on that date
- label-shuffle gate at date level (the GRU failure mode). Null path-loss must not look skilled
- feature lookahead + PIT universe lookahead, same as A0
- costs on, lag-0 close (house), lag-1 as the pessimistic bound

Keep criterion (draft, freeze before any number is produced):

> FuzzyX is VIABLE if, on the same OOS window as A0 Sleeve A (PIT top-20 is *not* the universe; report both top-30 volume and a top-20 restricted book): (i) all leakage gates pass; (ii) date-level label-shuffle is centered; (iii) net Sharpe ≥ 0 and MaxDD is reported; (iv) it does not lose more than 0.10 net Sharpe vs A0 on the identical-days top-20 book. Otherwise PARK. No retune.

This is a draft. It is not yet an addendum. Do not run the walk-forward until the keep rule is copied into an addendum and committed first.

---

## Prototype map

| file | role |
|---|---|
| `fuzzyx/membership.py` | Gaussian Low/Mid/High |
| `fuzzyx/gate.py` | MASTER-style α(m) |
| `fuzzyx/rules.py` | IGNORE/AND/NAND + product AND + prob-OR + `describe()` |
| `fuzzyx/encoder.py` | 1-layer asset-token attention |
| `fuzzyx/loss.py` | notebook path + occupancy + turnover/bias |
| `fuzzyx/model.py` | full forward, soft + hard positions |
| `fuzzyx/universe.py` | PIT top-30 + weekly calendar (pandas/btcb, optional) |
| `fuzzyx/smoke.py` | synthetic panel, no market data |
| `config_fuzzyx.yaml` | frozen v1 knobs |

The smoke does **not** claim skill. It checks shapes, membership range, gate simplex, masked names stay flat, and that the path loss is finite on both daily-hard and weekly-hard books.

---

## Explicit non-goals (v1)

- Complex-valued CFS / `torch.complex128`
- PyGAD / differential evolution over network weights
- Temporal transformer / GRU / 60-bar windows
- Kronos, microstructure, path signatures, context block (all previously killed or parked)
- Top-N sweeps, horizon sweeps, cost sweeps
- Live trading, schedules, leverage
- Replacing COMBO

---

## Suggested next step

1. Agree the keep rule and freeze it in an addendum.
2. Port `FuzzyX.forward` + `path_loss` to a tiny torch module (same shapes) so Adam can run.
3. Walk-forward on CMC PIT top-30, weekly, seed 42, one shot.
4. If gates fail or Sharpe is dead: PARK. Do not add heads.

The skeleton is there so the structure is executable, not so it can be tuned.