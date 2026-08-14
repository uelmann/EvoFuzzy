# System card — v2.0-combo-final

Daily research program closed. This card is the definitive description of the **reference book**. Backtest/analysis only; no live components. Frozen A0 SHA256: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa` (`config.yaml` = `config_frozen_a0.yaml` = `config_frozen_a0.sha256`). Tag: `v2.0-combo-final`.

Causal (training-window) τ (`tau_mode=fold_train`) is the house standard. Pooled/full-OOS τ numbers are deprecated (`reports/numbers_ledger.md`).

---

## Reference book

**COMBO** = 50% Sleeve A + 50% Sleeve B, fixed weights, no optimization. Each sleeve is already gross-limited to 1.0; 50/50 sums PnL on identical days and renormalizes total gross to 1.0.

Both sleeves share the frozen A0 LightGBM, seed 42, residualized labels, walk-forward CV, BTC beta hedge, Vision funding, lag-0 close execution, and hysteresis 0.6·τ. Config: `config.yaml` / `config_frozen_a0.yaml`. Features: `baseline/features.py` (`FEATURE_COLS`, 33 columns). Portfolio: `baseline/portfolio.py` (`run_tranche_portfolio`, `tau_mode="fold_train"`).

### Shared A0 model

| item | value |
|------|--------|
| Features | 33 price/volume, CS z-scored per date, clip ±5: `ret_7/14/28/56/90`, `mom_28_skip7`, `mom_90_skip14`, `rev_1/3`, `close_sma20/50/100`, `sma20_sma50`, `ema12_ema26`, `yz_vol_14/30/60`, `pk_vol_14`, `vol_ratio`, `vol_of_vol_30`, `max_ret_14`, `min_ret_14`, `dist_high_90`, `dist_low_90`, `range_pos_28`, `skew_28/60`, `beta_btc_60`, `idio_vol_60`, `corr_btc_28`, `amihud_14`, `dv_z_30`, `dv_trend` |
| Label | residualized forward log-return vs BTC, winsorized 1/99, horizons {7, 10} |
| LightGBM | Huber, `num_leaves=31`, `lr=0.03`, `n_estimators=3000`, `min_data_in_leaf=200`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1.0`; h=7 early-stop on mean daily RankIC, patience 100; h=10 **fixed 500 trees** (no early stop) |
| CV | expanding walk-forward, `min_train_days=730`, `val_days=90`, `step_days=90`, purge last *h* train days, embargo *h*+3 before val, inner holdout 90d |
| τ | causal: percentile of **training-window** \|score\| only; median-τ = house median of grid {60, 70, 80, 90} |
| Sizing | inverse `yz_vol_30`, gross cap 1.0 |
| Hedge | trailing BTC beta, applied daily |
| Funding | Binance Vision `fundingRate` dumps, longs pay when rate > 0 (`−w·f`) |
| Execution | lag 0 = trade at close *t* on score_t; no leverage; daily bars |
| Hysteresis | exit when \|score\| < 0.6·τ |
| Seed | 42 |
| Train universe | PIT top-120 by 30d median dollar volume ≤ *t* (`data.train_universe_n=120`) |

### Sleeve A (P1)

Frozen A0, residualized **h=7** label, execution **PIT top-20**, causal median-**τ = 80**, tranche **h=7**, hysteresis 0.6·τ, 1/`yz_vol_30` sizing, BTC beta hedge, funding accrued, **lag-0** close execution, costs **5 bps fee + 3 bps slip** (no tiering, no ADV cap).

### Sleeve B (P2)

Same frozen A0 model, **h=10** label and tranche horizon, execution **PIT top-40**, causal median-**τ = 70**, hysteresis 0.6·τ, same sizing/hedge/funding/lag. **Tiered costs:** ranks 1–20 use 5+3 bps; ranks 21–40 use **10+8 bps**. **Liquidity cap:** 0.5% of 30d ADV, nominal book **USD 1,000,000**. Rank universe for cost tier = PIT top-40.

Round F5 confirmed Sleeve B remains plain A0 (C0). Context, pruning, and the stacked P2′ (C3) did **not** replace it (`reports/roundF5_report.md`).

---

## Official numbers (causal τ, from the ledger)

Source: `reports/numbers_ledger.md`, Round F / F5 identical-days books. **These are backtest figures.** The program’s stated go-forward central estimate is **Sharpe 0.7–1.0** after a research-process haircut. Do not quote pooled-τ numbers (1.401 / 1.476 / 2026 −0.82); causal replacements are in the ledger footnote.

**COMBO:** net Sharpe full **1.711** / trailing-18m **0.997** / 2026 YTD **0.441** / MaxDD **−0.332** / sleeve daily-PnL correlation **0.254** / annual turnover **≈ 26.5**.

| book | full | trail-18m | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|------|-----------|------|------|------|------|------|
| COMBO | 1.711 | 0.997 | 1.097 | 2.544 | 2.843 | 1.444 | 0.441 |
| P1 (Sleeve A) | 1.207 | 1.009 | −0.370 | 2.755 | 1.391 | 1.148 | 0.721 |
| P2 (Sleeve B) | 1.470 | 0.723 | 2.317 | 1.547 | 3.445 | 1.216 | 0.241 |

COMBO per-year and MaxDD/corr/turnover: `reports/roundF_report.md`, reconfirmed `reports/roundF5_report.md`. P1/P2 headline rows: ledger (D.2 identical-days median-τ). P1 2026 on the COMBO identical-day index is +0.721; P2 2026 is +0.241.

---

## Assumptions & limitations

- **Trade-at-close lag-0.** Lag-1 (score_t traded at close t+1) is the documented pessimistic bound in the A0 stress pass (`artifacts/reports/baseline_report.md`, “Execution lag: lag-0 vs lag-1”). Example: tranche h=7 τ=80 lag-0 1.48 vs lag-1 1.68 (that cell); tranche h=10 τ=70 lag-0 1.24 vs lag-1 1.10. Production assumption remains lag-0 for 24/7 crypto; lag-1 is the robustness check, not the headline.
- **Flat slippage** plus taker fee; no impact model beyond the 0.5% ADV cap on Sleeve B.
- **Funding** from Binance Vision; missing rates treated as 0. Coverage starts ~2020-09.
- **Daily bars only.** No intraday fills, no queue, no partial days.
- **No leverage.** Gross 1.0 per sleeve; COMBO total gross 1.0.
- **Survivorship:** PIT universes (30d median DV ≤ *t*). LUNAUSDT / LUNA2USDT verified present in PIT top-20 over 2021–2022 (`artifacts/reports/baseline_report.md`).
- **OOS window 2022–2026** ≈ one crypto cycle. Trailing-18m is ~548 days. Four-candidate comparisons on that window used a raised hurdle in F5 (+0.15) for that reason.
- **Lookahead:** τ is causal (`fold_train`). Gates at current strictness include label-shuffle (and the empirical-null calibration where applicable), feature lookahead, universe lookahead (top-20/40/120), seed determinism.

---

## Kill list

One line each; verdicts mechanical as recorded.

- **Kronos frozen features — KILL** (`artifacts/reports/phaseB_report.md`): post-cutoff top-20 ΔRankIC failed the pre-registered keep rule at h=7 and h=10.
- **Kronos fine-tuned — contaminated reference** (`artifacts/reports/phaseB_report.md` §4): full-sample fine-tune, not walk-forward; not comparable to OOS books.
- **kr_sigma gate — REDUNDANT** (`artifacts/reports/phaseB1_report.md`, addendum): failed vs best control; production uses **no** entry gate. Archived as full-period-neutral, unproven.
- **Vol-control gates — worse than nothing** (`artifacts/reports/phaseB1_addendum.md`): best control (`C2_idio_vol_60_X10`) won a noisy ~250d post-cutoff cell but cost ≈ −0.44 Sharpe full-period; **no gate**.
- **Microstructure block — KILL top-20** (`reports/phaseD_report.md`); **REJECTED top-40** (`reports/phaseD2_report.md`): trail ΔIC positive on top-40, full ΔIC negative (regime-flip: hurts ≤2024, helps ≥2025).
- **Path signatures — KILL top-20 / marginal pit-120** (`reports/phaseE_report.md`): KEEP only on pit-120 h=7; KILL on the execution universe.
- **GRU/BLEND — PARKED-NO-SKILL** (`reports/phaseE1b_report.md`): empirical-null bias PASS, skill FAIL (2/4 folds at each horizon vs need 3).
- **Complexity block — KILL both universes** (`reports/roundF_report.md` F2).
- **Context block on top-20 — KILL** (`reports/roundF_report.md` F1): IC passed, P1-book trail Sharpe Δ −0.320. (Kept on top-40 IC/port vs A0, but did not replace P2 under the F5 +0.15/−0.10 sleeve hurdle.)
- **Stacked sleeve C3 (pruned+context) — not qualified** (`reports/roundF5_report.md`): IC vs A0 passed; full Sharpe 1.123 < incumbent − 0.10 (need 1.370).
- **Pooled-τ numbers — deprecated** (`reports/numbers_ledger.md`): 1.401 → 0.757 (τ60); 1.476 → 1.207 (median-τ); 2026 −0.82 → +0.72.

---

## Shelved leads (reopen only with fresh pre-registration)

1. **Context via portfolio-layer redesign.** IC-vs-Sharpe wedge: ΔRankIC positive (NW-t up to **3.27** full-period on top-40 h=10) yet threshold-τ books degrade in 2022/2024. Candidate fix: regime-conditional τ. Untested.
2. **Microstructure regime flip.** Hurts ≤2024, helps ≥2025 on both universes (`reports/phaseD2_report.md`). Needs more post-2025 data before a keep test.
3. **Intraday project.** 1h bars compressed into daily embeddings (sequence encoder or vision). The only remaining data-expansion direction after Kronos KILL.
4. **GRU line.** Faint unstable trajectory signal (`reports/phaseE1b_report.md`). Any revival needs a new design (more seeds, pooled skill test), not a retune of the parked ensemble.

---

## Retrain & discipline

- **Refit every 90 days** on the expanding window with the same purge/embargo (`cv.step_days=90`, purge *h*, embargo *h*+3).
- **Causal τ recomputed per fold** from that fold’s training-window \|score\| distribution. Never pooled OOS τ.
- **No parameter or feature changes** outside a new pre-registered experiment (addendum committed before results).
- **Gates at every refit**, current strictness: label-shuffle (including the empirical-null calibration where the object is a sequence/ensemble model), feature lookahead, PIT universe lookahead, seed determinism.
- **Reference book stays COMBO** (Sleeve A τ=80 h=7 top-20 + Sleeve B τ=70 h=10 top-40) until a pre-registered rule replaces it.
- Official numbers live only in `reports/numbers_ledger.md`. Cite this ledger.
