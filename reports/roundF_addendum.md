# Round F addendum — pre-registered design (before results)

**Written and frozen before any Round F ablation IC, portfolio, or combo number is observed.** Backtest/analysis only. Zero GPU. Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`. Causal (training-window) τ is the house standard; see `reports/numbers_ledger.md`.

## Pre-registered block KEEP criterion (verbatim)

> Block X is KEPT on universe U only if trailing-18m ΔRankIC on U ≥ +0.005 at h=7 or h=10 AND full-OOS ΔRankIC on U ≥ 0 AND Δ positive in ≥60% of trailing-18m folds on U AND the corresponding portfolio trailing-18m net Sharpe Δ on U ≥ 0. F4 (pruning) uses the same criterion with thresholds ΔRankIC ≥ 0 (trailing) and ≥ −0.002 (full): pruning is KEPT if it does not hurt. Verdicts per-universe, mechanical, no post-hoc adjustment.

## Pre-registered COMBO criterion (verbatim)

> COMBO is ADOPTED as the reference book only if its trailing-18m net Sharpe ≥ max(P1, P2 trailing) − 0.10 AND its full-period net Sharpe ≥ max(P1, P2 full) − 0.10. Otherwise the adopted book remains P2 with P1 as reference.

## Operationalization (pre-registered)

- Universes U ∈ {top-20, top-40} PIT. Horizons 7 and 10 for RankIC. Portfolio Δ is evaluated only on the two adopted books: top-20 h=7 (P1 config) and top-40 h=10 (P2 config, tiered costs + ADV cap).
- “Corresponding” portfolio: each candidate picks its own causal median-τ on {60,70,80,90}; A0 uses the ledger median-τ (P1 τ=80, P2 τ=70). ΔSharpe on identical days.
- F4 dropped columns = the 8 lowest mean LightGBM gain features across existing frozen A0 fold models (h=7 and h=10 pooled). Listed in the report; not chosen by looking at F4 OOS IC.
- catch22 is the pycatch22 22-feature set on the trailing 90d residual log-return window (min 60 observations, else NaN). Hurst is **single-scale R/S**: H = log(R/S) / log(n) on that window.
- COMBO = 0.5·P1_daily_net + 0.5·P2_daily_net on identical days (each sleeve already 1.0 gross; 50/50 ⇒ total gross 1.0). No weight optimization.
