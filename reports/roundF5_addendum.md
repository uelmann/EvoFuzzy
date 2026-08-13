# Round F5 addendum — pre-registered design (before results)

**Written and frozen before any Round F5 C3 IC, sleeve Sharpe, or COMBO′ number is observed.** Backtest/analysis only. Zero GPU. Frozen A0 hash: `e6b7407c8243ea49df3801ccaacedecd194315f45790e549a68c3368078b3faa`. Causal (training-window) τ is the house standard; see `reports/numbers_ledger.md`.

## Pre-registered P2 sleeve selection rule (verbatim)

> A candidate replaces the incumbent P2 sleeve only if its trailing-18m net Sharpe ≥ incumbent + 0.15 AND its full-period net Sharpe ≥ incumbent − 0.10 AND (for C3 only) its paired ΔRankIC vs plain A0 on top-40 satisfies the house block criterion at h=10 or h=7 (trail ≥ +0.005, full ≥ 0, ≥60% positive trailing folds). Among qualifying candidates, the sleeve with the highest trailing-18m net Sharpe is selected. If none qualify, the incumbent P2 stays. The +0.15 hurdle exists because four candidates are compared on a 548-day window; no post-hoc adjustment.

## Pre-registered COMBO′ rule (verbatim)

> COMBO′ becomes the reference book only if its trailing-18m net Sharpe ≥ COMBO trailing − 0.05 AND its full-period net Sharpe ≥ COMBO full − 0.05, where COMBO is the Round-F adopted book (full 1.711, trail 0.997). Otherwise the Round-F COMBO stays the reference.

## Operationalization (pre-registered)

- P2′ (C3) features = A0 minus the eight Round-F pruned columns (`rev_1`, `rev_3`, `dv_z_30`, `dv_trend`, `ret_28`, `skew_28`, `mom_28_skip7`, `ret_7`) plus the seven context columns. 32 features. Context is reused from the Round F volume cache; not recomputed.
- Four P2-sleeve candidates, top-40 PIT, h=10, tranche, funding on, lag 0, tiered costs, 0.5% ADV cap, causal median-τ (C0 uses ledger τ=70; C1/C2/C3 pick median-τ on {60,70,80,90}). Comparison on identical days.
- Incumbent for the +0.15 / −0.10 Sharpe hurdles = this-run C0 (plain A0, top-40, h=10, τ=70) on that common day index. Ledger C0 (full 1.470 / trail 0.723) is the published bookkeeping row.
- House block criterion for C3 is RankIC-only vs plain A0 on top-40 (no extra portfolio-Δ gate beyond the sleeve Sharpe hurdles): trail ΔRankIC ≥ +0.005, full ≥ 0, ≥60% positive trailing-18m folds, at h=10 **or** h=7.
- COMBO′ = 0.5·P1_daily_net + 0.5·selected_sleeve_daily_net on identical days. P1 is unchanged (A0, top-20, h=7, τ=80). The COMBO′ hurdle uses the frozen Round-F COMBO numbers (full 1.711, trail 0.997), not a re-run.
- Stability diagnostic (selected vs incumbent): per-year avg #positions, % flat, and the distribution of daily |Δ position count|. Information only; no verdict.
