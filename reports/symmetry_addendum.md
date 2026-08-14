# Symmetry audit — freeze addendum

**Status:** FROZEN before results. Diagnostic only.
**Reference book:** COMBO v2.0-combo-final. **UNCHANGED** by this task.
**Scope:** analysis only. Frozen A0 predictions and residual labels reused byte-identical. No retraining, no portfolio changes, no τ re-optimization. Master only. CPU only. Zero GPU. No live components.

## Pre-registered classification (verbatim)

> The engine is labeled SYMMETRIC if, at h=7 or h=10 on at least two of the three universes, the full-period TOP spread is positive with NW-t ≥ 2.0 AND the full-period symmetry ratio ≥ 0.4. It is labeled LONG-SIDE GAP otherwise. This is a diagnostic label: SYMMETRIC closes the question (long-leg economics are a raw-material property of the asset class, not a model defect); LONG-SIDE GAP opens a targeted research question on winner-side information. Neither label changes the reference book. No post-hoc adjustment.

Mechanical reading (frozen with this addendum): a (h, universe) cell **passes** if full-period mean(TOP spread) > 0 with Newey–West t ≥ 2.0 (HAC lag = h) **and** symmetry ratio = mean(TOP)/mean(BOTTOM) ≥ 0.4 and finite. The engine is **SYMMETRIC** if `n_pass(h=7) ≥ 2` **or** `n_pass(h=10) ≥ 2`; otherwise **LONG-SIDE GAP**.

## Definitions (frozen)

- Residual = frozen A0 label `y_h{h}` (h-day residualized forward log-return vs BTC, winsorized 1/99). Not recomputed.
- Score = frozen A0 `score` from `lgbm_price_only_h{h}.parquet`. Not recomputed.
- Universes: PIT top-20, PIT top-40, PIT top-120. Inner join on (date, symbol).
- Buckets: quintiles (top-20, top-40); deciles (pit-120). Rank-first `qcut` so ties do not drop bins. Days with fewer names than the bucket count are skipped.
- Top bucket = highest score bin; bottom = lowest; middle = Q3 (quintiles) or mean of D5 and D6 (deciles).
- TOP spread_t = mean residual(top)_t − mean residual(middle)_t.
- BOTTOM spread_t = mean residual(middle)_t − mean residual(bottom)_t.
- Fair LO benchmark: costless equal-weight PIT top-40 (and top-20) daily-rebalanced simple returns. BTC B&H kept as a reference only.
- Long-pick quality: mean residual of names with applied long weight in the frozen reference sleeves, minus the same-date cross-sectional mean residual on that sleeve's universe.

## What this freeze does not do

- Does not retrain, rescore, or rewrite labels.
- Does not change COMBO, sleeves, the ledger, or the system card.
- Does not adopt or kill a product; it only attaches a diagnostic label.
