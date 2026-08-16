"""Synthetic forward + path-loss smoke. No market data, no pandas required."""

from __future__ import annotations

import time

import numpy as np

from .constants import ENCODER, FEATURE_COLS, N_FEATURES, REBALANCE_DAYS, SEED, UNIVERSE_N
from .loss import path_loss
from .model import FuzzyX
from .universe import rebalance_dates


def _synthetic_panel(n_days: int = 84, n_assets: int = UNIVERSE_N, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_days, n_assets, N_FEATURES)).clip(-5, 5)
    # Plant a weak cross-sectional signal in ret_7 (col 0): tomorrow's residual
    # is slightly higher for names that are already high today.
    shock = rng.normal(0.0, 0.02, size=(n_days, n_assets))
    asset_ret = 0.04 * np.tanh(x[:, :, 0]) + shock
    mask = np.ones((n_days, n_assets), dtype=bool)
    mask[:, -2:] = False  # two slots empty (PIT universe < 30 some days)
    return x, asset_ret, mask


def run(encoder: str = ENCODER) -> dict:
    t0 = time.time()
    x, asset_ret, mask = _synthetic_panel()
    model = FuzzyX(encoder=encoder)
    out = model.forward(x, mask=mask)
    hard = out.hard_pos
    assert hard.shape == (x.shape[0], x.shape[1])
    assert set(np.unique(hard)).issubset({-1.0, 0.0, 1.0})
    assert out.memberships.min() >= 0.0 - 1e-12
    assert out.memberships.max() <= 1.0 + 1e-12
    assert np.allclose(out.alpha.sum(axis=-1), N_FEATURES, atol=1e-6)
    assert np.all(hard[:, -2:] == 0.0)
    metrics = path_loss(out.soft_pos, asset_ret, mask=mask)
    hard_metrics = path_loss(hard, asset_ret, mask=mask)
    dates = list(range(x.shape[0]))
    reb = rebalance_dates(dates, every=REBALANCE_DAYS)
    weekly_pos = np.zeros_like(hard)
    for i, d in enumerate(dates):
        src = max(r for r in reb if r <= d)
        weekly_pos[i] = hard[src]
    weekly_metrics = path_loss(weekly_pos, asset_ret, mask=mask)
    rules = model.rule_sheet()
    n_params = model.n_params()
    if not (5_000 <= n_params <= 50_000):
        raise AssertionError(f"param budget {n_params} outside 5–50k")
    report = {
        "encoder": encoder,
        "n_params": n_params,
        "n_features": N_FEATURES,
        "n_feature_names": len(FEATURE_COLS),
        "n_rules_readable": len(rules),
        "rebalance_n": len(reb),
        "soft_loss": metrics,
        "hard_loss": hard_metrics,
        "weekly_hard_loss": weekly_metrics,
        "sample_rules": rules[:8],
        "elapsed_s": round(time.time() - t0, 3),
    }
    return report


def main() -> None:
    r = run("deepsets")
    rx = run("xsec")
    print(
        f"FuzzyX smoke  encoder={r['encoder']}  params={r['n_params']}  "
        f"features={r['n_features']}  elapsed={r['elapsed_s']}s"
    )
    print(f"xsec ablation params={rx['n_params']}  elapsed={rx['elapsed_s']}s")
    print(f"readable rules={r['n_rules_readable']}  weekly rebalances={r['rebalance_n']}")
    sl = r["soft_loss"]
    print(
        f"soft  loss={sl['loss']:.4f}  core={sl['core']:.4f}  trend={sl['trend']:.3f}  "
        f"maxdd={sl['maxdd']:.3f}  L/S/T={sl['long_frac']:.2f}/{sl['short_frac']:.2f}/{sl['traded_frac']:.2f}"
    )
    hl = r["hard_loss"]
    print(
        f"hard  loss={hl['loss']:.4f}  core={hl['core']:.4f}  "
        f"L/S/T={hl['long_frac']:.2f}/{hl['short_frac']:.2f}/{hl['traded_frac']:.2f}"
    )
    wl = r["weekly_hard_loss"]
    print(
        f"week  loss={wl['loss']:.4f}  core={wl['core']:.4f}  turnover={wl['turnover']:.3f}"
    )
    print("sample rules:")
    for line in r["sample_rules"]:
        print(" ", line)


if __name__ == "__main__":
    main()
