"""Local unit tests for Phase 7.c NFN v1 craft (no Modal, no panel required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nfn.constants import N_PRIMITIVES, N_RULES
from nfn.constants_v1 import (
    ARCH_FROZEN,
    N_BAG,
    N_PARAMS_EXPECTED,
    PHASE7C_CRAFT_LEDGER,
    PHASE7C_CRITERION,
    assert_architecture_unchanged,
)
from nfn.firewall import assert_firewall
from nfn.craft_math import pick_swa_epochs, split_inner_holdout, swa_average, trailing_mean
from nfn.verdict_v1 import craft_confirmed, mechanical_verdict_v1


def test_architecture_frozen() -> None:
    rec = assert_architecture_unchanged()
    assert rec["passed"]
    assert ARCH_FROZEN["N_RULES"] == N_RULES
    assert ARCH_FROZEN["N_PRIMITIVES"] == N_PRIMITIVES
    assert N_BAG == 5


def test_n_params() -> None:
    try:
        from nfn.model import build_nfn
    except ImportError:
        print("skip test_n_params: no torch")
        return
    m = build_nfn(42)
    n = m.n_params()
    assert n == N_PARAMS_EXPECTED, n


def test_inner_holdout_120_purged() -> None:
    ids = np.arange(500, dtype=np.int32)
    tr, ho = split_inner_holdout(ids, n_ho=120, horizon=14, purge_extra=3)
    assert len(ho) == 120
    assert int(ho[0]) == 380
    assert int(tr[-1]) == 362
    purged = set(range(363, 380))
    assert purged.isdisjoint(set(int(x) for x in tr))
    assert purged.isdisjoint(set(int(x) for x in ho))


def test_trailing_and_swa() -> None:
    xs = [0.1, 0.2, 0.0, 0.3, 0.25]
    tm = trailing_mean(xs, 3)
    assert abs(tm[0] - 0.1) < 1e-9
    assert abs(tm[2] - (0.1 + 0.2 + 0.0) / 3) < 1e-9
    epochs = pick_swa_epochs(tm, 3)
    assert len(epochs) == 3
    assert epochs == sorted(epochs)
    try:
        import torch
    except ImportError:
        print("skip swa tensor test: no torch")
        return
    a = {"w": torch.tensor([0.0, 2.0]), "i": torch.tensor(1, dtype=torch.int64)}
    b = {"w": torch.tensor([2.0, 4.0]), "i": torch.tensor(3, dtype=torch.int64)}
    avg = swa_average([a, b])
    assert torch.allclose(avg["w"], torch.tensor([1.0, 3.0]))


def test_craft_confirmed_and_parked() -> None:
    grid = {
        "frozen_spread": {"tail_ic_top": 0.10, "overlap": 0.20},
        "nfn_ensemble": {"tail_ic_top": 0.105, "overlap": 0.21},
    }
    null = {"passed": True, "tail_ic_top": {"verdict": "GREEN"}}
    seeds = {42: {"tail_ic_top": 0.104}, 43: {"tail_ic_top": 0.106}, 44: {"tail_ic_top": 0.105}}
    hyg = [{"seed": s, "fold_id": f, "bag": b, "status": "ok", "selected_epoch": 12} for s in (42, 43, 44) for f in range(3) for b in range(5)]
    v = mechanical_verdict_v1(grid, null, seeds, hyg)
    assert v["live"] is False
    assert v["label"] == "NFN-v1 PARKED"
    assert v["craft_confirmed"] is True
    assert v["craft_ledger"] == PHASE7C_CRAFT_LEDGER


def test_craft_not_confirmed_early_stop() -> None:
    c = craft_confirmed(0.005, 4.0)
    assert c["craft_confirmed"] is False
    assert c["disp_ok"] is True
    assert c["epoch_ok"] is False


def test_live() -> None:
    grid = {
        "frozen_spread": {"tail_ic_top": 0.10, "overlap": 0.20},
        "nfn_ensemble": {"tail_ic_top": 0.12, "overlap": 0.22},
    }
    null = {"passed": True, "tail_ic_top": {"verdict": "GREEN"}}
    seeds = {42: {"tail_ic_top": 0.119}, 43: {"tail_ic_top": 0.121}, 44: {"tail_ic_top": 0.120}}
    hyg = [{"seed": 42, "fold_id": 0, "bag": 0, "status": "ok", "selected_epoch": 15}]
    v = mechanical_verdict_v1(grid, null, seeds, hyg)
    assert v["live"] is True
    assert v["label"] == "NFN-v1 LIVE"


def test_criterion_in_addendum() -> None:
    text = Path("reports/btcb_phase7c_nfn_v1_addendum.md").read_text()
    assert PHASE7C_CRITERION in text
    assert "NFN-v1 is LIVE if ALL of" in text
    assert "CRAFT-CONFIRMED" in text


def test_firewall() -> None:
    rec = assert_firewall()
    assert rec["passed"]


if __name__ == "__main__":
    test_architecture_frozen()
    test_n_params()
    test_inner_holdout_120_purged()
    test_trailing_and_swa()
    test_craft_confirmed_and_parked()
    test_craft_not_confirmed_early_stop()
    test_live()
    test_criterion_in_addendum()
    test_firewall()
    print("nfn v1 unit tests OK")
