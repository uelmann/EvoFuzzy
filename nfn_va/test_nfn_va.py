"""Local unit tests for Phase 7.d Variant A (no Modal, no panel required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfn_va.constants import (
    DEATH_CONVENTION,
    FIREWALL_PHASE7,
    HUBER_DELTA,
    MAG_COEF,
    N_FEATURES,
    N_MEMBERSHIPS,
    N_PRIMITIVES,
    NFN_V0_READONLY,
    PHASE7D_CRITERION,
    PHASE7_NULL_REGISTRATION,
    RANK_COEF,
    TAU,
    VARIANT_A_N_PARAMS,
)
from nfn_va.data import _cs_magnitude_labels
from nfn_va.firewall import assert_firewall
from nfn_va.interpret import decode_primitive, membership_movement, top_rules
from nfn_va.verdict import magnitude_gain, mechanical_verdict
from nfn_va.warmstart import primitive_index


def test_criterion_in_addendum() -> None:
    p = Path("reports/btcb_phase7d_addendum.md")
    if not p.exists():
        p = Path("/root/btcb_phase7d_addendum.md")
    text = p.read_text()
    assert PHASE7D_CRITERION in text
    assert FIREWALL_PHASE7 in text
    assert PHASE7_NULL_REGISTRATION in text
    assert DEATH_CONVENTION in text
    assert "VARIANT-A is LIVE if ALL of" in text


def test_firewall() -> None:
    rec = assert_firewall()
    assert rec["passed"]
    assert rec["n_files"] >= 5


def test_primitive_roundtrip() -> None:
    for j in range(N_FEATURES):
        for k in range(N_MEMBERSHIPS):
            for comp in (False, True):
                p = primitive_index(j, k, comp)
                assert 0 <= p < N_PRIMITIVES
                rec = decode_primitive(p)
                assert rec["k"] == k
                assert rec["complement"] is comp


def test_magnitude_labels_rank_and_winsor() -> None:
    excess = np.array([0.1, 0.2, 2.0, -0.5, 0.0, 0.05, 0.3, 0.15], dtype=np.float32)
    date_id = np.zeros(len(excess), dtype=np.int32)
    starts = np.array([0], dtype=np.int32)
    ends = np.array([len(excess)], dtype=np.int32)
    y_rank, y_win, y_z = _cs_magnitude_labels(excess, date_id, starts, ends)
    assert np.nanmin(y_rank) >= 0.0 - 1e-6
    assert np.nanmax(y_rank) <= 1.0 + 1e-6
    assert y_rank[np.argmax(excess)] == np.nanmax(y_rank)
    assert float(np.nanmax(y_win)) <= float(np.nanmax(excess)) + 1e-6
    assert abs(float(np.nanmean(y_z))) < 1e-5


def test_verdict_live_and_mag_gain() -> None:
    grid = {
        "frozen_spread": {"tail_ic_top": 0.10, "overlap": 0.20},
        "variant_a_ensemble": {"tail_ic_top": 0.12, "overlap": 0.22},
    }
    null = {"passed": True, "tail_ic_top": {"verdict": "GREEN"}}
    seeds = {42: {"tail_ic_top": 0.119}, 43: {"tail_ic_top": 0.121}, 44: {"tail_ic_top": 0.120}}
    mag = magnitude_gain(0.03, 0.02)
    v = mechanical_verdict(grid, null, seeds, mag)
    assert v["live"] is True
    assert v["label"] == "VARIANT-A LIVE"
    assert v["magnitude_gain"] is True


def test_verdict_parked_a() -> None:
    grid = {
        "frozen_spread": {"tail_ic_top": 0.10, "overlap": 0.20},
        "variant_a_ensemble": {"tail_ic_top": 0.101, "overlap": 0.22},
    }
    null = {"passed": True, "tail_ic_top": {"verdict": "GREEN"}}
    seeds = {42: {"tail_ic_top": 0.100}, 43: {"tail_ic_top": 0.101}, 44: {"tail_ic_top": 0.102}}
    mag = magnitude_gain(0.01, 0.02)
    v = mechanical_verdict(grid, null, seeds, mag)
    assert v["live"] is False
    assert "a" in v["failed_clauses"]
    assert v["magnitude_gain"] is False
    assert v["label"] == "VARIANT-A PARKED"


def test_top_rules_print() -> None:
    e = np.zeros((24, N_PRIMITIVES))
    e[0, primitive_index(0, 1, False)] = 1.0
    e[0, primitive_index(1, 1, True)] = 0.8
    w = np.zeros(24)
    w[0] = 0.5
    rules = top_rules(e, w, n_rules=3)
    assert rules
    assert "μ_" in rules[0]["formula"] or "1-μ" in rules[0]["formula"]


def test_membership_movement() -> None:
    c0 = np.tile(np.array([-0.67, 0.0, 0.67]), (33, 1))
    s0 = np.ones((33, 3))
    c = c0.copy()
    s = s0.copy()
    c[2, 0] = -1.5
    rows = membership_movement(c, s, c0, s0, [f"f{i}" for i in range(33)], top_n=5)
    assert rows[0]["feature"] == "f2"


def test_attach_excess_ignores_colliding_score_cols() -> None:
    import pandas as pd
    from nfn_va.magdiag import attach_excess

    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"], utc=True),
            "id": [2, 3],
            "spread": [0.2, 0.1],
            "excess_h14": [np.nan, np.nan],
        }
    )
    labeled = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"], utc=True),
            "id": [2, 3],
            "excess_h14": [0.05, -0.01],
        }
    )
    out = attach_excess(scores, labeled, "spread", "excess_h14")
    assert "excess_h14" in out.columns
    assert list(out["excess_h14"]) == [0.05, -0.01]


def test_nfn_v0_readonly_baked() -> None:
    assert NFN_V0_READONLY["n_params"] == 5488
    assert NFN_V0_READONLY["verdict"] == "PARKED"
    assert abs(float(NFN_V0_READONLY["tail_ic_top"]) - 0.06322926171542223) < 1e-12
    assert NFN_V0_READONLY.get("label") == "nfn_v0"


def test_loss_and_single_head_cpu() -> None:
    try:
        import torch
    except ImportError:
        print("skip test_loss_and_single_head_cpu: no torch")
        return
    from nfn_va.loss import huber_mag, listnet_mag, variant_a_loss
    from nfn_va.model import assert_arch_equal_phase7, build_nfn

    assert RANK_COEF == 1.0 and MAG_COEF == 0.5 and TAU == 1.0 and HUBER_DELTA == 1.0
    rec = assert_arch_equal_phase7()
    assert rec["passed"]
    m = build_nfn(42)
    assert m.n_params() == VARIANT_A_N_PARAMS
    assert m.n_params() == 5463
    z = torch.zeros(16, 33)
    md = torch.zeros(16, 3)
    score, extra = m(z, md)
    assert score.shape == (16,)
    assert extra["r"].shape == (16, 24)
    y_win = torch.linspace(-1.0, 2.0, 16)
    y_z = (y_win - y_win.mean()) / (y_win.std() + 1e-6)
    tot, parts = variant_a_loss(score, y_win, y_z, m)
    assert torch.isfinite(tot)
    assert "l_rank" in parts and "l_mag" in parts
    lr = listnet_mag(score, y_win, tau=1.0)
    hm = huber_mag(score, y_z, delta=1.0)
    assert torch.isfinite(lr) and torch.isfinite(hm)
    e = m.exponents().detach()
    nz = (e > 0.5).sum(dim=1)
    assert int((nz == 3).sum()) == 24


if __name__ == "__main__":
    test_criterion_in_addendum()
    test_firewall()
    test_primitive_roundtrip()
    test_magnitude_labels_rank_and_winsor()
    test_verdict_live_and_mag_gain()
    test_verdict_parked_a()
    test_top_rules_print()
    test_membership_movement()
    test_attach_excess_ignores_colliding_score_cols()
    test_nfn_v0_readonly_baked()
    test_loss_and_single_head_cpu()
    print("nfn_va unit tests OK")
