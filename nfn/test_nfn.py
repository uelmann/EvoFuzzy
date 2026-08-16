"""Local unit tests for Phase 7 NFN (no Modal, no panel required)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from nfn.constants import N_FEATURES, N_MEMBERSHIPS, N_PRIMITIVES, PHASE7_CRITERION, PHASE7_CEILING
from nfn.firewall import assert_firewall
from nfn.interpret import decode_primitive, membership_movement, top_rules
from nfn.verdict import mechanical_verdict
from nfn.warmstart import primitive_index


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


def test_verdict_parked_a_clean_bc() -> None:
    grid = {
        "frozen_spread": {"tail_ic_top": 0.10, "overlap": 0.20},
        "nfn_ensemble": {"tail_ic_top": 0.105, "overlap": 0.21},
    }
    null = {"passed": True, "tail_ic_top": {"verdict": "GREEN"}}
    seeds = {42: {"tail_ic_top": 0.104}, 43: {"tail_ic_top": 0.106}, 44: {"tail_ic_top": 0.105}}
    v = mechanical_verdict(grid, null, seeds)
    assert v["live"] is False
    assert v["clause_a"] is False
    assert v["clause_b"] is True
    assert v["clause_c"] is True
    assert v["ceiling"] == PHASE7_CEILING
    assert "a" in v["failed_clauses"]


def test_verdict_live() -> None:
    grid = {
        "frozen_spread": {"tail_ic_top": 0.10, "overlap": 0.20},
        "nfn_ensemble": {"tail_ic_top": 0.12, "overlap": 0.22},
    }
    null = {"passed": True, "tail_ic_top": {"verdict": "GREEN"}}
    seeds = {42: {"tail_ic_top": 0.119}, 43: {"tail_ic_top": 0.121}, 44: {"tail_ic_top": 0.120}}
    v = mechanical_verdict(grid, null, seeds)
    assert v["live"] is True
    assert v["label"] == "NFN LIVE"
    assert v["ceiling"] is None


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


def test_criterion_in_addendum() -> None:
    text = Path("reports/btcb_phase7_nfn_addendum.md").read_text()
    assert PHASE7_CRITERION in text
    assert "NFN is LIVE if ALL of" in text


def test_model_forward_cpu() -> None:
    try:
        import torch
    except ImportError:
        print("skip test_model_forward_cpu: no torch")
        return
    from nfn.model import build_nfn

    m = build_nfn(42)
    z = torch.zeros(16, 33)
    md = torch.zeros(16, 3)
    lt, lb, extra = m(z, md)
    assert lt.shape == (16,)
    assert lb.shape == (16,)
    assert extra["r"].shape == (16, 24)
    n = m.n_params()
    assert 3000 <= n <= 8000, n
    e = m.exponents().detach()
    # 3 primitives per rule near 1
    nz = (e > 0.5).sum(dim=1)
    assert int((nz == 3).sum()) == 24


if __name__ == "__main__":
    test_firewall()
    test_primitive_roundtrip()
    test_verdict_parked_a_clean_bc()
    test_verdict_live()
    test_top_rules_print()
    test_membership_movement()
    test_criterion_in_addendum()
    test_model_forward_cpu()
    print("nfn unit tests OK")
