"""FORGE F0 unit tests — search space, fitness, nested windows, no PI seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from btcb.constants import (
    DEATH_CONVENTION,
    FORGE_BINARY_OPS,
    FORGE_CRITERION,
    FORGE_FITNESS,
    FORGE_GENS,
    FORGE_HEADLINE_K,
    FORGE_JUDGE_END,
    FORGE_JUDGE_START,
    FORGE_K_SET,
    FORGE_KNIFE_SPREAD,
    FORGE_LAMBDA_C,
    FORGE_MAX_DEPTH,
    FORGE_MAX_NODES,
    FORGE_MINE_END,
    FORGE_MINE_START,
    FORGE_NONCONTAMINATION,
    FORGE_NULL_GENS,
    FORGE_P_CROSSOVER,
    FORGE_P_POINT,
    FORGE_P_SUBTREE,
    FORGE_POP,
    FORGE_PRIMITIVES,
    FORGE_SELECT_END,
    FORGE_SELECT_START,
    FORGE_TOURNAMENT,
    FORGE_TS_K,
    FORGE_TS_OPS,
    FORGE_UNARY_OPS,
)
from btcb.forge import (
    P_SPEC,
    _pdiv,
    champion_passes,
    eval_tree,
    evolve,
    fitness_details,
    formula_str,
    gen_tree,
    ingredient_census,
    init_population,
    make_synthetic_cube,
    mechanical_forge_verdict,
    node_count,
    parse_formula,
    run_book,
    shuffle_fwd_vol,
    tree_depth,
    walk_tokens,
)


ADDENDUM = Path("reports/forge_f0_addendum.md")


def test_addendum_freeze_strings():
    text = ADDENDUM.read_text()
    assert FORGE_NONCONTAMINATION in text
    assert FORGE_FITNESS in text
    assert FORGE_CRITERION in text
    assert DEATH_CONVENTION in text
    assert "No correlation metric appears anywhere in the fitness" in text


def test_hyperparams_frozen():
    assert FORGE_POP == 2000
    assert FORGE_GENS == 60
    assert FORGE_TOURNAMENT == 20
    assert FORGE_P_CROSSOVER == 0.70
    assert FORGE_P_SUBTREE == 0.25
    assert FORGE_P_POINT == 0.05
    assert abs(FORGE_P_CROSSOVER + FORGE_P_SUBTREE + FORGE_P_POINT - 1.0) < 1e-12
    assert FORGE_MAX_DEPTH == 8
    assert FORGE_MAX_NODES == 25
    assert FORGE_LAMBDA_C == 0.02
    assert FORGE_KNIFE_SPREAD == 1.0
    assert FORGE_K_SET == (3, 5, 8)
    assert FORGE_HEADLINE_K == 5
    assert FORGE_NULL_GENS == 30
    assert FORGE_MINE_START == "2019-10-20"
    assert FORGE_MINE_END == "2022-12-31"
    assert FORGE_SELECT_START == "2023-01-01"
    assert FORGE_SELECT_END == "2024-12-31"
    assert FORGE_JUDGE_START == "2025-01-01"
    assert FORGE_JUDGE_END == "2026-08-13"
    assert len(P_SPEC) == 12
    assert len(FORGE_PRIMITIVES) == 31
    assert "gauss" not in FORGE_PRIMITIVES


def test_parse_roundtrip():
    s = "add(mul(ret_14, rank_cs(std_63)), lag_5(volume))"
    t = parse_formula(s)
    assert formula_str(t) == s
    assert node_count(t) == 7
    assert tree_depth(t) <= FORGE_MAX_DEPTH
    prims, ops = walk_tokens(t)
    assert "ret_14" in prims
    assert "rank_cs" in ops
    assert "lag" in ops


def test_protected_div():
    a = np.array([[1.0, 2.0], [3.0, np.nan]])
    b = np.array([[0.0, 1e-12], [2.0, 4.0]])
    out = _pdiv(a, b)
    assert out[0, 0] == 1.0
    assert out[0, 1] == 1.0
    assert abs(out[1, 0] - 1.5) < 1e-12
    assert not np.isfinite(out[1, 1])


def test_no_pi_seed_in_init():
    rng = np.random.default_rng(42)
    pop = init_population(rng, 80)
    pi = "pdiv(mul(phi_cs(ret_14), phi_cs(ret_28)), phi_cs(std_63))"
    exprs = {formula_str(t) for t in pop}
    assert pi not in exprs
    for t in pop:
        assert node_count(t) <= FORGE_MAX_NODES
        assert tree_depth(t) <= FORGE_MAX_DEPTH
        parse_formula(formula_str(t))


def test_fitness_has_no_correlation():
    cube = make_synthetic_cube(T=28, N=10, seed=1)
    tree = parse_formula("rank_cs(ret_14)")
    d = fitness_details(tree, cube)
    blob = str(list(d.keys()))
    assert "corr" not in blob
    assert "ic" not in blob.lower()
    assert "fitness" in d
    assert "spread" in d
    assert len(d["ann"]) == 12


def test_book_and_knife_and_complexity():
    cube = make_synthetic_cube(T=30, N=10, seed=2)
    tree = parse_formula("mcap_rank")
    packed = run_book(
        eval_tree(tree, cube.prims, cube.mask_dv),
        cube,
        k=3,
        rebal_mode="daily",
        univ="dv100",
    )
    assert packed["daily_ret"] is not None
    assert len(packed["daily_ret"]) == len(cube.dates) - 1
    d = fitness_details(tree, cube)
    assert d["nodes"] == 1
    if not d["discarded"]:
        expected = d["median"] - FORGE_LAMBDA_C * (1 / 25)
        assert abs(d["fitness"] - expected) < 1e-12


def test_phi_cs_is_ndtr_z():
    cube = make_synthetic_cube(T=16, N=8, seed=3)
    tree = parse_formula("phi_cs(ret_1)")
    sc = eval_tree(tree, cube.prims, cube.mask_dv)
    finite = sc[np.isfinite(sc)]
    if finite.size:
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0


def test_null_shuffle_changes_fwd():
    cube = make_synthetic_cube(T=24, N=10, seed=4)
    sh = shuffle_fwd_vol(cube, seed_seq=(42, 7))
    assert sh.fwd.shape == cube.fwd.shape
    assert np.allclose(sh.prims["ret_14"], cube.prims["ret_14"], equal_nan=True)
    m = np.isfinite(cube.fwd) & np.isfinite(sh.fwd)
    assert not np.allclose(sh.fwd[m], cube.fwd[m])


def test_verdicts():
    btc = {"book_total": 0.5, "maxdd": -0.40, "total": 0.5}
    ok = {
        "book_total": 0.8,
        "rel_sharpe": 0.2,
        "maxdd": -0.30,
        "total": 0.8,
    }
    p = champion_passes(ok, btc)
    assert p["pass"] is True
    dead = champion_passes({"book_total": 0.1, "rel_sharpe": -0.1, "maxdd": -0.9}, btc)
    assert dead["pass"] is False
    v = mechanical_forge_verdict([p, dead, dead, dead, dead])
    assert v["label"] == "FORGE-ALIVE"
    strong = dict(ok)
    strong["rel_sharpe"] = 0.6
    ps = champion_passes(strong, btc)
    v2 = mechanical_forge_verdict([ps, dead, dead, dead, dead])
    assert v2["label"] == "FORGE-STRONG"
    v3 = mechanical_forge_verdict([dead] * 5)
    assert v3["label"] == "FORGE-DEAD"


def test_tiny_evolve_smoke(tmp_path):
    cube = make_synthetic_cube(T=22, N=8, seed=5)
    out = evolve(
        cube,
        n_pop=6,
        n_gen=2,
        seed=42,
        checkpoint_path=tmp_path / "ck.json",
        n_jobs=1,
        budget_sec=60,
        label="mine",
    )
    assert len(out["population"]) == 6
    assert len(out["history"]) >= 1
    assert (tmp_path / "ck.json").exists()
    census = ingredient_census([{"expr": formula_str(t)} for t in out["population"]])
    assert "one_liner" in census


def test_operators_are_generic():
    assert "gauss" not in FORGE_UNARY_OPS
    assert set(FORGE_TS_K) == {5, 14, 28, 63}
    rng = np.random.default_rng(0)
    t = gen_tree(rng, 4, True)
    assert tree_depth(t) <= 8
    src = Path("btcb/forge.py").read_text()
    assert "gauss(ret" not in src
    assert "MANUEL2_FORMULA" not in src
    assert FORGE_BINARY_OPS == ("add", "sub", "mul", "pdiv", "min", "max")
    assert "ts_mean" in FORGE_TS_OPS
