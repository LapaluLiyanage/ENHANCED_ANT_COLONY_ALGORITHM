import json
import os

import numpy as np
import pytest

from motp import MOTP, aco, combiners, exact, fixedprob, load_csvs, metrics
from motp.aco import ACOParams
from motp.localsearch import improve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


@pytest.fixture(scope="module")
def paper4x4():
    return load_csvs([os.path.join(DATA, "objective1.csv"), os.path.join(DATA, "objective2.csv")])


@pytest.fixture(scope="module")
def paper3obj():
    return load_csvs([os.path.join(DATA, f) for f in ["objective1_new.csv", "objective2_new.csv", "objective3_new.csv"]])


# ---------------------------------------------------------------- legacy parity
LEGACY = json.load(open(os.path.join(HERE, "fixtures", "legacy_outputs.json")))


@pytest.mark.parametrize("case", LEGACY, ids=[f"{c['instance']}-{c['combiner']}" for c in LEGACY])
def test_unified_code_reproduces_original_scripts(case):
    """Outputs recorded from the original four scripts before the refactor."""
    p = MOTP(case["costs"], case["supply"], case["demand"])
    res = fixedprob.run(p, case["combiner"])
    np.testing.assert_array_equal(res.allocation, np.array(case["allocation"]))
    assert res.as_dict()["objective_totals"] == case["totals"]


def test_legacy_wrappers_keep_their_api(paper4x4):
    from motp_geometric_fixedprob import run_algorithm

    r = run_algorithm(paper4x4.costs.tolist(), [21, 24, 18, 30], [15, 22, 26, 30])
    assert r["objective_totals"] == [1902, 1198]
    assert "geom_matrix" in r and "probability_matrix" in r


def test_known_results_on_sample_data(paper4x4, paper3obj):
    expect = {"arithmetic": [1902, 1228], "geometric": [1902, 1198], "minimum": [1898, 1212]}
    for c, tot in expect.items():
        assert fixedprob.run(paper4x4, c).as_dict()["objective_totals"] == tot
    assert fixedprob.run(paper3obj, "harmonic").as_dict()["objective_totals"] == [928, 95, 632]


# ----------------------------------------------------------------- feasibility
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("comb", sorted(combiners.COMBINERS))
def test_every_heuristic_is_feasible_and_basic(seed, comb):
    p = MOTP.random(6, 7, 3, seed=seed)
    for dyn in (False, True):
        x = fixedprob.run(p, comb, normalize="max", dynamic_penalties=dyn).allocation
        assert p.is_feasible(x)
        assert np.count_nonzero(x) <= sum(p.shape) - 1  # basic solution


def test_unbalanced_problem_gets_dummy():
    p = MOTP.balanced(np.ones((2, 2, 3)), [10, 5], [4, 4, 4])
    assert p.shape == (2, 4) and p.demand[-1] == 3
    with pytest.raises(ValueError):
        MOTP(np.ones((2, 2, 3)), [10, 5], [4, 4, 4])


# ----------------------------------------------------- VAM equivalence (finding)
@pytest.mark.parametrize("seed", range(15))
def test_dynamic_fixedprob_equals_textbook_vam(seed):
    """P is a decreasing affine map of C, so penalties on P == VAM penalties on C.

    Continuous costs avoid exact ties: with integer costs the two can differ only
    where floating-point rounding in P breaks a penalty tie differently.
    """
    p = MOTP.random(5, 6, 1, seed=seed)
    p.costs = p.costs + np.random.default_rng(seed).random(p.costs.shape)
    ours = fixedprob.run(p, "arithmetic", dynamic_penalties=True).allocation
    assert np.array_equal(ours, _textbook_vam(p.costs[0], p.supply, p.demand))


def _textbook_vam(c, supply, demand):
    s, d = supply.copy(), demand.copy()
    x = np.zeros_like(c)
    ar, ac = s > 0, d > 0
    while ar.any() and ac.any():
        def pen(v):
            v = np.sort(v)
            return v[0] if len(v) == 1 else v[1] - v[0]
        rp = np.array([pen(c[i, ac]) if ar[i] else -np.inf for i in range(len(s))])
        cp = np.array([pen(c[ar, j]) if ac[j] else -np.inf for j in range(len(d))])
        if rp.max() >= cp.max():
            i = int(np.argmax(rp)); cols = np.flatnonzero(ac); j = int(cols[np.argmin(c[i, cols])])
        else:
            j = int(np.argmax(cp)); rows = np.flatnonzero(ar); i = int(rows[np.argmin(c[rows, j])])
        q = min(s[i], d[j]); x[i, j] += q; s[i] -= q; d[j] -= q
        ar[i] = s[i] > 0; ac[j] = d[j] > 0
    return x


# --------------------------------------------------------------- exact baselines
def test_exact_front_of_sample_instance(paper4x4):
    ideal, nadir, _ = exact.ideal_and_nadir(paper4x4)
    assert ideal.tolist() == [1898, 1198]
    front = exact.epsilon_constraint_front(paper4x4)
    assert front.tolist() == [[1898, 1212], [1900, 1205], [1902, 1198]]


def test_target_in_search_solution_is_infeasible(paper4x4):
    """(1898, 1207), the target in legacy/search_solution.py, is not attainable."""
    front = exact.epsilon_constraint_front(paper4x4)
    assert not any(np.all(f <= [1898, 1207]) for f in front)


@pytest.mark.parametrize("seed", range(10))
def test_local_search_reaches_lp_optimum(seed):
    p = MOTP.random(7, 9, 1, seed=seed)
    x0 = fixedprob.run(p, "minimum").allocation
    x = improve(x0, p.costs[0], max_pivots=10_000)
    assert p.is_feasible(x)
    opt = exact.solve_weighted(p, [1.0])
    assert p.evaluate(x)[0] == pytest.approx(p.evaluate(opt)[0])


# ------------------------------------------------------------------------- ACO
def test_aco_is_reproducible_and_feasible():
    p = MOTP.random(6, 6, 2, seed=3)
    a = aco.run(p, ACOParams(seed=42, n_iter=15, ls_prob=0.2))
    b = aco.run(p, ACOParams(seed=42, n_iter=15, ls_prob=0.2))
    np.testing.assert_array_equal(a.archive_objectives, b.archive_objectives)
    assert all(p.is_feasible(x) for x in a.archive_allocations)
    F = a.archive_objectives
    assert len(metrics.nondominated(F)) == len(F)


def test_aco_finds_sample_front(paper4x4):
    r = aco.run(paper4x4, ACOParams(seed=0, n_iter=30))
    got = {tuple(f) for f in r.archive_objectives}
    assert {(1898.0, 1212.0), (1902.0, 1198.0)} <= got


# --------------------------------------------------------------------- metrics
def test_hypervolume_known_values():
    assert metrics.hypervolume([[1, 1]], [2, 2]) == 1
    assert metrics.hypervolume([[0, 1], [1, 0]], [2, 2]) == 3
    assert metrics.hypervolume([[0, 0, 0]], [1, 2, 3]) == 6
    assert metrics.hypervolume([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [2, 2, 2]) == pytest.approx(4)  # 3*2 - 3*1 + 1


def test_geometric_combiner_is_scale_invariant_in_ranking():
    c = np.random.default_rng(0).integers(1, 50, (2, 4, 5)).astype(float)
    a = combiners.geometric(c)
    b = combiners.geometric(combiners.normalize(c, "max"))
    assert np.array_equal(np.argsort(a, axis=None), np.argsort(b, axis=None))
