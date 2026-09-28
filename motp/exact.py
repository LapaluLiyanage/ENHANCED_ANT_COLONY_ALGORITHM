"""Exact reference solutions via linear / integer programming (SciPy HiGHS).

The transportation constraint matrix is totally unimodular, so with integer
supply/demand every LP vertex is integral. We still solve with ``milp`` so the
answers are guaranteed integral (and the epsilon-constraint front is exact).
"""

from __future__ import annotations

from itertools import product
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .problem import MOTP


def _constraints(problem: MOTP):
    m, n = problem.shape
    A = np.zeros((m + n, m * n))
    for i in range(m):
        A[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        A[m + j, j::n] = 1
    rhs = np.concatenate([problem.supply, problem.demand])
    return LinearConstraint(A, rhs, rhs)


def solve_weighted(problem: MOTP, weights: Sequence[float], extra=None, integral: bool = True) -> np.ndarray:
    """Minimise sum_k w_k * f_k(x). Returns the allocation matrix."""
    w = np.asarray(weights, dtype=float)
    c = np.tensordot(w, problem.costs, axes=1).ravel()
    cons = [_constraints(problem)] + list(extra or [])
    res = milp(
        c,
        constraints=cons,
        integrality=np.ones_like(c) if integral else np.zeros_like(c),
        bounds=Bounds(0, np.inf),
        options={"mip_rel_gap": 0.0, "time_limit": 120.0},
    )
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")
    return np.rint(res.x).reshape(problem.shape) if integral else res.x.reshape(problem.shape)


def single_objective_optima(problem: MOTP) -> List[np.ndarray]:
    """Lexicographic optimum of each objective (ties broken by the other objectives)."""
    sols = []
    K = problem.n_objectives
    for k in range(K):
        e = np.eye(K)[k]
        z = problem.evaluate(solve_weighted(problem, e))[k]
        # stage 2: among optima of f_k, minimise the (scaled) remaining objectives
        keep = LinearConstraint(problem.costs[k].ravel()[None, :], -np.inf, z + 1e-6)
        scale = problem.costs.max(axis=(1, 2))
        w = np.where(np.arange(K) == k, 0.0, 1.0 / np.where(scale > 0, scale, 1.0))
        sols.append(solve_weighted(problem, w, [keep]) if K > 1 else solve_weighted(problem, e))
    return sols


def ideal_and_nadir(problem: MOTP):
    """Ideal point (per-objective optimum) and payoff-table nadir estimate."""
    payoff = np.array([problem.evaluate(x) for x in single_objective_optima(problem)])
    return payoff.diagonal().copy(), payoff.max(axis=0), payoff


def weighted_sum_front(problem: MOTP, steps: int = 20) -> np.ndarray:
    """Supported Pareto points from weighted sums over a simplex grid.

    Objectives are scaled by their ideal values so the weights are meaningful.
    """
    ideal, _, _ = ideal_and_nadir(problem)
    scale = np.where(ideal > 0, ideal, 1.0)
    K = problem.n_objectives
    pts = []
    for combo in product(range(steps + 1), repeat=K - 1):
        if sum(combo) > steps:
            continue
        w = np.array(list(combo) + [steps - sum(combo)], dtype=float) / steps
        w = np.maximum(w, 1e-4) / scale
        pts.append(problem.evaluate(solve_weighted(problem, w)))
    from .metrics import nondominated

    return nondominated(np.array(pts))


def epsilon_constraint_front(problem: MOTP, max_points: int = 500, step: float = 1.0) -> np.ndarray:
    """Complete integer Pareto front for a bi-objective problem (incl. unsupported points).

    Repeatedly minimise f1 subject to f2 <= previous f2 - step. With integer costs
    and step=1 this enumerates every non-dominated objective vector.
    """
    if problem.n_objectives != 2:
        raise ValueError("epsilon_constraint_front supports exactly 2 objectives")
    f2_row = problem.costs[1].ravel()[None, :]
    pts = []
    bound = np.inf
    tie = 1.0 / (1.0 + problem.costs[1].sum() * problem.supply.sum())
    for _ in range(max_points):
        extra = [] if np.isinf(bound) else [LinearConstraint(f2_row, -np.inf, bound)]
        try:
            x = solve_weighted(problem, [1.0, tie], extra)
        except RuntimeError:
            break
        f = problem.evaluate(x)
        pts.append(f)
        bound = f[1] - step
    return np.array(pts)


def reference_front(problem: MOTP, extra_points=None, exact_limit: int = 300, try_exact: bool = False):
    """Best available approximation of the true Pareto front.

    Returns (front, is_exact). For 2 objectives the epsilon-constraint front is
    exact if it finishes within ``exact_limit`` points. Otherwise we use the
    supported (weighted-sum) points merged with ``extra_points`` (e.g. every
    solution found by every method) - the usual practice when the true front is
    unknown. Hypervolume ratios against a non-exact front can only be <= 1.
    """
    from .metrics import nondominated

    if problem.n_objectives == 2:
        if try_exact:
            front = epsilon_constraint_front(problem, max_points=exact_limit)
            if len(front) < exact_limit:
                return front, True
        supported = weighted_sum_front(problem, steps=50)
    else:
        supported = weighted_sum_front(problem, steps=12)
    pts = supported if extra_points is None else np.vstack([supported, np.atleast_2d(extra_points)])
    return nondominated(pts), False
