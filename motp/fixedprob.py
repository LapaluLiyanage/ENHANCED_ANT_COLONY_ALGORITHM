"""The repository's original "fixed-probability" allocation heuristic, unified.

This is the algorithm the four original ``motp_*_fixedprob.py`` scripts
implemented, now written once with a pluggable combiner:

1. Combine the K objective matrices into one matrix C (geometric mean, ...).
2. Fixed "pheromone" probability matrix  P = (max(C) - C) / sum(max(C) - C).
3. Row/column penalties = gap between the two largest P values in each line,
   computed ONCE over the full matrix (static penalties).
4. Repeatedly take the active line with the largest penalty (rows win ties),
   allocate min(supply, demand) to its highest-P active cell (lowest C, then
   lowest index, wins ties), and retire exhausted lines.

A note for the write-up: P is a strictly decreasing affine transform of C, so
"largest P" == "smallest C" and a P-penalty equals the Vogel penalty on C divided
by a constant. With ``dynamic_penalties=True`` the method is therefore
Vogel's Approximation Method on the combined matrix (identical up to how exact
ties are broken; see tests). The default
(static penalties) is a VAM variant whose penalties are never updated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .combiners import get_combiner, normalize as _normalize
from .problem import MOTP


@dataclass
class AllocationResult:
    allocation: np.ndarray
    objective_totals: np.ndarray
    combined_matrix: np.ndarray
    probability_matrix: Optional[np.ndarray] = None
    row_penalties: Optional[np.ndarray] = None
    col_penalties: Optional[np.ndarray] = None
    steps: Optional[list] = None

    def as_dict(self) -> dict:
        return {
            "allocation": self.allocation,
            "objective_totals": [_as_number(v) for v in self.objective_totals],
            "combined_matrix": self.combined_matrix,
            "probability_matrix": self.probability_matrix,
            "row_penalties": self.row_penalties,
            "col_penalties": self.col_penalties,
        }


def fixed_probability_matrix(combined: np.ndarray) -> np.ndarray:
    c = np.asarray(combined, dtype=float)
    diff = c.max() - c
    total = diff.sum()
    if total == 0:
        return np.full(c.shape, 1.0 / c.size)
    return diff / total


def line_penalties(P: np.ndarray, active_rows=None, active_cols=None):
    """Gap between the two largest values of each row/column (over active cells)."""
    rows, cols = P.shape
    ar = np.ones(rows, bool) if active_rows is None else active_rows
    ac = np.ones(cols, bool) if active_cols is None else active_cols
    rpen, cpen = np.zeros(rows), np.zeros(cols)
    for i in range(rows):
        rpen[i] = _top_two_gap(P[i, ac])
    for j in range(cols):
        cpen[j] = _top_two_gap(P[ar, j])
    return rpen, cpen


def _top_two_gap(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    top = np.sort(values)[::-1]
    return float(top[0] - top[1])


def run(
    problem: MOTP,
    combiner: Union[str, callable] = "geometric",
    weights: Optional[Sequence[float]] = None,
    normalize: Optional[str] = None,
    dynamic_penalties: bool = False,
    record_steps: bool = False,
) -> AllocationResult:
    """Run the fixed-probability heuristic.

    Parameters
    ----------
    combiner : name in ``motp.combiners.COMBINERS`` or a callable (K,m,n)->(m,n).
    weights : optional objective weights for the mean-type combiners.
    normalize : None (original behaviour), "max", "mean" or "range".
    dynamic_penalties : recompute penalties over active lines every step (== VAM).
    """
    comb = get_combiner(combiner) if isinstance(combiner, str) else combiner
    costs = _normalize(problem.costs, normalize)
    C = comb(costs, weights) if weights is not None else comb(costs)
    P = fixed_probability_matrix(C)

    supply = problem.supply.copy()
    demand = problem.demand.copy()
    m, n = C.shape
    x = np.zeros((m, n))
    active_r = supply > 0
    active_c = demand > 0
    rpen, cpen = line_penalties(P)
    steps = [] if record_steps else None

    while active_r.any() and active_c.any():
        if dynamic_penalties:
            rpen, cpen = line_penalties(P, active_r, active_c)
        rp = np.where(active_r, rpen, -np.inf)
        cp = np.where(active_c, cpen, -np.inf)
        if rp.max() >= cp.max():
            kind, idx = "row", int(np.argmax(rp))
            cand = np.flatnonzero(active_c)
            vals = P[idx, cand]
            ties = cand[vals == vals.max()]
            j = int(ties[np.argmin(C[idx, ties])])  # argmin keeps lowest index on ties
            i = idx
        else:
            kind, idx = "col", int(np.argmax(cp))
            cand = np.flatnonzero(active_r)
            vals = P[cand, idx]
            ties = cand[vals == vals.max()]
            i = int(ties[np.argmin(C[ties, idx])])
            j = idx
        q = min(supply[i], demand[j])
        x[i, j] += q
        supply[i] -= q
        demand[j] -= q
        if steps is not None:
            steps.append({"line": kind, "index": idx, "cell": (i, j), "qty": q,
                          "penalty": float(rpen[idx] if kind == "row" else cpen[idx])})
        if supply[i] <= 1e-12:
            active_r[i] = False
        if demand[j] <= 1e-12:
            active_c[j] = False

    return AllocationResult(
        allocation=_maybe_int(x),
        objective_totals=problem.evaluate(x),
        combined_matrix=C,
        probability_matrix=P,
        row_penalties=rpen,
        col_penalties=cpen,
        steps=steps,
    )


def vam(problem: MOTP, combiner="arithmetic", weights=None, normalize="max") -> AllocationResult:
    """Classic Vogel's Approximation Method on a combined cost matrix."""
    return run(problem, combiner, weights, normalize, dynamic_penalties=True)


def _maybe_int(x: np.ndarray) -> np.ndarray:
    return x.astype(int) if np.allclose(x, np.rint(x)) else x


def _as_number(v: float):
    return int(round(v)) if np.isclose(v, round(v)) else float(v)
