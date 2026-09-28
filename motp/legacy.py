"""Backwards-compatible API for the original ``motp_*_fixedprob.py`` scripts.

``run_algorithm(objective_matrices, supply, demand)`` returns the same dict keys
and the same allocations as the original per-combiner scripts (verified by
``tests/test_legacy_regression.py`` against recorded outputs).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from . import fixedprob
from .problem import MOTP, load_csvs

_MATRIX_KEY = {
    "arithmetic": "arith_matrix",
    "geometric": "geom_matrix",
    "harmonic": "harmonic_matrix",
    "minimum": "min_matrix",
}

_TITLE = {
    "arithmetic": "ARITHMETIC MEAN",
    "geometric": "GEOMETRIC MEAN",
    "harmonic": "HARMONIC MEAN",
    "minimum": "MINIMUM VALUE",
}


def make_run_algorithm(combiner: str):
    def run_algorithm(objective_matrices, supply, demand, debug=False):
        problem = MOTP(np.array(objective_matrices, dtype=float), supply, demand)
        res = fixedprob.run(problem, combiner, record_steps=debug)
        if debug:
            for t, s in enumerate(res.steps, 1):
                i, j = s["cell"]
                print(f"Iteration {t}: {s['line']} {s['index']} (penalty {s['penalty']:.6f}) "
                      f"-> S{i + 1}->D{j + 1}: {s['qty']:g} units")
        d = res.as_dict()
        d[_MATRIX_KEY[combiner]] = d.pop("combined_matrix")
        return d

    run_algorithm.__doc__ = f"Original fixed-probability algorithm with the {combiner} combiner."
    return run_algorithm


def main(combiner: str, default_files, argv=None) -> None:
    """Command-line entry point shared by the four legacy scripts."""
    argv = sys.argv[1:] if argv is None else argv
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = list(argv) or [os.path.join(here, "data", f) for f in default_files]
    print(f"Loading data from {', '.join(files)}...\n")
    problem = load_csvs(files)
    res = fixedprob.run(problem, combiner)

    for k, name in enumerate(problem.names):
        print(f"Objective {k + 1} Matrix:\n{problem.costs[k].astype(int) if np.all(problem.costs[k] % 1 == 0) else problem.costs[k]}\n")
    print("Supply:", problem.supply.astype(int).tolist())
    print("Demand:", problem.demand.astype(int).tolist())
    print("\n" + "=" * 60)
    print(f"RESULTS ({_TITLE[combiner]})")
    print("=" * 60)
    print(f"\nCombined ({_TITLE[combiner].lower()}) Matrix:\n{res.combined_matrix}")
    print(f"\nProbability Matrix:\n{res.probability_matrix}")
    print("\nRow Penalties:", res.row_penalties)
    print("Column Penalties:", res.col_penalties)
    print(f"\nAllocation Matrix:\n{res.allocation}")
    print("\nAllocation Details:")
    for i, j in zip(*np.nonzero(res.allocation)):
        per = ", ".join(f"{problem.costs[k, i, j]:g}" for k in range(problem.n_objectives))
        print(f"  S{i + 1} -> D{j + 1}: {res.allocation[i, j]} units (unit costs: {per})")
    totals = res.as_dict()["objective_totals"]
    print("\nObjective Totals:", totals)
    for k, v in enumerate(totals, 1):
        print(f"Objective {k} Total: {v}")
