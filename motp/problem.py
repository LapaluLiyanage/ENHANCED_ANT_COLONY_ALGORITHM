"""Problem definition, validation, and CSV I/O for the Multi-Objective Transportation Problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class MOTP:
    """A balanced multi-objective transportation problem.

    Attributes
    ----------
    costs : array of shape (K, m, n)
        One m x n cost matrix per objective.
    supply : array of shape (m,)
    demand : array of shape (n,)
    names : optional objective names (e.g. ["cost", "time", "distance"]).
    """

    costs: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.costs = np.asarray(self.costs, dtype=float)
        if self.costs.ndim == 2:
            self.costs = self.costs[None, :, :]
        if self.costs.ndim != 3:
            raise ValueError("costs must have shape (K, m, n)")
        self.supply = np.asarray(self.supply, dtype=float)
        self.demand = np.asarray(self.demand, dtype=float)
        k, m, n = self.costs.shape
        if self.supply.shape != (m,) or self.demand.shape != (n,):
            raise ValueError(
                f"supply/demand shapes {self.supply.shape}/{self.demand.shape} "
                f"do not match cost matrices of shape ({m}, {n})"
            )
        if np.any(self.supply < 0) or np.any(self.demand < 0):
            raise ValueError("supply and demand must be non-negative")
        if np.any(self.costs < 0):
            raise ValueError("costs must be non-negative")
        if not np.isclose(self.supply.sum(), self.demand.sum()):
            raise ValueError(
                f"unbalanced problem: total supply {self.supply.sum():g} != "
                f"total demand {self.demand.sum():g} (use MOTP.balanced())"
            )
        if not self.names:
            self.names = [f"obj{i + 1}" for i in range(k)]

    # ------------------------------------------------------------------ shape
    @property
    def n_objectives(self) -> int:
        return self.costs.shape[0]

    @property
    def shape(self) -> tuple:
        return self.costs.shape[1:]

    # -------------------------------------------------------------- helpers
    def evaluate(self, allocation: np.ndarray) -> np.ndarray:
        """Objective vector of an allocation matrix."""
        return np.einsum("kij,ij->k", self.costs, np.asarray(allocation, dtype=float))

    def is_feasible(self, allocation: np.ndarray, tol: float = 1e-6) -> bool:
        x = np.asarray(allocation, dtype=float)
        return (
            x.shape == self.shape
            and np.all(x >= -tol)
            and np.allclose(x.sum(axis=1), self.supply, atol=tol)
            and np.allclose(x.sum(axis=0), self.demand, atol=tol)
        )

    @classmethod
    def balanced(cls, costs, supply, demand, names=None) -> "MOTP":
        """Build a problem, adding a zero-cost dummy row/column if unbalanced."""
        costs = np.asarray(costs, dtype=float)
        if costs.ndim == 2:
            costs = costs[None]
        supply = np.asarray(supply, dtype=float)
        demand = np.asarray(demand, dtype=float)
        diff = supply.sum() - demand.sum()
        if diff > 0:  # dummy destination
            costs = np.concatenate([costs, np.zeros((costs.shape[0], costs.shape[1], 1))], axis=2)
            demand = np.append(demand, diff)
        elif diff < 0:  # dummy source
            costs = np.concatenate([costs, np.zeros((costs.shape[0], 1, costs.shape[2]))], axis=1)
            supply = np.append(supply, -diff)
        return cls(costs, supply, demand, list(names or []))

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        k: int = 2,
        cost_range=(1, 100),
        total: int | None = None,
        correlation: float = 0.0,
        seed: int | None = None,
    ) -> "MOTP":
        """Random balanced integer instance.

        ``correlation`` in [0, 1) mixes a shared base matrix into every objective,
        producing positively correlated objectives (0 = independent).
        """
        rng = np.random.default_rng(seed)
        lo, hi = cost_range
        base = rng.integers(lo, hi + 1, size=(m, n))
        costs = []
        for _ in range(k):
            own = rng.integers(lo, hi + 1, size=(m, n))
            costs.append(np.rint(correlation * base + (1 - correlation) * own))
        costs = np.maximum(np.array(costs), lo)
        total = total or 10 * m * n
        supply = _random_partition(rng, total, m)
        demand = _random_partition(rng, total, n)
        return cls(costs, supply, demand)


def _random_partition(rng: np.random.Generator, total: int, parts: int) -> np.ndarray:
    """Split an integer total into ``parts`` positive integers."""
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate([[0], cuts, [total]])).astype(float)


# ---------------------------------------------------------------------- CSV I/O
def read_objective_csv(path: str):
    """Read one objective CSV in the repo's format.

    Layout: first column = row labels, last column = ``Supply``,
    last row = ``Demand``. Returns (cost_matrix, supply, demand).
    """
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    cost = df.iloc[:-1, :-1].to_numpy(dtype=float)
    supply = df.iloc[:-1, -1].to_numpy(dtype=float)
    demand = df.iloc[-1, :-1].to_numpy(dtype=float)
    return cost, supply, demand


def load_csvs(paths: Sequence[str], names=None) -> MOTP:
    """Load a problem from one CSV per objective (supply/demand taken from the first)."""
    mats = []
    supply = demand = None
    for p in paths:
        cost, s, d = read_objective_csv(p)
        if supply is None:
            supply, demand = s, d
        elif not (np.allclose(s, supply) and np.allclose(d, demand)):
            raise ValueError(f"{p}: supply/demand differ from {paths[0]}")
        mats.append(cost)
    return MOTP(np.array(mats), supply, demand, list(names or []))
