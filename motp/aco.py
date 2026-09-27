"""A genuine multi-objective Ant Colony Optimisation for the MOTP.

Unlike the fixed-probability heuristic (a single deterministic pass), this is a
population-based, iterative ACO:

* one pheromone matrix per objective, tau_k (m x n), bounded MAX-MIN style;
* each ant gets its own weight vector lambda (spread evenly over the ants, as in
  Iredi, Merkle & Middendorf's bi-criterion ant), so the colony explores the
  whole trade-off surface instead of a single compromise;
* an ant builds a basic feasible solution by repeatedly picking an active cell
  (i, j) with probability  ~ (sum_k lambda_k tau_k)^alpha * (eta_lambda)^beta,
  where eta_lambda = 1 / (sum_k lambda_k * normalised c_k), and allocating
  min(supply_i, demand_j) there (ACS pseudo-random rule with q0);
* every iteration, pheromone evaporates, the iteration-best ant for each
  objective reinforces tau_k, and a random archive member reinforces all tau_k;
* an archive keeps every non-dominated solution found;
* optionally (``ls_prob``), an ant's solution is improved by a budgeted
  stepping-stone (MODI) local search on its own weighted cost.

Optionally, the fixed-probability heuristic's solution (or any other) can be
used to seed the archive and pheromone (``seed_allocations``), which is a clean
way to present the original method as the *initialisation* of a real ACO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .combiners import normalize as _normalize
from .localsearch import improve
from .metrics import dominates
from .problem import MOTP


@dataclass
class ACOParams:
    n_ants: int = 20
    n_iter: int = 100
    alpha: float = 1.0          # pheromone influence
    beta: float = 2.0           # heuristic influence
    rho: float = 0.1            # evaporation rate: tau <- (1 - rho) * tau
    q0: float = 0.2             # probability of greedy (argmax) choice
    tau_max: float = 1.0
    tau_min_ratio: float = 0.01  # tau_min = tau_max * ratio
    normalize: str = "max"
    ls_prob: float = 0.0         # probability an ant's solution gets stepping-stone local search
    ls_pivots: int = 20          # pivot budget per local search
    stagnation_reset: int = 30   # re-initialise pheromone after this many idle iterations (0 = off)
    seed: Optional[int] = None


@dataclass
class ACOResult:
    archive_allocations: List[np.ndarray]
    archive_objectives: np.ndarray
    history: List[int] = field(default_factory=list)  # archive size per iteration
    evaluations: int = 0

    def best_compromise(self, ideal=None):
        """Archive member with the smallest mean relative gap to ``ideal``
        (defaults to the archive's own ideal point)."""
        F = self.archive_objectives
        z = F.min(axis=0) if ideal is None else np.asarray(ideal, float)
        score = ((F - z) / np.where(z == 0, 1, z)).mean(axis=1)
        b = int(np.argmin(score))
        return self.archive_allocations[b], F[b]


class _Archive:
    def __init__(self):
        self.X: List[np.ndarray] = []
        self.F: List[np.ndarray] = []

    def add(self, x: np.ndarray, f: np.ndarray) -> bool:
        for g in self.F:
            if dominates(g, f) or np.array_equal(g, f):
                return False
        keep = [i for i, g in enumerate(self.F) if not dominates(f, g)]
        self.X = [self.X[i] for i in keep] + [x]
        self.F = [self.F[i] for i in keep] + [f]
        return True


def run(problem: MOTP, params: ACOParams = ACOParams(), seed_allocations: Sequence[np.ndarray] = ()) -> ACOResult:
    rng = np.random.default_rng(params.seed)
    K = problem.n_objectives
    m, n = problem.shape
    C = _normalize(problem.costs, params.normalize)
    tau_max = params.tau_max
    tau_min = tau_max * params.tau_min_ratio
    tau = np.full((K, m, n), tau_max)

    # evenly spread weight vectors, one per ant
    if K == 1:
        lambdas = np.ones((params.n_ants, 1))
    elif K == 2:
        t = np.linspace(0, 1, params.n_ants)
        lambdas = np.stack([t, 1 - t], axis=1)
    else:
        lambdas = rng.dirichlet(np.ones(K), size=params.n_ants)

    archive = _Archive()
    evals = 0
    for x in seed_allocations:
        x = np.asarray(x, dtype=float)
        if problem.is_feasible(x):
            archive.add(x, problem.evaluate(x))
            _deposit(tau, x, np.ones(K), tau_max)

    history, idle = [], 0
    for _ in range(params.n_iter):
        sols, objs = [], []
        for a in range(params.n_ants):
            lam = lambdas[a]
            x = _construct(problem, C, tau, lam, params, rng)
            if params.ls_prob > 0 and rng.random() < params.ls_prob:
                x = improve(x, np.tensordot(lam, C, axes=1), max_pivots=params.ls_pivots)
            f = problem.evaluate(x)
            evals += 1
            sols.append(x)
            objs.append(f)
        objs = np.array(objs)

        improved = False
        for x, f in zip(sols, objs):
            improved |= archive.add(x, f)
        idle = 0 if improved else idle + 1

        # --- pheromone update (MAX-MIN style)
        tau *= 1 - params.rho
        for k in range(K):
            b = int(np.argmin(objs[:, k]))
            _deposit(tau, sols[b], np.eye(K)[k], params.rho * tau_max)
        r = int(rng.integers(len(archive.X)))
        _deposit(tau, archive.X[r], np.ones(K), params.rho * tau_max)
        np.clip(tau, tau_min, tau_max, out=tau)

        if params.stagnation_reset and idle >= params.stagnation_reset:
            tau[:] = tau_max
            idle = 0
        history.append(len(archive.F))

    order = np.lexsort(np.array(archive.F).T[::-1])
    return ACOResult(
        archive_allocations=[archive.X[i] for i in order],
        archive_objectives=np.array(archive.F)[order],
        history=history,
        evaluations=evals,
    )


def _construct(problem, C, tau, lam, params, rng) -> np.ndarray:
    supply = problem.supply.copy()
    demand = problem.demand.copy()
    m, n = problem.shape
    x = np.zeros((m, n))
    eta = 1.0 / (np.tensordot(lam, C, axes=1) + 1e-9)
    ph = np.tensordot(lam, tau, axes=1)
    score = (ph ** params.alpha) * (eta ** params.beta)
    active_r = supply > 0
    active_c = demand > 0
    while active_r.any() and active_c.any():
        mask = np.outer(active_r, active_c)
        s = np.where(mask, score, 0.0).ravel()
        if rng.random() < params.q0:
            idx = int(np.argmax(s))
        else:
            idx = int(rng.choice(s.size, p=s / s.sum()))
        i, j = divmod(idx, n)
        q = min(supply[i], demand[j])
        x[i, j] += q
        supply[i] -= q
        demand[j] -= q
        if supply[i] <= 1e-12:
            active_r[i] = False
        if demand[j] <= 1e-12:
            active_c[j] = False
    return x


def _deposit(tau, x, per_objective, amount) -> None:
    used = x > 0
    for k, w in enumerate(per_objective):
        if w:
            tau[k][used] += amount * w
