"""Stepping-stone / MODI local search for transportation allocations.

Given a basic feasible allocation and a (scalar) cost matrix, perform up to
``max_pivots`` transportation-simplex pivots, each moving flow around a cycle
with negative reduced cost. With an unlimited budget this reaches the optimum
of the scalarised problem; with a small budget it is a classic local search that
can be embedded in the ACO (a hybrid "ACO + LS" as in most enhanced ACO work).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _complete_basis(x: np.ndarray, cost: np.ndarray):
    """Positive cells plus zero cells (cheapest first) to form a spanning tree."""
    m, n = x.shape
    parent = list(range(m + n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    basis = set()
    cells = list(zip(*np.nonzero(x > 1e-12)))
    zero_cells = sorted(
        ((i, j) for i in range(m) for j in range(n) if x[i, j] <= 1e-12), key=lambda c: cost[c]
    )
    for i, j in cells + zero_cells:
        a, b = find(i), find(m + j)
        if a != b:
            parent[a] = b
            basis.add((int(i), int(j)))
            if len(basis) == m + n - 1:
                break
    return basis


def _tree_adjacency(basis, m, n):
    adj = {k: [] for k in range(m + n)}
    for i, j in basis:
        adj[i].append(m + j)
        adj[m + j].append(i)
    return adj


def _potentials(basis, cost, m, n):
    adj = _tree_adjacency(basis, m, n)
    pot = np.full(m + n, np.nan)
    pot[0] = 0.0
    dq = deque([0])
    while dq:
        a = dq.popleft()
        for b in adj[a]:
            if np.isnan(pot[b]):
                i, j = (a, b - m) if a < m else (b, a - m)
                pot[b] = cost[i, j] - pot[a]
                dq.append(b)
    return pot[:m], pot[m:], adj


def _tree_path(adj, start, goal):
    prev = {start: None}
    dq = deque([start])
    while dq:
        a = dq.popleft()
        if a == goal:
            break
        for b in adj[a]:
            if b not in prev:
                prev[b] = a
                dq.append(b)
    path = [goal]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def improve(x: np.ndarray, cost: np.ndarray, max_pivots: int = 50, tol: float = 1e-9) -> np.ndarray:
    """Return an allocation no worse than ``x`` under ``cost``."""
    x = np.array(x, dtype=float)
    m, n = x.shape
    basis = _complete_basis(x, cost)
    if len(basis) != m + n - 1:
        return x
    for _ in range(max_pivots):
        u, v, adj = _potentials(basis, cost, m, n)
        red = cost - u[:, None] - v[None, :]
        for c in basis:
            red[c] = 0.0
        ei, ej = np.unravel_index(np.argmin(red), red.shape)
        if red[ei, ej] >= -tol:
            break
        # cycle: entering cell + tree path from column node back to row node
        path = _tree_path(adj, m + ej, ei)  # col_ej -> ... -> row_ei
        cells = []
        for a, b in zip(path[:-1], path[1:]):
            cells.append((a, b - m) if a < m else (b, a - m))
        # cells alternate starting with a "minus" cell
        minus = cells[0::2]
        plus = cells[1::2]
        theta_cell = min(minus, key=lambda c: (x[c], c))
        theta = x[theta_cell]
        x[ei, ej] += theta
        for c in minus:
            x[c] -= theta
        for c in plus:
            x[c] += theta
        basis.remove(theta_cell)
        basis.add((int(ei), int(ej)))
    x[np.abs(x) < 1e-9] = 0.0
    return x
