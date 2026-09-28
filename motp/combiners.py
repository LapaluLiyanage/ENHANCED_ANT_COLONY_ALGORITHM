"""Functions that merge K objective matrices into one scalar cost matrix.

Every combiner takes an array of shape (K, m, n) and returns shape (m, n).
Use :func:`normalize` first when objectives are measured in different units;
otherwise the objective with the largest numbers dominates the combined matrix.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

Combiner = Callable[[np.ndarray], np.ndarray]


def arithmetic(costs: np.ndarray, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    if weights is None:
        return np.mean(costs, axis=0)
    return np.tensordot(_weights(costs, weights), costs, axes=1)


def geometric(costs: np.ndarray, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """(Weighted) geometric mean. Any zero cost makes the cell zero."""
    if weights is None:  # exact legacy formula: prod(c) ** (1/K)
        out = np.prod(costs, axis=0) ** (1.0 / costs.shape[0])
    else:
        w = _weights(costs, weights)
        with np.errstate(divide="ignore"):
            out = np.exp(np.tensordot(w, np.log(costs), axes=1))
    return np.where(np.any(costs == 0, axis=0), 0.0, out)


def harmonic(costs: np.ndarray, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """(Weighted) harmonic mean. Any zero cost makes the cell zero."""
    safe = np.where(costs == 0, 1.0, costs)
    if weights is None:  # exact legacy formula: K / sum(1/c)
        out = costs.shape[0] / np.sum(1.0 / safe, axis=0)
    else:
        out = 1.0 / np.tensordot(_weights(costs, weights), 1.0 / safe, axes=1)
    return np.where(np.any(costs == 0, axis=0), 0.0, out)


def minimum(costs: np.ndarray, weights=None) -> np.ndarray:
    return np.min(costs, axis=0)


def maximum(costs: np.ndarray, weights=None) -> np.ndarray:
    """Pessimistic (worst-objective) aggregation."""
    return np.max(costs, axis=0)


def sum_of_squares(costs: np.ndarray, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Sum of squared costs, as used in Ekanayake's MACOA (1 / sum c^2)."""
    w = _weights(costs, weights) * costs.shape[0]
    return np.tensordot(w, costs**2, axes=1)


COMBINERS: Dict[str, Combiner] = {
    "arithmetic": arithmetic,
    "geometric": geometric,
    "harmonic": harmonic,
    "minimum": minimum,
    "maximum": maximum,
    "sum_of_squares": sum_of_squares,
}


def get_combiner(name: str) -> Combiner:
    try:
        return COMBINERS[name]
    except KeyError:
        raise ValueError(f"unknown combiner {name!r}; choose from {sorted(COMBINERS)}") from None


def normalize(costs: np.ndarray, method: Optional[str] = "max") -> np.ndarray:
    """Put every objective on a comparable scale.

    method:
      None     - no scaling (original behaviour)
      "max"    - divide each objective by its largest entry  -> (0, 1]
      "mean"   - divide each objective by its mean entry
      "range"  - min-max scale to [0, 1] (creates zeros: avoid with geometric/harmonic)
    """
    costs = np.asarray(costs, dtype=float)
    if method is None or method == "none":
        return costs
    axes = (1, 2)
    if method == "max":
        scale = costs.max(axis=axes, keepdims=True)
        return costs / np.where(scale == 0, 1, scale)
    if method == "mean":
        scale = costs.mean(axis=axes, keepdims=True)
        return costs / np.where(scale == 0, 1, scale)
    if method == "range":
        lo = costs.min(axis=axes, keepdims=True)
        hi = costs.max(axis=axes, keepdims=True)
        return (costs - lo) / np.where(hi - lo == 0, 1, hi - lo)
    raise ValueError(f"unknown normalization {method!r}")


def _weights(costs: np.ndarray, weights) -> np.ndarray:
    k = costs.shape[0]
    if weights is None:
        return np.full(k, 1.0 / k)
    w = np.asarray(weights, dtype=float)
    if w.shape != (k,) or np.any(w < 0) or w.sum() == 0:
        raise ValueError("weights must be K non-negative numbers, not all zero")
    return w / w.sum()
