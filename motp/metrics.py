"""Multi-objective quality indicators (all objectives minimised)."""

from __future__ import annotations

import numpy as np


def dominates(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated(points) -> np.ndarray:
    """Unique non-dominated rows, sorted lexicographically."""
    pts = np.unique(np.atleast_2d(np.asarray(points, dtype=float)), axis=0)
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        if keep[i]:
            le = np.all(pts[i] <= pts, axis=1) & np.any(pts[i] < pts, axis=1)
            keep &= ~le  # drop everything pts[i] dominates
    return pts[keep]


def hypervolume(points, reference) -> float:
    """Exact hypervolume dominated by ``points`` and bounded by ``reference``.

    Uses recursive slicing (fine for the small sets met here; any K).
    """
    ref = np.asarray(reference, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    return _hv(nondominated(pts), ref)


def _hv(pts: np.ndarray, ref: np.ndarray) -> float:
    if pts.shape[1] == 1:
        return float(ref[0] - pts[:, 0].min())
    if pts.shape[1] == 2:
        p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        vol, prev_y = 0.0, ref[1]
        for x, y in p:
            if y < prev_y:
                vol += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return vol
    # slice along the last objective
    order = np.argsort(pts[:, -1])
    p = pts[order]
    vol = 0.0
    for idx in range(len(p)):
        upper = p[idx + 1, -1] if idx + 1 < len(p) else ref[-1]
        depth = upper - p[idx, -1]
        if depth > 0:
            prefix = p[: idx + 1, :-1]
            vol += depth * _hv(prefix if prefix.shape[1] == 2 else nondominated(prefix), ref[:-1])
    return vol


def relative_gaps(point, ideal) -> np.ndarray:
    """Per-objective relative gap to the ideal point, (f - z*) / z*."""
    ideal = np.asarray(ideal, dtype=float)
    return (np.asarray(point, dtype=float) - ideal) / np.where(ideal == 0, 1, ideal)


def compromise_gap(points, ideal) -> float:
    """Smallest mean relative gap to the ideal over a set of solutions.

    A single-number score for 'how close is the best compromise to the ideal'.
    Works for one solution (a heuristic) or an archive (an ACO run).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return float(min(relative_gaps(p, ideal).mean() for p in pts))


def hv_ratio(points, front, ideal, nadir, margin: float = 0.1) -> float:
    """Hypervolume of ``points`` relative to the reference front, with a
    reference point placed ``margin`` beyond the nadir (normalised space)."""
    ideal, nadir = np.asarray(ideal, float), np.asarray(nadir, float)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    norm = lambda P: (np.atleast_2d(np.asarray(P, float)) - ideal) / span
    ref = np.full(len(ideal), 1.0 + margin)
    denom = hypervolume(norm(front), ref)
    return hypervolume(norm(points), ref) / denom if denom > 0 else float("nan")
