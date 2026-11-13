"""Kabsch / quaternion superposition and RMSD_OR (GM-DockZn)."""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return rotation R, translation t so that mobile @ R.T + t ~ target.

    Translation is applied after rotation of centered coordinates.
    RMSD is computed on the aligned points.
    """
    a = np.asarray(mobile, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("mobile and target must be (N, 3) with equal N")
    n = a.shape[0]
    if n == 0:
        raise ValueError("need at least one point")
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    a0 = a - ca
    b0 = b - cb
    h = a0.T @ b0
    u, _s, vt = svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = cb - r @ ca
    aligned = (r @ a.T).T + t
    rmsd = float(np.sqrt(((aligned - b) ** 2).sum() / n))
    return r, t, rmsd


def apply_rt(xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (rotation @ np.asarray(xyz, dtype=float).T).T + translation


def unit_vectors(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    vec = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    nrm = np.linalg.norm(vec, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1.0e-12, None)
    return vec / nrm


def rmsd_or(observed: np.ndarray, metal_xyz: np.ndarray, ideal_unit: np.ndarray) -> tuple[float, np.ndarray]:
    """Orientational RMSD of unit vectors to an ideal polyhedron.

    Permutes ideal vertices to the observed set by greedy matching after
    a Kabsch alignment of the two unit-vector clouds.
    """
    obs = unit_vectors(observed, metal_xyz)
    ideal = np.asarray(ideal_unit, dtype=float)
    if obs.shape[0] > ideal.shape[0]:
        return float("inf"), np.arange(ideal.shape[0])
    best = float("inf")
    best_perm = np.arange(obs.shape[0])
    # Try aligning obs to every combination-sized subset via Kabsch on unit vectors.
    from itertools import combinations, permutations

    n_obs = obs.shape[0]
    for subset in combinations(range(ideal.shape[0]), n_obs):
        ideal_sub = ideal[list(subset)]
        if n_obs == 1:
            rmsd = float(np.linalg.norm(obs[0] - ideal_sub[0]))
            perm = np.array(subset)
            if rmsd < best:
                best, best_perm = rmsd, perm
            continue
        if n_obs <= 4:
            for perm in permutations(range(n_obs)):
                tgt = ideal_sub[list(perm)]
                _r, _t, rmsd = kabsch(obs, tgt)
                if rmsd < best:
                    best = rmsd
                    best_perm = np.array(subset)[list(perm)]
        else:
            _r, _t, rmsd = kabsch(obs, ideal_sub)
            if rmsd < best:
                best = rmsd
                best_perm = np.array(subset)
    return best, best_perm


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation mapping unit vector a onto unit vector b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + 1.0e-12)
    b = b / (np.linalg.norm(b) + 1.0e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1.0e-12:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis = axis / (np.linalg.norm(axis) + 1.0e-12)
        k = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * k @ k
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


def axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1.0e-12)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )
