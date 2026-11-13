"""Ideal coordination polyhedra used by GM-DockZn (Wang et al., Bioinformatics 2020).

Motifs are labeled MET_P,L following GM-DockZn: P protein donors, L ligand donors.
Geometries: tetrahedral (td), trigonal bipyramidal (tbp), octahedral (oh),
square planar (sp), square pyramidal (spy).
"""

from __future__ import annotations

import numpy as np

_SQRT2 = np.sqrt(2.0)
_SQRT3 = np.sqrt(3.0)

IDEAL: dict[str, np.ndarray] = {
    "td": np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    ),
    "tbp": np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-0.5, _SQRT3 / 2.0, 0.0],
            [-0.5, -_SQRT3 / 2.0, 0.0],
        ],
        dtype=float,
    ),
    "oh": np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    ),
    "sp": np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    ),
    "spy": np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
}

for _key, _arr in list(IDEAL.items()):
    nrm = np.linalg.norm(_arr, axis=1, keepdims=True)
    IDEAL[_key] = _arr / nrm

# GM-DockZn seven motifs generalized as (geometry, protein_CN, ligand_CN).
GM_MOTIFS: tuple[tuple[str, int, int], ...] = (
    ("td", 3, 1),
    ("td", 2, 2),
    ("tbp", 2, 3),
    ("tbp", 3, 2),
    ("tbp", 4, 1),
    ("oh", 3, 3),
    ("oh", 4, 2),
    ("oh", 5, 1),
    ("sp", 3, 1),
    ("sp", 2, 2),
    ("spy", 4, 1),
    ("spy", 3, 2),
)


def geometry_for_counts(n_protein: int, n_ligand: int, preferred: tuple[str, ...]) -> str | None:
    total = n_protein + n_ligand
    mapping = {4: "td", 5: "tbp", 6: "oh"}
    geo = mapping.get(total)
    if geo is None:
        if total == 4 and "sp" in preferred:
            return "sp"
        return preferred[0] if preferred else None
    if geo in preferred:
        return geo
    return preferred[0]


def vacant_vertices(
    protein_unit: np.ndarray,
    geometry: str,
    n_vacant: int,
) -> tuple[np.ndarray, float]:
    """Return the unused ideal vertices after matching protein donors."""
    from .geometry import rmsd_or

    ideal = IDEAL[geometry]
    rmsd, perm = rmsd_or(protein_unit + np.zeros((protein_unit.shape[0], 3)), np.zeros(3), ideal)
    # protein_unit is already unit vectors at origin.
    occupied = set(int(i) for i in perm)
    vacant = [i for i in range(ideal.shape[0]) if i not in occupied]
    if len(vacant) < n_vacant:
        return np.zeros((0, 3)), rmsd
    return ideal[vacant[:n_vacant]], rmsd
