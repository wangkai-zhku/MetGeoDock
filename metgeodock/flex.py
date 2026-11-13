"""Cheap ligand semi-flex: rotate donor-adjacent torsions only.

Protein atoms are not moved. Pocket coordination-density overlap is used as
a filter: keep a torsion only if map_overlap does not fall. Typical cost is
a few extra MetGeo evaluations on the top seeds, not a conformer ensemble.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import axis_angle
from .io import Molecule
from .maps import map_overlap

_COV = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "SE": 1.20,
    "B": 0.84,
    "SI": 1.11,
}


def _el_key(element: str) -> str:
    e = str(element).upper()
    if e[:2] in _COV:
        return e[:2]
    return e[:1]


def infer_bonds(ligand: Molecule, slack_A: float = 0.45) -> list[tuple[int, int]]:
    """Distance-based bonds when MOL2 connectivity is missing (PDB ligands)."""
    xyz = np.asarray(ligand.xyz, dtype=float)
    keys = [_el_key(e) for e in ligand.elements]
    n = len(ligand)
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        ri = _COV.get(keys[i], 0.77)
        for j in range(i + 1, n):
            if keys[i] == "H" and keys[j] == "H":
                continue
            rj = _COV.get(keys[j], 0.77)
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            if 0.40 < d < ri + rj + slack_A:
                bonds.append((i, j))
    return bonds


def with_bonds(ligand: Molecule) -> Molecule:
    if ligand.bonds:
        return ligand
    mol = ligand.copy_with_xyz(ligand.xyz)
    mol.bonds = infer_bonds(mol)
    return mol


def _adjacency(n: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        if 0 <= a < n and 0 <= b < n and a != b:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def _component_from(adj: list[list[int]], start: int, blocked: int) -> list[int]:
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v == blocked or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return sorted(seen)


def donor_torsions(ligand: Molecule, donor_indices: tuple[int, ...], max_bonds: int = 2) -> list[tuple[int, int, np.ndarray]]:
    """Return (i, j, moving_atom_indices) for rotatable bonds next to donors."""
    bonds = ligand.bonds or infer_bonds(ligand)
    if not bonds or not donor_indices:
        return []
    adj = _adjacency(len(ligand), bonds)
    donors = {int(i) for i in donor_indices}
    out: list[tuple[int, int, np.ndarray]] = []
    seen_bond: set[tuple[int, int]] = set()
    for d in donors:
        if d < 0 or d >= len(ligand):
            continue
        for nbr in adj[d]:
            key = (min(d, nbr), max(d, nbr))
            if key in seen_bond:
                continue
            seen_bond.add(key)
            if _el_key(str(ligand.elements[d])) == "H" or _el_key(str(ligand.elements[nbr])) == "H":
                continue
            moving = _component_from(adj, nbr, d)
            if len(moving) < 2 or len(moving) > min(24, len(ligand) - 2):
                continue
            out.append((d, nbr, np.asarray(moving, dtype=int)))
            if len(out) >= max_bonds:
                return out
    return out


def rotate_about_bond(ligand: Molecule, i: int, j: int, moving: np.ndarray, angle_rad: float) -> Molecule:
    axis = ligand.xyz[j] - ligand.xyz[i]
    r = axis_angle(axis, angle_rad)
    xyz = ligand.xyz.copy()
    origin = ligand.xyz[i]
    xyz[moving] = (r @ (ligand.xyz[moving] - origin).T).T + origin
    mol = ligand.copy_with_xyz(xyz)
    mol.bonds = list(ligand.bonds or [])
    return mol


def flex_variants(
    ligand: Molecule,
    donor_indices: tuple[int, ...],
    metal: str,
    metal_xyz: np.ndarray,
    angles_deg: tuple[float, ...] = (-30.0, -15.0, 15.0, 30.0),
    max_bonds: int = 2,
    max_keep: int = 8,
) -> list[Molecule]:
    """Torsion copies that do not lower coordination-map overlap."""
    ligand = with_bonds(ligand)
    bonds = donor_torsions(ligand, donor_indices, max_bonds=max_bonds)
    if not bonds:
        return []
    base = map_overlap(metal, ligand, metal_xyz)
    kept: list[Molecule] = []
    for i, j, moving in bonds:
        for ang in angles_deg:
            mol = rotate_about_bond(ligand, i, j, moving, np.deg2rad(float(ang)))
            ov = map_overlap(metal, mol, metal_xyz)
            if ov + 1.0e-6 < base:
                continue
            kept.append(mol)
            if len(kept) >= max_keep:
                return kept
    return kept
