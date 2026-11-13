"""First-shell metal coordination detection (GPDOCK / GM-DockZn cutoffs)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import unit_vectors
from .io import METAL_RESNAMES, WATER_RESNAMES, Molecule
from .metals import METALS, donor_cutoffs, metal_from_name


@dataclass
class CoordinationSphere:
    metal_name: str
    metal_index: int
    metal_xyz: np.ndarray
    protein_idx: np.ndarray
    protein_xyz: np.ndarray
    protein_elements: np.ndarray
    water_idx: np.ndarray
    water_xyz: np.ndarray
    other_metal_idx: np.ndarray
    other_metal_xyz: np.ndarray
    distances: np.ndarray

    @property
    def n_protein(self) -> int:
        return int(self.protein_idx.size)

    @property
    def n_water(self) -> int:
        return int(self.water_idx.size)

    @property
    def n_other_metal(self) -> int:
        return int(self.other_metal_idx.size)

    @property
    def protein_unit(self) -> np.ndarray:
        if self.n_protein == 0:
            return np.zeros((0, 3))
        return unit_vectors(self.protein_xyz, self.metal_xyz)


def _is_water(mol: Molecule, i: int) -> bool:
    return str(mol.resnames[i]).strip().upper() in WATER_RESNAMES


def detect_coordination(
    protein: Molecule,
    metal: str | None = None,
    metal_index: int | None = None,
) -> CoordinationSphere:
    """Identify the first coordination shell of a metal ion in ``protein``."""
    metal_idx = protein.metal_indices(metal=metal)
    if metal_idx.size == 0:
        metal_idx = protein.metal_indices()
    if metal_idx.size == 0:
        raise ValueError("No metal ion found in protein coordinates")
    if metal_index is None:
        metal_index = int(metal_idx[0])
    res = str(protein.resnames[metal_index])
    el = str(protein.elements[metal_index])
    try:
        metal_name = metal_from_name(res or el)
    except Exception:
        metal_name = metal_from_name(metal or res or el)
    params = METALS[metal_name]
    cut = donor_cutoffs(metal_name)
    origin = protein.xyz[metal_index]
    p_idx, p_xyz, p_el, dlist = [], [], [], []
    w_idx, w_xyz = [], []
    om_idx, om_xyz = [], []
    for i, (xyz, element) in enumerate(zip(protein.xyz, protein.elements)):
        if i == metal_index:
            continue
        resn = str(protein.resnames[i]).strip().upper()
        dist = float(np.linalg.norm(xyz - origin))
        if resn in METAL_RESNAMES:
            if dist <= 5.50:
                om_idx.append(i)
                om_xyz.append(xyz)
            continue
        el1 = str(element).upper()[:1]
        if el1 not in cut:
            continue
        lo, hi = cut[el1]
        # GM-DockZn used 2.5 A (non-S) / 2.8 A (S) for Zn; GPDOCK uses metal-specific max.
        if dist < lo - 0.15 or dist > hi:
            continue
        if _is_water(protein, i):
            w_idx.append(i)
            w_xyz.append(xyz)
        else:
            p_idx.append(i)
            p_xyz.append(xyz)
            p_el.append(el1)
            dlist.append(dist)
    return CoordinationSphere(
        metal_name=metal_name,
        metal_index=metal_index,
        metal_xyz=origin.copy(),
        protein_idx=np.asarray(p_idx, dtype=int),
        protein_xyz=np.asarray(p_xyz, dtype=float).reshape(-1, 3),
        protein_elements=np.asarray(p_el) if p_el else np.asarray([], dtype=object),
        water_idx=np.asarray(w_idx, dtype=int),
        water_xyz=np.asarray(w_xyz, dtype=float).reshape(-1, 3),
        other_metal_idx=np.asarray(om_idx, dtype=int),
        other_metal_xyz=np.asarray(om_xyz, dtype=float).reshape(-1, 3),
        distances=np.asarray(dlist, dtype=float),
    )


def pocket_metal_indices(
    protein: Molecule,
    ligand: Molecule | None = None,
    ligand_radius_A: float = 6.0,
    cluster_A: float = 5.50,
) -> np.ndarray:
    """Metal ions in the binding pocket: near the ligand, plus binuclear partners."""
    mets = protein.metal_indices()
    if mets.size == 0:
        return mets
    if ligand is None or len(ligand) == 0:
        return mets
    heavy = ligand.xyz[ligand.heavy_mask()] if ligand.heavy_mask().any() else ligand.xyz
    near: list[int] = []
    for i in mets:
        dmin = float(np.linalg.norm(heavy - protein.xyz[int(i)], axis=1).min())
        if dmin <= ligand_radius_A:
            near.append(int(i))
    if not near:
        # fall back to the metal closest to the ligand
        d = [float(np.linalg.norm(heavy - protein.xyz[int(i)], axis=1).min()) for i in mets]
        return np.array([int(mets[int(np.argmin(d))])], dtype=int)
    changed = True
    cluster = set(near)
    while changed:
        changed = False
        for i in mets:
            ii = int(i)
            if ii in cluster:
                continue
            if any(float(np.linalg.norm(protein.xyz[ii] - protein.xyz[j])) <= cluster_A for j in cluster):
                cluster.add(ii)
                changed = True
    return np.array(sorted(cluster), dtype=int)


def ligand_donors(ligand: Molecule) -> tuple[np.ndarray, np.ndarray]:
    idx = ligand.donor_indices()
    heavy = [i for i in idx if str(ligand.elements[i]).upper() != "H"]
    return np.asarray(heavy, dtype=int), ligand.xyz[heavy] if heavy else np.zeros((0, 3))
