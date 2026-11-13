"""Vina-style chemistry terms for MetalGeoDock: hydrogen bonding, hydrophobic contact and
desolvation. These are the terms MetalGeoDock lacked relative to AutoDock Vina, which is why
Vina out-ranked it on the HDAC1 library screen (0.720 versus 0.687 on identical molecules).

Each function returns an energy (lower = better is preserved by making favourable terms
negative) plus diagnostics, so nothing is hidden from the caller.

  hbond_energy        donor/acceptor pairs within 3.6 A; Vina-style distance weighting
  hydrophobic_energy  carbon/sulfur contacts in the 3.4-4.6 A shell
  desolvation_energy  penalty for burying polar ligand atoms without a partner

Reference for the functional forms and default weights: Trott & Olson, J Comput Chem 2010.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .io import Molecule

# Vina-like reference weights (kcal/mol scale, before any dataset calibration)
W_HBOND = -0.587
W_HYDROPHOBIC = -0.035
W_DESOLV = 0.015

HBOND_CUT = 3.60
HBOND_MIN = 2.20
HYDROPHOBIC_LO = 3.40
HYDROPHOBIC_HI = 4.60
POLAR = {"N", "O", "F"}
APOLAR = {"C", "S", "CL", "BR", "I"}


def _ligand_polar_and_apolar(lig: Molecule):
    """Return (polar_idx, donor_idx, acceptor_idx, apolar_idx) as index arrays."""
    el = np.array([str(e).upper() for e in lig.elements])
    polar = np.where(np.isin(el, list(POLAR)))[0]
    apolar = np.where(np.isin(el, list(APOLAR)))[0]
    # donors/acceptors are approximated: every N/O/F is both (Vina does the same by atom type)
    donors = polar
    acceptors = polar
    return polar, donors, acceptors, apolar


def _protein_polar_apolar(protein: Molecule, metal_index: int, cutoff: float = 12.0):
    """Protein atoms near the metal, split into polar and apolar, with element labels."""
    mxyz = protein.xyz[metal_index]
    mask = protein.heavy_mask()
    idx = np.where(mask)[0]
    d = np.linalg.norm(protein.xyz[idx] - mxyz[None, :], axis=1)
    near = idx[d <= cutoff]
    el = np.array([str(protein.elements[i]).upper() for i in near])
    polar = near[np.isin(el, list(POLAR))]
    apolar = near[np.isin(el, list(APOLAR))]
    return polar, apolar


def hbond_energy(protein: Molecule, ligand: Molecule, metal_index: int) -> tuple[float, int]:
    """Hydrogen-bond energy and the number of hydrogen bonds.

    Distance-only: a donor-acceptor pair scores -0.587 * exp(-(d/3.6)^2) inside 2.2-3.6 A.
    Angle checks are omitted deliberately - MetalGeoDock poses do not carry hydrogens here,
    so an angle criterion would be noise.
    """
    _, donors, acceptors, _ = _ligand_polar_and_apolar(ligand)
    if donors.size == 0:
        return 0.0, 0
    p_polar, _ = _protein_polar_apolar(protein, metal_index)
    if p_polar.size == 0:
        return 0.0, 0
    pxyz = protein.xyz[p_polar]
    lxyz = ligand.xyz[donors]
    tree = cKDTree(pxyz)
    pairs = tree.query_ball_point(lxyz, HBOND_CUT)
    e = 0.0
    n = 0
    for i, js in enumerate(pairs):
        for j in js:
            d = float(np.linalg.norm(lxyz[i] - pxyz[j]))
            if d < HBOND_MIN:
                continue
            e += W_HBOND * np.exp(-((d / HBOND_CUT) ** 2))
            n += 1
    return float(e), n


def hydrophobic_energy(protein: Molecule, ligand: Molecule, metal_index: int) -> tuple[float, int]:
    """Hydrophobic contact energy and the number of carbon/sulfur contacts in 3.4-4.6 A."""
    _, _, _, apolar_l = _ligand_polar_and_apolar(ligand)
    if apolar_l.size == 0:
        return 0.0, 0
    _, apolar_p = _protein_polar_apolar(protein, metal_index)
    if apolar_p.size == 0:
        return 0.0, 0
    pxyz = protein.xyz[apolar_p]
    lxyz = ligand.xyz[apolar_l]
    tree = cKDTree(pxyz)
    pairs = tree.query_ball_point(lxyz, HYDROPHOBIC_HI)
    e = 0.0
    n = 0
    for i, js in enumerate(pairs):
        for j in js:
            d = float(np.linalg.norm(lxyz[i] - pxyz[j]))
            if HYDROPHOBIC_LO <= d <= HYDROPHOBIC_HI:
                e += W_HYDROPHOBIC
                n += 1
    return float(e), n


def desolvation_energy(protein: Molecule, ligand: Molecule, metal_index: int,
                       burial_cut: float = 4.5) -> tuple[float, float]:
    """Penalty for polar ligand atoms that are buried without a nearby partner.

    A polar atom with no protein polar atom within ``burial_cut`` but several protein apolar
    atoms around it is desolvated unfavourably; charge it W_DESOLV per apolar neighbour.
    """
    polar_l, _, _, _ = _ligand_polar_and_apolar(ligand)
    if polar_l.size == 0:
        return 0.0, 0.0
    p_polar, p_apolar = _protein_polar_apolar(protein, metal_index)
    lxyz = ligand.xyz[polar_l]
    e = 0.0
    buried = 0
    if p_polar.size:
        d_polar = cKDTree(protein.xyz[p_polar]).query(lxyz, k=1)[0]
    else:
        d_polar = np.full(len(lxyz), 99.0)
    if p_apolar.size:
        d_apolar_tree = cKDTree(protein.xyz[p_apolar])
        n_apolar = d_apolar_tree.query_ball_point(lxyz, 5.0)
    else:
        n_apolar = [[] for _ in lxyz]
    for i in range(len(lxyz)):
        if d_polar[i] <= burial_cut:
            continue
        k = len(n_apolar[i])
        if k >= 2:
            e += W_DESOLV * k
            buried += 1
    return float(e), float(buried / max(1, len(lxyz)))


def chemistry_terms(protein: Molecule, ligand: Molecule, metal_index: int) -> dict[str, float]:
    """All three terms in one call, returning raw energies and counts."""
    hb, n_hb = hbond_energy(protein, ligand, metal_index)
    hp, n_hp = hydrophobic_energy(protein, ligand, metal_index)
    ds, frac = desolvation_energy(protein, ligand, metal_index)
    return {
        "hbond": hb, "n_hbond": float(n_hb),
        "hydrophobic": hp, "n_hydrophobic": float(n_hp),
        "desolvation": ds, "desolv_frac": frac,
    }
