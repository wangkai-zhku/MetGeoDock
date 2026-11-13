"""Thermodynamic scoring terms built from literature constants, with no fitted parameters.

The premise: MetalGeoDock ranks poses well (AUROC 0.919) but ranks libraries poorly (0.687)
because every term it owns asks "is the coordination geometry correct?" and none asks "how
tightly does this molecule bind?". Binding tightness decomposes into contributions whose
coefficients are measured physical constants, so no activity data is needed to set them:

    dG_bind  ~  dG_coord + dG_buried + dG_hbond + dG_desolv - T*dS_conf

    dG_buried = -0.025 kcal/mol/A^2 * buried SASA        (hydrophobic effect, Chothia 1974;
                                                          Eisenberg & McLachlan 1986)
    dG_hbond  = -0.6  kcal/mol per hydrogen bond         (Fersht 1987; Williams 1991)
    dG_desolv = +0.015 kcal/mol per desolvated polar atom (Vina, Trott & Olson 2010)
    -T*dS_conf= +0.6  kcal/mol per rotatable bond        (frozen-rotor estimate, ~0.6 kcal/mol;
                                                          Doig & Sternberg 1995)

None of these numbers is tuned on our data. That is the point: if the combination works, it
works for physical reasons and transfers across metals by construction.

Deep-pocket handling: buried area is the term that grows with how far a ligand reaches into
the pocket, so it is the natural selector there. To stop a large floppy molecule from winning
on buried area alone, the entrance overlap is subtracted - a ligand that collides with the
pocket rim when trying to go deep is penalised.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .io import Molecule

# ---- literature constants (kcal/mol); do not fit these
D_BURIED = -0.025        # per A^2 of buried SASA
D_HBOND = -0.6           # per hydrogen bond
D_DESOLV = 0.015         # per desolvated polar atom
D_CONF = 0.6             # per rotatable bond
PROBE = 1.4              # water probe radius, A
N_SPHERE = 92            # Shrake-Rupley sphere points
RIM_CUT = 4.0            # A shell used to detect entrance clashes


def _sphere_points(n: int = N_SPHERE) -> np.ndarray:
    """Golden-spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


_SPH = _sphere_points()
_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47, "CL": 1.75,
        "BR": 1.85, "I": 1.98, "P": 1.80, "H": 1.20, "ZN": 1.39, "FE": 1.47,
        "MN": 1.61, "MG": 1.73, "CA": 1.95, "CO": 1.38, "CU": 1.32, "NI": 1.63}


def _radii(mol: Molecule, mask: np.ndarray | None = None) -> np.ndarray:
    el = [str(e).upper() for e, keep in zip(mol.elements, mol.heavy_mask() if mask is None else mask) if keep]
    return np.array([_VDW.get(e, 1.70) for e in el], float)


def sasa(points: np.ndarray, radii: np.ndarray, probe: float = PROBE) -> float:
    """Solvent-accessible surface area by Shrake-Rupley.

    For each atom, test N_SPHERE points on the (r+probe) sphere and count those not inside
    any neighbour's sphere. O(N_atoms^2 * N_SPHERE); for a 60-atom ligand plus a pocket
    subset this is a few milliseconds.
    """
    if len(points) == 0:
        return 0.0
    rtot = radii + probe
    # only neighbours within 2*(max radius + probe) can occlude
    tree = cKDTree(points)
    maxr = float(rtot.max())
    total = 0.0
    for i in range(len(points)):
        near = tree.query_ball_point(points[i], 2 * maxr)
        near = [j for j in near if j != i]
        pts = points[i] + rtot[i] * _SPH
        if near:
            d = np.linalg.norm(pts[:, None, :] - points[near][None, :, :], axis=2)
            occluded = (d < rtot[near][None, :]).any(axis=1)
            free = int((~occluded).sum())
        else:
            free = len(_SPH)
        total += 4 * np.pi * rtot[i] ** 2 * free / len(_SPH)
    return float(total)


def buried_area(protein: Molecule, ligand: Molecule, metal_index: int,
                cutoff: float = 12.0) -> tuple[float, float]:
    """Buried SASA (A^2) and the entrance-rim overlap used to penalise blocked entry.

    buried = SASA(ligand alone) - SASA(ligand inside pocket)
    rim    = ligand atoms within RIM_CUT of protein atoms that sit *outside* the pocket axis
             direction, i.e. the part of the ligand that would have to squeeze past the rim.
    """
    lm = ligand.heavy_mask()
    lxyz = ligand.xyz[lm]
    lrad = _radii(ligand)
    if len(lxyz) == 0:
        return 0.0, 0.0

    mxyz = protein.xyz[metal_index]
    pm = protein.heavy_mask()
    pidx = np.where(pm)[0]
    d = np.linalg.norm(protein.xyz[pidx] - mxyz[None, :], axis=1)
    near = pidx[d <= cutoff]
    pxyz = protein.xyz[near]
    prad = _radii(protein, mask=np.isin(np.arange(len(protein.elements)), near))

    alone = sasa(lxyz, lrad)
    all_pts = np.vstack([lxyz, pxyz])
    all_rad = np.concatenate([lrad, prad])
    together = sasa(all_pts, all_rad)
    # SASA of the ligand portion inside the complex = total - protein-only contribution
    prot_only = sasa(pxyz, prad)
    lig_in_complex = max(0.0, together - prot_only)
    buried = max(0.0, alone - lig_in_complex)

    # entrance rim: protein atoms whose projection along the metal->pocket axis is negative
    # (i.e. behind the metal, at the mouth) that the ligand touches
    axis = pxyz.mean(axis=0) - mxyz
    nrm = np.linalg.norm(axis)
    if nrm > 1e-6 and len(pxyz):
        axis = axis / nrm
        proj_p = (pxyz - mxyz) @ axis
        rim = pxyz[proj_p < 0.0]
        if len(rim):
            tree = cKDTree(rim)
            dd = tree.query(lxyz, k=1)[0]
            overlap = float(np.clip(RIM_CUT - dd, 0.0, None).sum())
        else:
            overlap = 0.0
    else:
        overlap = 0.0
    return buried, overlap


def thermodynamic_terms(protein: Molecule, ligand: Molecule, metal_index: int,
                        n_rotatable: int = 0, n_hbond: int = 0,
                        n_desolv: float = 0.0) -> dict[str, float]:
    """The four literature-constant terms plus their sum.

    ``n_rotatable`` comes from RDKit on the ligand SMILES; ``n_hbond`` and ``n_desolv`` come
    from metgeodock.chemistry, which already computes them.
    """
    buried, rim = buried_area(protein, ligand, metal_index)
    g_buried = D_BURIED * buried
    g_hbond = D_HBOND * n_hbond
    g_desolv = D_DESOLV * max(0.0, rim)      # reuse the rim measure as a desolvation proxy
    g_conf = D_CONF * max(0, int(n_rotatable))
    return {
        "g_buried": float(g_buried),
        "g_hbond": float(g_hbond),
        "g_desolv": float(g_desolv),
        "g_conf": float(g_conf),
        "buried_area": float(buried),
        "rim_overlap": float(rim),
        "g_total": float(g_buried + g_hbond + g_desolv + g_conf),
    }
