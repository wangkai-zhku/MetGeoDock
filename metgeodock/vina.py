"""AutoDock Vina empirical terms (Trott & Olson, JCC 2010).

This is a metal-agnostic protein–ligand score. Metal coordination is handled
separately by geometric matching / GPDOCK / SLEF terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Molecule

# Vina van der Waals radii (Angstrom) by element.
VDW = {
    "H": 1.00,
    "C": 1.90,
    "N": 1.80,
    "O": 1.70,
    "F": 1.50,
    "P": 2.10,
    "S": 2.00,
    "CL": 1.80,
    "BR": 2.00,
    "I": 2.20,
    "ZN": 1.48,
    "MG": 1.30,
    "CA": 1.74,
    "MN": 1.30,
    "FE": 1.30,
    "CU": 1.40,
    "CO": 1.30,
    "CD": 1.58,
    "NI": 1.30,
}

WEIGHTS = {
    "gauss1": -0.035579,
    "gauss2": -0.005156,
    "repulsion": 0.840245,
    "hydrophobic": -0.035069,
    "hbond": -0.587439,
}

HYDROPHOBIC = {"C"}
HBOND_DONOR_ACCEPTOR = {"N", "O"}


def _radius(element: str) -> float:
    return VDW.get(element.upper(), 1.80)


def _hydrophobic(element: str) -> bool:
    return element.upper() in HYDROPHOBIC


def _hbond(element: str) -> bool:
    return element.upper() in HBOND_DONOR_ACCEPTOR


@dataclass
class VinaTerms:
    gauss1: float
    gauss2: float
    repulsion: float
    hydrophobic: float
    hbond: float

    @property
    def total(self) -> float:
        return (
            WEIGHTS["gauss1"] * self.gauss1
            + WEIGHTS["gauss2"] * self.gauss2
            + WEIGHTS["repulsion"] * self.repulsion
            + WEIGHTS["hydrophobic"] * self.hydrophobic
            + WEIGHTS["hbond"] * self.hbond
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "vina_gauss1": self.gauss1,
            "vina_gauss2": self.gauss2,
            "vina_repulsion": self.repulsion,
            "vina_hydrophobic": self.hydrophobic,
            "vina_hbond": self.hbond,
            "vina_total": self.total,
        }


def vina_terms(
    protein: Molecule,
    ligand: Molecule,
    cutoff: float = 8.0,
    skip_metal: bool = True,
) -> VinaTerms:
    p_mask = protein.heavy_mask()
    l_mask = ligand.heavy_mask()
    if skip_metal:
        metal = np.zeros(len(protein), dtype=bool)
        midx = protein.metal_indices()
        if midx.size:
            metal[midx] = True
        p_mask = p_mask & ~metal
    p_xyz = protein.xyz[p_mask]
    l_xyz = ligand.xyz[l_mask]
    if p_xyz.size == 0 or l_xyz.size == 0:
        return VinaTerms(0.0, 0.0, 0.0, 0.0, 0.0)
    p_r = np.array([_radius(str(el)) for el in protein.elements[p_mask]])
    l_r = np.array([_radius(str(el)) for el in ligand.elements[l_mask]])
    p_h = np.array([_hydrophobic(str(el)) for el in protein.elements[p_mask]])
    l_h = np.array([_hydrophobic(str(el)) for el in ligand.elements[l_mask]])
    p_b = np.array([_hbond(str(el)) for el in protein.elements[p_mask]])
    l_b = np.array([_hbond(str(el)) for el in ligand.elements[l_mask]])
    delta = l_xyz[:, None, :] - p_xyz[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=2))
    surface = dist - (l_r[:, None] + p_r[None, :])
    close = dist <= cutoff
    s = np.where(close, surface, 0.0)
    g1 = float(np.exp(-(s / 0.5) ** 2)[close].sum()) if close.any() else 0.0
    g2 = float(np.exp(-((s - 3.0) ** 2) / 4.0)[close].sum()) if close.any() else 0.0
    rep_mask = close & (surface < 0.0)
    rep = float((surface[rep_mask] ** 2).sum()) if rep_mask.any() else 0.0
    hyd = 0.0
    hb = 0.0
    hyd_pair = close & l_h[:, None] & p_h[None, :]
    if hyd_pair.any():
        sh = surface[hyd_pair]
        hyd = float(np.where(sh < 0.5, 1.0, np.where(sh < 1.5, 1.5 - sh, 0.0)).sum())
    hb_pair = close & l_b[:, None] & p_b[None, :]
    if hb_pair.any():
        sb = surface[hb_pair]
        hb = float(np.where(sb < -0.7, 1.0, np.where(sb < 0.0, -sb / 0.7, 0.0)).sum())
    return VinaTerms(g1, g2, rep, hyd, hb)
