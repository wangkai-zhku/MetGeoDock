"""MetGeo score: GPDOCK-derived force field (no SLEF, no Vina fusion).

The default total is ``ff_energy`` from ``metgeodock.forcefield``:
tabulated −log P(d) + harmonic-to-mode + steric + pocket contacts + RMSD_OR.
SLEF remains in this module only as a disabled diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coordination import CoordinationSphere, ligand_donors
from .geometry import rmsd_or
from .io import Molecule
from .metals import (
    FORMAL_CHARGE,
    expected_coordination,
    gp_probability,
    histogram_mode_distance,
    metal_from_name,
)
from .templates import IDEAL, geometry_for_counts

ALPHA = 0.11
BETA = 0.81
GAMMA = 0.74
COULOMB = 332.0637

# Lower total is better. vina remains in the dict so ablation scripts can
# re-enable it, but the default fusion weight is zero.
DEFAULT_WEIGHTS = {
    "vina": 0.00,
    "slef": 0.00,
    "geo": 0.00,
    "gp": 0.00,
    "clash": 0.00,
    "dist": 0.00,
    "cn": 0.00,
    "ff": 1.00,
}


@dataclass
class HybridScore:
    total: float
    vina: float
    slef: float
    geo: float
    gp: float
    clash: float
    rmsd_or: float
    n_coord_lig: int
    n_coord_pro: int
    dist: float = 0.0
    cn: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)


def slef_metal_energy(
    ligand: Molecule,
    metal_xyz: np.ndarray,
    metal_name: str,
    metal_vdw: float = 1.48,
) -> float:
    """Non-bonded SLEF energy between ligand atoms and the metal ion."""
    qz = FORMAL_CHARGE[metal_from_name(metal_name)]
    energy = 0.0
    for xyz, charge, el in zip(ligand.xyz, ligand.charges, ligand.elements):
        if str(el).upper() == "H":
            continue
        r = float(np.linalg.norm(xyz - metal_xyz))
        r = max(r, 0.60)
        q = float(charge)
        lj_sigma = 0.5 * (metal_vdw + _el_sigma(str(el)))
        rsq = lj_sigma / r
        r6 = rsq ** 6
        vdw = np.sqrt(0.2 * 0.2) * (r6 * r6 - 2.0 * r6)
        chdvdw = (abs(q) + abs(qz)) ** 2 / (lj_sigma * lj_sigma)
        vdwchd = (abs(q) + abs(qz)) / max(abs(qz), 1.0e-6)
        cs = 1.0 / (1.0 + ALPHA * chdvdw * np.exp(min(BETA * r * r, 40.0)))
        cl = 1.0 / (1.0 + vdwchd * np.exp(min(1.0 - GAMMA * r, 40.0)))
        ele = COULOMB * (cs + cl) * q * qz / r
        energy += vdw + ele
    return float(energy)


def _el_sigma(element: str) -> float:
    return {
        "C": 1.90,
        "N": 1.80,
        "O": 1.70,
        "S": 2.00,
        "P": 2.10,
        "F": 1.50,
        "CL": 1.80,
        "BR": 2.00,
        "I": 2.20,
    }.get(element.upper(), 1.80)


def clash_energy(protein: Molecule, ligand: Molecule, metal_index: int, cutoff: float = 2.0) -> float:
    p_mask = protein.heavy_mask()
    p_mask[metal_index] = False
    p = protein.xyz[p_mask]
    l = ligand.xyz[ligand.heavy_mask()]
    if p.size == 0 or l.size == 0:
        return 0.0
    dist = np.sqrt(((l[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
    hit = dist < cutoff
    if not hit.any():
        return 0.0
    return float(((cutoff - dist[hit]) ** 2).sum())


def gp_log_score(metal: str, ligand: Molecule, metal_xyz: np.ndarray, max_d: float = 2.8) -> tuple[float, int]:
    idx, xyz = ligand_donors(ligand)
    if idx.size == 0:
        return 6.0, 0
    logp = 0.0
    n = 0
    for i, coord in zip(idx, xyz):
        d = float(np.linalg.norm(coord - metal_xyz))
        if d > max_d:
            continue
        p = gp_probability(metal, str(ligand.elements[i]), d)
        logp += -np.log(p)
        n += 1
    if n == 0:
        return 6.0, 0
    return float(logp), n


def distance_harmonic(
    metal: str,
    ligand: Molecule,
    metal_xyz: np.ndarray,
    max_d: float = 2.9,
) -> tuple[float, int, float, float]:
    """Sum of squared deviations from GPDOCK mode distances.

    Returns (harmonic_energy, n_coord, min_dist, mean_dist).
    """
    idx, xyz = ligand_donors(ligand)
    if idx.size == 0:
        return 1.50, 0, float("nan"), float("nan")
    energy = 0.0
    ds: list[float] = []
    for i, coord in zip(idx, xyz):
        d = float(np.linalg.norm(coord - metal_xyz))
        if d > max_d:
            continue
        d0 = histogram_mode_distance(metal, str(ligand.elements[i]))
        energy += (d - d0) ** 2
        ds.append(d)
    if not ds:
        return 1.50, 0, float("nan"), float("nan")
    arr = np.asarray(ds, dtype=float)
    return float(energy), int(arr.size), float(arr.min()), float(arr.mean())


def coordination_penalty(metal: str, n_protein: int, n_ligand: int) -> float:
    lo, hi = expected_coordination(metal)
    total = int(n_protein + n_ligand)
    if total < lo:
        return float(lo - total)
    if total > hi:
        return float(total - hi)
    return 0.0


def geometry_penalty(
    sphere: CoordinationSphere,
    ligand: Molecule,
    geometry: str | None = None,
) -> tuple[float, str]:
    lig_idx, lig_xyz = ligand_donors(ligand)
    if lig_idx.size == 0:
        return 1.50, "none"
    close = []
    for coord, i in zip(lig_xyz, lig_idx):
        d = float(np.linalg.norm(coord - sphere.metal_xyz))
        if d <= sphere.distances.max() + 0.6 if sphere.distances.size else 2.8:
            if d < 2.9:
                close.append(coord)
    n_lig = max(len(close), 1)
    n_pro = sphere.n_protein
    preferred = {
        "zn": ("td", "tbp", "oh"),
        "cu": ("sp", "spy", "td"),
        "ni": ("sp", "oh", "td"),
        "mg": ("oh", "tbp", "td"),
        "ca": ("oh", "tbp"),
        "mn": ("oh", "tbp", "td"),
        "fe2": ("oh", "tbp", "td"),
        "fe3": ("oh", "tbp", "td"),
        "co": ("oh", "td"),
        "cd": ("oh", "td"),
    }.get(sphere.metal_name, ("td", "oh"))
    geo = geometry or geometry_for_counts(n_pro, n_lig, preferred) or preferred[0]
    donors = np.vstack([sphere.protein_xyz, np.asarray(close).reshape(-1, 3)]) if close else sphere.protein_xyz
    if donors.size == 0:
        return 1.50, geo
    rmsd, _perm = rmsd_or(donors, sphere.metal_xyz, IDEAL[geo])
    if not np.isfinite(rmsd):
        rmsd = 1.5
    return float(rmsd), geo


def score_pose(
    protein: Molecule,
    ligand: Molecule,
    sphere: CoordinationSphere,
    weights: dict[str, float] | None = None,
    include_vina: bool = False,
) -> HybridScore:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    from .forcefield import ff_energy

    vina_total = 0.0
    vina_terms: dict[str, float] = {}
    need_vina = include_vina or abs(float(w.get("vina", 0.0))) > 0.0
    if need_vina:
        from .vina import vina_terms

        vt = vina_terms(protein, ligand)
        vina_total = float(vt.total)
        vina_terms = vt.as_dict()
    ff = ff_energy(protein, ligand, sphere)
    gp, n_lig = gp_log_score(sphere.metal_name, ligand, sphere.metal_xyz)
    geo_rmsd, geo_name = geometry_penalty(sphere, ligand)
    clash = float(ff["steric"])
    dist_e, n_dist, min_d, mean_d = distance_harmonic(sphere.metal_name, ligand, sphere.metal_xyz)
    n_use = n_lig if n_lig else n_dist
    cn = coordination_penalty(sphere.metal_name, sphere.n_protein, n_use)
    total = float(w.get("ff", 1.0)) * float(ff["total"]) + w["vina"] * vina_total
    terms = dict(vina_terms)
    terms.update(
        {
            "ff_total": ff["total"],
            "ff_coord": ff["coord"],
            "ff_steric": ff["steric"],
            "ff_contact": ff["contact"],
            "gp_nll": gp,
            "geo_rmsd_or": geo_rmsd,
            "geometry": geo_name,
            "clash": clash,
            "dist_harm": dist_e,
            "cn_pen": cn,
            "min_donor_dist": min_d if np.isfinite(min_d) else 9.0,
            "mean_donor_dist": mean_d if np.isfinite(mean_d) else 9.0,
            "n_coord_lig": n_use,
            "n_coord_pro": sphere.n_protein,
            "slef": 0.0,
            "map_overlap": float(ff.get("map_overlap", 0.0)),
            "centroid_pen": float(ff.get("centroid_pen", 0.0)),
            "contact": float(ff.get("contact", 0.0)),
            "n_water": float(ff.get("n_water", sphere.n_water)),
            "n_other_metal": float(ff.get("n_other_metal", sphere.n_other_metal)),
            "n_water_displaced": float(ff.get("n_water_displaced", 0.0)),
            "water_min_dist": float(ff.get("water_min_dist", 9.0)),
        }
    )
    return HybridScore(
        total=float(total),
        vina=float(vina_total),
        slef=0.0,
        geo=float(geo_rmsd),
        gp=float(gp),
        clash=float(clash),
        rmsd_or=float(geo_rmsd),
        n_coord_lig=int(n_use),
        n_coord_pro=int(sphere.n_protein),
        dist=float(dist_e),
        cn=float(cn),
        terms=terms,
    )


def feature_vector(score: HybridScore, metal: str, ligand: Molecule) -> dict[str, float]:
    """Numeric features for the ML scorer (no Vina fusion)."""
    metal = metal_from_name(metal)
    feats = {
        "slef": 0.0,
        "gp_nll": score.gp,
        "geo_rmsd_or": score.geo,
        "clash": score.clash,
        "dist_harm": score.dist,
        "cn_pen": score.cn,
        "ff_coord": float(score.terms.get("ff_coord", 0.0)),
        "ff_steric": float(score.terms.get("ff_steric", 0.0)),
        "ff_contact": float(score.terms.get("ff_contact", 0.0)),
        "n_coord_lig": float(score.n_coord_lig),
        "n_coord_pro": float(score.n_coord_pro),
        "n_heavy": float(ligand.heavy_mask().sum()),
        "n_donor": float(ligand.donor_indices().size),
        "charge_sum": float(ligand.charges[ligand.heavy_mask()].sum()) if ligand.charges.size else 0.0,
        "min_donor_dist": float(score.terms.get("min_donor_dist", 9.0)),
        "mean_donor_dist": float(score.terms.get("mean_donor_dist", 9.0)),
        "slef_tanh": 0.0,
        "map_overlap": float(score.terms.get("map_overlap", 0.0)),
    }
    for name in (
        "zn",
        "ca",
        "mg",
        "mn",
        "fe2",
        "fe3",
        "cu",
        "co",
        "cd",
        "ni",
    ):
        feats[f"metal_{name}"] = 1.0 if name == metal else 0.0
    return feats
