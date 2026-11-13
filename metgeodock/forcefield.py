"""MetGeo force field derived from GPDOCK distance histograms.

Replaces SLEF. Coordination energy is a tabulated −log P(d) plus a harmonic
restraint to the histogram mode. Steric repulsion is a quadratic clash below
element-wise VDW radii. A contact well rewards non-clashing pocket packing.
Parameters are produced by scripts/analysis/fit_metalgeo_ff.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .coordination import CoordinationSphere, ligand_donors
from .geometry import rmsd_or
from .io import Molecule
from .metals import gp_probability, histogram_mode_distance, metal_from_name
from .templates import IDEAL, geometry_for_counts

_ROOT = Path(__file__).resolve().parents[1]
_PKG_JSON = Path(__file__).resolve().parent / "data" / "metalgeo_params.json"
_ROOT_JSON = _ROOT / "processed" / "ff" / "metalgeo_params.json"
_DEFAULT_JSON = _PKG_JSON if _PKG_JSON.exists() else _ROOT_JSON

_VDW = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "H": 1.10,
}

_K_FALLBACK = 8.0


@lru_cache(maxsize=1)
def load_params(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_JSON
    if p.exists():
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "weights": {"coord": 1.0, "steric": 12.0, "contact": 0.02, "geo": 2.0, "uncoord": 12.0},
        "rank_weights": {"coord": 0.35, "geo": 0.0, "cpen": 0.0, "contact": 0.0, "map": 0.0},
        "vdw_sigma_angstrom": _VDW,
        "steric_cutoff_angstrom": 2.30,
        "contact_lo_angstrom": 3.10,
        "contact_hi_angstrom": 4.60,
        "pairs": {},
    }


def _sigma(element: str, params: dict) -> float:
    table = params.get("vdw_sigma_angstrom") or _VDW
    return float(table.get(str(element).upper(), 1.70))


def _pair(metal: str, element: str, params: dict) -> dict:
    metal = metal_from_name(metal)
    el = str(element).upper()[0]
    return (params.get("pairs") or {}).get(metal, {}).get(el, {})


def coord_energy(
    metal: str,
    ligand: Molecule,
    metal_xyz: np.ndarray,
    params: dict | None = None,
    n_protein: int = 0,
    donor_chem_scale: float | None = None,
) -> tuple[float, int, float]:
    """Coordination energy using the best remaining vacant-site contacts only.

    ``donor_chem_scale`` > 0 adds a per-donor penalty ``-ln(strength(metal, group)) * scale``
    from :mod:`metgeodock.donor_chem`, and drops donors whose group cannot coordinate
    that metal (amide N, pyrrole N). ``None`` falls back to the module-level switch;
    0.0 reproduces the original purely geometric behaviour exactly.
    """
    params = params or load_params()
    idx, xyz = ligand_donors(ligand)
    w = params["weights"]
    if idx.size == 0:
        return float(w["uncoord"]), 0, 1.5
    groups = getattr(ligand, "donor_groups", None)
    if donor_chem_scale is None:
        from .donor_chem import active_scale

        donor_chem_scale = active_scale()
    use_chem = bool(groups) and donor_chem_scale > 0.0
    n_expect = max(1, min(2, 4 - int(n_protein)))
    terms: list[tuple[float, float]] = []
    for i, coord in zip(idx, xyz):
        d = float(np.linalg.norm(coord - metal_xyz))
        pair = _pair(metal, str(ligand.elements[i]), params)
        dmax = float(pair.get("dmax_angstrom", 2.9))
        if d > dmax + 0.15:
            continue
        d0 = float(pair.get("mode_angstrom", histogram_mode_distance(metal, str(ligand.elements[i]))))
        k = min(float(pair.get("k_kcal_per_a2", _K_FALLBACK)), 8.0)
        p = gp_probability(metal, str(ligand.elements[i]), d)
        p_mode = max(float(pair.get("p_mode", 0.2)), 1.0e-4)
        nll = -np.log(max(p, 1.0e-4) / p_mode)
        h = 0.5 * k * (d - d0) ** 2
        if use_chem:
            group = groups.get(int(i))
            if group is not None:
                from .donor_chem import donor_penalty, donor_strength

                if donor_strength(group, metal) < 0.1:
                    continue  # chemically cannot coordinate this metal
                nll += donor_penalty(group, metal, scale=donor_chem_scale)
        terms.append((nll + h, h))
    if not terms:
        return float(w["uncoord"]), 0, 1.5
    terms.sort(key=lambda t: t[0])
    used = terms[:n_expect]
    energy = float(sum(t[0] for t in used))
    harm = float(sum(t[1] for t in used))
    return float(w["coord"] * energy), int(len(used)), harm


def displaced_water_mask(
    ligand: Molecule,
    water_xyz: np.ndarray | None,
    cut_A: float = 0.85,
) -> tuple[np.ndarray, float]:
    """Boolean mask of first-shell waters sitting on ligand donors, plus leftover min distance."""
    if water_xyz is None or np.asarray(water_xyz).size == 0:
        return np.zeros(0, dtype=bool), 9.0
    wxyz = np.asarray(water_xyz, dtype=float).reshape(-1, 3)
    donors_idx = ligand.donor_indices()
    donors = ligand.xyz[donors_idx] if donors_idx.size else ligand.xyz[ligand.heavy_mask()]
    if donors.size == 0:
        return np.zeros(len(wxyz), dtype=bool), 9.0
    dmin = cdist(wxyz, donors).min(axis=1)
    mask = dmin < float(cut_A)
    remain = dmin[~mask]
    wmin = float(remain.min()) if remain.size else 9.0
    return mask, wmin


def displaced_water_count(ligand: Molecule, water_xyz: np.ndarray | None, cut_A: float = 0.85) -> tuple[int, float]:
    mask, wmin = displaced_water_mask(ligand, water_xyz, cut_A=cut_A)
    return int(mask.sum()), float(wmin)


def protein_steric_cache(
    protein: Molecule,
    metal_index: int,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-atom protein coordinates and VDW radii, metal ion excluded."""
    params = params or load_params()
    p_mask = protein.heavy_mask()
    if 0 <= metal_index < len(p_mask):
        p_mask[metal_index] = False
    pxyz = protein.xyz[p_mask]
    pel = [str(e).upper() for e, keep in zip(protein.elements, p_mask) if keep]
    sig_p = np.array([_sigma(e, params) for e in pel], dtype=float)
    return pxyz, sig_p


def steric_and_contact(
    protein: Molecule,
    ligand: Molecule,
    metal_index: int,
    params: dict | None = None,
    protein_cache: tuple[np.ndarray, np.ndarray] | None = None,
    metal_xyz: np.ndarray | None = None,
    pocket_axis: np.ndarray | None = None,
    displace_xyz: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Return (steric_energy, n_contact, n_clash_pairs)."""
    params = params or load_params()
    if protein_cache is None:
        pxyz, sig_p = protein_steric_cache(protein, metal_index, params)
    else:
        pxyz, sig_p = protein_cache
    if displace_xyz is not None and np.asarray(displace_xyz).size and pxyz.size:
        dw = np.asarray(displace_xyz, dtype=float).reshape(-1, 3)
        keep = cdist(pxyz, dw).min(axis=1) > 0.40
        pxyz, sig_p = pxyz[keep], sig_p[keep]
    l_mask = ligand.heavy_mask()
    lxyz = ligand.xyz[l_mask]
    if pxyz.size == 0 or lxyz.size == 0:
        return 0.0, 0.0, 0
    dist = cdist(lxyz, pxyz)
    lel = [str(e).upper() for e, keep in zip(ligand.elements, l_mask) if keep]
    sig_l = np.array([_sigma(e, params) for e in lel], dtype=float)
    sigma = 0.5 * (sig_l[:, None] + sig_p[None, :])
    overlap = np.clip(sigma - dist, 0.0, None)
    core = np.clip(1.80 - dist, 0.0, None)
    steric = float((overlap * overlap).sum() + 4.0 * (core * core).sum())
    lo = float(params.get("contact_lo_angstrom", 3.10))
    hi = float(params.get("contact_hi_angstrom", 4.60))
    contact_mask = (dist > np.maximum(sigma, lo)) & (dist < hi)
    if pocket_axis is not None and metal_xyz is not None:
        axis = np.asarray(pocket_axis, dtype=float)
        nrm = float(np.linalg.norm(axis))
        if nrm > 1.0e-8:
            side = (pxyz - np.asarray(metal_xyz, dtype=float)) @ (axis / nrm)
            contact_mask = contact_mask & (side[None, :] > 0.10)
    n_contact = int(contact_mask.sum())
    n_clash = int((overlap > 0).sum())
    w = params["weights"]
    return float(w["steric"] * steric), float(w["contact"] * n_contact), n_clash


def geometry_energy(sphere: CoordinationSphere, ligand: Molecule, params: dict | None = None) -> tuple[float, str]:
    params = params or load_params()
    lig_idx, lig_xyz = ligand_donors(ligand)
    scored = [
        (float(np.linalg.norm(coord - sphere.metal_xyz)), coord)
        for coord in lig_xyz
        if float(np.linalg.norm(coord - sphere.metal_xyz)) < 2.9
    ]
    scored.sort(key=lambda t: t[0])
    n_expect = max(1, min(2, 4 - int(sphere.n_protein)))
    close = [coord for _d, coord in scored[:n_expect]]
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
    n_lig = max(len(close), 1)
    geo = geometry_for_counts(sphere.n_protein, n_lig, preferred) or preferred[0]
    donors = np.vstack([sphere.protein_xyz, np.asarray(close).reshape(-1, 3)]) if close else sphere.protein_xyz
    if donors.size == 0:
        return float(params["weights"]["geo"] * 1.5), geo
    rmsd, _ = rmsd_or(donors, sphere.metal_xyz, IDEAL[geo])
    if not np.isfinite(rmsd):
        rmsd = 1.5
    return float(params["weights"]["geo"] * rmsd), geo


def ff_energy(
    protein: Molecule,
    ligand: Molecule,
    sphere: CoordinationSphere,
    params: dict | None = None,
    skip_geo: bool = False,
    protein_cache: tuple[np.ndarray, np.ndarray] | None = None,
    include_chem: bool = False,
) -> dict[str, float]:
    """Lower is better. total = steric + 0.35*coord + geo - contact."""
    params = params or load_params()
    coord, n_lig, harm = coord_energy(
        sphere.metal_name, ligand, sphere.metal_xyz, params, n_protein=sphere.n_protein
    )
    pocket_axis = -sphere.protein_unit.mean(axis=0) if sphere.n_protein else None
    wmask, wmin = displaced_water_mask(ligand, sphere.water_xyz)
    n_disp = int(wmask.sum())
    displace_xyz = None
    if n_disp and sphere.n_water:
        displace_xyz = np.asarray(sphere.water_xyz, dtype=float).reshape(-1, 3)[wmask]
    steric, contact, n_clash = steric_and_contact(
        protein,
        ligand,
        sphere.metal_index,
        params,
        protein_cache=protein_cache,
        metal_xyz=sphere.metal_xyz,
        pocket_axis=pocket_axis,
        displace_xyz=displace_xyz,
    )
    _idx, dxyz = ligand_donors(ligand)
    if dxyz.size:
        dd = np.linalg.norm(dxyz - sphere.metal_xyz[None, :], axis=1)
        min_d = float(dd.min())
        mean_d = float(dd.mean())
    else:
        min_d, mean_d = 9.0, 9.0
    if skip_geo:
        geo, geo_name = 0.0, "skip"
    else:
        geo, geo_name = geometry_energy(sphere, ligand, params)
    cpen = 0.0
    if pocket_axis is not None:
        axis = np.asarray(pocket_axis, dtype=float)
        nrm = float(np.linalg.norm(axis))
        if nrm > 1.0e-8:
            lmask = ligand.heavy_mask()
            lmean = ligand.xyz[lmask].mean(axis=0) if lmask.any() else ligand.xyz.mean(axis=0)
            pocket_center = sphere.metal_xyz + 3.20 * (axis / nrm)
            cpen = float(np.linalg.norm(lmean - pocket_center))
    from .maps import map_overlap

    overlap = map_overlap(sphere.metal_name, ligand, sphere.metal_xyz)
    # Vina-style chemistry terms (hydrogen bonding, hydrophobic contact, desolvation).
    # Called only when the caller asks for them so the default score is bit-identical.
    chem = {}
    if include_chem:
        from .chemistry import chemistry_terms
        chem = chemistry_terms(protein, ligand, sphere.metal_index)
    rw = params.get("rank_weights") or {
        "coord": 0.35,
        "geo": 1.0,
        "cpen": 0.0,
        "contact": 0.0,
        "map": 0.0,
    }
    total = (
        steric
        + float(rw.get("coord", 0.35)) * coord
        + float(rw.get("geo", 1.0)) * geo
        + float(rw.get("cpen", 0.0)) * cpen
        - float(rw.get("contact", 0.0)) * contact
        - float(rw.get("map", 0.0)) * overlap
    )
    if chem:
        total += (float(rw.get("hbond", 1.0)) * chem["hbond"]
                  + float(rw.get("hydrophobic", 1.0)) * chem["hydrophobic"]
                  + float(rw.get("desolvation", 1.0)) * chem["desolvation"])
    return {
        "total": float(total),
        "coord": float(coord),
        "steric": float(steric),
        "contact": float(contact),
        "geo": float(geo),
        "harm": float(harm),
        "n_coord_lig": float(n_lig),
        "n_clash": float(n_clash),
        "centroid_pen": float(cpen),
        "map_overlap": float(overlap),
        "geometry": geo_name,
        "n_water_displaced": float(n_disp),
        "water_min_dist": float(wmin),
        "n_water": float(sphere.n_water),
        "n_other_metal": float(sphere.n_other_metal),
        "min_donor_dist": float(min_d),
        "mean_donor_dist": float(mean_d),
        **({k: float(v) for k, v in chem.items()} if chem else {}),
    }
