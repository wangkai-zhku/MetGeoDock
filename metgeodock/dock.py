"""Geometry-matching pose generation (GM-DockZn sampling + GPDOCK probabilities).

Donor atoms are ranked by GPDOCK peak probability rather than crystal
proximity, then mapped onto vacant ideal vertices (Hungarian assignment).
A short radial refinement pulls coordinating atoms toward histogram-mode
distances. Ranking uses the Vina-free metal-aware score.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .coordination import CoordinationSphere, detect_coordination, ligand_donors, pocket_metal_indices
from .geometry import apply_rt, axis_angle, kabsch, rotation_matrix_from_vectors, unit_vectors
from .io import Molecule, merge_with_waters, strip_waters
from .metals import METALS, histogram_mode_distance, peak_probability
from .scoring import HybridScore, score_pose
from .templates import GM_MOTIFS, IDEAL


@dataclass
class DockPose:
    ligand: Molecule
    score: HybridScore
    motif: str
    donor_indices: tuple[int, ...]
    sphere: CoordinationSphere | None = None


@dataclass
class DockResult:
    sphere: CoordinationSphere
    poses: list[DockPose] = field(default_factory=list)
    spheres: list[CoordinationSphere] = field(default_factory=list)

    @property
    def best(self) -> DockPose | None:
        return self.poses[0] if self.poses else None


def rank_coordination_atoms(ligand: Molecule, metal: str) -> np.ndarray:
    """Order ligand O/N/S atoms by GPDOCK peak probability (coordination prior)."""
    idx, _ = ligand_donors(ligand)
    if idx.size == 0:
        return idx
    scores = np.array([peak_probability(metal, str(ligand.elements[i])) for i in idx], dtype=float)
    return idx[np.argsort(-scores, kind="mergesort")]


def _protein_aligned_template(sphere: CoordinationSphere, geometry: str) -> tuple[np.ndarray, float]:
    """Rotate the ideal polyhedron so protein donors occupy matched vertices."""
    from .geometry import rmsd_or

    ideal = IDEAL[geometry]
    if sphere.n_protein == 0:
        return ideal.copy(), 0.0
    rmsd, perm = rmsd_or(sphere.protein_xyz, sphere.metal_xyz, ideal)
    perm = np.asarray(perm, dtype=int)
    perm = perm[(perm >= 0) & (perm < ideal.shape[0])]
    if perm.size == 0:
        return ideal.copy(), float(rmsd) if np.isfinite(rmsd) else 1.5
    obs = unit_vectors(sphere.protein_xyz[: perm.size], sphere.metal_xyz)
    tgt = ideal[perm]
    if obs.shape[0] >= 2:
        r, _t, _ = kabsch(obs, tgt)
        aligned = (r @ ideal.T).T
    else:
        r = rotation_matrix_from_vectors(ideal[perm[0]], obs[0])
        aligned = (r @ ideal.T).T
    return aligned, rmsd


def _vacant_sites(aligned_ideal: np.ndarray, protein_unit: np.ndarray, n_need: int) -> np.ndarray:
    if n_need <= 0:
        return np.zeros((0, 3))
    if protein_unit.size == 0:
        return aligned_ideal[:n_need]
    used = set()
    for pu in protein_unit:
        d = np.linalg.norm(aligned_ideal - pu, axis=1)
        used.add(int(np.argmin(d)))
    vacant = [i for i in range(aligned_ideal.shape[0]) if i not in used]
    if len(vacant) < n_need:
        order = np.argsort(
            [np.min(np.linalg.norm(aligned_ideal[i] - protein_unit, axis=1)) for i in range(aligned_ideal.shape[0])]
        )
        extra = [int(i) for i in order if i not in used]
        vacant.extend(extra)
    return aligned_ideal[vacant[:n_need]]


def _assign_vacant(donor_xyz: np.ndarray, metal_xyz: np.ndarray, vacant_unit: np.ndarray) -> np.ndarray:
    """Hungarian match of donor unit vectors to vacant coordination vertices."""
    n = min(len(donor_xyz), len(vacant_unit))
    if n == 0:
        return vacant_unit
    d_unit = unit_vectors(donor_xyz[:n], metal_xyz)
    vac = vacant_unit[: max(n, vacant_unit.shape[0])]
    cost = 1.0 - (d_unit @ vac.T)
    _ri, ci = linear_sum_assignment(cost)
    return vac[ci]


def _place_ligand(
    ligand: Molecule,
    donor_idx: np.ndarray,
    metal_xyz: np.ndarray,
    vacant_unit: np.ndarray,
    metal: str,
) -> Molecule | None:
    """Map selected ligand donors onto vacant coordination vertices."""
    if donor_idx.size == 0 or vacant_unit.shape[0] == 0:
        return None
    n = min(donor_idx.size, vacant_unit.shape[0])
    donors = ligand.xyz[donor_idx[:n]]
    els = [str(ligand.elements[i]) for i in donor_idx[:n]]
    vac = _assign_vacant(donors, metal_xyz, vacant_unit)
    target = np.array([metal_xyz + vac[k] * histogram_mode_distance(metal, els[k]) for k in range(n)])
    if n == 1:
        src = donors[0]
        vec_src = src - metal_xyz
        if np.linalg.norm(vec_src) < 1.0e-6:
            return None
        r = rotation_matrix_from_vectors(vec_src, vac[0])
        xyz = (r @ (ligand.xyz - metal_xyz).T).T + metal_xyz
        donor_pos = xyz[donor_idx[0]]
        shift = target[0] - donor_pos
        xyz = xyz + shift
        return ligand.copy_with_xyz(xyz)
    r, t, _rmsd = kabsch(donors, target)
    xyz = apply_rt(ligand.xyz, r, t)
    return ligand.copy_with_xyz(xyz)


def _refine_distances(ligand: Molecule, donor_idx: np.ndarray, metal_xyz: np.ndarray, metal: str) -> Molecule:
    """Translate the ligand so coordinating donors approach GPDOCK mode distances."""
    if donor_idx.size == 0:
        return ligand
    shifts = []
    for i in donor_idx:
        vec = ligand.xyz[i] - metal_xyz
        nrm = float(np.linalg.norm(vec))
        if nrm < 1.0e-6:
            continue
        target = histogram_mode_distance(metal, str(ligand.elements[i]))
        shifts.append(((target - nrm) / nrm) * vec)
    if not shifts:
        return ligand
    shift = np.mean(np.asarray(shifts), axis=0)
    return ligand.copy_with_xyz(ligand.xyz + shift)


def _spin_around_axis(ligand: Molecule, metal_xyz: np.ndarray, axis: np.ndarray, n_spin: int) -> list[Molecule]:
    poses = []
    for k in range(n_spin):
        ang = 2.0 * np.pi * k / n_spin
        r = axis_angle(axis, ang)
        xyz = (r @ (ligand.xyz - metal_xyz).T).T + metal_xyz
        poses.append(ligand.copy_with_xyz(xyz))
    return poses


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(max(n, 1), dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * i + 1.0) / max(n, 1)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    th = golden * i
    return np.column_stack((r * np.cos(th), r * np.sin(th), z))


def _cone_around(axis: np.ndarray, n_ring: int = 6, angles_deg: tuple[float, ...] = (0.0, 8.0, 16.0, 26.0)) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    nrm = float(np.linalg.norm(axis))
    if nrm < 1.0e-8:
        return np.zeros((0, 3))
    axis = axis / nrm
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    out: list[np.ndarray] = []
    for ang in angles_deg:
        if ang <= 1.0e-8:
            out.append(axis)
            continue
        t = np.deg2rad(float(ang))
        ct, st = np.cos(t), np.sin(t)
        for k in range(n_ring):
            phi = 2.0 * np.pi * k / n_ring
            vec = ct * axis + st * (np.cos(phi) * e1 + np.sin(phi) * e2)
            out.append(vec / np.linalg.norm(vec))
    return np.asarray(out, dtype=float)


def _unique_dirs(dirs: np.ndarray, min_cos: float = 0.985) -> np.ndarray:
    keep: list[np.ndarray] = []
    for vec in np.asarray(dirs, dtype=float).reshape(-1, 3):
        nrm = float(np.linalg.norm(vec))
        if nrm < 1.0e-8:
            continue
        vec = vec / nrm
        if keep and max(float(vec @ k) for k in keep) > min_cos:
            continue
        keep.append(vec)
    return np.asarray(keep, dtype=float) if keep else np.zeros((0, 3))


def _open_hemisphere(protein_unit: np.ndarray, n: int = 32, max_cos: float = 0.55) -> np.ndarray:
    pts = _fibonacci_sphere(n)
    if protein_unit.size == 0:
        return pts
    dots = pts @ protein_unit.T
    kept = pts[dots.max(axis=1) < max_cos]
    return kept if kept.size else pts[: max(6, n // 4)]


def _occupied_xyz(sphere: CoordinationSphere) -> np.ndarray:
    parts = []
    if sphere.n_protein:
        parts.append(sphere.protein_xyz)
    if sphere.n_water:
        parts.append(sphere.water_xyz)
    if sphere.n_other_metal:
        parts.append(sphere.other_metal_xyz)
    if not parts:
        return np.zeros((0, 3))
    return np.vstack(parts)


def _occupied_unit(sphere: CoordinationSphere) -> np.ndarray:
    xyz = _occupied_xyz(sphere)
    if xyz.size == 0:
        return np.zeros((0, 3))
    return unit_vectors(xyz, sphere.metal_xyz)


def _sphere_without_waters(sphere: CoordinationSphere) -> CoordinationSphere:
    return replace(
        sphere,
        water_idx=np.zeros(0, dtype=int),
        water_xyz=np.zeros((0, 3), dtype=float),
    )


def _sphere_with_water_xyz(sphere: CoordinationSphere, water_xyz: np.ndarray) -> CoordinationSphere:
    xyz = np.asarray(water_xyz, dtype=float).reshape(-1, 3)
    return replace(sphere, water_idx=np.arange(len(xyz), dtype=int), water_xyz=xyz)


def _open_vacant_vertices(sphere: CoordinationSphere, preferred: tuple[str, ...], k: int = 2) -> np.ndarray:
    """Most open polyhedron vertices (away from protein/other-metal donors)."""
    dry = _sphere_without_waters(sphere)
    occupied = _occupied_unit(dry)
    geo = preferred[0] if preferred else "oh"
    aligned, _ = _protein_aligned_template(dry, geo)
    n_vert = IDEAL[geo].shape[0]
    n_vac = max(n_vert - int(occupied.shape[0]), 1)
    vacant = _vacant_sites(aligned, occupied, n_vac)
    if vacant.size == 0:
        vacant = _open_hemisphere(occupied, n=max(8, k * 4), max_cos=0.50)
    if vacant.size == 0:
        return np.zeros((0, 3))
    vacant = np.asarray(vacant, dtype=float).reshape(-1, 3)
    if occupied.size:
        openness = -((vacant @ occupied.T).max(axis=1))
    else:
        openness = np.zeros(vacant.shape[0])
    order = np.argsort(-openness, kind="mergesort")
    return vacant[order[: max(1, int(k))]]


def coord_water_hypotheses(
    sphere: CoordinationSphere,
    n_max: int,
    use_crystal: bool,
) -> list[tuple[int, CoordinationSphere]]:
    """Occupancy hypotheses: 0, 1 or 2 waters on vacant vertices.

    Waters occupy polyhedron sites so the ligand is sampled on remaining
    vacant directions (mono- vs bidentate changes). This is not bulk pocket
    hydration. Crystal first-shell O atoms are used only if ``use_crystal``
    and at most ``n_max`` closest waters are kept.
    """
    n_max = max(0, int(n_max))
    dry = _sphere_without_waters(sphere)
    hyps: list[tuple[int, CoordinationSphere]] = [(0, dry)]
    if n_max <= 0:
        return hyps
    if use_crystal and sphere.n_water:
        xyz = np.asarray(sphere.water_xyz, dtype=float).reshape(-1, 3)
        dist = np.linalg.norm(xyz - sphere.metal_xyz[None, :], axis=1)
        order = np.argsort(dist, kind="mergesort")
        for n_w in range(1, min(n_max, int(order.size)) + 1):
            hyps.append((n_w, _sphere_with_water_xyz(dry, xyz[order[:n_w]])))
        return hyps
    verts = _open_vacant_vertices(dry, METALS[sphere.metal_name].geometries, k=min(2, n_max + 1))
    if verts.shape[0] == 0:
        return hyps
    d0 = float(histogram_mode_distance(sphere.metal_name, "O"))
    placed = sphere.metal_xyz[None, :] + d0 * verts
    if n_max >= 1:
        hyps.append((1, _sphere_with_water_xyz(dry, placed[:1])))
        if placed.shape[0] >= 2:
            hyps.append((1, _sphere_with_water_xyz(dry, placed[1:2])))
    if n_max >= 2 and placed.shape[0] >= 2:
        hyps.append((2, _sphere_with_water_xyz(dry, placed[:2])))
    return hyps


def _collect_rays(sphere: CoordinationSphere, preferred: tuple[str, ...]) -> np.ndarray:
    """Vacant GM vertices plus a cone/hemisphere covering distorted sites."""
    chunks: list[np.ndarray] = []
    occupied = _occupied_unit(sphere)
    n_occ = int(occupied.shape[0])
    geos = list(dict.fromkeys(list(preferred) + ["td", "oh", "tbp", "spy", "sp"]))
    for geometry in geos:
        n_vert = IDEAL[geometry].shape[0]
        if n_occ > n_vert:
            continue
        aligned, _ = _protein_aligned_template(sphere, geometry)
        n_vac = max(n_vert - max(n_occ, 0), 1)
        vacant = _vacant_sites(aligned, occupied, n_vac)
        if vacant.size == 0:
            continue
        chunks.append(vacant)
        for vtx in vacant:
            chunks.append(_cone_around(vtx))
    open_h = _open_hemisphere(occupied, n=36, max_cos=0.55)
    if open_h.size:
        chunks.append(open_h)
    if not chunks:
        return _fibonacci_sphere(24)
    return _unique_dirs(np.vstack(chunks), min_cos=0.982)


def _vacant_groups(vacant: np.ndarray, l_need: int) -> list[np.ndarray]:
    """Monodentate: try every vacant vertex. Polydentate: one Hungarian set."""
    if l_need <= 1:
        return [vacant[i : i + 1] for i in range(vacant.shape[0])]
    return [vacant]


def _pose_sort_key(pose: DockPose) -> tuple[float, float, float, float, float]:
    """Rank score first (MetGeo total unless reranked); pocket terms break ties."""
    t = pose.score.terms
    return (
        float(t.get("rank_score", pose.score.total)),
        float(t.get("centroid_pen", 0.0)),
        float(t.get("geo", pose.score.geo)),
        -float(t.get("contact", 0.0)),
        -float(t.get("map_overlap", 0.0)),
    )


def _hybrid_from_ff(ff: dict, sphere: CoordinationSphere) -> HybridScore:
    return HybridScore(
        total=float(ff["total"]),
        vina=0.0,
        slef=0.0,
        geo=float(ff["geo"]),
        gp=float(ff["coord"]),
        clash=float(ff["steric"]),
        rmsd_or=float(ff["geo"]),
        n_coord_lig=int(ff["n_coord_lig"]),
        n_coord_pro=int(sphere.n_protein),
        dist=float(ff["harm"]),
        cn=0.0,
        terms=dict(ff),
    )


def _apply_rerank(poses: list[DockPose], mode: str, model_path: str | None = None) -> list[DockPose]:
    mode = (mode or "metgeo").lower()
    if mode in {"metgeo", "metalgeo", "hybrid"}:
        for pose in poses:
            pose.score.terms["rank_score"] = float(pose.score.total)
            pose.score.terms["rerank"] = mode
        poses.sort(key=_pose_sort_key)
        return poses
    if mode == "posenet":
        from .nnscore import posenet_rank_value

        for pose in poses:
            pose.score.terms["rank_score"] = float(posenet_rank_value(pose.score, pose.ligand, model_path=model_path))
            pose.score.terms["rerank"] = mode
        poses.sort(key=_pose_sort_key)
        return poses
    from .posescore import rank_value

    use_nn = mode in {"posescore-nn", "posescore_nn", "posenet-out"}
    use_ranker = mode in {"ranker", "geolist", "best", "ranker-blend"}
    raw = []
    for pose in poses:
        metal = pose.sphere.metal_name if pose.sphere is not None else None
        val = float(
            rank_value(
                pose.score,
                pose.ligand,
                model_path=model_path,
                use_nn=use_nn,
                metal=metal,
                use_ranker=use_ranker,
            )
        )
        raw.append(val)
        pose.score.terms["ranker_raw"] = val
        pose.score.terms["rerank"] = mode
    if mode == "ranker-blend" and len(poses) >= 2:
        import json as _json
        from pathlib import Path as _Path

        alpha = 0.25
        cfg = _Path(__file__).resolve().parents[1] / "processed" / "ml" / "ranker_blend.json"
        if cfg.exists():
            try:
                alpha = float(_json.loads(cfg.read_text(encoding="utf-8")).get("alpha", alpha))
            except Exception:
                alpha = 0.25
        h = np.array([float(p.score.total) for p in poses], dtype=float)
        r = np.asarray(raw, dtype=float)
        zh = (h - h.mean()) / (float(h.std()) + 1.0e-6)
        zr = (r - r.mean()) / (float(r.std()) + 1.0e-6)
        blend = zh + float(alpha) * zr
        for pose, s in zip(poses, blend):
            pose.score.terms["rank_score"] = float(s)
            pose.score.terms["blend_alpha"] = float(alpha)
    else:
        for pose, val in zip(poses, raw):
            pose.score.terms["rank_score"] = float(val)
    poses.sort(key=_pose_sort_key)
    return poses


def _append_scored(
    scored: list[DockPose],
    protein: Molecule,
    pose: Molecule,
    sphere: CoordinationSphere,
    combo: tuple[int, ...],
    motif: str,
    weights: dict[str, float] | None,
    include_vina: bool,
    clash_skip: float,
) -> None:
    sc = score_pose(protein, pose, sphere, weights=weights, include_vina=include_vina)
    if sc.clash > clash_skip:
        return
    scored.append(DockPose(ligand=pose, score=sc, motif=motif, donor_indices=combo))


def _dock_one_metal(
    protein: Molecule,
    ligand: Molecule,
    metal: str | None = None,
    metal_index: int | None = None,
    n_spin: int = 18,
    max_poses: int = 40,
    max_donor_combos: int = 16,
    weights: dict[str, float] | None = None,
    include_vina: bool = False,
    clash_skip: float = 500.0,
    coord_water: int = 2,
    use_crystal_water: bool = False,
    include_chem: bool = False,
    chem_weights: dict[str, float] | None = None,
    sphere_override: CoordinationSphere | None = None,
) -> DockResult:
    """Sample poses onto vacant vertices of one metal ion.

    ``coord_water`` enumerates 0/1/2 coordinating waters on vacant sites so
    the ligand denticity can change. Pocket HOH is not kept as bulk solvent.
    """
    from .forcefield import ff_energy, protein_steric_cache, load_params

    sphere = sphere_override or detect_coordination(protein, metal=metal, metal_index=metal_index)
    if sphere_override is None and int(coord_water) > 0:
        merged: list[DockPose] = []
        last = sphere
        for n_w, sph in coord_water_hypotheses(sphere, int(coord_water), use_crystal_water):
            spin = int(n_spin) if n_w == 0 else max(4, int(n_spin) // 2)
            one = _dock_one_metal(
                protein,
                ligand,
                metal=metal,
                metal_index=metal_index,
                n_spin=spin,
                max_poses=max_poses,
                max_donor_combos=max_donor_combos,
                weights=weights,
                include_vina=include_vina,
                clash_skip=clash_skip,
                coord_water=0,
                use_crystal_water=False,
                sphere_override=sph,
                include_chem=include_chem,
                chem_weights=chem_weights,
            )
            last = sph
            for pose in one.poses:
                pose.sphere = sph
                pose.motif = f"{pose.motif}_w{n_w}"
                pose.score.terms["n_coord_water"] = float(n_w)
                pose.score.terms["n_water"] = float(n_w)
                merged.append(pose)
        merged.sort(key=_pose_sort_key)
        uniq = _unique_by_centroid(merged, 0.35, max_poses)
        return DockResult(sphere=uniq[0].sphere if uniq else last, poses=uniq)
    metal_name = sphere.metal_name
    preferred = METALS[metal_name].geometries
    ranked = rank_coordination_atoms(ligand, metal_name)
    if ranked.size == 0:
        raise ValueError("Ligand has no O/N/S donor atoms")
    result = DockResult(sphere=sphere)
    scored: list[DockPose] = []
    pcache = protein_steric_cache(protein, sphere.metal_index)

    _chem_p = None
    if include_chem:
        _base = load_params()
        rw = dict(_base.get("rank_weights") or {})
        if chem_weights:
            rw.update(chem_weights)
        _chem_p = {**_base, "rank_weights": rw}

    def _ff_pose(mol: Molecule, combo: tuple[int, ...], motif: str, skip_geo: bool) -> DockPose | None:
        ff = ff_energy(protein, mol, sphere, skip_geo=skip_geo, protein_cache=pcache,
                       include_chem=include_chem, params=_chem_p)
        if float(ff["steric"]) > clash_skip:
            return None
        return DockPose(
            ligand=mol,
            score=_hybrid_from_ff(ff, sphere),
            motif=motif,
            donor_indices=combo,
        )

    rays = _collect_rays(sphere, preferred)
    o_all = [int(i) for i in ranked if str(ligand.elements[int(i)]).upper().startswith("O")]
    ns_all = [int(i) for i in ranked if str(ligand.elements[int(i)]).upper()[0] in "NS"]
    donors_mono = list(dict.fromkeys(o_all[:16] + ns_all[:8] + [int(i) for i in ranked[:12]]))
    for ray in rays:
        vac = ray.reshape(1, 3)
        for donor_i in donors_mono:
            combo = (donor_i,)
            placed = _place_ligand(ligand, np.asarray(combo), sphere.metal_xyz, vac, metal_name)
            if placed is None:
                continue
            placed = _refine_distances(placed, np.asarray(combo), sphere.metal_xyz, metal_name)
            for pose in _spin_around_axis(placed, sphere.metal_xyz, ray, n_spin):
                item = _ff_pose(pose, combo, "cone_1", skip_geo=True)
                if item is not None:
                    scored.append(item)

    occupied = _occupied_unit(sphere)
    n_occ = int(occupied.shape[0])
    motifs = [m for m in GM_MOTIFS if m[0] in preferred or m[0] in ("td", "oh", "sp", "tbp", "spy")]
    o_donors = [int(i) for i in ranked if str(ligand.elements[int(i)]).upper().startswith("O")]
    pair_pool = list(dict.fromkeys(o_donors[:8] + donors_mono[:6]))
    pair_combos = list(combinations(pair_pool, 2))[:max_donor_combos]
    for a, b in zip(ranked.tolist(), ranked.tolist()[1:]):
        pair = (int(a), int(b))
        if pair not in pair_combos and (pair[1], pair[0]) not in pair_combos:
            pair_combos.append(pair)
        if len(pair_combos) >= max(max_donor_combos, 24):
            break
    for geometry, p_need, l_need in motifs:
        if l_need != 2:
            continue
        n_vert = IDEAL[geometry].shape[0]
        if n_occ > n_vert:
            continue
        aligned, _ = _protein_aligned_template(sphere, geometry)
        vacant = _vacant_sites(aligned, occupied, 2)
        if vacant.shape[0] < 2:
            continue
        for combo in pair_combos:
            placed = _place_ligand(ligand, np.asarray(combo), sphere.metal_xyz, vacant, metal_name)
            if placed is None:
                continue
            placed = _refine_distances(placed, np.asarray(combo), sphere.metal_xyz, metal_name)
            axis = vacant[0]
            for pose in _spin_around_axis(placed, sphere.metal_xyz, axis, max(6, n_spin // 3)):
                item = _ff_pose(pose, tuple(combo), f"{geometry}_{p_need}_{l_need}", skip_geo=True)
                if item is not None:
                    scored.append(item)

    scored.sort(key=_pose_sort_key)
    seed: list[DockPose] = []
    cents = []
    for pose in scored:
        c = pose.ligand.xyz.mean(axis=0)
        if cents and min(np.linalg.norm(c - x) for x in cents) < 0.50:
            continue
        seed.append(pose)
        cents.append(c)
        if len(seed) >= min(12, max(max_poses, 8)):
            break
    for pose in seed:
        if not pose.donor_indices:
            continue
        axis = pose.ligand.xyz[pose.donor_indices[0]] - sphere.metal_xyz
        if float(np.linalg.norm(axis)) < 1.0e-6:
            continue
        for fine in _spin_around_axis(pose.ligand, sphere.metal_xyz, axis, 12):
            item = _ff_pose(fine, pose.donor_indices, pose.motif + "_fine", skip_geo=True)
            if item is not None:
                scored.append(item)
    scored.sort(key=_pose_sort_key)
    by_donor: dict[tuple[int, ...], list[DockPose]] = {}
    for pose in scored:
        by_donor.setdefault(pose.donor_indices, []).append(pose)
    pool: list[DockPose] = []
    seen_ids: set[int] = set()
    for lst in by_donor.values():
        lst.sort(key=_pose_sort_key)
        for pose in lst[:12]:
            key = id(pose)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            pool.append(pose)
    by_contact = sorted(scored, key=lambda p: -float(p.score.terms.get("contact", 0.0)))
    for pose in scored[:80] + by_contact[:30]:
        key = id(pose)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        pool.append(pose)
    rescored: list[DockPose] = []
    for pose in pool:
        item = _ff_pose(pose.ligand, pose.donor_indices, pose.motif, skip_geo=False)
        if item is not None:
            rescored.append(item)
    scored = rescored or scored
    scored.sort(key=_pose_sort_key)
    uniq: list[DockPose] = []
    cents = []
    for pose in scored:
        c = pose.ligand.xyz.mean(axis=0)
        if cents and min(np.linalg.norm(c - x) for x in cents) < 0.35:
            continue
        uniq.append(pose)
        cents.append(c)
        if len(uniq) >= max_poses:
            break
    result.poses = uniq
    return result


def _annotate_pose(pose: DockPose, sphere: CoordinationSphere) -> DockPose:
    pose.sphere = sphere
    pose.score.terms["dock_metal_index"] = float(sphere.metal_index)
    pose.score.terms["n_water"] = float(sphere.n_water)
    pose.score.terms["n_other_metal"] = float(sphere.n_other_metal)
    return pose


def _unique_by_centroid(poses: list[DockPose], min_A: float, max_keep: int) -> list[DockPose]:
    uniq: list[DockPose] = []
    cents: list[np.ndarray] = []
    for pose in poses:
        c = pose.ligand.xyz.mean(axis=0)
        if cents and min(float(np.linalg.norm(c - x)) for x in cents) < min_A:
            continue
        uniq.append(pose)
        cents.append(c)
        if len(uniq) >= max_keep:
            break
    return uniq


def dock_ligand(
    protein: Molecule,
    ligand: Molecule,
    metal: str | None = None,
    n_spin: int = 18,
    max_poses: int = 40,
    max_donor_combos: int = 16,
    weights: dict[str, float] | None = None,
    include_vina: bool = False,
    clash_skip: float = 500.0,
    keep_water: bool = False,
    multi_metal: bool = True,
    metal_index: int | None = None,
    water: Molecule | None = None,
    max_pocket_metals: int = 4,
    semi_flex: bool = False,
    rerank: str = "metgeo",
    flex_seeds: int = 6,
    model_path: str | None = None,
    coord_water: int = 2,
    donor_chem_scale: float = 0.0,
    include_chem: bool = False,
    chem_weights: dict[str, float] | None = None,
) -> DockResult:
    """Sample ligand poses along vacant-site cones, then rank by MetGeo FF.

    Coordinating waters (0–``coord_water``, default 2) occupy polyhedron
    vertices so the ligand denticity can change. This is not bulk pocket
    hydration: ``keep_water`` only optionally uses 1–2 crystal first-shell
    oxygens as those occupancy hypotheses. Extra pocket metals occupy vertices
    and are not stripped.

    ``semi_flex`` rotates donor-adjacent torsions on the top seeds, keeping
    copies whose coordination-map overlap does not fall. ``rerank`` is
    ``metgeo`` (default MetGeo total), ``posescore`` (same-ligand ranker,
    water-aware), or ``posenet``.
    """
    from .donor_chem import set_donor_chem as _set_donor_chem

    _set_donor_chem(donor_chem_scale)

    from .flex import with_bonds

    ligand = with_bonds(ligand)
    protein_use = protein
    if water is not None:
        protein_use = merge_with_waters(protein_use, water)
    if not keep_water:
        protein_use = strip_waters(protein_use)

    if metal_index is not None:
        indices = [int(metal_index)]
    elif multi_metal:
        pocket = pocket_metal_indices(protein_use, ligand)
        indices = [int(i) for i in pocket]
        if len(indices) > max(1, int(max_pocket_metals)):
            heavy = ligand.xyz[ligand.heavy_mask()] if ligand.heavy_mask().any() else ligand.xyz
            d = [float(np.linalg.norm(heavy - protein_use.xyz[i], axis=1).min()) for i in indices]
            order = np.argsort(np.asarray(d))
            indices = [indices[int(j)] for j in order[: int(max_pocket_metals)]]
    else:
        sphere0 = detect_coordination(protein_use, metal=metal)
        indices = [int(sphere0.metal_index)]
    if not indices:
        raise ValueError("No metal ion found in protein coordinates")

    all_poses: list[DockPose] = []
    spheres: list[CoordinationSphere] = []
    for mi in indices:
        one = _dock_one_metal(
            protein_use,
            ligand,
            metal=metal,
            metal_index=mi,
            n_spin=n_spin,
            max_poses=max_poses,
            max_donor_combos=max_donor_combos,
            weights=weights,
            include_vina=include_vina,
            clash_skip=clash_skip,
            coord_water=int(coord_water),
            use_crystal_water=bool(keep_water),
            include_chem=include_chem,
            chem_weights=chem_weights,
        )
        spheres.append(one.sphere)
        for pose in one.poses:
            all_poses.append(_annotate_pose(pose, one.sphere))

    all_poses.sort(key=_pose_sort_key)
    by_metal: dict[int, list[DockPose]] = {}
    for pose in all_poses:
        mi = int(pose.score.terms.get("dock_metal_index", -1))
        by_metal.setdefault(mi, []).append(pose)
    merged: list[DockPose] = []
    for lst in by_metal.values():
        merged.extend(_unique_by_centroid(lst, 0.35, max_poses))
    merged.sort(key=_pose_sort_key)
    uniq = merged[:max_poses]
    if semi_flex and uniq:
        from .flex import flex_variants
        from .forcefield import ff_energy, protein_steric_cache, load_params

        caches: dict[int, tuple] = {}
        extra: list[DockPose] = []
        for pose in uniq[: max(1, int(flex_seeds))]:
            sph = pose.sphere or (spheres[0] if spheres else None)
            if sph is None:
                continue
            mi = int(sph.metal_index)
            if mi not in caches:
                caches[mi] = protein_steric_cache(protein_use, mi)
            variants = flex_variants(pose.ligand, pose.donor_indices, sph.metal_name, sph.metal_xyz)
            for mol in variants:
                if include_chem and chem_weights:
                    _p = load_params()
                    _p = {**_p, "rank_weights": {**(_p.get("rank_weights") or {}), **chem_weights}}
                else:
                    _p = None
                ff = ff_energy(protein_use, mol, sph, skip_geo=False,
                               protein_cache=caches[mi], include_chem=include_chem,
                               params=_p)
                if float(ff["steric"]) > clash_skip:
                    continue
                item = DockPose(
                    ligand=mol,
                    score=_hybrid_from_ff(ff, sph),
                    motif=pose.motif + "_flex",
                    donor_indices=pose.donor_indices,
                    sphere=sph,
                )
                item.score.terms["semi_flex"] = 1.0
                extra.append(_annotate_pose(item, sph))
        if extra:
            uniq = _unique_by_centroid(sorted(uniq + extra, key=_pose_sort_key), 0.35, max_poses)
    uniq = _apply_rerank(uniq, rerank, model_path=model_path)
    uniq = uniq[:max_poses]
    if uniq and uniq[0].sphere is not None:
        sphere = uniq[0].sphere
    elif spheres:
        sphere = spheres[0]
    else:
        sphere = detect_coordination(protein_use, metal=metal)
    return DockResult(sphere=sphere, poses=uniq, spheres=spheres)


def redock_rmsd(pred: Molecule, crystal: Molecule) -> float:
    """Heavy-atom RMSD after pairing by atom order (crystal files share topology)."""
    a = pred.xyz[pred.heavy_mask()]
    b = crystal.xyz[crystal.heavy_mask()]
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.sqrt(((a[:n] - b[:n]) ** 2).sum() / n))
