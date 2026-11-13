"""Batch virtual screening for one metalloprotein and a ligand library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dock import dock_ligand
from .io import Molecule, iter_sdf_molecules, load_molecule, write_pdb
from .scoring import feature_vector


@dataclass
class ScreenHit:
    ligand_id: str
    score: float
    gp: float
    geo: float
    n_coord_lig: int
    motif: str
    pose_path: str | None = None
    pactivity: float | None = None


def _ligand_files(library: str | Path | list[Path]) -> list[Path]:
    if isinstance(library, list):
        return [Path(p) for p in library]
    path = Path(library)
    if path.is_file():
        return [path]
    files = []
    for pat in ("*.mol2", "*.MOL2", "*.sdf", "*.SDF", "*.pdb", "*.PDB"):
        files.extend(sorted(path.glob(pat)))
    return files


def iter_library(library: str | Path | list[Path]):
    """Yield ``(ligand_id, Molecule)`` for every compound in a library."""
    for lig_path in _ligand_files(library):
        if lig_path.suffix.lower() == ".sdf":
            yield from iter_sdf_molecules(lig_path)
            continue
        yield lig_path.stem, load_molecule(lig_path)


def screen_library(
    protein: Molecule,
    library: str | Path | list[Path],
    metal: str | None = None,
    scorer: str = "hybrid",
    model_path: str | Path | None = None,
    n_spin: int = 8,
    max_poses: int = 5,
    out_dir: str | Path | None = None,
    top_n: int = 0,
    write_poses: bool = False,
    keep_water: bool = False,
    multi_metal: bool = True,
    semi_flex: bool = False,
    rerank: str = "metgeo",
    coord_water: int = 2,
    donor_chem_scale: float = 0.0,
    thermo: bool = False,
) -> pd.DataFrame:
    """Dock each ligand and rank by hybrid or ML score.

    Every library molecule is scored. The returned table is sorted
    best-to-worst (lower ``score`` is better). ``top_n`` of 100 or 1000
    keeps only that many rows; ``top_n <= 0`` (default) writes all.

    ``scorer='hybrid'`` (alias ``metgeo``) uses the Vina-free metal-aware
    score (lower is better). ``scorer='ml'`` / ``coordnet`` store
    ``score = -pactivity`` so the same ascending sort still means best first.
    """
    if scorer in {"hybrid", "metgeo", "metalgeo"}:
        scorer = "hybrid"
    if scorer == "ml" and model_path:
        from .ml import load_model          # lazy: scikit-learn/joblib are optional
        model = load_model(model_path)
    else:
        model = None
    coordnet_path = model_path
    if scorer == "coordnet" and coordnet_path is None:
        from .nnscore import DEFAULT_COORDNET

        coordnet_path = DEFAULT_COORDNET if DEFAULT_COORDNET.exists() else None
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    hits: list[dict] = []
    for ligand_id, ligand in iter_library(library):
        try:
            result = dock_ligand(
                protein,
                ligand,
                metal=metal,
                n_spin=n_spin,
                max_poses=max_poses,
                keep_water=keep_water,
                multi_metal=multi_metal,
                semi_flex=semi_flex,
                rerank=rerank,
                coord_water=int(coord_water),
                donor_chem_scale=float(donor_chem_scale),
            )
        except Exception as exc:  # noqa: BLE001 — keep screening going
            hits.append(
                {
                    "ligand_id": ligand_id,
                    "score": float("nan"),
                    "error": str(exc),
                }
            )
            continue
        if not result.poses:
            hits.append({"ligand_id": ligand_id, "score": float("nan"), "error": "no_pose"})
            continue
        best = result.best
        assert best is not None
        pactivity = None
        rank_score = best.score.total
        if model is not None:
            feats = feature_vector(best.score, result.sphere.metal_name, best.ligand)
            from .ml import predict_pactivity   # lazy
            pactivity = predict_pactivity(model, feats)
            rank_score = -pactivity
        if scorer == "coordnet":
            from .nnscore import coordnet_score_ligand

            p_act = coordnet_score_ligand(
                result.sphere.metal_name,
                ligand=best.ligand,
                metal_xyz=result.sphere.metal_xyz,
                model_path=coordnet_path,
            )
            pactivity = p_act
            rank_score = -float(p_act) if np.isfinite(p_act) else float("nan")
        pose_path = None
        if write_poses and out:
            pose_path = str(out / f"{ligand_id}_best.pdb")
            write_pdb(best.ligand, pose_path)
        n_rot = 0
        try:
            from rdkit import Chem as _Chem
            from rdkit.Chem import rdMolDescriptors as _rd
            _m = _Chem.MolFromPDBBlock(_pdb_block(best.ligand), sanitize=False, removeHs=True)
            if _m is not None:
                _m.UpdatePropertyCache(strict=False)
                try:
                    _Chem.SanitizeMol(_m, _Chem.SANITIZE_ALL ^ _Chem.SANITIZE_PROPERTIES)
                except Exception:
                    pass
                n_rot = int(_rd.CalcNumRotatableBonds(_m))
        except Exception:
            n_rot = 0
        th = {}
        if thermo:
            try:
                from .thermo import thermodynamic_terms
                from .chemistry import chemistry_terms
                chem = chemistry_terms(protein, best.ligand, result.sphere.metal_index)
                th = thermodynamic_terms(protein, best.ligand, result.sphere.metal_index,
                                         n_rotatable=n_rot,
                                         n_hbond=int(chem["n_hbond"]),
                                         n_desolv=chem["desolvation"])
            except Exception:
                th = {}
        if thermo and th:
            # rank-normalised combination: geometry + hydrophobic burial + conformational
            # adaptability, all with literature constants (see metgeodock/thermo.py)
            A_REF = 600.0
            rank_score = (float(best.score.total)
                          + th.get("g_buried", 0.0)
                          - 0.6 * n_rot * min(1.0, th.get("buried_area", 0.0) / A_REF) / 100.0)
        hits.append(
            {
                "ligand_id": ligand_id,
                "score": rank_score,
                "n_rotatable": n_rot,
                "buried_area": th.get("buried_area", np.nan),
                "g_buried": th.get("g_buried", np.nan),
                "g_total": th.get("g_total", np.nan),
                "hybrid_total": best.score.total,
                "slef": best.score.slef,
                "gp_nll": best.score.gp,
                "geo_rmsd_or": best.score.geo,
                "clash": best.score.clash,
                "dist_harm": best.score.dist,
                "cn_pen": best.score.cn,
                "n_coord_lig": best.score.n_coord_lig,
                "n_coord_pro": best.score.n_coord_pro,
                "motif": best.motif,
                "metal": result.sphere.metal_name,
                "pactivity": pactivity,
                "pose_path": pose_path,
                "error": "",
            }
        )
    frame = pd.DataFrame(hits)
    from .ranking import take_top

    if "score" in frame.columns:
        frame = take_top(frame, top_n=top_n, score_col="score")
    else:
        frame = frame.reset_index(drop=True)
    return frame


def _pdb_block(mol: "Molecule") -> str:
    """Minimal PDB text for a Molecule, used to rebuild an RDKit graph when no SMILES exists."""
    lines = []
    for i, (xyz, el) in enumerate(zip(mol.xyz, mol.elements), start=1):
        e = str(el)[:2]
        lines.append(
            f"HETATM{i:5d} {e:>2s}   LIG A   1    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {e:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)
