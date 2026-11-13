#!/usr/bin/env python3
"""Two virtual-screening schemes for MetGeoDock.

Scheme A (MetGeo, semi-empirical): dock the crystal ligand (active) and
ChEMBL non-chelator decoys into the same pocket; rank by MetGeo FF
(lower better).

Scheme B (CoordNet, ML): ligand-level MLP on warhead probes + Morgan bits,
no docking. Evaluated by GroupKFold OOF scores already computed in
train_coordnet.py, plus a held-out per-metal screen against donor-count
and random baselines.

Decoys for A are ChEMBL molecules with zero metal-warhead probes so the
test asks whether coordination physics enriches true metalloligands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.dock import dock_ligand  # noqa: E402
from metgeodock.fingerprint import probe_counts  # noqa: E402
from metgeodock.io import Molecule, read_mol2, read_pdb  # noqa: E402
from metgeodock.nnscore import coordnet_score_ligand  # noqa: E402

LOG = logging.getLogger("benchmark_two_vs")


def auroc(labels: np.ndarray, scores_higher_better: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores_higher_better, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(tpr, fpr))


def enrichment(labels: np.ndarray, scores_higher_better: np.ndarray, frac: float) -> float:
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-np.asarray(scores_higher_better, dtype=float))
    k = max(1, int(np.ceil(frac * len(y))))
    hit = float(y[order][:k].mean())
    base = float(y.mean())
    return float(hit / base) if base > 0 else float("nan")


def smiles_to_ligand(smiles: str, seed: int = 0) -> Molecule | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) % 10000
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=50)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    els = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    dummy = np.array(["X"] * len(els))
    zeros = np.zeros(len(els))
    seqs = np.zeros(len(els), dtype=int)
    return Molecule(
        name="embed",
        xyz=xyz,
        elements=els,
        names=els,
        resnames=dummy,
        resseqs=seqs,
        charges=zeros,
        atom_types=els,
    )


def n_warheads(smiles: str) -> int:
    c = probe_counts(smiles)
    return int(sum(c[k] for k in ("hydroxamate", "carboxylate", "imidazole", "thiol", "phosphate", "catechol")))


def mol2_smiles(path: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(path), sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:  # noqa: BLE001
        try:
            return Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001
            return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--chembl", type=Path, default=Path("processed/chembl/curated/all_targets_molecules.csv"))
    p.add_argument("--coordnet", type=Path, default=Path("processed/ml/coordnet.joblib"))
    p.add_argument("--coordnet-et", type=Path, default=Path("processed/ml/coordnet_et.joblib"))
    p.add_argument("--coordnet-report", type=Path, default=Path("processed/ml/coordnet_report.json"))
    p.add_argument("--receptors-per-metal", type=int, default=1)
    p.add_argument("--decoys", type=int, default=8)
    p.add_argument("--n-spin", type=int, default=8)
    p.add_argument("--max-poses", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=Path("results/two_vs_benchmark.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/two_vs_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_two_vs.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    rng = np.random.default_rng(args.seed)
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1].copy()
    if args.split_csv.exists():
        split = pd.read_csv(args.split_csv)
        test_ids = set(split.loc[split["time_split"] == "test", "pdb_id"].astype(str).str.upper())
        inv_a = inv[inv["pdb_id"].astype(str).str.upper().isin(test_ids)].copy()
        if len(inv_a) < 10:
            inv_a = inv
    else:
        inv_a = inv
    chembl = pd.read_csv(args.chembl)
    chembl = chembl[chembl["inconsistent"] == False].dropna(subset=["canonical_smiles"])  # noqa: E712
    decoy_pool = []
    for smi in chembl["canonical_smiles"].astype(str).tolist():
        if n_warheads(smi) == 0:
            decoy_pool.append(smi)
        if len(decoy_pool) >= 4000:
            break
    LOG.info("non-chelator decoy pool %s", len(decoy_pool))

    receptors = []
    for metal, sub in inv_a.groupby("metal"):
        n = min(args.receptors_per_metal, len(sub))
        pick = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        receptors.extend(sub.loc[pick].to_dict(orient="records"))

    rows = []
    for rec in receptors:
        LOG.info("A dock %s %s", rec["metal"], rec["pdb_id"])
        try:
            protein = read_pdb(rec["protein_path"])
            native = read_mol2(rec["ligand_path"])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("skip receptor %s", exc)
            continue
        labels = [1]
        scores_ff = []
        scores_nn = []
        scores_et = []
        scores_donor = []
        native_smi = mol2_smiles(rec["ligand_path"])
        try:
            result = dock_ligand(
                protein, native, metal=rec["metal"], n_spin=args.n_spin, max_poses=args.max_poses
            )
            if not result.poses:
                continue
            scores_ff.append(-float(result.best.score.total))
            nn = coordnet_score_ligand(
                rec["metal"],
                smiles=native_smi,
                ligand=result.best.ligand,
                metal_xyz=result.sphere.metal_xyz,
                model_path=args.coordnet,
            )
            scores_nn.append(0.0 if not np.isfinite(nn) else nn)
            et = coordnet_score_ligand(
                rec["metal"],
                smiles=native_smi,
                ligand=result.best.ligand,
                metal_xyz=result.sphere.metal_xyz,
                model_path=args.coordnet_et,
            )
            scores_et.append(0.0 if not np.isfinite(et) else et)
            scores_donor.append(float(native.donor_indices().size))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("native dock fail %s", exc)
            continue
        chosen = rng.choice(np.arange(len(decoy_pool)), size=min(args.decoys, len(decoy_pool)), replace=False)
        for j, di in enumerate(chosen):
            lig = smiles_to_ligand(decoy_pool[int(di)], seed=args.seed + j)
            if lig is None:
                continue
            try:
                result = dock_ligand(
                    protein, lig, metal=rec["metal"], n_spin=args.n_spin, max_poses=args.max_poses
                )
            except Exception:  # noqa: BLE001
                continue
            if not result.poses:
                continue
            labels.append(0)
            scores_ff.append(-float(result.best.score.total))
            nn = coordnet_score_ligand(
                rec["metal"],
                smiles=decoy_pool[int(di)],
                ligand=result.best.ligand,
                metal_xyz=result.sphere.metal_xyz,
                model_path=args.coordnet,
            )
            scores_nn.append(0.0 if not np.isfinite(nn) else nn)
            et = coordnet_score_ligand(
                rec["metal"],
                smiles=decoy_pool[int(di)],
                ligand=result.best.ligand,
                metal_xyz=result.sphere.metal_xyz,
                model_path=args.coordnet_et,
            )
            scores_et.append(0.0 if not np.isfinite(et) else et)
            scores_donor.append(float(lig.donor_indices().size))
        y = np.asarray(labels, dtype=int)
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        rows.append(
            {
                "metal": rec["metal"],
                "pdb_id": rec["pdb_id"],
                "n_library": int(len(y)),
                "metalgeo_auroc": auroc(y, np.asarray(scores_ff)),
                "coordnet_auroc": auroc(y, np.asarray(scores_nn)),
                "coordnet_et_auroc": auroc(y, np.asarray(scores_et)),
                "donor_auroc": auroc(y, np.asarray(scores_donor)),
                "random_auroc": 0.5,
                "metalgeo_ef5": enrichment(y, np.asarray(scores_ff), 0.05),
                "coordnet_ef5": enrichment(y, np.asarray(scores_nn), 0.05),
                "coordnet_et_ef5": enrichment(y, np.asarray(scores_et), 0.05),
            }
        )

    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    coordnet_cv = {}
    if args.coordnet_report.exists():
        coordnet_cv = json.loads(args.coordnet_report.read_text(encoding="utf-8")).get("models", {})
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scheme_a": "MetGeo dock + FF rank vs non-chelator ChEMBL decoys",
        "scheme_b": "CoordNet MLP/ExtraTrees on warhead probes + Morgan + map overlap; GroupKFold by target",
        "n_receptors": int(len(frame)),
        "structure_based": {
            "metalgeo_mean_auroc": float(frame["metalgeo_auroc"].mean()) if len(frame) else None,
            "coordnet_mean_auroc": float(frame["coordnet_auroc"].mean()) if len(frame) else None,
            "coordnet_et_mean_auroc": float(frame["coordnet_et_auroc"].mean()) if len(frame) else None,
            "donor_mean_auroc": float(frame["donor_auroc"].mean()) if len(frame) else None,
            "random_auroc": 0.5,
            "metalgeo_beats_random": (
                bool(float(frame["metalgeo_auroc"].mean()) > 0.5) if len(frame) else None
            ),
            "coordnet_beats_random": (
                bool(float(frame["coordnet_auroc"].mean()) > 0.5) if len(frame) else None
            ),
            "coordnet_et_beats_random": (
                bool(float(frame["coordnet_et_auroc"].mean()) > 0.5) if len(frame) else None
            ),
            "metalgeo_beats_donor": (
                bool(float(frame["metalgeo_auroc"].mean()) > float(frame["donor_auroc"].mean()))
                if len(frame)
                else None
            ),
        },
        "ligand_based_cv": coordnet_cv,
        "ligand_based": {
            "coordnet_mlp_oof_auroc": (coordnet_cv.get("coordnet_mlp") or {}).get("oof_auroc"),
            "extratrees_oof_auroc": (coordnet_cv.get("extratrees") or {}).get("oof_auroc"),
            "donor_logreg_oof_auroc": (coordnet_cv.get("donor_logreg") or {}).get("oof_auroc"),
            "random_auroc": 0.5,
            "mlp_beats_random": (
                bool(float((coordnet_cv.get("coordnet_mlp") or {}).get("oof_auroc", 0)) > 0.5)
                if coordnet_cv.get("coordnet_mlp")
                else None
            ),
            "et_beats_random": (
                bool(float((coordnet_cv.get("extratrees") or {}).get("oof_auroc", 0)) > 0.5)
                if coordnet_cv.get("extratrees")
                else None
            ),
            "mlp_beats_donor": (
                bool(
                    float((coordnet_cv.get("coordnet_mlp") or {}).get("oof_auroc", 0))
                    > float((coordnet_cv.get("donor_logreg") or {}).get("oof_auroc", 0))
                )
                if coordnet_cv.get("coordnet_mlp") and coordnet_cv.get("donor_logreg")
                else None
            ),
            "et_beats_donor": (
                bool(
                    float((coordnet_cv.get("extratrees") or {}).get("oof_auroc", 0))
                    > float((coordnet_cv.get("donor_logreg") or {}).get("oof_auroc", 0))
                )
                if coordnet_cv.get("extratrees") and coordnet_cv.get("donor_logreg")
                else None
            ),
        },
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary %s", json.dumps(summary["structure_based"]))


if __name__ == "__main__":
    main()
