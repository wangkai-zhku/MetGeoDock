#!/usr/bin/env python3
"""Build ML feature tables from crystal poses and optional ChEMBL affinities.

If ChEMBL curated molecules are present, PDB codes from the inventory are
joined to target-level pActivity (best effort). Native crystal poses always
contribute geometry/Vina/GP features. Decoy poses of the same ligand are
written for ranking-model training (binary native vs decoy).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.coordination import detect_coordination  # noqa: E402
from metgeodock.dock import dock_ligand  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402
from metgeodock.scoring import feature_vector, score_pose  # noqa: E402

LOG = logging.getLogger("build_ml_features")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--chembl", type=Path, default=Path("processed/chembl/curated/all_targets_molecules.csv"))
    p.add_argument("--per-metal", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-spin", type=int, default=6)
    p.add_argument("--out-native", type=Path, default=Path("processed/ml/native_features.csv"))
    p.add_argument("--out-rank", type=Path, default=Path("processed/ml/pose_rank_features.csv"))
    p.add_argument("--log", type=Path, default=Path("logs/build_ml_features.log"))
    args = p.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    rng = np.random.default_rng(args.seed)
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1]
    chembl = pd.read_csv(args.chembl) if args.chembl.exists() else pd.DataFrame()
    pdb_to_p = {}
    if len(chembl) and "pdb" in chembl.columns:
        tmp = chembl.dropna(subset=["pdb"]).copy()
        tmp["pdb"] = tmp["pdb"].astype(str).str.upper()
        pdb_to_p = tmp.groupby("pdb")["pactivity"].median().to_dict()

    native_rows = []
    rank_rows = []
    for metal, sub in inv.groupby("metal"):
        n = min(args.per_metal, len(sub))
        pick = sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)]
        for rec in pick.itertuples(index=False):
            try:
                protein = read_pdb(rec.protein_path)
                ligand = read_mol2(rec.ligand_path)
                sphere = detect_coordination(protein, metal=rec.metal)
                native = score_pose(protein, ligand, sphere)
                feats = feature_vector(native, rec.metal, ligand)
                feats.update(
                    {
                        "pdb_id": rec.pdb_id,
                        "metal": rec.metal,
                        "target_id": rec.pdb_id,
                        "is_native": 1,
                        "pactivity": pdb_to_p.get(str(rec.pdb_id).upper(), np.nan),
                    }
                )
                native_rows.append(feats)
                result = dock_ligand(protein, ligand, metal=rec.metal, n_spin=args.n_spin, max_poses=8)
                for pose in result.poses:
                    f = feature_vector(pose.score, rec.metal, pose.ligand)
                    f.update(
                        {
                            "pdb_id": rec.pdb_id,
                            "metal": rec.metal,
                            "target_id": rec.pdb_id,
                            "is_native": 0,
                            "hybrid_total": pose.score.total,
                        }
                    )
                    rank_rows.append(f)
                nfeat = dict(feats)
                nfeat["is_native"] = 1
                nfeat["hybrid_total"] = native.total
                rank_rows.append(nfeat)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s %s: %s", rec.metal, rec.pdb_id, exc)
    native = pd.DataFrame(native_rows)
    rank = pd.DataFrame(rank_rows)
    args.out_native.parent.mkdir(parents=True, exist_ok=True)
    native.to_csv(args.out_native, index=False)
    rank.to_csv(args.out_rank, index=False)
    LOG.info("native_rows=%s rank_rows=%s chembl_mapped=%s", len(native), len(rank), int(native["pactivity"].notna().sum()) if len(native) else 0)


if __name__ == "__main__":
    main()
