#!/usr/bin/env python3
"""Dump MetGeo FF terms vs RMSD for public MetalloDock complexes.

Used to retune ranking weights. Does not mutate raw data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import load_molecule, read_pdb  # noqa: E402

LOG = logging.getLogger("diagnose_rank_terms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("processed/metallodock_ref/dataset/metallo_complex"))
    p.add_argument("--ids", nargs="+", default=["8jao", "8bdb", "8qvu", "8hz7", "8a7k", "8elk"])
    p.add_argument("--n-spin", type=int, default=12)
    p.add_argument("--max-poses", type=int, default=20)
    p.add_argument("--out-csv", type=Path, default=Path("results/diagnose_rank_terms.csv"))
    p.add_argument("--log", type=Path, default=Path("logs/diagnose_rank_terms.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    rows = []
    for pdb in args.ids:
        folder = args.root / pdb
        protein = read_pdb(folder / f"{pdb}_pro_pocket.pdb")
        lig_path = folder / f"{pdb}_ligand.pdb"
        if not lig_path.exists():
            lig_path = folder / f"{pdb}_ligand.sdf"
        ligand = load_molecule(lig_path)
        result = dock_ligand(protein, ligand, n_spin=args.n_spin, max_poses=args.max_poses)
        LOG.info("%s n_poses=%s", pdb, len(result.poses))
        for rank, pose in enumerate(result.poses, start=1):
            rmsd = redock_rmsd(pose.ligand, ligand)
            t = pose.score.terms
            rows.append(
                {
                    "pdb_id": pdb,
                    "rank": rank,
                    "rmsd": rmsd,
                    "total": pose.score.total,
                    "steric": t.get("steric", pose.score.clash),
                    "coord": t.get("coord", pose.score.gp),
                    "geo": t.get("geo", pose.score.geo),
                    "contact": t.get("contact", 0.0),
                    "harm": t.get("harm", pose.score.dist),
                    "n_coord_lig": t.get("n_coord_lig", pose.score.n_coord_lig),
                    "motif": pose.motif,
                }
            )
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))


if __name__ == "__main__":
    main()
