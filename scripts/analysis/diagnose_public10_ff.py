#!/usr/bin/env python3
"""Per-complex redock diagnostic on the public MetalloDock 10-set.

Reports top-1 / top-5 / best-of-returned RMSD and wall time. Does not
invent literature numbers; MetalloDock 78.6% is cited for comparison only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import load_molecule, read_pdb  # noqa: E402

LOG = logging.getLogger("diagnose_public10_ff")


def _ligand_path(folder: Path, pdb: str) -> Path | None:
    for name in (f"{pdb}_ligand.pdb", f"{pdb}_ligand.sdf"):
        cand = folder / name
        if cand.exists():
            return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("processed/metallodock_ref/dataset/metallo_complex"))
    p.add_argument("--csv", type=Path, default=Path("processed/metallodock_ref/dataset/test_pdbid.csv"))
    p.add_argument("--n-spin", type=int, default=16)
    p.add_argument("--max-poses", type=int, default=40)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out-csv", type=Path, default=Path("results/diagnose_public10_ff.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/diagnose_public10_ff.json"))
    p.add_argument("--log", type=Path, default=Path("logs/diagnose_public10_ff.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    ids = pd.read_csv(args.csv)["pdb_id"].astype(str).str.lower().tolist()
    rows = []
    for pdb in ids:
        folder = args.root / pdb
        pro = folder / f"{pdb}_pro_pocket.pdb"
        lig_path = _ligand_path(folder, pdb)
        if not pro.exists() or lig_path is None:
            rows.append({"pdb_id": pdb, "error": "missing_files"})
            continue
        protein = read_pdb(pro)
        ligand = load_molecule(lig_path)
        t0 = time.perf_counter()
        result = dock_ligand(protein, ligand, n_spin=args.n_spin, max_poses=args.max_poses)
        elapsed = time.perf_counter() - t0
        if not result.poses:
            rows.append({"pdb_id": pdb, "error": "no_pose", "seconds": elapsed})
            continue
        rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
        top1 = float(rmsds[0])
        top3 = float(min(rmsds[:3]))
        top5 = float(min(rmsds[:5]))
        best = float(np.nanmin(rmsds))
        LOG.info(
            "%s metal=%s n_pose=%s top1=%.3f top3=%.3f top5=%.3f best=%.3f t=%.1fs",
            pdb,
            result.sphere.metal_name,
            len(result.poses),
            top1,
            top3,
            top5,
            best,
            elapsed,
        )
        rows.append(
            {
                "pdb_id": pdb,
                "metal": result.sphere.metal_name,
                "n_protein_coord": result.sphere.n_protein,
                "n_poses": len(result.poses),
                "top1_rmsd": top1,
                "top3_rmsd": top3,
                "top5_rmsd": top5,
                "best_sampled_rmsd": best,
                "success_top1": int(top1 <= args.threshold),
                "success_top3": int(top3 <= args.threshold),
                "success_top5": int(top5 <= args.threshold),
                "success_sampled": int(best <= args.threshold),
                "best_score": result.best.score.total if result.best else float("nan"),
                "seconds": elapsed,
                "error": "",
            }
        )
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    ok = frame[frame["error"].fillna("") == ""]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_ok": int(len(ok)),
        "threshold_angstrom": args.threshold,
        "n_spin": args.n_spin,
        "max_poses": args.max_poses,
        "literature_MetalloDock": 0.786,
        "metgeodock_success": {
            "top1": float(ok["success_top1"].mean()) if len(ok) else None,
            "top3": float(ok["success_top3"].mean()) if len(ok) else None,
            "top5": float(ok["success_top5"].mean()) if len(ok) else None,
            "sampled": float(ok["success_sampled"].mean()) if len(ok) else None,
        },
        "median_seconds": float(ok["seconds"].median()) if len(ok) else None,
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary %s", json.dumps(summary))


if __name__ == "__main__":
    main()
