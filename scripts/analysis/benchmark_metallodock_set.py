#!/usr/bin/env python3
"""Redock the public MetalloDock 10-complex set with MetGeoDock.

Published MetalloDock time-split success is 78.6% (RMSD < 2 A) on 541
complexes. GitHub only ships 10 PDB IDs; this script reports MetGeoDock
success on that public subset and does not invent numbers for the 541.
Protein = *_pro_pocket.pdb; ligand = *_ligand.pdb (fallback *_ligand.sdf).
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

from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import load_molecule, read_pdb  # noqa: E402

LOG = logging.getLogger("benchmark_metallodock_set")


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
    p.add_argument("--out-csv", type=Path, default=Path("results/metallodock_public_redock.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/metallodock_public_redock.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_metallodock_set.log"))
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
        LOG.info("dock %s", pdb)
        try:
            protein = read_pdb(pro)
            ligand = load_molecule(lig_path)
            result = dock_ligand(protein, ligand, n_spin=args.n_spin, max_poses=args.max_poses)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s: %s", pdb, exc)
            rows.append({"pdb_id": pdb, "error": str(exc)})
            continue
        if not result.poses:
            rows.append({"pdb_id": pdb, "error": "no_pose", "metal": result.sphere.metal_name})
            continue
        rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
        top1 = float(rmsds[0])
        top5 = float(min(rmsds[:5]))
        best = float(np.nanmin(rmsds))
        rows.append(
            {
                "pdb_id": pdb,
                "metal": result.sphere.metal_name,
                "n_protein_coord": result.sphere.n_protein,
                "n_poses": len(result.poses),
                "top1_rmsd": top1,
                "top5_rmsd": top5,
                "best_sampled_rmsd": best,
                "success_top1": int(top1 <= args.threshold) if np.isfinite(top1) else 0,
                "success_top5": int(top5 <= args.threshold) if np.isfinite(top5) else 0,
                "success_sampled": int(best <= args.threshold) if np.isfinite(best) else 0,
                "best_score": result.best.score.total if result.best else float("nan"),
                "error": "",
            }
        )
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    ok = frame[frame["error"].fillna("") == ""]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "MetalloDock GitHub public 10-complex subset (Zhang et al. JACS 2026)",
        "published_literature": {
            "citation": "Zhang et al., J. Am. Chem. Soc. 2026, 148, 3086-3101",
            "MetalloDock": 0.786,
            "SurfDock": 0.754,
            "AlphaFold3": 0.689,
            "Glide_SP": 0.484,
        },
        "published_metallodock_success_541": 0.786,
        "n_attempted": int(len(frame)),
        "n_ok": int(len(ok)),
        "threshold_angstrom": args.threshold,
        "metgeodock_success": {
            "top1": float(ok["success_top1"].mean()) if len(ok) else None,
            "top5": float(ok["success_top5"].mean()) if len(ok) else None,
            "sampled": float(ok["success_sampled"].mean()) if len(ok) else None,
        },
        "by_metal": (
            ok.groupby("metal")["success_top1"].mean().astype(float).to_dict() if len(ok) and "metal" in ok else {}
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(summary["metgeodock_success"]))


if __name__ == "__main__":
    main()
