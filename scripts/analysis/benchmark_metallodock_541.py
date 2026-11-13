#!/usr/bin/env python3
"""Redock the reconstructed MetalloDock 2022 time-split set.

Inventory from scripts/data/build_metallodock_2022_set.py. Reports top-1/3/5
vs literature 78.6% (Zhang 2026). Checkpoint CSV for resume. Ranked pose
tables go to processed/dock/metallodock_541/<rec_id>/ranks.csv.
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
from metgeodock.io import crop_pocket, load_molecule, read_pdb, write_pdb  # noqa: E402
from metgeodock.ranking import take_top, write_rank_table  # noqa: E402

LOG = logging.getLogger("benchmark_metallodock_541")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/metallodock_541/inventory.csv"))
    p.add_argument("--n-spin", type=int, default=16)
    p.add_argument("--max-poses", type=int, default=40)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out-csv", type=Path, default=Path("results/metallodock_541_redock.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/metallodock_541_redock.json"))
    p.add_argument("--rank-dir", type=Path, default=Path("processed/dock/metallodock_541"))
    p.add_argument("--pocket-radius", type=float, default=15.0)
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_metallodock_541.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    inv = pd.read_csv(args.inventory)
    done = set()
    rows: list[dict] = []
    if args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        rows = prev.to_dict(orient="records")
        done = {str(r.get("rec_id")) for r in rows}
        LOG.info("resume %s", len(done))
    n_total = len(inv)
    for rec in inv.itertuples(index=False):
        rec_id = str(rec.rec_id)
        if rec_id in done:
            continue
        LOG.info("[%s/%s] %s %s", len(rows) + 1, n_total, rec.metal, rec.pdb_id)
        try:
            protein = read_pdb(rec.protein_path)
            ligand = load_molecule(rec.ligand_path)
            metals = protein.metal_indices(metal=str(rec.metal))
            if metals.size:
                pocket_path = Path(rec.protein_path).with_name(Path(rec.protein_path).stem + "_pocket.pdb")
                if pocket_path.exists():
                    protein = read_pdb(pocket_path)
                else:
                    protein = crop_pocket(protein, protein.xyz[int(metals[0])], radius_A=args.pocket_radius)
                    write_pdb(protein, pocket_path, record="ATOM")
            result = dock_ligand(
                protein, ligand, metal=str(rec.metal), n_spin=args.n_spin, max_poses=args.max_poses
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s %s", rec_id, exc)
            rows.append({"rec_id": rec_id, "pdb_id": rec.pdb_id, "metal": rec.metal, "error": str(exc)})
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            continue
        if not result.poses:
            rows.append({"rec_id": rec_id, "pdb_id": rec.pdb_id, "metal": rec.metal, "error": "no_pose"})
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            continue
        rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
        top1 = float(rmsds[0])
        top3 = float(min(rmsds[:3]))
        top5 = float(min(rmsds[:5]))
        best = float(np.nanmin(rmsds))
        table = take_top(
            pd.DataFrame(
                [
                    {"score": pose.score.total, "motif": pose.motif, "rmsd": rmsd}
                    for pose, rmsd in zip(result.poses, rmsds)
                ]
            ),
            args.top_n,
        )
        rank_path = args.rank_dir / rec_id / "ranks.csv"
        write_rank_table(table, rank_path)
        rows.append(
            {
                "rec_id": rec_id,
                "pdb_id": rec.pdb_id,
                "metal": rec.metal,
                "n_poses": len(result.poses),
                "top1_rmsd": top1,
                "top3_rmsd": top3,
                "top5_rmsd": top5,
                "best_sampled_rmsd": best,
                "success_top1": int(top1 <= args.threshold) if np.isfinite(top1) else 0,
                "success_top3": int(top3 <= args.threshold) if np.isfinite(top3) else 0,
                "success_top5": int(top5 <= args.threshold) if np.isfinite(top5) else 0,
                "success_sampled": int(best <= args.threshold) if np.isfinite(best) else 0,
                "best_score": result.best.score.total if result.best else float("nan"),
                "rank_csv": str(rank_path),
                "error": "",
            }
        )
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    frame = pd.DataFrame(rows)
    ok = frame[frame["error"].fillna("") == ""]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "RCSB reconstruction of MetalloDock 2022 time-split (SI has protocol, not PDB list)",
        "literature_MetalloDock": 0.786,
        "n_attempted": int(len(frame)),
        "n_ok": int(len(ok)),
        "n_unique_pdb": int(ok["pdb_id"].nunique()) if len(ok) else 0,
        "threshold_angstrom": args.threshold,
        "metgeodock_success": {
            "top1": float(ok["success_top1"].mean()) if len(ok) else None,
            "top3": float(ok["success_top3"].mean()) if len(ok) else None,
            "top5": float(ok["success_top5"].mean()) if len(ok) else None,
            "sampled": float(ok["success_sampled"].mean()) if len(ok) else None,
        },
        "by_metal": (
            {
                str(m): {"n": int(len(g)), "top1": float(g["success_top1"].mean()), "top3": float(g["success_top3"].mean())}
                for m, g in ok.groupby("metal")
            }
            if len(ok)
            else {}
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary %s", json.dumps(summary["metgeodock_success"]))


if __name__ == "__main__":
    main()
