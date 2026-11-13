#!/usr/bin/env python3
"""Redock the FULL time-split test set (all complete complexes).

GitHub MetalloDock only ships 10 PDBs; ACS SI (541 IDs) is bot-blocked.
This runs every complete test-split row in processed/splits/time_split.csv
(currently 552 complexes, same order of magnitude as the paper 541).
Pass --pdb-list to restrict to an SI-extracted PDB ID file when available.

Writes a checkpoint CSV so the job can resume. Does not mutate data/.
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
from metgeodock.io import read_mol2, read_pdb  # noqa: E402
from metgeodock.ranking import take_top, write_rank_table  # noqa: E402

LOG = logging.getLogger("benchmark_timesplit_full")


def _key(rec: pd.Series) -> str:
    return f"{rec.metal}_{str(rec.pdb_id).upper()}_{int(rec.rec_id)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--pdb-list", type=Path, default=None, help="optional SI PDB IDs (one per line or csv col pdb_id)")
    p.add_argument("--n-spin", type=int, default=16)
    p.add_argument("--max-poses", type=int, default=40)
    p.add_argument("--top-n", type=int, default=40, help="ranked poses stored per complex")
    p.add_argument("--write-poses", action="store_true")
    p.add_argument("--pose-dir", type=Path, default=Path("processed/dock/timesplit_full"))
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out-csv", type=Path, default=Path("results/timesplit_full_redock.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/timesplit_full_redock.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_timesplit_full.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    inv = pd.read_csv(args.split_csv)
    inv = inv[(inv["time_split"] == "test") & (inv["complete"] == 1)].copy()
    if args.pdb_list and args.pdb_list.exists():
        raw = args.pdb_list.read_text(encoding="utf-8")
        if args.pdb_list.suffix.lower() == ".csv":
            ids = set(pd.read_csv(args.pdb_list)["pdb_id"].astype(str).str.upper())
        else:
            ids = {ln.strip().upper() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")}
        inv = inv[inv["pdb_id"].astype(str).str.upper().isin(ids)].copy()
        LOG.info("restricted to SI list n_ids=%s n_complexes=%s", len(ids), len(inv))
    done = set()
    rows: list[dict] = []
    if args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        rows = prev.to_dict(orient="records")
        for rec in rows:
            done.add(f"{rec.get('metal')}_{str(rec.get('pdb_id')).upper()}_{int(rec.get('rec_id', 0))}")
        LOG.info("resume %s already done", len(done))
    n_total = len(inv)
    for i, rec in inv.iterrows():
        key = _key(rec)
        if key in done:
            continue
        LOG.info("[%s/%s] %s %s rec=%s", len(rows) + 1, n_total, rec.metal, rec.pdb_id, rec.rec_id)
        try:
            protein = read_pdb(rec.protein_path)
            ligand = read_mol2(rec.ligand_path)
            result = dock_ligand(
                protein,
                ligand,
                metal=str(rec.metal),
                n_spin=args.n_spin,
                max_poses=args.max_poses,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s: %s", key, exc)
            rows.append({"metal": rec.metal, "pdb_id": rec.pdb_id, "rec_id": rec.rec_id, "error": str(exc)})
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            continue
        if not result.poses:
            rows.append(
                {
                    "metal": rec.metal,
                    "pdb_id": rec.pdb_id,
                    "rec_id": rec.rec_id,
                    "error": "no_pose",
                    "n_protein_coord": result.sphere.n_protein,
                }
            )
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            continue
        rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
        top1 = float(rmsds[0])
        top3 = float(min(rmsds[:3]))
        top5 = float(min(rmsds[:5]))
        best = float(np.nanmin(rmsds))
        pose_table = take_top(
            pd.DataFrame(
                [
                    {
                        "score": pose.score.total,
                        "geo": pose.score.geo,
                        "clash": pose.score.clash,
                        "motif": pose.motif,
                        "rmsd": rmsd,
                    }
                    for pose, rmsd in zip(result.poses, rmsds)
                ]
            ),
            args.top_n,
        )
        rank_path = args.pose_dir / key / "ranks.csv"
        write_rank_table(pose_table, rank_path)
        if args.write_poses:
            from metgeodock.io import write_pdb

            pdir = args.pose_dir / key / "poses"
            pdir.mkdir(parents=True, exist_ok=True)
            for j, pose in enumerate(result.poses[: len(pose_table)], start=1):
                write_pdb(pose.ligand, pdir / f"pose_{j:04d}.pdb")
        rows.append(
            {
                "metal": rec.metal,
                "pdb_id": rec.pdb_id,
                "rec_id": rec.rec_id,
                "n_protein_coord": result.sphere.n_protein,
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
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    ok = frame[frame["error"].fillna("") == ""]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "full time-split test on GPDOCK inventory; not the unpublished MetalloDock SI 541 unless --pdb-list",
        "n_attempted": int(len(frame)),
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
        "by_metal": (
            {
                str(m): {
                    "n": int(len(g)),
                    "top1": float(g["success_top1"].mean()),
                    "top3": float(g["success_top3"].mean()),
                    "sampled": float(g["success_sampled"].mean()),
                }
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
