#!/usr/bin/env python3
"""Redock the RCSB time-split test complexes with MetGeoDock.

Split is processed/splits/time_split.csv (last 15% by initial release date).
This is the same *style* of split as MetalloDock (Zhang 2026), not the same
541 PDB IDs. Literature 78.6% is cited for comparison; MetGeoDock numbers
are computed here. Native crystal poses are NOT mixed into ranking.
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

LOG = logging.getLogger("benchmark_timesplit_redock")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--n-spin", type=int, default=16)
    p.add_argument("--max-poses", type=int, default=40)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0, help="optional cap for smoke tests")
    p.add_argument("--per-metal", type=int, default=0, help="if >0, stratified sample this many per metal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=Path("results/timesplit_redock.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/timesplit_redock.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_timesplit_redock.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    inv = pd.read_csv(args.split_csv)
    inv = inv[(inv["time_split"] == "test") & (inv["complete"] == 1)].copy()
    if args.per_metal:
        rng = np.random.default_rng(args.seed)
        parts = []
        for _metal, sub in inv.groupby("metal"):
            n = min(args.per_metal, len(sub))
            parts.append(sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)])
        inv = pd.concat(parts, axis=0).reset_index(drop=True)
    if args.limit:
        inv = inv.iloc[: args.limit].copy()
    LOG.info("n_test=%s", len(inv))
    rows = []
    for i, rec in inv.iterrows():
        LOG.info("[%s/%s] %s %s", len(rows) + 1, len(inv), rec.metal, rec.pdb_id)
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
            LOG.warning("fail %s %s: %s", rec.metal, rec.pdb_id, exc)
            rows.append({"metal": rec.metal, "pdb_id": rec.pdb_id, "error": str(exc)})
            continue
        if not result.poses:
            rows.append(
                {
                    "metal": rec.metal,
                    "pdb_id": rec.pdb_id,
                    "error": "no_pose",
                    "n_protein_coord": result.sphere.n_protein,
                }
            )
            continue
        rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
        top1 = float(rmsds[0])
        top3 = float(min(rmsds[:3]))
        top5 = float(min(rmsds[:5]))
        best = float(np.nanmin(rmsds))
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
                "error": "",
            }
        )
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    ok = frame[frame["error"].fillna("") == ""]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "RCSB time-split test on GPDOCK inventory (not MetalloDock SI 541)",
        "published_literature": {
            "citation": "Zhang et al., J. Am. Chem. Soc. 2026, 148, 3086-3101",
            "MetalloDock": 0.786,
            "SurfDock": 0.754,
            "AlphaFold3": 0.689,
            "Glide_SP": 0.484,
        },
        "n_attempted": int(len(frame)),
        "n_ok": int(len(ok)),
        "threshold_angstrom": args.threshold,
        "n_spin": args.n_spin,
        "max_poses": args.max_poses,
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
