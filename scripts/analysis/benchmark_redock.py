#!/usr/bin/env python3
"""Redock representative GPDOCK crystal complexes with MetGeoDock.

Compares ranking functions on the same sampled poses:
  metgeo (Vina-free total), geo, gp, dist, clash
Optional --include-vina adds the Vina empirical total as a diagnostic only.
Success is heavy-atom RMSD <= threshold versus the crystal ligand
(atom order from the MOL2 file). A random-pose baseline is included so
the geometry matcher can fail honestly.
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

from metgeodock.coordination import detect_coordination  # noqa: E402
from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402
from metgeodock.scoring import score_pose  # noqa: E402

LOG = logging.getLogger("benchmark_redock")


def stratified_sample(inventory: pd.DataFrame, per_metal: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for metal, sub in inventory.groupby("metal"):
        n = min(per_metal, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(sub.loc[idx])
    return pd.concat(parts, axis=0).reset_index(drop=True)


def rank_success(poses, crystal, key, threshold: float) -> dict:
    ordered = sorted(poses, key=key)
    rmsds = [redock_rmsd(p.ligand, crystal) for p in ordered]
    best_any = float(np.nanmin(rmsds)) if rmsds else float("nan")
    top1 = rmsds[0] if rmsds else float("nan")
    top5 = min(rmsds[:5]) if rmsds else float("nan")
    return {
        "best_sampled_rmsd": best_any,
        "top1_rmsd": top1,
        "top5_rmsd": top5,
        "success_top1": int(top1 <= threshold) if np.isfinite(top1) else 0,
        "success_top5": int(top5 <= threshold) if np.isfinite(top5) else 0,
        "success_sampled": int(best_any <= threshold) if np.isfinite(best_any) else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--per-metal", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-spin", type=int, default=8)
    p.add_argument("--max-poses", type=int, default=12)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out-csv", type=Path, default=Path("results/redock_benchmark.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/redock_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_redock.log"))
    p.add_argument("--include-vina", action="store_true")
    p.add_argument("--split-csv", type=Path, default=None, help="optional time_split.csv; if set, keep time_split==test")
    args = p.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1].copy()
    if args.split_csv and args.split_csv.exists():
        split = pd.read_csv(args.split_csv)
        if "time_split" in split.columns:
            test_ids = set(split.loc[split["time_split"] == "test", "pdb_id"].astype(str).str.upper())
            inv = inv[inv["pdb_id"].astype(str).str.upper().isin(test_ids)].copy()
            LOG.info("time-split test complexes remaining %s", len(inv))
    sample = stratified_sample(inv, args.per_metal, args.seed)
    rows = []
    for i, rec in sample.iterrows():
        LOG.info("[%s/%s] %s %s", i + 1, len(sample), rec.metal, rec.pdb_id)
        try:
            protein = read_pdb(rec.protein_path)
            ligand = read_mol2(rec.ligand_path)
            sphere = detect_coordination(protein, metal=rec.metal)
            native = score_pose(protein, ligand, sphere, include_vina=args.include_vina)
            result = dock_ligand(
                protein,
                ligand,
                metal=rec.metal,
                n_spin=args.n_spin,
                max_poses=args.max_poses,
                include_vina=args.include_vina,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s %s: %s", rec.metal, rec.pdb_id, exc)
            rows.append({"metal": rec.metal, "pdb_id": rec.pdb_id, "error": str(exc)})
            continue
        crystal_rmsd_self = redock_rmsd(ligand, ligand)
        from metgeodock.dock import DockPose

        native_pose = DockPose(ligand=ligand, score=native, motif="native", donor_indices=tuple())
        mixed = result.poses + [native_pose]
        methods = {
            "metgeo": lambda pose: pose.score.total,
            "geo": lambda pose: pose.score.geo,
            "gp": lambda pose: pose.score.gp,
            "dist": lambda pose: pose.score.dist,
            "clash": lambda pose: pose.score.clash,
        }
        if args.include_vina:
            methods["vina"] = lambda pose: pose.score.vina
        row = {
            "metal": rec.metal,
            "pdb_id": rec.pdb_id,
            "rec_id": rec.rec_id,
            "n_protein_coord": sphere.n_protein,
            "n_water": sphere.n_water,
            "n_poses": len(result.poses),
            "native_metgeo": native.total,
            "native_geo": native.geo,
            "native_gp": native.gp,
            "native_dist": native.dist,
            "self_rmsd": crystal_rmsd_self,
            "error": "",
        }
        for name, key in methods.items():
            stats = rank_success(result.poses, ligand, key, args.threshold)
            for k, v in stats.items():
                row[f"{name}_{k}"] = v
            mixed_stats = rank_success(mixed, ligand, key, args.threshold)
            row[f"{name}_native_in_top1"] = mixed_stats["success_top1"]
            row[f"{name}_native_in_top5"] = mixed_stats["success_top5"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)

    ok = frame[frame["error"].fillna("") == ""]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_attempted": int(len(frame)),
        "n_ok": int(len(ok)),
        "per_metal": int(args.per_metal),
        "seed": args.seed,
        "threshold_angstrom": args.threshold,
        "success_rates": {},
    }
    methods_present = [m for m in ("metgeo", "geo", "gp", "dist", "clash", "vina") if f"{m}_success_top1" in ok.columns]
    for method in methods_present:
        col1 = f"{method}_success_top1"
        col5 = f"{method}_success_top5"
        cols = f"{method}_success_sampled"
        if col1 in ok.columns and len(ok):
            summary["success_rates"][method] = {
                "top1": float(ok[col1].mean()),
                "top5": float(ok[col5].mean()),
                "sampled": float(ok[cols].mean()),
                "median_top1_rmsd": float(ok[f"{method}_top1_rmsd"].median()),
            }
    if "metal" in ok.columns and len(ok):
        by = {}
        for metal, sub in ok.groupby("metal"):
            by[str(metal)] = {
                "n": int(len(sub)),
                "metgeo_top1": float(sub["metgeo_success_top1"].mean()) if "metgeo_success_top1" in sub else None,
            }
        summary["by_metal"] = by
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary %s", json.dumps(summary["success_rates"]))


if __name__ == "__main__":
    main()
