#!/usr/bin/env python3
"""Public-10 redock: MetGeo vs PoseScore rerank vs semi-flex.

Does not mutate data/. Reports top-1 / top-3 / sampled RMSD<=2 A. Literature
MetalloDock 78.6% is cited only as a published number, not recomputed here.
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

LOG = logging.getLogger("benchmark_posescore_flex")

PROTOCOLS = (
    {"name": "metgeo", "semi_flex": False, "rerank": "metgeo"},
    {"name": "posescore", "semi_flex": False, "rerank": "posescore"},
    {"name": "posescore_nn", "semi_flex": False, "rerank": "posescore-nn"},
    {"name": "semiflex", "semi_flex": True, "rerank": "metgeo"},
    {"name": "posescore_semiflex", "semi_flex": True, "rerank": "posescore"},
)


def _ligand_path(folder: Path, pdb: str) -> Path | None:
    for name in (f"{pdb}_ligand.pdb", f"{pdb}_ligand.sdf", f"{pdb}_ligand.mol2"):
        cand = folder / name
        if cand.exists():
            return cand
    return None


def _rates(rmsds: list[float], threshold: float) -> dict:
    arr = np.asarray(rmsds, dtype=float)
    if arr.size == 0:
        return {"top1": float("nan"), "top3": float("nan"), "top5": float("nan"), "sampled": float("nan")}
    return {
        "top1": float(arr[0] <= threshold),
        "top3": float(np.nanmin(arr[:3]) <= threshold),
        "top5": float(np.nanmin(arr[:5]) <= threshold),
        "sampled": float(np.nanmin(arr) <= threshold),
        "top1_rmsd": float(arr[0]),
        "best_rmsd": float(np.nanmin(arr)),
        "n_pose": int(arr.size),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("processed/metallodock_ref/dataset/metallo_complex"))
    p.add_argument("--csv", type=Path, default=Path("processed/metallodock_ref/dataset/test_pdbid.csv"))
    p.add_argument("--n-spin", type=int, default=16)
    p.add_argument("--max-poses", type=int, default=40)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out-csv", type=Path, default=Path("results/posescore_flex_public10.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/posescore_flex_public10.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_posescore_flex.log"))
    p.add_argument("--model", type=Path, default=Path("processed/ml/posescore.pt"))
    p.add_argument(
        "--protocols",
        default="all",
        help="comma list of protocol names, or all",
    )
    args = p.parse_args()
    wanted = {x.strip() for x in str(args.protocols).split(",") if x.strip()}
    protocols = PROTOCOLS if "all" in wanted else tuple(item for item in PROTOCOLS if item["name"] in wanted)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    ids = pd.read_csv(args.csv)["pdb_id"].astype(str).str.lower().tolist()
    model_path = str(args.model) if args.model.exists() else None
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
        for proto in protocols:
            t0 = time.perf_counter()
            try:
                result = dock_ligand(
                    protein,
                    ligand,
                    n_spin=args.n_spin,
                    max_poses=args.max_poses,
                    semi_flex=bool(proto["semi_flex"]),
                    rerank=str(proto["rerank"]),
                    model_path=model_path,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "pdb_id": pdb,
                        "protocol": proto["name"],
                        "error": str(exc),
                        "seconds": time.perf_counter() - t0,
                    }
                )
                continue
            elapsed = time.perf_counter() - t0
            if not result.poses:
                rows.append({"pdb_id": pdb, "protocol": proto["name"], "error": "no_pose", "seconds": elapsed})
                continue
            rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
            rec = _rates(rmsds, args.threshold)
            rec.update(
                {
                    "pdb_id": pdb,
                    "protocol": proto["name"],
                    "metal": result.sphere.metal_name,
                    "n_water": float(result.sphere.n_water),
                    "seconds": elapsed,
                    "error": "",
                    "has_posescore_ckpt": bool(model_path),
                }
            )
            LOG.info(
                "%s %s top1=%.3f top3=%s sampled=%s t=%.1fs",
                pdb,
                proto["name"],
                rec["top1_rmsd"],
                rec["top3"],
                rec["sampled"],
                elapsed,
            )
            rows.append(rec)
    frame = pd.DataFrame(rows)
    if args.out_csv.exists() and "all" not in wanted:
        prev = pd.read_csv(args.out_csv)
        keep = prev[~prev.get("protocol", pd.Series(dtype=str)).isin({x["name"] for x in protocols})]
        frame = pd.concat([keep, frame], ignore_index=True)
    frame.to_csv(args.out_csv, index=False)
    summary = {}
    ok = frame[frame.get("error", pd.Series([""] * len(frame))).fillna("") == ""]
    for name, sub in ok.groupby("protocol"):
        summary[str(name)] = {
            "n": int(len(sub)),
            "top1": float(sub["top1"].mean()) if "top1" in sub else None,
            "top3": float(sub["top3"].mean()) if "top3" in sub else None,
            "top5": float(sub["top5"].mean()) if "top5" in sub else None,
            "sampled": float(sub["sampled"].mean()) if "sampled" in sub else None,
            "median_seconds": float(sub["seconds"].median()) if "seconds" in sub else None,
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_complexes": int(len(ids)),
        "threshold_angstrom": args.threshold,
        "has_posescore_ckpt": bool(model_path),
        "by_protocol": summary,
        "literature_metallodock_top1": 0.786,
        "note": "Public 10 only. PoseScore trained on OUTPUT_DOCK, grouped by pdb_id.",
        "csv": str(args.out_csv),
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.out_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
