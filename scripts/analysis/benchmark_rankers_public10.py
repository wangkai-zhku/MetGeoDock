#!/usr/bin/env python3
"""Public-10: MetGeo vs trained ranker, with and without semi-flex.

Hold-out set (not used in ranker training). Success = heavy-atom RMSD<=2 A.
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

LOG = logging.getLogger("benchmark_rankers_public10")

PROTOCOLS = (
    {"name": "metgeo", "semi_flex": False, "rerank": "metgeo", "model": None},
    {"name": "deepres", "semi_flex": False, "rerank": "ranker", "model": "processed/ml/ranker_best.pt"},
    {"name": "geolist", "semi_flex": False, "rerank": "ranker", "model": "processed/ml/rankers/geolist_ranknet.pt"},
    {"name": "geolist_semiflex", "semi_flex": True, "rerank": "ranker", "model": "processed/ml/rankers/geolist_ranknet.pt"},
    {"name": "deepres_semiflex", "semi_flex": True, "rerank": "ranker", "model": "processed/ml/ranker_best.pt"},
    {"name": "ranker_blend", "semi_flex": False, "rerank": "ranker-blend", "model": "processed/ml/ranker_best.pt"},
    {"name": "ranker_blend_semiflex", "semi_flex": True, "rerank": "ranker-blend", "model": "processed/ml/ranker_best.pt"},
    {"name": "joint_geolist", "semi_flex": False, "rerank": "ranker", "model": "processed/ml/ranker_joint.pt"},
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
    p.add_argument("--model", type=Path, default=Path("processed/ml/ranker_best.pt"))
    p.add_argument("--out-csv", type=Path, default=Path("results/ranker_public10.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/ranker_public10.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_rankers_public10.log"))
    p.add_argument("--protocols", default="all", help="comma list or all")
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
                mpath = proto.get("model") or (str(args.model) if args.model.exists() else None)
                result = dock_ligand(
                    protein,
                    ligand,
                    n_spin=args.n_spin,
                    max_poses=args.max_poses,
                    semi_flex=bool(proto["semi_flex"]),
                    rerank=str(proto["rerank"]),
                    model_path=mpath,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"pdb_id": pdb, "protocol": proto["name"], "error": str(exc), "seconds": time.perf_counter() - t0})
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
                    "seconds": elapsed,
                    "error": "",
                    "has_ranker": bool(mpath),
                }
            )
            LOG.info("%s %s top1=%.3f top3=%s t=%.1fs", pdb, proto["name"], rec["top1_rmsd"], rec["top3"], elapsed)
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
    met = summary.get("metgeo") or {}
    deep = summary.get("deepres") or {}
    geo = summary.get("geolist") or {}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_complexes": int(len(ids)),
        "by_protocol": summary,
        "deepres_beats_metgeo_top1": bool((deep.get("top1") or -1) > (met.get("top1") or 0)),
        "deepres_beats_metgeo_top3": bool((deep.get("top3") or -1) > (met.get("top3") or 0)),
        "geolist_beats_metgeo_top1": bool((geo.get("top1") or -1) > (met.get("top1") or 0)),
        "geolist_beats_metgeo_top3": bool((geo.get("top3") or -1) > (met.get("top3") or 0)),
        "blend_beats_metgeo_top1": bool(((summary.get("ranker_blend") or {}).get("top1") or -1) > (met.get("top1") or 0)),
        "blend_beats_metgeo_top3": bool(((summary.get("ranker_blend") or {}).get("top3") or -1) > (met.get("top3") or 0)),
        "literature_metallodock_top1": 0.786,
        "csv": str(args.out_csv),
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
