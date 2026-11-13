#!/usr/bin/env python3
"""Ablate coordinating water and pocket multi-metal docking.

Two protocols on the same crystal complexes:

  water:   keep first-shell HOH as occupied vertices vs strip HOH
           (MetalloDock README requires removing water)

  multi:   dock every pocket metal (extras occupy vertices) vs keep only
           the ligand-closest metal (MetalloDock README: remove extraneous metals)

Success is heavy-atom RMSD <= threshold vs the crystal ligand. Native poses
are not mixed into ranking. Numbers are computed here; MetalloDock literature
rates are cited, not re-run.
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

from metgeodock.coordination import pocket_metal_indices  # noqa: E402
from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import crop_pocket, keep_only_metal, load_molecule, merge_with_waters, read_pdb  # noqa: E402

LOG = logging.getLogger("benchmark_water_multimetal")


def _stratify(frame: pd.DataFrame, per_metal: int, seed: int) -> pd.DataFrame:
    if per_metal <= 0 or frame.empty:
        return frame
    rng = np.random.default_rng(seed)
    parts = []
    for _metal, sub in frame.groupby("metal"):
        n = min(int(per_metal), len(sub))
        parts.append(sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)])
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _success_from_rmsds(rmsds: list[float], threshold: float) -> dict:
    arr = np.asarray(rmsds, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return {
            "top1_rmsd": float("nan"),
            "top3_rmsd": float("nan"),
            "top5_rmsd": float("nan"),
            "best_sampled_rmsd": float("nan"),
            "success_top1": 0,
            "success_top3": 0,
            "success_top5": 0,
            "success_sampled": 0,
            "n_poses": 0,
        }
    finite = arr[np.isfinite(arr)]
    top1 = float(finite[0])
    top3 = float(np.min(finite[: min(3, finite.size)]))
    top5 = float(np.min(finite[: min(5, finite.size)]))
    best = float(np.min(finite))
    return {
        "top1_rmsd": top1,
        "top3_rmsd": top3,
        "top5_rmsd": top5,
        "best_sampled_rmsd": best,
        "success_top1": int(top1 <= threshold),
        "success_top3": int(top3 <= threshold),
        "success_top5": int(top5 <= threshold),
        "success_sampled": int(best <= threshold),
        "n_poses": int(arr.size),
    }


def _dock_record(
    rec,
    *,
    keep_water: bool,
    multi_metal: bool,
    strip_extra_metals: bool,
    n_spin: int,
    max_poses: int,
    pocket_A: float,
    threshold: float,
    protocol: str,
) -> dict:
    protein = read_pdb(rec.protein_path)
    ligand = load_molecule(rec.ligand_path)
    water = None
    water_path = str(rec.water_path) if "water_path" in rec.index and pd.notna(rec.water_path) else ""
    if water_path and Path(water_path).is_file():
        water = read_pdb(water_path)
    merged = merge_with_waters(protein, water) if keep_water else protein
    pocket = pocket_metal_indices(merged, ligand)
    if pocket.size == 0:
        raise ValueError("no pocket metal")
    if strip_extra_metals:
        typed = merged.metal_indices(metal=str(rec.metal))
        keep_idx = int(typed[0]) if typed.size else int(merged.metal_indices()[0])
        merged = keep_only_metal(merged, keep_idx)
        remain = merged.metal_indices()
        if remain.size == 0:
            raise ValueError("no metal after strip")
        pocket_xyz = merged.xyz[int(remain[0])]
        receptor = crop_pocket(merged, pocket_xyz, radius_A=pocket_A)
        remain2 = receptor.metal_indices()
        result = dock_ligand(
            receptor,
            ligand,
            metal=str(rec.metal),
            n_spin=n_spin,
            max_poses=max_poses,
            keep_water=keep_water,
            multi_metal=False,
            metal_index=int(remain2[0]) if remain2.size else None,
            coord_water=0,
        )
    else:
        pocket_xyz = merged.xyz[pocket]
        receptor = crop_pocket(merged, pocket_xyz, radius_A=pocket_A)
        result = dock_ligand(
            receptor,
            ligand,
            metal=str(rec.metal),
            n_spin=n_spin,
            max_poses=max_poses,
            keep_water=keep_water,
            multi_metal=multi_metal,
            coord_water=0,
        )
    if not result.poses:
        raise ValueError("no_pose")
    rmsds = [redock_rmsd(pose.ligand, ligand) for pose in result.poses]
    out = {
        "protocol": protocol,
        "metal": rec.metal,
        "pdb_id": rec.pdb_id,
        "rec_id": rec.rec_id if "rec_id" in rec.index else "",
        "keep_water": int(keep_water),
        "multi_metal": int(multi_metal and not strip_extra_metals),
        "strip_extra_metals": int(strip_extra_metals),
        "n_pocket_metal": int(pocket.size),
        "n_water": int(result.sphere.n_water),
        "n_other_metal": int(result.sphere.n_other_metal),
        "n_spheres": int(len(result.spheres)),
        "best_metal_index": int(result.sphere.metal_index),
        "gpdock_label": rec.gpdock_label if "gpdock_label" in rec.index else "",
        "error": "",
    }
    out.update(_success_from_rmsds(rmsds, threshold))
    return out


def _rate(frame: pd.DataFrame, col: str) -> float | None:
    sub = frame[frame["error"] == ""]
    if sub.empty or col not in sub.columns:
        return None
    return float(sub[col].mean())


def _summarize(frame: pd.DataFrame) -> dict:
    ok = frame[frame["error"] == ""] if "error" in frame.columns else frame
    return {
        "n": int(len(ok)),
        "n_fail": int((frame["error"] != "").sum()) if "error" in frame.columns else 0,
        "top1": _rate(frame, "success_top1"),
        "top3": _rate(frame, "success_top3"),
        "top5": _rate(frame, "success_top5"),
        "sampled": _rate(frame, "success_sampled"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--water-csv", type=Path, default=Path("processed/water_multimetal/inventory_coord_water_gpdock.csv"))
    p.add_argument("--multi-csv", type=Path, default=Path("processed/water_multimetal/inventory_multimetal_rcsb.csv"))
    p.add_argument("--n-spin", type=int, default=12)
    p.add_argument("--max-poses", type=int, default=24)
    p.add_argument("--pocket-A", dest="pocket_A", type=float, default=15.0)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--limit-water", type=int, default=0)
    p.add_argument("--limit-multi", type=int, default=0)
    p.add_argument("--per-metal", type=int, default=8, help="stratified cap per metal; 0 = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=Path("results/water_multimetal_redock.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/water_multimetal_redock.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_water_multimetal.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    jobs: list[tuple[str, pd.Series, dict]] = []
    if args.water_csv.exists():
        water = pd.read_csv(args.water_csv)
        if "complete" in water.columns:
            water = water[water["complete"] == 1].copy()
        if "has_water" in water.columns:
            water = water[water["has_water"] == 1].copy()
        water = _stratify(water, args.per_metal, args.seed)
        if args.limit_water and len(water) > args.limit_water:
            water = water.sample(n=args.limit_water, random_state=int(args.seed))
        LOG.info("water set n=%s", len(water))
        for _, rec in water.iterrows():
            jobs.append(("water_keep", rec, {"keep_water": True, "multi_metal": True, "strip_extra_metals": False}))
            jobs.append(("water_strip", rec, {"keep_water": False, "multi_metal": True, "strip_extra_metals": False}))
    else:
        LOG.warning("missing %s", args.water_csv)
    if args.multi_csv.exists():
        multi = pd.read_csv(args.multi_csv)
        if "n_metal" in multi.columns:
            multi = multi[multi["n_metal"] >= 2].copy()
        if "complete" in multi.columns:
            multi = multi[multi["complete"] == 1].copy()
        multi = _stratify(multi, args.per_metal, args.seed)
        if args.limit_multi and len(multi) > args.limit_multi:
            multi = multi.sample(n=args.limit_multi, random_state=int(args.seed))
        LOG.info("multi set n=%s", len(multi))
        for _, rec in multi.iterrows():
            jobs.append(("multi_keep", rec, {"keep_water": True, "multi_metal": True, "strip_extra_metals": False}))
            jobs.append(("multi_strip", rec, {"keep_water": True, "multi_metal": False, "strip_extra_metals": True}))
    else:
        LOG.warning("missing %s", args.multi_csv)

    rows = []
    for k, (protocol, rec, flags) in enumerate(jobs, start=1):
        LOG.info("[%s/%s] %s %s %s", k, len(jobs), protocol, rec.metal, rec.pdb_id)
        try:
            row = _dock_record(
                rec,
                n_spin=args.n_spin,
                max_poses=args.max_poses,
                pocket_A=args.pocket_A,
                threshold=args.threshold,
                protocol=protocol,
                **flags,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s %s %s: %s", protocol, rec.metal, rec.pdb_id, exc)
            row = {
                "protocol": protocol,
                "metal": rec.metal,
                "pdb_id": rec.pdb_id,
                "error": str(exc),
                "success_top1": 0,
                "success_top3": 0,
                "success_top5": 0,
                "success_sampled": 0,
            }
        rows.append(row)
        if k % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_csv, index=False)
    by_protocol = {}
    for proto, sub in frame.groupby("protocol"):
        by_protocol[str(proto)] = _summarize(sub)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_A": args.threshold,
        "n_spin": args.n_spin,
        "max_poses": args.max_poses,
        "pocket_A": args.pocket_A,
        "n_jobs": len(jobs),
        "by_protocol": by_protocol,
        "water_keep_beats_strip_top3": None,
        "multi_keep_beats_strip_top3": None,
        "csv": str(args.out_csv),
        "note": (
            "MetalloDock GitHub README requires removing water/heteroatoms/"
            "extraneous metals before docking. MetalloDock_bimetal 74.4% is a "
            "literature rate on prepared bimetal complexes, not a keep-water run."
        ),
    }
    wk = by_protocol.get("water_keep") or {}
    ws = by_protocol.get("water_strip") or {}
    mk = by_protocol.get("multi_keep") or {}
    ms = by_protocol.get("multi_strip") or {}
    if wk.get("top3") is not None and ws.get("top3") is not None:
        report["water_keep_beats_strip_top3"] = bool(float(wk["top3"]) > float(ws["top3"]))
    if mk.get("top3") is not None and ms.get("top3") is not None:
        report["multi_keep_beats_strip_top3"] = bool(float(mk["top3"]) > float(ms["top3"]))
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
