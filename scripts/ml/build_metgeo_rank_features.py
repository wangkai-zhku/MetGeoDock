#!/usr/bin/env python3
"""Build MetGeoDock pose-ranking features (hold out public-10 PDBs).

Docks time-split TRAIN complexes with optional semi-flex, labels each pose
by crystal heavy-atom RMSD. Does not mutate ORI_PDB_LIGAND / data/.
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
from metgeodock.io import load_molecule, merge_with_waters, read_pdb  # noqa: E402
from metgeodock.posescore import pose_feature_vector  # noqa: E402

LOG = logging.getLogger("build_metgeo_rank_features")


def _public10(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    col = "pdb_id" if "pdb_id" in frame.columns else frame.columns[0]
    return set(frame[col].astype(str).str.lower())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--holdout", type=Path, default=Path("processed/metallodock_ref/dataset/test_pdbid.csv"))
    p.add_argument("--per-metal", type=int, default=10)
    p.add_argument("--n-spin", type=int, default=12)
    p.add_argument("--max-poses", type=int, default=24)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--semi-flex", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=Path, default=Path("processed/ml/metgeo_rank_features.csv"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/metgeo_rank_features_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/build_metgeo_rank_features.log"))
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--split", default="train", help="time_split value: train or test")
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    hold = _public10(args.holdout)
    inv = pd.read_csv(args.split_csv)
    inv = inv[(inv["time_split"] == str(args.split)) & (inv["complete"] == 1)].copy()
    inv["pdb_l"] = inv["pdb_id"].astype(str).str.lower()
    inv = inv[~inv["pdb_l"].isin(hold)].copy()
    rng = np.random.default_rng(args.seed)
    parts = []
    for _metal, sub in inv.groupby("metal"):
        n = min(int(args.per_metal), len(sub))
        parts.append(sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)])
    pick = pd.concat(parts, axis=0).reset_index(drop=True)
    done: set[str] = set()
    rows: list[dict] = []
    if args.out.exists() and not args.rebuild:
        prev = pd.read_csv(args.out)
        rows = prev.to_dict(orient="records")
        done = set(prev["pdb_id"].astype(str).str.lower() + "_" + prev["rec_id"].astype(str))
        LOG.info("resume n_rows=%s n_complexes=%s", len(prev), prev.groupby(["pdb_id", "rec_id"]).ngroups)
    for rec in pick.itertuples(index=False):
        key = f"{str(rec.pdb_id).lower()}_{rec.rec_id}"
        if key in done:
            continue
        try:
            protein = read_pdb(rec.protein_path)
            water_path = getattr(rec, "water_path", "") or ""
            water = None
            if str(water_path) and Path(str(water_path)).exists():
                water = read_pdb(water_path)
            ligand = load_molecule(rec.ligand_path)
            result = dock_ligand(
                protein,
                ligand,
                metal=str(rec.metal),
                n_spin=args.n_spin,
                max_poses=args.max_poses,
                keep_water=True,
                water=water,
                semi_flex=bool(args.semi_flex),
                rerank="metgeo",
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s %s: %s", rec.metal, rec.pdb_id, exc)
            continue
        if not result.poses:
            continue
        metal = result.sphere.metal_name
        n_ok = 0
        for rank_i, pose in enumerate(result.poses, start=1):
            rmsd = redock_rmsd(pose.ligand, ligand)
            if not np.isfinite(rmsd):
                continue
            feats = pose_feature_vector(pose.score, pose.ligand, metal=metal)
            feats.update(
                {
                    "pdb_id": rec.pdb_id,
                    "rec_id": str(rec.rec_id),
                    "metal": metal,
                    "source": "metgeodock",
                    "semi_flex_run": int(bool(args.semi_flex)),
                    "metalgeo_rank": int(rank_i),
                    "rmsd_angstrom": float(rmsd),
                    "y": int(rmsd <= args.threshold),
                    "has_water": int(bool(water_path) and Path(str(water_path)).exists()),
                    "group": f"{rec.pdb_id}_{rec.rec_id}",
                }
            )
            rows.append(feats)
            n_ok += 1
        LOG.info("%s %s n_pose=%s n_succ=%s", rec.metal, rec.pdb_id, n_ok, sum(1 for r in rows[-n_ok:] if r["y"]))
        done.add(key)
        if len(rows) and len(rows) % 400 < 30:
            pd.DataFrame(rows).to_csv(args.out, index=False)
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    n_groups = int(frame.groupby("group").ngroups) if len(frame) else 0
    n_pos_g = 0
    if len(frame):
        n_pos_g = int((frame.groupby("group")["y"].sum() > 0).sum())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv": str(args.out),
        "n_rows": int(len(frame)),
        "n_pos": int(frame["y"].sum()) if len(frame) else 0,
        "n_groups": n_groups,
        "n_groups_with_success": n_pos_g,
        "holdout_public10": sorted(hold),
        "semi_flex": bool(args.semi_flex),
        "per_metal": args.per_metal,
        "n_spin": args.n_spin,
        "max_poses": args.max_poses,
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
