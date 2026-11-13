#!/usr/bin/env python3
"""Index time-split test complexes that have coordinating water or pocket multi-metals.

Reads processed/splits/time_split.csv (paths only). Merges HOH files in memory
and never writes back to ORI_PDB_LIGAND / data/. Inventories go to processed/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.coordination import detect_coordination, pocket_metal_indices  # noqa: E402
from metgeodock.io import load_molecule, merge_with_waters, read_pdb  # noqa: E402

LOG = logging.getLogger("build_water_multimetal_sets")


def _row_features(rec) -> dict:
    protein = read_pdb(rec.protein_path)
    ligand = load_molecule(rec.ligand_path)
    water = None
    water_path = ""
    if "water_path" in rec.index:
        water_path = str(rec.water_path) if pd.notna(rec.water_path) else ""
        if water_path and Path(water_path).is_file():
            water = read_pdb(water_path)
    merged = merge_with_waters(protein, water)
    n_metal = int(protein.metal_indices().size)
    pocket = pocket_metal_indices(merged, ligand)
    n_pocket = int(pocket.size)
    n_water_atoms = 0
    n_shell_water = 0
    n_other_metal = 0
    if pocket.size:
        for mi in pocket:
            sphere = detect_coordination(merged, metal=str(rec.metal), metal_index=int(mi))
            n_shell_water = max(n_shell_water, int(sphere.n_water))
            n_other_metal = max(n_other_metal, int(sphere.n_other_metal))
        n_water_atoms = int(sum(1 for i in range(len(merged)) if merged.is_water(i)))
    return {
        "metal": rec.metal,
        "pdb_id": rec.pdb_id,
        "rec_id": rec.rec_id,
        "protein_path": rec.protein_path,
        "ligand_path": rec.ligand_path,
        "water_path": water_path,
        "source": rec.source if "source" in rec.index else "gpdock",
        "time_split": rec.time_split if "time_split" in rec.index else "",
        "n_metal": n_metal,
        "n_pocket_metal": n_pocket,
        "n_water_atoms": n_water_atoms,
        "n_shell_water": n_shell_water,
        "n_other_metal": n_other_metal,
        "has_coord_water": int(n_shell_water > 0),
        "has_multimetal_pocket": int(n_pocket >= 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument(
        "--rcsb-csv",
        type=Path,
        default=Path("processed/metallodock_541/inventory_unique_pdb.csv"),
        help="full-ASU RCSB 2022 complexes (waters and extra metals still in the protein PDB)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("processed/water_multimetal"))
    p.add_argument("--log", type=Path, default=Path("logs/build_water_multimetal_sets.log"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rcsb-limit", type=int, default=0)
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    frames = []
    if args.split_csv.exists():
        gpdock = pd.read_csv(args.split_csv)
        gpdock = gpdock[(gpdock["time_split"] == "test") & (gpdock["complete"] == 1)].copy()
        gpdock["source"] = "gpdock_timesplit"
        if args.limit:
            gpdock = gpdock.iloc[: args.limit].copy()
        frames.append(gpdock)
        LOG.info("n_gpdock_test_complete=%s", len(gpdock))
    if args.rcsb_csv.exists():
        rcsb = pd.read_csv(args.rcsb_csv)
        rcsb["source"] = "rcsb_2022"
        if args.rcsb_limit:
            rcsb = rcsb.iloc[: args.rcsb_limit].copy()
        frames.append(rcsb)
        LOG.info("n_rcsb=%s", len(rcsb))
    if not frames:
        raise SystemExit("no inventory csv found")
    inv = pd.concat(frames, axis=0, ignore_index=True)
    rows = []
    n_fail = 0
    for i, rec in inv.iterrows():
        try:
            feat = _row_features(rec)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            LOG.warning("skip %s %s: %s", rec.metal, rec.pdb_id, exc)
            continue
        rows.append(feat)
        if len(rows) % 50 == 0:
            LOG.info("indexed %s / %s", len(rows), len(inv))
    frame = pd.DataFrame(rows)
    all_path = args.out_dir / "inventory_test.csv"
    water_path = args.out_dir / "inventory_coord_water.csv"
    multi_path = args.out_dir / "inventory_multimetal.csv"
    both_path = args.out_dir / "inventory_water_and_multimetal.csv"
    frame.to_csv(all_path, index=False)
    water = frame[frame["has_coord_water"] == 1].copy()
    multi = frame[frame["has_multimetal_pocket"] == 1].copy()
    both = frame[(frame["has_coord_water"] == 1) & (frame["has_multimetal_pocket"] == 1)].copy()
    water.to_csv(water_path, index=False)
    multi.to_csv(multi_path, index=False)
    both.to_csv(both_path, index=False)
    gpdock_water = water[water["source"] == "gpdock_timesplit"].copy()
    rcsb_water = water[water["source"] == "rcsb_2022"].copy()
    rcsb_multi = multi[multi["source"] == "rcsb_2022"].copy()
    gpdock_water.to_csv(args.out_dir / "inventory_coord_water_gpdock.csv", index=False)
    rcsb_water.to_csv(args.out_dir / "inventory_coord_water_rcsb.csv", index=False)
    rcsb_multi.to_csv(args.out_dir / "inventory_multimetal_rcsb.csv", index=False)
    by_source = frame.groupby("source").size().to_dict() if len(frame) else {}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_input": int(len(inv)),
        "n_indexed": int(len(frame)),
        "n_fail": int(n_fail),
        "n_by_source": by_source,
        "n_coord_water": int(len(water)),
        "n_coord_water_gpdock": int(len(gpdock_water)),
        "n_coord_water_rcsb": int(len(rcsb_water)),
        "n_multimetal_pocket": int(len(multi)),
        "n_multimetal_rcsb": int(len(rcsb_multi)),
        "n_both": int(len(both)),
        "n_coord_water_by_metal": water.groupby("metal").size().to_dict() if len(water) else {},
        "n_multimetal_by_metal": multi.groupby("metal").size().to_dict() if len(multi) else {},
        "paths": {
            "all": str(all_path),
            "coord_water": str(water_path),
            "multimetal": str(multi_path),
            "both": str(both_path),
        },
    }
    out_json = args.out_dir / "build_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
