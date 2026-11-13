#!/usr/bin/env python3
"""Index GPDOCK ORI_PDB_LIGAND crystal complexes without mutating raw files.

Reads filenames of the form {metal}_{PDB}_{pro|lig|hoh}_{id}.{pdb|mol2}
and writes a compact inventory plus per-metal counts.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("index_crystal_set")


def parse_name(path: Path) -> dict | None:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    metal = parts[0].lower()
    pdb_id = parts[1].upper()
    kind = parts[2].lower()
    rec_id = parts[3]
    if kind not in {"pro", "lig", "hoh"}:
        return None
    return {
        "metal": metal,
        "pdb_id": pdb_id,
        "kind": kind,
        "rec_id": rec_id,
        "path": str(path),
        "suffix": path.suffix.lower(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=Path("ORI_PDB_LIGAND/SW"))
    p.add_argument("--out-csv", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--out-json", type=Path, default=Path("processed/crystal/inventory_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/index_crystal_set.log"))
    args = p.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    if not args.raw_dir.is_dir():
        raise SystemExit(f"raw dir not found: {args.raw_dir}")

    rows = []
    for path in sorted(args.raw_dir.iterdir()):
        if not path.is_file():
            continue
        rec = parse_name(path)
        if rec:
            rows.append(rec)
    files = pd.DataFrame(rows)
    grouped = defaultdict(dict)
    for rec in rows:
        key = (rec["metal"], rec["pdb_id"], rec["rec_id"])
        grouped[key][rec["kind"]] = rec["path"]

    complexes = []
    for (metal, pdb_id, rec_id), parts in grouped.items():
        complexes.append(
            {
                "metal": metal,
                "pdb_id": pdb_id,
                "rec_id": rec_id,
                "protein_path": parts.get("pro"),
                "ligand_path": parts.get("lig"),
                "water_path": parts.get("hoh"),
                "complete": int("pro" in parts and "lig" in parts),
                "has_water": int("hoh" in parts),
            }
        )
    inventory = pd.DataFrame(complexes).sort_values(["metal", "pdb_id", "rec_id"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.out_csv, index=False)

    complete = inventory[inventory["complete"] == 1]
    per_metal = (
        complete.groupby("metal")
        .agg(
            n_complete=("pdb_id", "size"),
            n_unique_pdb=("pdb_id", "nunique"),
            n_with_water=("has_water", "sum"),
        )
        .reset_index()
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(args.raw_dir.resolve()),
        "n_files": int(len(files)),
        "n_complex_keys": int(len(inventory)),
        "n_complete": int(complete.shape[0]),
        "n_unique_pdb": int(complete["pdb_id"].nunique()),
        "per_metal": per_metal.to_dict(orient="records"),
        "file_kind_counts": dict(Counter(files["kind"].tolist())) if len(files) else {},
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("files=%s complete_complexes=%s unique_pdb=%s", summary["n_files"], summary["n_complete"], summary["n_unique_pdb"])
    LOG.info("wrote %s and %s", args.out_csv, args.out_json)


if __name__ == "__main__":
    main()
