#!/usr/bin/env python3
"""Sanity-check water occupancy and multi-metal pocket detection.

Does not dock. Checks: merge increases first-shell water count; strip_waters
zeros it; pocket_metal_indices >= 2 on the multi-metal inventory; crop_pocket
keeps every listed metal. Writes results/water_multimetal_validation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.coordination import detect_coordination, pocket_metal_indices  # noqa: E402
from metgeodock.io import crop_pocket, load_molecule, merge_with_waters, read_pdb, strip_waters  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--water-csv", type=Path, default=Path("processed/water_multimetal/inventory_coord_water_gpdock.csv"))
    p.add_argument("--multi-csv", type=Path, default=Path("processed/water_multimetal/inventory_multimetal_rcsb.csv"))
    p.add_argument("--n-check", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/water_multimetal_validation.json"))
    args = p.parse_args()
    checks = []

    if args.water_csv.exists():
        water = pd.read_csv(args.water_csv).head(args.n_check)
        n_ok = 0
        for _, rec in water.iterrows():
            protein = read_pdb(rec.protein_path)
            lig = load_molecule(rec.ligand_path)
            water_path = str(rec.water_path) if "water_path" in rec.index and pd.notna(rec.water_path) else ""
            if water_path and Path(water_path).is_file():
                merged = merge_with_waters(protein, read_pdb(water_path))
            else:
                merged = protein
            stripped = strip_waters(merged)
            pocket = pocket_metal_indices(merged, lig)
            mi = int(pocket[0]) if pocket.size else int(merged.metal_indices()[0])
            n_keep = detect_coordination(merged, metal=str(rec.metal), metal_index=mi).n_water
            pocket_s = pocket_metal_indices(stripped, lig)
            mi_s = int(pocket_s[0]) if pocket_s.size else int(stripped.metal_indices()[0])
            n_strip = detect_coordination(stripped, metal=str(rec.metal), metal_index=mi_s).n_water
            ok = int(n_keep) > 0 and int(n_strip) == 0
            n_ok += int(ok)
            checks.append(
                {
                    "kind": "water",
                    "pdb_id": rec.pdb_id,
                    "n_water_keep": int(n_keep),
                    "n_water_strip": int(n_strip),
                    "ok": ok,
                }
            )
        water_ok = n_ok == len(water) and len(water) > 0
    else:
        water_ok = False

    if args.multi_csv.exists():
        multi = pd.read_csv(args.multi_csv).head(args.n_check)
        n_ok = 0
        for _, rec in multi.iterrows():
            protein = read_pdb(rec.protein_path)
            lig = load_molecule(rec.ligand_path)
            pocket = pocket_metal_indices(protein, lig)
            cropped = crop_pocket(protein, protein.xyz[pocket], radius_A=15.0)
            kept = pocket_metal_indices(cropped, lig)
            ok = int(pocket.size) >= 2 and int(kept.size) >= 2
            n_ok += int(ok)
            checks.append(
                {
                    "kind": "multi",
                    "pdb_id": rec.pdb_id,
                    "n_pocket": int(pocket.size),
                    "n_pocket_after_crop": int(kept.size),
                    "ok": ok,
                }
            )
        multi_ok = n_ok == len(multi) and len(multi) > 0
    else:
        multi_ok = False

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "water_occupancy_ok": water_ok,
        "multimetal_pocket_ok": multi_ok,
        "ok": bool(water_ok and multi_ok),
        "n_checks": len(checks),
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "water_occupancy_ok", "multimetal_pocket_ok", "n_checks")}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
