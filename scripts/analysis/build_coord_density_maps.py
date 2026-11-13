#!/usr/bin/env python3
"""Build 3D metal-coordination density maps from train-split crystal complexes.

Reads ORI_PDB_LIGAND via inventory paths (read-only). Accumulates ligand
donor unit vectors on a Fibonacci sphere, per metal and element (O/N/S).
Writes processed/maps/coord_density.npz. Does not use the time-split test set.
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

from metgeodock.coordination import detect_coordination, ligand_donors  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402
from metgeodock.maps import N_RAYS, fibonacci_sphere, nearest_ray_index  # noqa: E402
from metgeodock.metals import METAL_ORDER  # noqa: E402

LOG = logging.getLogger("build_coord_density_maps")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--max-dist", type=float, default=2.90)
    p.add_argument("--out", type=Path, default=Path("processed/maps/coord_density.npz"))
    p.add_argument("--log", type=Path, default=Path("logs/build_coord_density_maps.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1].copy()
    if args.split_csv.exists():
        split = pd.read_csv(args.split_csv)
        train_ids = set(split.loc[split["time_split"] == "train", "pdb_id"].astype(str).str.upper())
        inv = inv[inv["pdb_id"].astype(str).str.upper().isin(train_ids)].copy()
        LOG.info("train complexes %s", len(inv))
    rays = fibonacci_sphere(N_RAYS)
    counts = {m: {el: np.zeros(N_RAYS, dtype=float) for el in "ONS"} for m in METAL_ORDER}
    n_donors = {m: 0 for m in METAL_ORDER}
    n_ok = 0
    for rec in inv.itertuples(index=False):
        try:
            protein = read_pdb(rec.protein_path)
            ligand = read_mol2(rec.ligand_path)
            sphere = detect_coordination(protein, metal=str(rec.metal))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("skip %s %s: %s", rec.metal, rec.pdb_id, exc)
            continue
        idx, xyz = ligand_donors(ligand)
        if idx.size == 0:
            continue
        metal = str(rec.metal)
        used = False
        for i, coord in zip(idx, xyz):
            vec = coord - sphere.metal_xyz
            dist = float(np.linalg.norm(vec))
            if dist < 1.0e-6 or dist > args.max_dist:
                continue
            unit = (vec / dist).reshape(1, 3)
            ray_i = int(nearest_ray_index(unit, rays)[0])
            el = str(ligand.elements[int(i)]).upper()[:1]
            if el not in counts[metal]:
                continue
            counts[metal][el][ray_i] += 1.0
            n_donors[metal] += 1
            used = True
        if used:
            n_ok += 1
    payload = {"rays": rays}
    density_summary = {}
    for metal in METAL_ORDER:
        for el in "ONS":
            hist = counts[metal][el]
            total = float(hist.sum())
            dens = hist / total if total > 0 else np.ones(N_RAYS) / N_RAYS
            payload[f"{metal}_{el}"] = dens
            density_summary[f"{metal}_{el}"] = {"n": int(total), "max": float(dens.max())}
        payload[f"{metal}_n"] = np.array(n_donors[metal])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_complexes": int(n_ok),
        "n_rays": N_RAYS,
        "max_dist_angstrom": args.max_dist,
        "split": "time_split==train",
        "n_donors": n_donors,
        "density_summary": density_summary,
        "out": str(args.out),
    }
    (args.out.parent / "coord_density_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("wrote %s complexes=%s", args.out, n_ok)


if __name__ == "__main__":
    main()
