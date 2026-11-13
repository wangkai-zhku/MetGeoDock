#!/usr/bin/env python3
"""Scoring-power test: rank the crystal pose among rigid-body decoys.

Decoys keep the ligand internal coordinates and are randomly rotated about
the metal, then radially shifted. A metal-aware score should prefer the
crystal geometry; a metal-agnostic Vina-like score often does not.
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
from metgeodock.geometry import axis_angle  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402
from metgeodock.scoring import score_pose  # noqa: E402

LOG = logging.getLogger("scoring_power")


def clash_decoy(ligand, metal_xyz, rng) -> object:
    axis = rng.normal(size=3)
    ang = float(rng.uniform(0.4, 3.0))
    r = axis_angle(axis, ang)
    xyz = (r @ (ligand.xyz - metal_xyz).T).T + metal_xyz
    shift = rng.normal(size=3)
    shift = 0.6 * shift / (np.linalg.norm(shift) + 1e-12)
    return ligand.copy_with_xyz(xyz + shift)


def coordination_broken_decoy(ligand, metal_xyz, rng) -> object:
    """Keep pocket occupancy but push donors off the first shell (2.0 -> ~4.2 A)."""
    centroid = ligand.xyz[ligand.heavy_mask()].mean(axis=0)
    vec = centroid - metal_xyz
    nrm = np.linalg.norm(vec)
    if nrm < 1e-6:
        vec = rng.normal(size=3)
        nrm = np.linalg.norm(vec)
    direction = vec / nrm
    dist = float(rng.uniform(2.2, 3.2))
    xyz = ligand.xyz + direction * dist
    axis = np.cross(direction, rng.normal(size=3))
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
    xyz = (axis_angle(axis, float(rng.uniform(-0.4, 0.4))) @ (xyz - metal_xyz).T).T + metal_xyz
    return ligand.copy_with_xyz(xyz)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--per-metal", type=int, default=10)
    p.add_argument("--n-decoy", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=Path("results/scoring_power.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/scoring_power_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_scoring_power.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    rng = np.random.default_rng(args.seed)
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1]
    rows = []
    for metal, sub in inv.groupby("metal"):
        n = min(args.per_metal, len(sub))
        pick = sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)]
        for rec in pick.itertuples(index=False):
            try:
                protein = read_pdb(rec.protein_path)
                ligand = read_mol2(rec.ligand_path)
                sphere = detect_coordination(protein, metal=rec.metal)
                native = score_pose(protein, ligand, sphere)
                n_clash = args.n_decoy // 3
                n_broke = args.n_decoy - n_clash
                decoy_mols = [
                    clash_decoy(ligand, sphere.metal_xyz, rng) for _ in range(n_clash)
                ] + [
                    coordination_broken_decoy(ligand, sphere.metal_xyz, rng) for _ in range(n_broke)
                ]
                decoys = [score_pose(protein, mol, sphere) for mol in decoy_mols]
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s %s: %s", rec.metal, rec.pdb_id, exc)
                continue
            recs = {"hybrid": native.total, "vina": native.vina, "geo": native.geo, "gp": native.gp, "consensus": native.total + 0.25 * native.vina}
            dvals = {
                "hybrid": [d.total for d in decoys],
                "vina": [d.vina for d in decoys],
                "geo": [d.geo for d in decoys],
                "gp": [d.gp for d in decoys],
            }
            dvals["consensus"] = [t + 0.25 * v for t, v in zip(dvals["hybrid"], dvals["vina"])]
            row = {"metal": rec.metal, "pdb_id": rec.pdb_id, "n_decoy": args.n_decoy}
            for name, nat in recs.items():
                better = sum(1 for x in dvals[name] if x < nat)
                row[f"{name}_rank"] = better + 1
                row[f"{name}_top1"] = int(better == 0)
                row[f"{name}_native"] = nat
                row[f"{name}_decoy_median"] = float(np.median(dvals[name]))
            rows.append(row)
            LOG.info("%s %s hybrid_rank=%s vina_rank=%s", rec.metal, rec.pdb_id, row["hybrid_rank"], row["vina_rank"])
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "n": int(len(frame)), "methods": {}}
    for name in ("hybrid", "vina", "geo", "gp", "consensus"):
        if f"{name}_top1" in frame:
            summary["methods"][name] = {
                "top1": float(frame[f"{name}_top1"].mean()),
                "median_rank": float(frame[f"{name}_rank"].median()),
            }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(summary["methods"]))


if __name__ == "__main__":
    main()
