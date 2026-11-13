#!/usr/bin/env python3
"""Extract first-shell metal clusters for QM / QM/MM validation.

Raw PDB/MOL2 files are read-only. Each cluster contains the metal ion,
protein donor heavy atoms (and their attached hydrogens if present), and
ligand donor atoms within the GPDOCK cutoff. Outputs:

* XYZ for xTB / ORCA
* ORCA input using hybrid DFT (B3LYP-D3BJ/def2-SVP) as a default
  cluster-model protocol, not a full protein QM/MM unless an MM topology
  is supplied by the user
* JSON with observed vs GPDOCK-mode distances (always computed)
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
from metgeodock.metals import FORMAL_CHARGE, METAL_INDEX, PROB_N, PROB_O, PROB_S  # noqa: E402
from metgeodock.scoring import geometry_penalty  # noqa: E402

LOG = logging.getLogger("prepare_qmmm")

ORCA_TMPL = """! B3LYP D3BJ def2-SVP CPCM(water) TightSCF
%maxcore 4000
%pal nprocs {nprocs} end
* xyz {charge} {mult}
{xyz}
*
"""


def mode_distance(metal: str, element: str) -> float:
    idx = METAL_INDEX[metal]
    el = element.upper()[0]
    table = PROB_O if el == "O" else PROB_N if el == "N" else PROB_S
    mode = int(np.argmax(table[idx]))
    return 1.7 + (mode + 0.5) * 0.1


def cluster_xyz(protein, ligand, sphere) -> tuple[list[str], int]:
    lines = []
    metal_el = {
        "zn": "Zn",
        "ca": "Ca",
        "mg": "Mg",
        "mn": "Mn",
        "fe2": "Fe",
        "fe3": "Fe",
        "cu": "Cu",
        "co": "Co",
        "cd": "Cd",
        "ni": "Ni",
    }[sphere.metal_name]
    xyz = sphere.metal_xyz
    lines.append(f"{metal_el:2s}  {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}")
    for coord, el in zip(sphere.protein_xyz, sphere.protein_elements):
        lines.append(f"{str(el):2s}  {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
    idx, lxyz = ligand_donors(ligand)
    n_lig = 0
    for i, coord in zip(idx, lxyz):
        d = float(np.linalg.norm(coord - sphere.metal_xyz))
        if d <= 2.9:
            el = str(ligand.elements[i])
            lines.append(f"{el:2s}  {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
            n_lig += 1
    charge = int(round(FORMAL_CHARGE[sphere.metal_name]))
    # crude: carboxylate O often -0.5 each; we keep metal formal charge only
    return lines, charge


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--per-metal", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("processed/qm/clusters"))
    p.add_argument("--nprocs", type=int, default=4)
    p.add_argument("--log", type=Path, default=Path("logs/prepare_qmmm_clusters.log"))
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for metal, sub in inv.groupby("metal"):
        n = min(args.per_metal, len(sub))
        pick = sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)]
        for rec in pick.itertuples(index=False):
            try:
                protein = read_pdb(rec.protein_path)
                ligand = read_mol2(rec.ligand_path)
                sphere = detect_coordination(protein, metal=rec.metal)
                rmsd, geo = geometry_penalty(sphere, ligand)
                lines, charge = cluster_xyz(protein, ligand, sphere)
                stem = f"{rec.metal}_{rec.pdb_id}_{rec.rec_id}"
                xyz_path = args.out_dir / f"{stem}.xyz"
                xyz_body = "\n".join(lines)
                xyz_path.write_text(f"{len(lines)}\n{stem}\n{xyz_body}\n", encoding="utf-8")
                orca_path = args.out_dir / f"{stem}.inp"
                orca_path.write_text(
                    ORCA_TMPL.format(nprocs=args.nprocs, charge=charge, mult=1, xyz=xyz_body),
                    encoding="utf-8",
                )
                dist_rows = []
                for coord, el, d in zip(sphere.protein_xyz, sphere.protein_elements, sphere.distances):
                    dist_rows.append(
                        {
                            "source": "protein",
                            "element": str(el),
                            "distance_angstrom": float(d),
                            "gpdock_mode_angstrom": mode_distance(rec.metal, str(el)),
                        }
                    )
                rec_out = {
                    "metal": rec.metal,
                    "pdb_id": rec.pdb_id,
                    "n_protein": sphere.n_protein,
                    "n_water": sphere.n_water,
                    "rmsd_or": rmsd,
                    "geometry": geo,
                    "xyz": str(xyz_path),
                    "orca_inp": str(orca_path),
                    "charge": charge,
                    "mean_abs_dev_from_mode": float(
                        np.mean([abs(r["distance_angstrom"] - r["gpdock_mode_angstrom"]) for r in dist_rows])
                    )
                    if dist_rows
                    else None,
                    "contacts": dist_rows,
                }
                (args.out_dir / f"{stem}.json").write_text(json.dumps(rec_out, indent=2), encoding="utf-8")
                records.append({k: rec_out[k] for k in rec_out if k != "contacts"})
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s %s: %s", rec.metal, rec.pdb_id, exc)
    frame = pd.DataFrame(records)
    frame.to_csv(args.out_dir / "cluster_index.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_clusters": int(len(frame)),
        "mean_rmsd_or": float(frame["rmsd_or"].mean()) if len(frame) else None,
        "mean_abs_dev_from_mode": float(frame["mean_abs_dev_from_mode"].mean()) if len(frame) else None,
        "note": "ORCA/xTB binaries are optional; geometry vs GPDOCK-mode distances are always reported.",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/qm_cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("%s", summary)


if __name__ == "__main__":
    main()
