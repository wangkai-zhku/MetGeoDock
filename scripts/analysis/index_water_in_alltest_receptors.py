#!/usr/bin/env python3
"""Count HOH already inside receptors used by the all-test redocks.

Timesplit_full / RCSB-2022 / public10 called dock_ligand(protein, ligand)
without water=; only HOH present in that protein PDB can participate.
Does not mutate ORI / data / agri raw.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metgeodock.io import WATER_RESNAMES


def _n_water_pdb(path: Path) -> int:
    if not path.is_file():
        return -1
    n = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[17:20].strip().upper() in WATER_RESNAMES:
                n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesplit", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--rcsb", type=Path, default=Path("processed/metallodock_541/inventory_unique_pdb.csv"))
    p.add_argument("--public-root", type=Path, default=Path("processed/metallodock_ref/dataset/metallo_complex"))
    p.add_argument("--public-csv", type=Path, default=Path("processed/metallodock_ref/dataset/test_pdbid.csv"))
    p.add_argument("--out", type=Path, default=Path("results/all_test_water_in_receptors.json"))
    args = p.parse_args()

    def _frac(n_pos: int, n: int) -> float | None:
        return float(n_pos / n) if n else None

    report: dict = {"created_utc": datetime.now(timezone.utc).isoformat(), "sets": {}}

    ts = pd.read_csv(args.timesplit)
    ts = ts[(ts["time_split"] == "test") & (ts["complete"] == 1)].copy()
    ts_water = []
    ts_has_hoh_file = 0
    for rec in ts.itertuples(index=False):
        n = _n_water_pdb(Path(rec.protein_path))
        ts_water.append(n)
        wp = getattr(rec, "water_path", "")
        if str(wp) and Path(str(wp)).is_file():
            ts_has_hoh_file += 1
    ts_n = [x for x in ts_water if x >= 0]
    report["sets"]["timesplit_full"] = {
        "n_protein": len(ts_n),
        "n_with_hoh_in_protein_pdb": int(sum(x > 0 for x in ts_n)),
        "frac_hoh_in_protein_pdb": _frac(sum(x > 0 for x in ts_n), len(ts_n)),
        "n_separate_hoh_file": int(ts_has_hoh_file),
        "note": "benchmark_timesplit_full.py did not pass water=; separate *_hoh_*.pdb were unused.",
    }

    rcsb = pd.read_csv(args.rcsb)
    rc_paths = []
    for pth in rcsb["protein_path"].astype(str):
        pocket = Path(pth.replace("_pro.pdb", "_pro_pocket.pdb"))
        rc_paths.append(pocket if pocket.is_file() else Path(pth))
    rc_water = [_n_water_pdb(p) for p in rc_paths]
    rc_n = [x for x in rc_water if x >= 0]
    report["sets"]["rcsb2022"] = {
        "n_protein": len(rc_n),
        "n_with_hoh_in_protein_pdb": int(sum(x > 0 for x in rc_n)),
        "frac_hoh_in_protein_pdb": _frac(sum(x > 0 for x in rc_n), len(rc_n)),
        "note": "cropped ASU pockets; HOH kept if present. dock_ligand keep_water=True (default).",
    }

    pub = {"n_protein": 0, "n_with_hoh_in_protein_pdb": 0}
    if args.public_csv.exists():
        ids = pd.read_csv(args.public_csv)["pdb_id"].astype(str).str.lower().tolist()
        hits = []
        for pdb in ids:
            path = args.public_root / pdb / f"{pdb}_pro_pocket.pdb"
            hits.append(_n_water_pdb(path))
        hits = [x for x in hits if x >= 0]
        pub = {
            "n_protein": len(hits),
            "n_with_hoh_in_protein_pdb": int(sum(x > 0 for x in hits)),
            "frac_hoh_in_protein_pdb": _frac(sum(x > 0 for x in hits), len(hits)),
            "note": "MetalloDock public pockets; keep_water=True only if HOH is already in the pocket PDB.",
        }
    report["sets"]["public10"] = pub
    report["code_default"] = {
        "dock_ligand_keep_water": True,
        "mechanism": "HOH/WAT in the receptor occupy vacant polyhedron vertices.",
        "timesplit_water_files_used_in_all_test": False,
        "dedicated_water_ablation": "results/water_multimetal_redock.json",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
