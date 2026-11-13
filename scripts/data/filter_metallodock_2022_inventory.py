#!/usr/bin/env python3
"""Collapse reconstructed 2022 complexes toward a MetalloDock-like test set.

Paper SI: 7-metal MP-ligand time-split, year 2022 boundary, n_test=541.
This keeps 7 metals (zn/mg/mn/fe2/fe3/co/ni/ca), drops tiny fragments, then
one site per PDB (largest n_heavy). Writes inventory_unique_pdb.csv.
Does not mutate data/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METALS7 = {"zn", "mg", "mn", "fe2", "fe3", "co", "ni", "ca"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/metallodock_541/inventory.csv"))
    p.add_argument("--out", type=Path, default=Path("processed/metallodock_541/inventory_unique_pdb.csv"))
    p.add_argument("--summary", type=Path, default=Path("processed/metallodock_541/filter_summary.json"))
    p.add_argument("--min-heavy", type=int, default=8)
    p.add_argument("--max-heavy", type=int, default=70)
    p.add_argument("--min-protein-coord", type=int, default=2)
    args = p.parse_args()
    inv = pd.read_csv(args.inventory)
    seven = inv[inv["metal"].astype(str).str.lower().isin(METALS7)].copy()
    sized = seven[
        (seven["n_heavy"] >= args.min_heavy)
        & (seven["n_heavy"] <= args.max_heavy)
        & (seven["n_protein_coord"] >= args.min_protein_coord)
    ].copy()
    ranked = sized.sort_values(["n_heavy", "n_protein_coord"], ascending=False)
    uniq = ranked.drop_duplicates(subset=["pdb_id"], keep="first").copy()
    uniq.to_csv(args.out, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_sites_raw": int(len(inv)),
        "n_sites_7metal": int(len(seven)),
        "n_sites_sized": int(len(sized)),
        "n_unique_pdb": int(len(uniq)),
        "literature_n_test": 541,
        "by_metal_unique_pdb": uniq["metal"].value_counts().to_dict(),
        "min_heavy": args.min_heavy,
        "max_heavy": args.max_heavy,
        "min_protein_coord": args.min_protein_coord,
        "out": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

