#!/usr/bin/env python3
"""Summarize curated ChEMBL tables already on disk (no new downloads)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curated", type=Path, default=Path("processed/chembl/curated"))
    p.add_argument("--out-csv", type=Path, default=Path("processed/chembl/curated/all_targets_molecules.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/chembl_summary.json"))
    args = p.parse_args()
    files = sorted(args.curated.glob("*_CHEMBL*.csv"))
    frames = [pd.read_csv(f) for f in files]
    if not frames:
        raise SystemExit("no curated tables")
    cat = pd.concat(frames, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(args.out_csv, index=False)
    by_metal = (
        cat.groupby("metal")
        .agg(n_molecules=("molecule_chembl_id", "nunique"), n_targets=("target_id", "nunique"))
        .reset_index()
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": int(len(files)),
        "n_molecule_target_rows": int(len(cat)),
        "n_unique_molecules": int(cat["molecule_chembl_id"].nunique()),
        "n_targets": int(cat["target_id"].nunique()),
        "by_metal": by_metal.to_dict(orient="records"),
        "targets": cat.groupby(["metal", "target_id", "target_name"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict(orient="records"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "targets"}, indent=2))


if __name__ == "__main__":
    main()
