#!/usr/bin/env python3
"""Summarize train-sample docking RMSD from fit_ff_rank_weights poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("processed/ff/rank_weight_poses.csv"))
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("results/rank_weight_train_summary.json"))
    args = p.parse_args()
    frame = pd.read_csv(args.csv)
    grp = frame.groupby(["metal", "pdb_id"], sort=False)
    best = grp["rmsd"].min().reset_index(name="best_rmsd")
    top1 = grp.head(1)[["metal", "pdb_id", "rmsd"]].rename(columns={"rmsd": "top1_rmsd"})
    merged = best.merge(top1, on=["metal", "pdb_id"])
    merged["success_sampled"] = (merged["best_rmsd"] <= args.threshold).astype(int)
    merged["success_top1"] = (merged["top1_rmsd"] <= args.threshold).astype(int)
    summary = {
        "n": int(len(merged)),
        "sampled": float(merged["success_sampled"].mean()),
        "top1": float(merged["success_top1"].mean()),
        "median_best_rmsd": float(merged["best_rmsd"].median()),
        "median_top1_rmsd": float(merged["top1_rmsd"].median()),
        "failures_sampled": merged.loc[merged["success_sampled"] == 0, ["metal", "pdb_id", "best_rmsd"]].to_dict(
            "records"
        ),
        "by_metal": {
            str(m): {
                "n": int(len(g)),
                "sampled": float(g["success_sampled"].mean()),
                "top1": float(g["success_top1"].mean()),
            }
            for m, g in merged.groupby("metal")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
