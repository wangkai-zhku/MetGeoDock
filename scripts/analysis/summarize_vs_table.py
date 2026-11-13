#!/usr/bin/env python3
"""Rebuild a VS summary JSON from an already-written benchmark CSV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/vs_ml_benchmark.csv"))
    p.add_argument("--out", type=Path, default=Path("results/vs_ml_summary.json"))
    args = p.parse_args()
    out = pd.read_csv(args.csv)
    methods = []
    for col in out.columns:
        if col.endswith("_auroc"):
            methods.append(col[: -len("_auroc")])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_receptors": int(len(out)),
        "n_metals": int(out["metal"].nunique()) if "metal" in out else None,
        "source_csv": str(args.csv),
        "methods": {},
        "by_metal": {},
    }
    for method in methods:
        summary["methods"][method] = {
            "mean_auroc": float(out[f"{method}_auroc"].mean()),
            "mean_ef": float(out[f"{method}_ef1"].mean()) if f"{method}_ef1" in out else None,
            "median_active_rank": float(out[f"{method}_active_rank"].median()) if f"{method}_active_rank" in out else None,
        }
    if "metal" in out:
        for metal, sub in out.groupby("metal"):
            rec = {"n": int(len(sub))}
            for method in methods:
                rec[f"{method}_auroc"] = float(sub[f"{method}_auroc"].mean())
            summary["by_metal"][str(metal)] = rec
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
