#!/usr/bin/env python3
"""Same-ligand listwise ranking on OUTPUT_DOCK PoseScore features.

GPDOCK file order is unranked. This script measures whether MetGeo or
PoseScore analytic puts an RMSD<=2 A model first within each complex.
Does not mutate ORI_PDB_LIGAND or OUTPUT_DOCK.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _top1_rate(frame: pd.DataFrame, score_col: str, y_col: str = "y") -> float | None:
    hits = []
    for _, sub in frame.groupby(["subset", "pdb_id", "rec_id", "protocol"], sort=False):
        if len(sub) < 2:
            continue
        if int(sub[y_col].sum()) == 0:
            continue
        order = np.argsort(sub[score_col].to_numpy(dtype=float), kind="mergesort")
        y = sub[y_col].to_numpy(dtype=int)[order]
        hits.append(int(y[0] == 1))
    if not hits:
        return None
    return float(np.mean(hits))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("processed/ml/posescore_features.csv"))
    p.add_argument("--out", type=Path, default=Path("results/posescore_listwise.json"))
    args = p.parse_args()
    feats = pd.read_csv(args.features)
    feats = feats.copy()
    feats["analytic"] = (
        feats["hybrid_total"].astype(float)
        - 0.20 * feats["map_overlap"].astype(float)
        + 0.08 * np.clip(2.20 - feats["water_min_dist"].astype(float), 0.0, None)
    )
    ceiling = []
    n_groups = 0
    n_groups_with_pos = 0
    for _, sub in feats.groupby(["subset", "pdb_id", "rec_id", "protocol"], sort=False):
        if len(sub) < 2:
            continue
        n_groups += 1
        if int(sub["y"].sum()) == 0:
            continue
        n_groups_with_pos += 1
        ceiling.append(1.0)
    by_subset = {}
    for subset, sub in feats.groupby("subset"):
        by_subset[str(subset)] = {
            "n_rows": int(len(sub)),
            "n_pos": int(sub["y"].sum()),
            "metalgeo_top1": _top1_rate(sub, "hybrid_total"),
            "analytic_top1": _top1_rate(sub, "analytic"),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(feats)),
        "n_pos": int(feats["y"].sum()),
        "n_listwise_groups": int(n_groups),
        "n_groups_with_success": int(n_groups_with_pos),
        "sampled_ceiling": float(np.mean(ceiling)) if ceiling else None,
        "metalgeo_top1": _top1_rate(feats, "hybrid_total"),
        "analytic_top1": _top1_rate(feats, "analytic"),
        "by_subset": by_subset,
        "note": (
            "Groups are same-ligand OUTPUT_DOCK models. "
            "GPDOCK file order is not a ranker; MetGeo/analytic top-1 is computed here."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
