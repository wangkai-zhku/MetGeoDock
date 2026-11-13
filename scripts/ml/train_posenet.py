#!/usr/bin/env python3
"""Train PoseNet (sklearn MLP) to rank docked poses from MetGeo terms.

Label is is_native (crystal-like). Split by pdb_id GroupKFold. Writes
processed/ml/posenet.joblib. Does not mutate data/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from metgeodock.nnscore import build_posenet, save_nn

FEATS = [
    "gp_nll",
    "geo_rmsd_or",
    "clash",
    "dist_harm",
    "cn_pen",
    "n_coord_lig",
    "n_coord_pro",
    "n_heavy",
    "n_donor",
    "min_donor_dist",
    "mean_donor_dist",
    "hybrid_total",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("processed/ml/pose_rank_features.csv"))
    p.add_argument("--out-model", type=Path, default=Path("processed/ml/posenet.joblib"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/posenet_report.json"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    frame = pd.read_csv(args.csv)
    cols = [c for c in FEATS if c in frame.columns]
    if "is_native" not in frame.columns:
        raise SystemExit("pose_rank_features.csv missing is_native")
    x = np.nan_to_num(frame[cols].to_numpy(dtype=float), nan=0.0)
    y = frame["is_native"].to_numpy(dtype=int)
    groups = frame["pdb_id"].astype(str).to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    n_splits = min(5, max(2, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    aucs, aps = [], []
    for tr, te in cv.split(x, y, groups):
        model = build_posenet()
        # regressor trained on 0/1; ranking uses negative prediction as score
        model.fit(x[tr], y[tr].astype(float))
        pred = model.predict(x[te])
        oof[te] = pred
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], pred))
        aps.append(average_precision_score(y[te], pred))
    final = build_posenet()
    final.fit(x, y.astype(float))
    save_nn(final, args.out_model, extra={"features": cols, "seed": args.seed, "kind": "posenet_mlp"})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_poses": int(len(frame)),
        "n_complexes": n_groups,
        "n_pos": int(y.sum()),
        "oof_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "oof_ap": float(np.mean(aps)) if aps else float("nan"),
        "n_cv_folds_scored": int(len(aucs)),
        "features": cols,
        "model": str(args.out_model),
        "split": "GroupKFold by pdb_id",
        "baseline_random_auroc": 0.5,
        "baseline_random_ap": float(y.mean()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
