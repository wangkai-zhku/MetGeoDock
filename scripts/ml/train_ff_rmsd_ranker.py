#!/usr/bin/env python3
"""Train ExtraTrees to predict pose RMSD from MetGeo FF terms.

Uses processed/ff/rank_weight_poses.csv (time-split train sample only).
Reports grouped OOF top-1 success and writes processed/ml/ff_rmsd_ranker.joblib.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

FEATS = ["steric", "coord", "geo", "contact", "cpen"]


def top1_from_pred(frame: pd.DataFrame, pred: np.ndarray, threshold: float) -> float:
    tmp = frame.assign(pred=pred)
    hits = []
    for _, sub in tmp.groupby(["metal", "pdb_id"], sort=False):
        best = sub.loc[sub["pred"].idxmin()]
        hits.append(float(best["rmsd"]) <= threshold)
    return float(np.mean(hits)) if hits else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("processed/ff/rank_weight_poses.csv"))
    p.add_argument("--out-model", type=Path, default=Path("processed/ml/ff_rmsd_ranker.joblib"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/ff_rmsd_ranker_report.json"))
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    frame = pd.read_csv(args.csv).dropna(subset=FEATS + ["rmsd"])
    x = frame[FEATS].to_numpy(dtype=float)
    y = frame["rmsd"].to_numpy(dtype=float)
    groups = frame["pdb_id"].astype(str).to_numpy()
    n_groups = frame["pdb_id"].nunique()
    n_splits = min(5, max(2, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(frame), dtype=float)
    for fold, (tr, te) in enumerate(cv.split(x, y, groups), start=1):
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=args.seed + fold,
            n_jobs=1,
        )
        model.fit(x[tr], y[tr])
        oof[te] = model.predict(x[te])
    oof_top1 = top1_from_pred(frame, oof, args.threshold)
    sampled = float(frame.groupby(["metal", "pdb_id"])["rmsd"].min().le(args.threshold).mean())
    final = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=1,
    )
    final.fit(x, y)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": FEATS}, args.out_model)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_poses": int(len(frame)),
        "n_complexes": int(n_groups),
        "oof_top1": oof_top1,
        "sampled": sampled,
        "importances": dict(zip(FEATS, (float(v) for v in final.feature_importances_))),
        "model": str(args.out_model),
        "split": "GroupKFold by pdb_id on time-split train sample",
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
