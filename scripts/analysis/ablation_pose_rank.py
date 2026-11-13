#!/usr/bin/env python3
"""Ablate metal-aware terms vs the ML pose-ranker on native/decoy poses.

Lower metgeo/geo/gp/dist/clash is better, so scores are negated for AUROC.
The ML native_score_oof is higher-better. Vina is not part of the default model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metgeodock.ml import FEATURE_COLUMNS


def auc(y, s, higher_better: bool) -> float:
    score = np.asarray(s, dtype=float)
    if not higher_better:
        score = -score
    mask = np.isfinite(score)
    if mask.sum() < 10 or len(set(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], score[mask]))


def ap(y, s, higher_better: bool) -> float:
    score = np.asarray(s, dtype=float)
    if not higher_better:
        score = -score
    mask = np.isfinite(score)
    if mask.sum() < 10 or len(set(y[mask])) < 2:
        return float("nan")
    return float(average_precision_score(y[mask], score[mask]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=Path, default=Path("processed/ml/pose_rank_oof.csv"))
    p.add_argument("--out", type=Path, default=Path("results/pose_rank_ablation.json"))
    args = p.parse_args()
    frame = pd.read_csv(args.rank)
    y = frame["is_native"].to_numpy(dtype=int)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": int(len(frame)),
        "n_native": int(y.sum()),
        "vina_fused": False,
        "methods": {},
    }
    mapping = {
        "hybrid_total": False,
        "geo_rmsd_or": False,
        "gp_nll": False,
        "clash": False,
        "dist_harm": False,
        "cn_pen": False,
        "slef": False,
        "native_score_oof": True,
    }
    for col, higher in mapping.items():
        if col not in frame.columns:
            continue
        report["methods"][col] = {"auroc": auc(y, frame[col], higher), "ap": ap(y, frame[col], higher)}
    feat_cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    x = np.nan_to_num(frame[feat_cols].to_numpy(dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
    groups = frame["target_id"].astype(str).to_numpy() if "target_id" in frame.columns else np.arange(len(frame))
    oof = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=max(2, min(5, int(np.unique(groups).size))))
    for tr, va in gkf.split(x, y, groups):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=400, class_weight="balanced", C=0.5)),
            ]
        )
        pipe.fit(x[tr], y[tr])
        oof[va] = pipe.predict_proba(x[va])[:, 1]
    report["methods"]["learned_logreg"] = {"auroc": auc(y, oof, True), "ap": ap(y, oof, True)}
    if "hybrid_total" in report["methods"] and "native_score_oof" in report["methods"]:
        report["delta_ap_ml_minus_metgeo"] = (
            report["methods"]["native_score_oof"]["ap"] - report["methods"]["hybrid_total"]["ap"]
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
