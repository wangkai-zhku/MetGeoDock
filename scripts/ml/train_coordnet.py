#!/usr/bin/env python3
"""Train CoordNet (MLP) on ChEMBL metalloprotein inhibitor data.

Actives: pactivity >= 6. Decoys: pactivity <= 5. Split by target_id
(GroupKFold). Features are metal-warhead probes + donor counts + Morgan bits.
Baselines: ExtraTrees and a donor-count logistic model. Writes
processed/ml/coordnet.joblib. Does not mutate data/.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.fingerprint import COORDNET_COLUMNS, ligand_feature_vector  # noqa: E402
from metgeodock.nnscore import build_coordnet, coordnet_matrix, save_nn  # noqa: E402

LOG = logging.getLogger("train_coordnet")


def featurize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in frame.itertuples(index=False):
        feats = ligand_feature_vector(str(rec.metal), smiles=str(rec.canonical_smiles), include_map=False)
        feats["y"] = int(rec.y)
        feats["target_id"] = rec.target_id
        feats["metal"] = rec.metal
        feats["smiles"] = rec.canonical_smiles
        rows.append(feats)
    return pd.DataFrame(rows)


def grouped_metrics(model, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    n_groups = int(pd.Series(groups).nunique())
    n_splits = min(5, max(2, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    aucs, aps = [], []
    oof = np.zeros(len(y), dtype=float)
    for tr, te in cv.split(x, y, groups):
        model.fit(x[tr], y[tr])
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(x[te])[:, 1]
        else:
            pred = model.predict(x[te])
        oof[te] = pred
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], pred))
        aps.append(average_precision_score(y[te], pred))
    return {
        "oof_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "oof_ap": float(np.mean(aps)) if aps else float("nan"),
        "n_cv_folds_scored": int(len(aucs)),
        "n_samples": int(len(y)),
        "n_groups": n_groups,
        "n_pos": int(y.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chembl", type=Path, default=Path("processed/chembl/curated/all_targets_molecules.csv"))
    p.add_argument("--active-cut", type=float, default=6.0)
    p.add_argument("--decoy-cut", type=float, default=5.0)
    p.add_argument("--max-per-class-per-target", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-model", type=Path, default=Path("processed/ml/coordnet.joblib"))
    p.add_argument("--out-features", type=Path, default=Path("processed/ml/coordnet_features.csv"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/coordnet_report.json"))
    p.add_argument("--log", type=Path, default=Path("logs/train_coordnet.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    raw = pd.read_csv(args.chembl)
    raw = raw[raw["inconsistent"] == False].copy()  # noqa: E712
    raw = raw.dropna(subset=["canonical_smiles", "pactivity", "metal", "target_id"])
    rng = np.random.default_rng(args.seed)
    parts = []
    for tid, sub in raw.groupby("target_id"):
        act = sub[sub["pactivity"] >= args.active_cut]
        dec = sub[sub["pactivity"] <= args.decoy_cut]
        if len(act) < 8 or len(dec) < 8:
            continue
        if len(act) > args.max_per_class_per_target:
            act = act.iloc[rng.choice(len(act), args.max_per_class_per_target, replace=False)]
        if len(dec) > args.max_per_class_per_target:
            dec = dec.iloc[rng.choice(len(dec), args.max_per_class_per_target, replace=False)]
        act = act.assign(y=1)
        dec = dec.assign(y=0)
        parts.append(pd.concat([act, dec], axis=0))
    if not parts:
        raise SystemExit("no target with both actives and decoys")
    labeled = pd.concat(parts, axis=0).reset_index(drop=True)
    LOG.info("labeled n=%s targets=%s pos=%s", len(labeled), labeled["target_id"].nunique(), int(labeled["y"].sum()))
    feats = featurize_rows(labeled)
    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.out_features, index=False)
    x = coordnet_matrix(feats)
    y = feats["y"].to_numpy(dtype=int)
    groups = feats["target_id"].astype(str).to_numpy()
    models = {
        "coordnet_mlp": build_coordnet(),
        "extratrees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=args.seed, n_jobs=1),
        "donor_logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=400, random_state=args.seed)),
            ]
        ),
    }
    reports = {}
    for name, model in models.items():
        LOG.info("cv %s", name)
        if name == "donor_logreg":
            cols = [c for c in ("n_donor", "n_o", "n_n", "n_s", "coord_prior") if c in feats.columns]
            xd = np.nan_to_num(feats[cols].to_numpy(dtype=float))
            reports[name] = grouped_metrics(model, xd, y, groups)
        else:
            reports[name] = grouped_metrics(model, x, y, groups)
        LOG.info("%s %s", name, reports[name])
    final = build_coordnet()
    final.fit(x, y)
    save_nn(final, args.out_model, extra={"columns": list(COORDNET_COLUMNS), "seed": args.seed})
    blob = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "active_cut": args.active_cut,
        "decoy_cut": args.decoy_cut,
        "n_labeled": int(len(labeled)),
        "n_targets": int(labeled["target_id"].nunique()),
        "models": reports,
        "model_path": str(args.out_model),
        "split": "GroupKFold by target_id",
        "baseline_random_auroc": 0.5,
    }
    args.out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.out_json)


if __name__ == "__main__":
    main()
