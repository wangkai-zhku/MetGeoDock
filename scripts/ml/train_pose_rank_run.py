#!/usr/bin/env python3
"""Retrain pose-rank models into processed/ml_runs/pose_rank_v1/ with plottable logs.

Inputs (read-only): processed/ml/metgeo_rank_features.csv
Outputs:
  processed/ml_runs/pose_rank_v1/
    run_manifest.json
    metrics_by_model.json
    oof_predictions.csv
    history_torch_posenet.csv          # epoch, fold, train_loss, val_loss, val_auroc, val_ap
    history_sklearn_mlp.json           # loss_curve_ per fold
    roc_points.csv / pr_points.csv     # best model OOF curves
    feature_importance_rf.csv
    feature_corr_with_y.csv
    ablation_terms.json
    rmsd_vs_score_sample.csv

Split: GroupKFold on ``group`` (complex id). Public-10 never in this CSV.
Seed logged. Does not mutate data/ or ORI trees.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("train_pose_rank_run")

POSE_FEATURES = [
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
    "map_overlap",
    "contact",
    "centroid_pen",
    "n_water",
    "semi_flex",
    "probe_hydroxamate",
    "probe_carboxylate",
    "probe_imidazole",
    "probe_thiol",
    "probe_phosphate",
    "metal_zn",
    "metal_ca",
    "metal_mg",
    "metal_mn",
    "metal_fe2",
    "metal_fe3",
    "metal_cu",
    "metal_co",
    "metal_cd",
    "metal_ni",
]

ABLATION_TERMS = [
    "hybrid_total",
    "geo_rmsd_or",
    "gp_nll",
    "clash",
    "dist_harm",
    "cn_pen",
    "map_overlap",
]


def _xy(frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    use = [c for c in cols if c in frame.columns]
    x = np.nan_to_num(frame[use].to_numpy(dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
    y = frame["y"].to_numpy(dtype=int)
    return x, y, use


def _metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.min() == y_true.max():
        return {"auroc": float("nan"), "ap": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "ap": float(average_precision_score(y_true, score)),
    }


def train_sklearn_models(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_names: list[str],
    n_splits: int,
    seed: int,
    out: Path,
) -> tuple[dict, np.ndarray, str]:
    gkf = GroupKFold(n_splits=n_splits)
    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=800, class_weight="balanced", C=0.5, random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    oof = {name: np.zeros(len(y), dtype=float) for name in models}
    mlp_hist: dict = {"folds": []}
    for fold, (tr, va) in enumerate(gkf.split(x, y, groups)):
        for name, proto in models.items():
            # fresh clone via re-init
            if name == "logreg":
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=800, class_weight="balanced", C=0.5, random_state=seed + fold)),
                    ]
                )
            elif name == "rf":
                model = RandomForestClassifier(
                    n_estimators=400, max_depth=12, min_samples_leaf=4, class_weight="balanced_subsample",
                    random_state=seed + fold, n_jobs=1,
                )
            elif name == "extratrees":
                model = ExtraTreesClassifier(
                    n_estimators=400, max_depth=12, min_samples_leaf=4, class_weight="balanced_subsample",
                    random_state=seed + fold, n_jobs=1,
                )
            else:
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            MLPClassifier(
                                hidden_layer_sizes=(64, 32),
                                activation="relu",
                                max_iter=200,
                                early_stopping=True,
                                validation_fraction=0.15,
                                random_state=seed + fold,
                            ),
                        ),
                    ]
                )
            model.fit(x[tr], y[tr])
            if hasattr(model, "predict_proba"):
                oof[name][va] = model.predict_proba(x[va])[:, 1]
            else:
                oof[name][va] = model.predict_proba(x[va])[:, 1]
            if name == "mlp":
                clf = model.named_steps["clf"]
                mlp_hist["folds"].append(
                    {
                        "fold": fold,
                        "loss_curve": [float(v) for v in getattr(clf, "loss_curve_", [])],
                        "validation_scores": [float(v) for v in getattr(clf, "validation_scores_", [])]
                        if hasattr(clf, "validation_scores_")
                        else [],
                        "n_iter": int(getattr(clf, "n_iter_", 0)),
                    }
                )
    (out / "history_sklearn_mlp.json").write_text(json.dumps(mlp_hist, indent=2) + "\n", encoding="utf-8")

    metrics = {}
    best_name, best_ap = "extratrees", -1.0
    for name, scores in oof.items():
        m = _metrics(y, scores)
        metrics[name] = {**m, "n": int(len(y)), "n_pos": int(y.sum())}
        if m["ap"] == m["ap"] and m["ap"] > best_ap:
            best_ap, best_name = m["ap"], name
        LOG.info("sklearn %s auroc=%.4f ap=%.4f", name, m["auroc"], m["ap"])
    if best_name not in oof:
        best_name = next(iter(oof))

    # RF importance on full data
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4, class_weight="balanced_subsample",
        random_state=seed, n_jobs=1,
    )
    rf.fit(x, y)
    imp = pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp.to_csv(out / "feature_importance_rf.csv", index=False)

    return metrics, oof[best_name], best_name


def train_torch_posenet(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    epochs: int,
    out: Path,
) -> tuple[dict, np.ndarray]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from metgeodock.nnscore import MLP

    torch.manual_seed(seed)
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    hist_rows: list[dict] = []
    device = torch.device("cpu")

    for fold, (tr, va) in enumerate(gkf.split(x, y, groups)):
        mean = x[tr].mean(axis=0)
        std = np.clip(x[tr].std(axis=0), 1e-6, None)
        xtr = (x[tr] - mean) / std
        xva = (x[va] - mean) / std
        ytr = y[tr].astype(float)
        yva = y[va].astype(float)

        model = MLP(x.shape[1], hidden=(128, 64, 32), dropout=0.2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = max(float(ytr.sum()), 1.0)
        neg = max(float(len(ytr) - ytr.sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        ds = TensorDataset(
            torch.tensor(xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for xb, yb in loader:
                opt.zero_grad()
                logit = model(xb)
                loss = crit(logit, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                tr_logit = model(torch.tensor(xtr, dtype=torch.float32)).numpy()
                va_logit = model(torch.tensor(xva, dtype=torch.float32)).numpy()
            tr_prob = 1.0 / (1.0 + np.exp(-tr_logit))
            va_prob = 1.0 / (1.0 + np.exp(-va_logit))
            tr_m = _metrics(ytr.astype(int), tr_prob)
            va_m = _metrics(yva.astype(int), va_prob)
            # train BCE mean
            hist_rows.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_auroc": va_m["auroc"],
                    "val_ap": va_m["ap"],
                    "train_auroc": tr_m["auroc"],
                    "train_ap": tr_m["ap"],
                }
            )
        with torch.no_grad():
            oof[va] = 1.0 / (1.0 + np.exp(-model(torch.tensor(xva, dtype=torch.float32)).numpy()))
        LOG.info("torch fold=%s oof_partial auroc=%.4f", fold, _metrics(y[va], oof[va])["auroc"])

    pd.DataFrame(hist_rows).to_csv(out / "history_torch_posenet.csv", index=False)
    # save last-fold style full-data checkpoint for reuse
    mean = x.mean(axis=0)
    std = np.clip(x.std(axis=0), 1e-6, None)
    xs = (x - mean) / std
    model = MLP(x.shape[1], hidden=(128, 64, 32), dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos]))
    xt = torch.tensor(xs, dtype=torch.float32)
    yt = torch.tensor(y.astype(float), dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(xt), yt)
        loss.backward()
        opt.step()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "hidden": (128, 64, 32),
            "dropout": 0.2,
            "feature_names": None,
            "kind": "posenet_ml_run",
        },
        out / "posenet_full.pt",
    )
    return _metrics(y, oof), oof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=ROOT / "processed" / "ml" / "metgeo_rank_features.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "processed" / "ml_runs" / "pose_rank_v1")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "train_pose_rank_run.log")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )

    frame = pd.read_csv(args.features)
    if "group" not in frame.columns:
        frame["group"] = frame["pdb_id"].astype(str) + "_" + frame["rec_id"].astype(str)
    x, y, feat_names = _xy(frame, POSE_FEATURES)
    groups = frame["group"].astype(str).to_numpy()
    n_groups = int(np.unique(groups).size)
    n_splits = max(2, min(args.n_splits, n_groups))
    LOG.info("n=%s n_pos=%s n_groups=%s n_feat=%s splits=%s", len(y), int(y.sum()), n_groups, len(feat_names), n_splits)

    # feature correlation with label
    corr_rows = []
    for i, name in enumerate(feat_names):
        if y.std() == 0 or x[:, i].std() == 0:
            c = float("nan")
        else:
            c = float(np.corrcoef(x[:, i], y.astype(float))[0, 1])
        corr_rows.append({"feature": name, "pearson_y": c})
    corr_df = pd.DataFrame(corr_rows)
    corr_df["abs_pearson"] = corr_df["pearson_y"].abs()
    corr_df.sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"]).to_csv(
        args.out / "feature_corr_with_y.csv", index=False
    )

    sk_metrics, best_oof, best_name = train_sklearn_models(
        x, y, groups, feat_names, n_splits, args.seed, args.out
    )
    torch_metrics, torch_oof = train_torch_posenet(x, y, groups, n_splits, args.seed, args.epochs, args.out)
    sk_metrics["torch_posenet"] = {**torch_metrics, "n": int(len(y)), "n_pos": int(y.sum())}

    # ablation single-term AUROC (score = -term for lower-better physics terms; use as-is for map_overlap)
    abl = {}
    for term in ABLATION_TERMS:
        if term not in frame.columns:
            continue
        raw = frame[term].to_numpy(dtype=float)
        # native poses should have low hybrid / geo; invert those
        score = -raw if term in {"hybrid_total", "geo_rmsd_or", "gp_nll", "clash", "dist_harm", "cn_pen"} else raw
        abl[term] = _metrics(y, score)
    abl["best_sklearn"] = sk_metrics[best_name]
    abl["torch_posenet"] = torch_metrics
    (args.out / "ablation_terms.json").write_text(json.dumps(abl, indent=2) + "\n", encoding="utf-8")

    # OOF table
    oof_df = frame[["pdb_id", "rec_id", "metal", "group", "rmsd_angstrom", "y", "hybrid_total", "geo_rmsd_or"]].copy()
    oof_df["oof_best_sklearn"] = best_oof
    oof_df["oof_torch_posenet"] = torch_oof
    oof_df["best_sklearn_name"] = best_name
    oof_df.to_csv(args.out / "oof_predictions.csv", index=False)

    # ROC / PR for torch (usually best for curves storytelling)
    fpr, tpr, _ = roc_curve(y, torch_oof)
    prec, rec, _ = precision_recall_curve(y, torch_oof)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(args.out / "roc_points_torch.csv", index=False)
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(args.out / "pr_points_torch.csv", index=False)
    fpr2, tpr2, _ = roc_curve(y, best_oof)
    pd.DataFrame({"fpr": fpr2, "tpr": tpr2}).to_csv(args.out / "roc_points_sklearn_best.csv", index=False)

    # RMSD vs score sample for scatter
    sample = oof_df.sample(n=min(800, len(oof_df)), random_state=args.seed)
    sample.to_csv(args.out / "rmsd_vs_score_sample.csv", index=False)

    (args.out / "metrics_by_model.json").write_text(json.dumps(sk_metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features_csv": str(args.features.relative_to(ROOT)),
        "out_dir": str(args.out.relative_to(ROOT)),
        "seed": args.seed,
        "n_splits": n_splits,
        "epochs_torch": args.epochs,
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "n_groups": n_groups,
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "best_sklearn": best_name,
        "metrics": sk_metrics,
        "note": "GroupKFold by complex; public-10 held out upstream in feature build.",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # also mirror summary into results/
    results = ROOT / "results" / "ml_run_pose_rank_v1.json"
    results.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
