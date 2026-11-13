#!/usr/bin/env python3
"""Train CoordNet and PoseNet with PyTorch (no scikit-learn).

CoordNet: ChEMBL p>=6 vs p<=5, grouped by target_id.
PoseNet: native vs sampled poses, grouped by pdb_id.
Writes processed/ml/coordnet.pt and processed/ml/posenet.pt.
Compares against random and a linear donor baseline implemented in torch.
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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.fingerprint import COORDNET_COLUMNS  # noqa: E402
from metgeodock.nnscore import MLP, coordnet_matrix, save_nn, standardize  # noqa: E402

LOG = logging.getLogger("train_torch_nets")
POSE_FEATS = [
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


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(tpr, fpr))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    tps = np.cumsum(ys)
    prec = tps / np.arange(1, len(ys) + 1)
    rec_delta = ys / n_pos
    return float(np.sum(prec * rec_delta))


def group_kfold(groups: np.ndarray, n_splits: int, seed: int = 42):
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for i in range(n_splits):
        te_g = set(folds[i].tolist())
        te = np.array([g in te_g for g in groups])
        yield np.where(~te)[0], np.where(te)[0]


def fit_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray | None,
    y_va: np.ndarray | None,
    hidden: tuple[int, ...],
    dropout: float,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    pos_weight: bool = False,
) -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std[std < 1.0e-6] = 1.0
    xt = torch.tensor(standardize(x_tr, mean, std), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    model = MLP(xt.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1.0e-4)
    pw = None
    if pos_weight:
        n_pos = float(max(y_tr.sum(), 1))
        n_neg = float(max(len(y_tr) - y_tr.sum(), 1))
        pw = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(256, len(xt)), shuffle=True)
    best_state = None
    best_auc = -1.0
    bad = 0
    use_val = x_va is not None and y_va is not None and len(np.unique(y_va)) > 1
    if use_val:
        xv = torch.tensor(standardize(x_va, mean, std), dtype=torch.float32)
        yv = y_va
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if not use_val:
            continue
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(xv)).numpy()
        auc = auroc(yv, pred)
        if np.isfinite(auc) and auc > best_auc + 1.0e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    extra = {"mean": mean, "std": std, "hidden": list(hidden), "dropout": dropout, "best_val_auroc": best_auc}
    return model, extra


def oof_and_fit(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    hidden: tuple[int, ...],
    dropout: float,
    seed: int,
    use_val: bool = True,
    pos_weight: bool = False,
    epochs: int = 80,
) -> tuple[dict, dict]:
    n_groups = int(len(set(groups.tolist())))
    n_splits = min(5, max(2, n_groups))
    oof = np.zeros(len(y), dtype=float)
    aucs, aps = [], []
    for fold, (tr, te) in enumerate(group_kfold(groups, n_splits, seed), start=1):
        if use_val:
            rng = np.random.default_rng(seed + fold)
            tr_g = np.unique(groups[tr])
            rng.shuffle(tr_g)
            n_va = max(1, int(0.15 * len(tr_g)))
            va_g = set(tr_g[:n_va].tolist())
            va_mask = np.array([g in va_g for g in groups[tr]])
            tr_idx = tr[~va_mask] if va_mask.any() and (~va_mask).any() else tr
            va_idx = tr[va_mask] if va_mask.any() and (~va_mask).any() else None
            x_va = x[va_idx] if va_idx is not None else None
            y_va = y[va_idx] if va_idx is not None else None
        else:
            tr_idx = tr
            x_va, y_va = None, None
        model, extra = fit_mlp(
            x[tr_idx], y[tr_idx], x_va, y_va, hidden, dropout, seed + fold, epochs, 12, 1.0e-3, pos_weight=pos_weight
        )
        xs = torch.tensor(standardize(x[te], extra["mean"], extra["std"]), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(xs)).numpy()
        oof[te] = pred
        if len(np.unique(y[te])) > 1:
            aucs.append(auroc(y[te], pred))
            aps.append(average_precision(y[te], pred))
        LOG.info("fold %s auroc=%s", fold, aucs[-1] if aucs else None)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    if use_val:
        va_g = set(uniq[: max(1, int(0.15 * len(uniq)))].tolist())
        va = np.array([g in va_g for g in groups])
        tr = ~va
        if not tr.any() or not va.any():
            tr = np.ones(len(y), dtype=bool)
            va = tr
        model, extra = fit_mlp(x[tr], y[tr], x[va], y[va], hidden, dropout, seed, epochs, 12, 1.0e-3, pos_weight=pos_weight)
    else:
        model, extra = fit_mlp(x, y, None, None, hidden, dropout, seed, epochs, 12, 1.0e-3, pos_weight=pos_weight)
    bundle = {
        "state_dict": model.state_dict(),
        "mean": extra["mean"],
        "std": extra["std"],
        "hidden": list(hidden),
        "dropout": dropout,
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "seed": seed,
    }
    metrics = {
        "oof_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "oof_ap": float(np.mean(aps)) if aps else float("nan"),
        "n_cv_folds_scored": int(len(aucs)),
        "n_samples": int(len(y)),
        "n_groups": n_groups,
        "n_pos": int(y.sum()),
    }
    return bundle, metrics


def donor_baseline(x_donor: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """One-layer logistic in torch on donor counts."""
    n_groups = int(len(set(groups.tolist())))
    n_splits = min(5, max(2, n_groups))
    oof = np.zeros(len(y), dtype=float)
    aucs, aps = [], []
    for tr, te in group_kfold(groups, n_splits, seed):
        model, extra = fit_mlp(
            x_donor[tr], y[tr], x_donor[tr], y[tr], hidden=(), dropout=0.0, seed=seed, epochs=40, patience=8, lr=5.0e-3
        )
        xs = torch.tensor(standardize(x_donor[te], extra["mean"], extra["std"]), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(xs)).numpy()
        oof[te] = pred
        if len(np.unique(y[te])) > 1:
            aucs.append(auroc(y[te], pred))
            aps.append(average_precision(y[te], pred))
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
    p.add_argument("--coordnet-features", type=Path, default=Path("processed/ml/coordnet_features.csv"))
    p.add_argument("--pose-features", type=Path, default=Path("processed/ml/pose_rank_features.csv"))
    p.add_argument("--out-coordnet", type=Path, default=Path("processed/ml/coordnet.pt"))
    p.add_argument("--out-posenet", type=Path, default=Path("processed/ml/posenet.pt"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/torch_report.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", type=Path, default=Path("logs/train_torch_nets.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    LOG.info("torch %s", torch.__version__)
    feats = pd.read_csv(args.coordnet_features)
    x = coordnet_matrix(feats)
    y = feats["y"].to_numpy(dtype=int)
    groups = feats["target_id"].astype(str).to_numpy()
    LOG.info("coordnet n=%s pos=%s groups=%s", len(y), int(y.sum()), len(set(groups.tolist())))
    bundle, metrics = oof_and_fit(x, y, groups, hidden=(128, 64, 32), dropout=0.20, seed=args.seed)
    bundle["columns"] = list(COORDNET_COLUMNS)
    save_nn(bundle, args.out_coordnet)
    donor_cols = [c for c in ("n_donor", "n_o", "n_n", "n_s", "coord_prior") if c in feats.columns]
    donor = donor_baseline(np.nan_to_num(feats[donor_cols].to_numpy(dtype=float)), y, groups, args.seed)

    pose_metrics = {}
    if args.pose_features.exists():
        pose = pd.read_csv(args.pose_features)
        cols = [c for c in POSE_FEATS if c in pose.columns]
        xp = np.nan_to_num(pose[cols].to_numpy(dtype=float), nan=0.0)
        yp = pose["is_native"].to_numpy(dtype=int)
        gp = pose["pdb_id"].astype(str).to_numpy()
        LOG.info("posenet n=%s pos=%s groups=%s", len(yp), int(yp.sum()), len(set(gp.tolist())))
        pbundle, pose_metrics = oof_and_fit(
            xp, yp, gp, hidden=(64, 32), dropout=0.10, seed=args.seed, use_val=False, pos_weight=True, epochs=60
        )
        pbundle["columns"] = cols
        save_nn(pbundle, args.out_posenet)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "coordnet": {**metrics, "model": str(args.out_coordnet), "hidden": [128, 64, 32]},
        "donor_logistic_torch": donor,
        "posenet": {**pose_metrics, "model": str(args.out_posenet)} if pose_metrics else {},
        "sklearn_mlp_previous_oof_auroc": 0.6261199091219717,
        "extratrees_previous_oof_auroc": 0.6771910270487415,
        "baseline_random_auroc": 0.5,
        "torch_beats_sklearn_mlp": bool(metrics["oof_auroc"] > 0.6261199091219717) if np.isfinite(metrics["oof_auroc"]) else None,
        "torch_beats_extratrees": bool(metrics["oof_auroc"] > 0.6771910270487415) if np.isfinite(metrics["oof_auroc"]) else None,
        "torch_beats_random": bool(metrics["oof_auroc"] > 0.5) if np.isfinite(metrics["oof_auroc"]) else None,
        "posenet_sklearn_previous_oof_auroc": 0.7840447048611112,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.out_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
