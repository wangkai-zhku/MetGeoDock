#!/usr/bin/env python3
"""Train CoordNet on the agri-168 library (ligand-level) with epoch history.

Builds features from SMILES via metgeodock.fingerprint (warhead probes + Morgan
+ metal one-hot; map_overlap=0 without pose). Labels: per-target binary from
library_all.csv. Group key = ligand_id (never split the same molecule).

Outputs → processed/ml_runs/coordnet_agri_v1/
  features.csv
  history_torch_coordnet.csv
  metrics_by_target.json
  oof_predictions.csv
  roc_points_*.csv
  run_manifest.json

Does not mutate data/ or agri raw trees.
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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.fingerprint import COORDNET_COLUMNS, ligand_feature_vector  # noqa: E402
from metgeodock.nnscore import MLP  # noqa: E402

LOG = logging.getLogger("train_coordnet_agri_run")
TARGETS = ("hppd", "gs", "igpd")
TARGET_METAL = {"hppd": "fe2", "gs": "mn", "igpd": "mn"}


def _metrics(y: np.ndarray, s: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if y.min() == y.max():
        return {"auroc": float("nan"), "ap": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def build_feature_table(lib: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in lib.itertuples(index=False):
        for tid in TARGETS:
            metal = TARGET_METAL[tid]
            feats = ligand_feature_vector(metal, smiles=str(rec.smiles), ligand=None, include_map=False)
            label_col = f"label_{tid}"
            y = int(getattr(rec, label_col))
            row = {
                "ligand_id": str(rec.ligand_id),
                "target": tid,
                "metal": metal,
                "y": y,
                "group": str(rec.ligand_id),
                "smiles": str(rec.smiles),
            }
            row.update({c: float(feats.get(c, 0.0)) for c in COORDNET_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def train_one_target(
    frame: pd.DataFrame,
    tid: str,
    n_splits: int,
    epochs: int,
    seed: int,
    out: Path,
) -> dict:
    sub = frame[frame["target"] == tid].reset_index(drop=True)
    x = np.nan_to_num(sub[list(COORDNET_COLUMNS)].to_numpy(dtype=float), nan=0.0)
    y = sub["y"].to_numpy(dtype=int)
    groups = sub["group"].astype(str).to_numpy()
    n_groups = int(np.unique(groups).size)
    splits = max(2, min(n_splits, n_groups, int(y.sum()), int((1 - y).sum()) or 1))
    if int(y.sum()) < 2 or int((1 - y).sum()) < 2:
        LOG.warning("%s too few pos/neg for CV", tid)
        return {"target": tid, "skipped": True, "n": len(y), "n_pos": int(y.sum())}

    gkf = GroupKFold(n_splits=splits)
    oof = np.zeros(len(y), dtype=float)
    hist_rows: list[dict] = []
    torch.manual_seed(seed)
    fold_ok = 0

    for fold, (tr, va) in enumerate(gkf.split(x, y, groups)):
        if y[tr].sum() < 1 or y[va].sum() < 1 or (1 - y[tr]).sum() < 1:
            LOG.warning("%s fold=%s skipped (class missing in split)", tid, fold)
            continue
        fold_ok += 1
        mean, std = x[tr].mean(0), np.clip(x[tr].std(0), 1e-6, None)
        xtr, xva = (x[tr] - mean) / std, (x[va] - mean) / std
        ytr, yva = y[tr].astype(float), y[va].astype(float)
        model = MLP(x.shape[1], hidden=(128, 64, 32), dropout=0.2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos = max(float(ytr.sum()), 1.0)
        neg = max(float(len(ytr) - ytr.sum()), 1.0)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos]))
        loader = DataLoader(
            TensorDataset(torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
        )
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                va_logit = model(torch.tensor(xva, dtype=torch.float32)).numpy()
            va_prob = 1.0 / (1.0 + np.exp(-va_logit))
            vm = _metrics(yva.astype(int), va_prob)
            hist_rows.append(
                {
                    "target": tid,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_auroc": vm["auroc"],
                    "val_ap": vm["ap"],
                }
            )
        with torch.no_grad():
            oof[va] = 1.0 / (1.0 + np.exp(-model(torch.tensor(xva, dtype=torch.float32)).numpy()))

    hist_path = out / f"history_torch_coordnet_{tid}.csv"
    pd.DataFrame(hist_rows).to_csv(hist_path, index=False)
    if fold_ok == 0 or not hist_rows:
        return {"target": tid, "skipped": True, "reason": "no_valid_fold", "n": int(len(y)), "n_pos": int(y.sum())}
    # evaluate on rows that appeared in at least one validation fold with training
    # reconstruct covered indices from non-default: re-run coverage via oof assignment only on trained folds
    m = _metrics(y, oof)
    oof_df = sub[["ligand_id", "target", "metal", "y", "group"]].copy()
    oof_df["oof_proba"] = oof
    oof_df.to_csv(out / f"oof_{tid}.csv", index=False)
    if y.min() != y.max():
        fpr, tpr, _ = roc_curve(y, oof)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out / f"roc_points_{tid}.csv", index=False)
    LOG.info("%s auroc=%.4f ap=%.4f n=%s n_pos=%s folds_ok=%s", tid, m["auroc"], m["ap"], len(y), int(y.sum()), fold_ok)
    return {
        "target": tid,
        "skipped": False,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_splits": splits,
        "folds_ok": fold_ok,
        **m,
        "history_csv": str(hist_path.relative_to(ROOT)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, default=ROOT / "processed" / "agri_vs" / "library_all.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "processed" / "ml_runs" / "coordnet_agri_v1")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "train_coordnet_agri_run.log")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )

    lib = pd.read_csv(args.lib)
    LOG.info("building features n_mol=%s", len(lib))
    feats = build_feature_table(lib)
    feats.to_csv(args.out / "features.csv", index=False)

    by_target = {}
    hist_all = []
    for tid in TARGETS:
        by_target[tid] = train_one_target(feats, tid, args.n_splits, args.epochs, args.seed, args.out)
        hp = args.out / f"history_torch_coordnet_{tid}.csv"
        if hp.exists():
            hist_all.append(pd.read_csv(hp))
    if hist_all:
        pd.concat(hist_all, axis=0).to_csv(args.out / "history_torch_coordnet.csv", index=False)

    (args.out / "metrics_by_target.json").write_text(json.dumps(by_target, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lib": str(args.lib.relative_to(ROOT)),
        "out_dir": str(args.out.relative_to(ROOT)),
        "seed": args.seed,
        "epochs": args.epochs,
        "n_splits": args.n_splits,
        "n_feature_rows": int(len(feats)),
        "n_molecules": int(lib.shape[0]),
        "n_coordnet_dims": len(COORDNET_COLUMNS),
        "coordnet_columns_head": list(COORDNET_COLUMNS)[:20],
        "by_target": by_target,
        "note": "Ligand-level CoordNet; map_overlap=0 (no pose). GroupKFold by ligand_id.",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results" / "ml_run_coordnet_agri_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info("done %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
