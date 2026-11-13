#!/usr/bin/env python3
"""Train PoseScore on GPDOCK OUTPUT_DOCK same-ligand models vs crystal RMSD.

Does not mutate ORI_PDB_LIGAND or OUTPUT_DOCK. Labels y=1 if heavy-atom
RMSD <= 2 A. Groups are pdb_id so CV does not leak the same complex.
Includes SW water protocols. Writes processed/ml/posescore.pt.
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

from metgeodock.coordination import detect_coordination  # noqa: E402
from metgeodock.dock import redock_rmsd  # noqa: E402
from metgeodock.forcefield import ff_energy  # noqa: E402
from metgeodock.io import WATER_RESNAMES, iter_pdb_models, load_molecule, merge_with_waters, read_pdb, select_atoms  # noqa: E402
from metgeodock.nnscore import MLP, save_nn, standardize  # noqa: E402
from metgeodock.posescore import POSE_COLUMNS, pose_feature_vector  # noqa: E402
from metgeodock.scoring import HybridScore  # noqa: E402

LOG = logging.getLogger("train_posescore")


def _ligand_only(mol):
    keep = [i for i in range(len(mol)) if str(mol.resnames[i]).strip().upper() not in WATER_RESNAMES]
    if not keep:
        return mol
    return select_atoms(mol, keep, name=mol.name)


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


def group_kfold(groups: np.ndarray, n_splits: int, seed: int = 42):
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for i in range(n_splits):
        te_g = set(folds[i].tolist())
        te = np.array([g in te_g for g in groups])
        yield np.where(~te)[0], np.where(te)[0]


def _hybrid(ff: dict, sphere) -> HybridScore:
    return HybridScore(
        total=float(ff["total"]),
        vina=0.0,
        slef=0.0,
        geo=float(ff["geo"]),
        gp=float(ff["coord"]),
        clash=float(ff["steric"]),
        rmsd_or=float(ff["geo"]),
        n_coord_lig=int(ff["n_coord_lig"]),
        n_coord_pro=int(sphere.n_protein),
        dist=float(ff["harm"]),
        cn=0.0,
        terms=dict(ff),
    )


def _sample_rows(joined: pd.DataFrame, output: pd.DataFrame, per_subset: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for subset in ("SW", "SM", "MM"):
        ori = joined[(joined["subset"] == subset) & (joined["complete"] == 1)].copy()
        out = output[output["subset"] == subset].copy()
        if subset == "SW":
            out = out[out["protocol"].isin(("out", "wat_1"))]
        else:
            out = out[out["protocol"] == "out"]
        key = ["metal", "pdb_id", "rec_id"]
        merged = ori.merge(out, on=key + ["subset"], how="inner", suffixes=("", "_out"))
        if merged.empty:
            continue
        cx = merged[key].drop_duplicates()
        n = min(int(per_subset), len(cx))
        pick = cx.iloc[rng.choice(len(cx), size=n, replace=False)]
        parts.append(merged.merge(pick, on=key, how="inner"))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_features(args) -> pd.DataFrame:
    joined = pd.read_csv(args.joined)
    output = pd.read_csv(args.output_inv)
    sample = _sample_rows(joined, output, args.per_subset, args.seed)
    LOG.info("sampled files=%s complexes≈%s", len(sample), sample.groupby(["subset", "pdb_id", "rec_id"]).ngroups)
    rows = []
    for rec in sample.itertuples(index=False):
        try:
            protein = read_pdb(rec.protein_path)
            water_path = getattr(rec, "water_path", "") or ""
            if str(water_path) and Path(str(water_path)).exists() and str(rec.protocol) in {"wat_1", "wat_2", "out"}:
                if rec.subset == "SW" and str(rec.protocol) != "out":
                    protein = merge_with_waters(protein, read_pdb(water_path))
            crystal = load_molecule(rec.ligand_path)
            pose_path = rec.pose_path
            sphere = detect_coordination(protein, metal=rec.metal)
            n_kept = 0
            for _i, pose in iter_pdb_models(pose_path):
                lig = _ligand_only(pose)
                if crystal.bonds and len(lig) == len(crystal):
                    lig.bonds = list(crystal.bonds)
                rmsd = redock_rmsd(lig, crystal)
                if not np.isfinite(rmsd):
                    continue
                ff = ff_energy(protein, lig, sphere)
                sc = _hybrid(ff, sphere)
                sc.terms["n_water"] = float(sphere.n_water)
                sc.terms["n_other_metal"] = float(sphere.n_other_metal)
                feats = pose_feature_vector(sc, lig)
                feats.update(
                    {
                        "subset": rec.subset,
                        "pdb_id": rec.pdb_id,
                        "rec_id": str(rec.rec_id),
                        "metal": rec.metal,
                        "protocol": rec.protocol,
                        "gpdock_label": rec.gpdock_label,
                        "rmsd_angstrom": float(rmsd),
                        "y": int(rmsd <= args.threshold),
                    }
                )
                rows.append(feats)
                n_kept += 1
                if n_kept >= args.max_models:
                    break
        except Exception as exc:  # noqa: BLE001
            LOG.warning("skip %s %s: %s", rec.pdb_id, rec.pose_path, exc)
            continue
    return pd.DataFrame(rows)


def fit_mlp(x_tr, y_tr, hidden, dropout, seed, epochs, pos_weight: bool) -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std[std < 1.0e-6] = 1.0
    xt = torch.tensor(standardize(x_tr, mean, std), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    model = MLP(xt.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    pw = None
    if pos_weight:
        n_pos = float(max(y_tr.sum(), 1))
        n_neg = float(max(len(y_tr) - y_tr.sum(), 1))
        pw = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(256, len(xt)), shuffle=True)
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model, {"mean": mean, "std": std, "hidden": list(hidden), "dropout": dropout}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joined", type=Path, default=Path("processed/gpdock_orig/inventory_joined.csv"))
    p.add_argument("--output-inv", type=Path, default=Path("processed/gpdock_orig/inventory_output.csv"))
    p.add_argument("--features", type=Path, default=Path("processed/ml/posescore_features.csv"))
    p.add_argument("--out", type=Path, default=Path("processed/ml/posescore.pt"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/posescore_report.json"))
    p.add_argument("--log", type=Path, default=Path("logs/train_posescore.log"))
    p.add_argument("--per-subset", type=int, default=80)
    p.add_argument("--max-models", type=int, default=6)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    if args.rebuild or not args.features.exists():
        feats = build_features(args)
        args.features.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(args.features, index=False)
        LOG.info("wrote %s n=%s", args.features, len(feats))
    else:
        feats = pd.read_csv(args.features)
        LOG.info("loaded %s n=%s", args.features, len(feats))
    if feats.empty:
        raise SystemExit("no PoseScore features")
    cols = [c for c in POSE_COLUMNS if c in feats.columns]
    x = np.nan_to_num(feats[cols].to_numpy(dtype=float), nan=0.0)
    y = feats["y"].to_numpy(dtype=int)
    groups = feats["pdb_id"].astype(str).to_numpy()
    n_groups = int(len(set(groups.tolist())))
    n_splits = min(5, max(2, n_groups))
    oof = np.zeros(len(y), dtype=float)
    aucs = []
    for fold, (tr, te) in enumerate(group_kfold(groups, n_splits, args.seed), start=1):
        model, extra = fit_mlp(x[tr], y[tr], (64, 32), 0.10, args.seed + fold, args.epochs, pos_weight=True)
        xs = torch.tensor(standardize(x[te], extra["mean"], extra["std"]), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(xs)).numpy()
        oof[te] = pred
        if len(np.unique(y[te])) > 1:
            aucs.append(auroc(y[te], pred))
        LOG.info("fold %s auroc=%s", fold, aucs[-1] if aucs else None)
    model, extra = fit_mlp(x, y, (64, 32), 0.10, args.seed, args.epochs, pos_weight=True)
    bundle = {
        "state_dict": model.state_dict(),
        "mean": extra["mean"],
        "std": extra["std"],
        "hidden": [64, 32],
        "dropout": 0.10,
        "columns": cols,
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "seed": args.seed,
        "kind": "posescore",
    }
    save_nn(bundle, args.out)
    by_subset = {}
    if "subset" in feats.columns:
        for subset, sub in feats.groupby("subset"):
            by_subset[str(subset)] = {
                "n": int(len(sub)),
                "n_pos": int(sub["y"].sum()),
                "n_groups": int(sub["pdb_id"].nunique()),
            }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.out),
        "features": str(args.features),
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_groups": n_groups,
        "n_water_rows": int((feats["n_water"] > 0).sum()) if "n_water" in feats.columns else 0,
        "oof_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_cv_folds_scored": int(len(aucs)),
        "by_subset": by_subset,
        "columns": cols,
        "label": "OUTPUT_DOCK model RMSD <= 2 A vs ORI crystal",
        "grouping": "pdb_id",
        "sklearn": False,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s oof_auroc=%s", args.out_json, report["oof_auroc"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
