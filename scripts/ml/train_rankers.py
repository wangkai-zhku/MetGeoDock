#!/usr/bin/env python3
"""Train several PyTorch pose-rankers on MetGeoDock samples (no sklearn).

Losses: BCE, RMSD regression, RankNet, ListNet, ListMLE.
Architectures: linear, MLP, residual, FiLM, GeoList residual (custom).
Select by grouped OOF listwise top-1 among complexes that have a success.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.nnscore import standardize  # noqa: E402
from metgeodock.posescore import POSE_COLUMNS  # noqa: E402
from metgeodock.rankers import build_model, forward_ranker, save_ranker  # noqa: E402

LOG = logging.getLogger("train_rankers")

RECIPES: tuple[dict, ...] = (
    {"name": "linear_ranknet", "arch": "linear", "loss": "ranknet", "epochs": 50},
    {"name": "mlp_bce", "arch": "mlp", "loss": "bce", "epochs": 50, "hidden": [64, 32]},
    {"name": "mlp_rmsd", "arch": "mlp", "loss": "rmsd", "epochs": 50, "hidden": [64, 32]},
    {"name": "resmlp_ranknet", "arch": "residual", "loss": "ranknet", "epochs": 60},
    {"name": "resmlp_listnet", "arch": "residual", "loss": "listnet", "epochs": 60},
    {"name": "film_ranknet", "arch": "film", "loss": "ranknet", "epochs": 60},
    {"name": "geolist_ranknet", "arch": "geolist", "loss": "ranknet", "epochs": 70},
    {"name": "geolist_listnet", "arch": "geolist", "loss": "listnet", "epochs": 70},
    {"name": "deepres_listmle", "arch": "deepres", "loss": "listmle", "epochs": 70},
)


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


def group_kfold(groups: np.ndarray, n_splits: int, seed: int):
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for i in range(n_splits):
        te_g = set(folds[i].tolist())
        te = np.array([g in te_g for g in groups])
        yield np.where(~te)[0], np.where(te)[0]


def listwise_rates(frame: pd.DataFrame, score_col: str, lower_better: bool) -> dict:
    hits1, hits3, ceil = [], [], []
    for _, sub in frame.groupby("group", sort=False):
        if len(sub) < 2:
            continue
        y = sub["y"].to_numpy(dtype=int)
        if int(y.sum()) == 0:
            continue
        ceil.append(1.0)
        s = sub[score_col].to_numpy(dtype=float)
        order = np.argsort(s if lower_better else -s, kind="mergesort")
        yo = y[order]
        hits1.append(int(yo[0] == 1))
        hits3.append(int(yo[:3].max() == 1))
    if not hits1:
        return {"n": 0, "top1": None, "top3": None, "sampled": None}
    return {
        "n": int(len(hits1)),
        "top1": float(np.mean(hits1)),
        "top3": float(np.mean(hits3)),
        "sampled": float(np.mean(ceil)),
    }


def _pairs_loss(s: torch.Tensor, rmsd: torch.Tensor, margin: float = 0.30) -> torch.Tensor:
    if s.numel() < 2:
        return s.sum() * 0.0
    diff = s[:, None] - s[None, :]
    better = (rmsd[:, None] + margin) < rmsd[None, :]
    if not bool(better.any()):
        return s.sum() * 0.0
    return torch.nn.functional.softplus(diff[better]).mean()


def _listnet_loss(s: torch.Tensor, rmsd: torch.Tensor, tau: float = 1.50) -> torch.Tensor:
    u = -s
    rel = torch.softmax(-rmsd / tau, dim=0)
    pred = torch.softmax(u, dim=0)
    return -(rel * torch.log(pred.clamp_min(1.0e-8))).sum()


def _listmle_loss(s: torch.Tensor, rmsd: torch.Tensor) -> torch.Tensor:
    u = -s
    order = torch.argsort(rmsd)
    loss = s.sum() * 0.0
    n = int(order.numel())
    for k in range(n):
        rest = u[order[k:]]
        loss = loss + (torch.logsumexp(rest, 0) - u[order[k]])
    return loss / max(n, 1)


def _score_kind(loss: str) -> str:
    if loss == "bce":
        return "neg_logit"
    if loss == "rmsd":
        return "rmsd"
    return "energy"


def _lower_better(kind: str) -> bool:
    return kind != "logit"


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    rmsd: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
    recipe: dict,
    seed: int,
    cv_groups: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    torch.manual_seed(seed)
    arch = recipe["arch"]
    loss_name = recipe["loss"]
    extra = {
        "hidden": recipe.get("hidden", (64, 32) if arch == "mlp" else 96),
        "dropout": 0.10,
        "depth": 3,
        "n_metal": int(sum(c.startswith("metal_") for c in columns)),
        "hybrid_index": int(columns.index("hybrid_total")) if "hybrid_total" in columns else 0,
    }
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-6] = 1.0
    xs = torch.tensor(standardize(x, mean, std), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    rt = torch.tensor(rmsd, dtype=torch.float32)
    model = build_model(arch, xs.shape[1], extra)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    bce = nn.BCEWithLogitsLoss()
    uniq = list(dict.fromkeys(groups.tolist()))
    epochs = int(recipe.get("epochs", 50))
    for _epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + _epoch)
        order = rng.permutation(len(uniq))
        for gi in order:
            g = uniq[int(gi)]
            idx = np.where(groups == g)[0]
            if idx.size < 2 and loss_name != "bce" and loss_name != "rmsd":
                continue
            xb = xs[idx]
            pred = forward_ranker(model, xb, columns, arch)
            if loss_name == "bce":
                loss = bce(pred, yt[idx])
            elif loss_name == "rmsd":
                loss = nn.functional.smooth_l1_loss(pred, rt[idx])
            elif loss_name == "listnet":
                loss = _listnet_loss(pred, rt[idx])
            elif loss_name == "listmle":
                loss = _listmle_loss(pred, rt[idx])
            else:
                loss = _pairs_loss(pred, rt[idx])
            if float(loss.detach()) == 0.0:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_all = forward_ranker(model, xs, columns, arch).cpu().numpy()
    kind = _score_kind(loss_name)
    bundle = {
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "arch": arch,
        "extra": extra,
        "columns": columns,
        "score_kind": kind,
        "loss": loss_name,
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "seed": seed,
        "name": recipe["name"],
    }
    return bundle, pred_all


def oof_for_recipe(x, y, rmsd, groups, columns, recipe, seed, cv_groups=None) -> tuple[np.ndarray, dict]:
    split_g = cv_groups if cv_groups is not None else groups
    n_groups = int(len(set(split_g.tolist())))
    n_splits = min(5, max(2, n_groups))
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(group_kfold(split_g, n_splits, seed), start=1):
        bundle, _ = train_one(x[tr], y[tr], rmsd[tr], groups[tr], columns, recipe, seed + fold)
        xs = torch.tensor(standardize(x[te], bundle["mean"], bundle["std"]), dtype=torch.float32)
        model = build_model(bundle["arch"], xs.shape[1], bundle["extra"])
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        with torch.no_grad():
            oof[te] = forward_ranker(model, xs, columns, bundle["arch"]).cpu().numpy()
        LOG.info("%s fold %s", recipe["name"], fold)
    return oof, {"n_splits": n_splits}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("processed/ml/metgeo_rank_features.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/ml/rankers"))
    p.add_argument("--best", type=Path, default=Path("processed/ml/ranker_best.pt"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ml/ranker_compare.json"))
    p.add_argument("--log", type=Path, default=Path("logs/train_rankers.log"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipes", default="all", help="comma names or all")
    p.add_argument("--extra-features", type=Path, default=None, help="optional extra CSV to concat")
    args = p.parse_args()
    wanted_r = {x.strip() for x in str(args.recipes).split(",") if x.strip()}
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    feats = pd.read_csv(args.features)
    if args.extra_features and Path(args.extra_features).exists():
        extra_f = pd.read_csv(args.extra_features)
        feats = pd.concat([feats, extra_f], ignore_index=True)
    if "group" not in feats.columns:
        feats["group"] = feats["pdb_id"].astype(str) + "_" + feats["rec_id"].astype(str)
    cols = [c for c in POSE_COLUMNS if c in feats.columns]
    x = np.nan_to_num(feats[cols].to_numpy(dtype=float), nan=0.0)
    y = feats["y"].to_numpy(dtype=int)
    rmsd = feats["rmsd_angstrom"].to_numpy(dtype=float)
    groups = feats["group"].astype(str).to_numpy()
    cv_groups = feats["pdb_id"].astype(str).to_numpy()
    LOG.info("n=%s pos=%s groups=%s pdb=%s d=%s", len(y), int(y.sum()), len(set(groups.tolist())), len(set(cv_groups.tolist())), x.shape[1])
    base = feats.copy()
    base["score_metalgeo"] = feats["hybrid_total"].astype(float)
    metalgeo = listwise_rates(base, "score_metalgeo", True)
    compared = {"metalgeo": {**metalgeo, "kind": "baseline"}}
    best_name = "metalgeo"
    best_top1 = float(metalgeo.get("top1") or -1.0)
    best_top3 = float(metalgeo.get("top3") or -1.0)
    oof_store = {}
    recipes = RECIPES if "all" in wanted_r else tuple(r for r in RECIPES if r["name"] in wanted_r)
    if not recipes:
        recipes = RECIPES
    for recipe in recipes:
        oof, _meta = oof_for_recipe(x, y, rmsd, groups, cols, recipe, args.seed, cv_groups=cv_groups)
        kind = _score_kind(recipe["loss"])
        lower = True
        scores = oof.copy()
        if kind == "neg_logit":
            scores = -oof
        tmp = feats.copy()
        tmp["oof"] = scores
        rates = listwise_rates(tmp, "oof", lower)
        y_score = -scores if lower else scores
        auc = auroc(y, y_score)
        LOG.info("%s listwise top1=%s top3=%s auroc=%s", recipe["name"], rates.get("top1"), rates.get("top3"), auc)
        compared[recipe["name"]] = {**rates, "oof_auroc": auc, "arch": recipe["arch"], "loss": recipe["loss"]}
        oof_store[recipe["name"]] = oof
        t1 = float(rates.get("top1") or -1.0)
        t3 = float(rates.get("top3") or -1.0)
        if t1 > best_top1 + 1.0e-6 or (abs(t1 - best_top1) <= 1.0e-6 and t3 > best_top3):
            best_top1, best_top3, best_name = t1, t3, recipe["name"]
        bundle, _ = train_one(x, y, rmsd, groups, cols, recipe, args.seed)
        save_ranker(bundle, args.out_dir / f"{recipe['name']}.pt")
    beats = best_name != "metalgeo" and best_top1 > float(metalgeo.get("top1") or 0.0)
    if beats:
        chosen = next(r for r in RECIPES if r["name"] == best_name)
        bundle, _ = train_one(x, y, rmsd, groups, cols, chosen, args.seed)
        bundle["selected_because"] = "oof_listwise_top1"
        save_ranker(bundle, args.best)
    else:
        LOG.warning("no model beat MetGeo OOF top-1; writing geolist_ranknet as exploratory best")
        fallback = next(r for r in RECIPES if r["name"] == "geolist_ranknet")
        bundle, _ = train_one(x, y, rmsd, groups, cols, fallback, args.seed)
        bundle["selected_because"] = "fallback_geolist_no_oof_gain"
        save_ranker(bundle, args.best)
        best_name = "geolist_ranknet"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_groups": int(len(set(groups.tolist()))),
        "columns": cols,
        "metalgeo_listwise": metalgeo,
        "models": compared,
        "best_name": best_name,
        "best_beats_metalgeo_oof_top1": bool(beats),
        "best_path": str(args.best),
        "sklearn": False,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s best=%s", args.out_json, best_name)
    print(json.dumps({k: report[k] for k in ("best_name", "best_beats_metalgeo_oof_top1", "metalgeo_listwise", "models")}, indent=2))


if __name__ == "__main__":
    main()
