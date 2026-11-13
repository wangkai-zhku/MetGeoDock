#!/usr/bin/env python3
"""Virtual-screening enrichment on crystal ligands (structure-based proxy).

For each receptor, the native ligand is the active. Decoys are ligands taken
from other complexes (same split seed). Each molecule is docked with
MetGeoDock and ranked by metgeo / geo / gp. Reports AUROC, EF1%
and EF5%. Covers all 10 GPDOCK metals present in the inventory.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.dock import dock_ligand  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402

LOG = logging.getLogger("benchmark_vs")


def auroc(labels: np.ndarray, scores_lower_better: np.ndarray) -> float:
    """ROC AUC for scores where more negative is better."""
    y = np.asarray(labels, dtype=int)
    s = -np.asarray(scores_lower_better, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(tpr, fpr))


def enrichment(labels: np.ndarray, scores_lower_better: np.ndarray, frac: float) -> float:
    y = np.asarray(labels, dtype=int)
    order = np.argsort(np.asarray(scores_lower_better, dtype=float))
    k = max(1, int(np.ceil(frac * len(y))))
    top = y[order][:k]
    hit = float(top.mean()) if k else 0.0
    base = float(y.mean()) if len(y) else 0.0
    return float(hit / base) if base > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--receptors-per-metal", type=int, default=4)
    p.add_argument("--decoys", type=int, default=19)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-spin", type=int, default=6)
    p.add_argument("--max-poses", type=int, default=6)
    p.add_argument("--out-csv", type=Path, default=Path("results/vs_benchmark.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/vs_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_vs.log"))
    p.add_argument("--model", type=Path, default=None, help="optional pose-rank joblib; adds ml AUROC")
    args = p.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    rng = np.random.default_rng(args.seed)
    model = None
    if args.model and args.model.exists():
        from metgeodock.ml import load_model
        from metgeodock.scoring import feature_vector

        model = load_model(args.model)
        LOG.info("loaded ML scorer %s", args.model)
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1].copy().reset_index(drop=True)

    receptors = []
    for metal, sub in inv.groupby("metal"):
        n = min(args.receptors_per_metal, len(sub))
        pick = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        receptors.extend(sub.loc[pick].to_dict(orient="records"))

    rows = []
    for rec in receptors:
        LOG.info("receptor %s %s", rec["metal"], rec["pdb_id"])
        try:
            protein = read_pdb(rec["protein_path"])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("protein fail: %s", exc)
            continue
        others = inv[(inv["pdb_id"] != rec["pdb_id"])].copy()
        n_dec = min(args.decoys, len(others))
        dec_idx = rng.choice(others.index.to_numpy(), size=n_dec, replace=False)
        library = [("active", rec["ligand_path"])] + [
            ("decoy", others.loc[i, "ligand_path"]) for i in dec_idx
        ]
        scored = []
        for label, lig_path in library:
            try:
                ligand = read_mol2(lig_path)
                result = dock_ligand(
                    protein,
                    ligand,
                    metal=rec["metal"],
                    n_spin=args.n_spin,
                    max_poses=args.max_poses,
                )
                if not result.poses:
                    continue
                best = result.best
                rec_score = {
                    "label": 1 if label == "active" else 0,
                    "metgeo": best.score.total,
                    "geo": best.score.geo,
                    "gp": best.score.gp,
                    "dist": best.score.dist,
                    "clash": best.score.clash,
                }
                if model is not None:
                    from metgeodock.scoring import feature_vector
                    from metgeodock.ml import predict_pactivity

                    feats = feature_vector(best.score, rec["metal"], best.ligand)
                    rec_score["ml"] = -predict_pactivity(model, feats)
                scored.append(rec_score)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("lig fail %s: %s", lig_path, exc)
                continue
        if not scored or sum(r["label"] for r in scored) == 0:
            continue
        frame = pd.DataFrame(scored)
        row = {"metal": rec["metal"], "pdb_id": rec["pdb_id"], "n_library": int(len(frame))}
        y = frame["label"].to_numpy()
        methods = [c for c in ("metgeo", "geo", "gp", "dist", "clash", "ml") if c in frame.columns]
        for method in methods:
            s = frame[method].to_numpy()
            row[f"{method}_auroc"] = auroc(y, s)
            row[f"{method}_ef1"] = enrichment(y, s, 0.01) if len(y) >= 100 else enrichment(y, s, max(1 / len(y), 0.05))
            row[f"{method}_ef5"] = enrichment(y, s, 0.05) if len(y) >= 20 else enrichment(y, s, 1 / len(y))
            row[f"{method}_active_rank"] = int(np.argsort(s).tolist().index(int(np.where(y == 1)[0][0])) + 1)
        rows.append(row)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_receptors": int(len(out)),
        "seed": args.seed,
        "methods": {},
    }
    for method in ("metgeo", "geo", "gp", "dist", "clash", "ml"):
        if f"{method}_auroc" in out:
            rec_m = {
                "mean_auroc": float(out[f"{method}_auroc"].mean()),
                "mean_ef": float(out[f"{method}_ef1"].mean()),
                "median_active_rank": float(out[f"{method}_active_rank"].median()),
            }
            if f"{method}_auroc" in out.columns and "metal" in out.columns:
                rec_m["by_metal"] = {
                    str(metal): float(sub[f"{method}_auroc"].mean()) for metal, sub in out.groupby("metal")
                }
            summary["methods"][method] = rec_m
    if len(out) and "metal" in out.columns:
        summary["by_metal"] = {
            str(metal): {
                "n": int(len(sub)),
                "metgeo_auroc": float(sub["metgeo_auroc"].mean()) if "metgeo_auroc" in sub else None,
                "ml_auroc": float(sub["ml_auroc"].mean()) if "ml_auroc" in sub else None,
            }
            for metal, sub in out.groupby("metal")
        }
        summary["n_metals"] = int(out["metal"].nunique())
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(summary["methods"]))


if __name__ == "__main__":
    main()
