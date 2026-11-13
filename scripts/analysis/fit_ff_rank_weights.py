#!/usr/bin/env python3
"""Fit MetGeo ranking weights on the RCSB time-split *train* set.

Docks a stratified train sample, records pose-level FF terms and RMSD, then
grid-searches linear weights to maximize top-1 success (RMSD < 2 A).
Writes processed/ff/rank_weights.json and patches metalgeo_params.json.
Does not use the MetalloDock public 10 or the time-split test set.
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

from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.forcefield import load_params  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402

LOG = logging.getLogger("fit_ff_rank_weights")


def stratified_train(split: pd.DataFrame, per_metal: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    train = split[(split["time_split"] == "train") & (split["complete"] == 1)]
    for metal, sub in train.groupby("metal"):
        n = min(per_metal, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(sub.loc[idx])
    return pd.concat(parts, axis=0).reset_index(drop=True)


def collect_poses(inv: pd.DataFrame, n_spin: int, max_poses: int) -> pd.DataFrame:
    rows = []
    for i, rec in inv.iterrows():
        LOG.info("[%s/%s] %s %s", i + 1, len(inv), rec.metal, rec.pdb_id)
        try:
            protein = read_pdb(rec.protein_path)
            ligand = read_mol2(rec.ligand_path)
            result = dock_ligand(
                protein,
                ligand,
                metal=str(rec.metal),
                n_spin=n_spin,
                max_poses=max_poses,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("fail %s: %s", rec.pdb_id, exc)
            continue
        for rank, pose in enumerate(result.poses, start=1):
            t = pose.score.terms
            rows.append(
                {
                    "metal": rec.metal,
                    "pdb_id": rec.pdb_id,
                    "rank": rank,
                    "rmsd": redock_rmsd(pose.ligand, ligand),
                    "steric": float(t.get("steric", 0.0)),
                    "coord": float(t.get("coord", 0.0)),
                    "geo": float(t.get("geo", 0.0)),
                    "contact": float(t.get("contact", 0.0)),
                    "cpen": float(t.get("centroid_pen", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def top1_success(frame: pd.DataFrame, weights: dict[str, float], threshold: float) -> float:
    if frame.empty:
        return 0.0
    score = (
        frame["steric"]
        + weights["coord"] * frame["coord"]
        + weights["geo"] * frame["geo"]
        + weights["cpen"] * frame["cpen"]
        - weights["contact"] * frame["contact"]
    )
    tmp = frame.assign(fit_score=score)
    hits = []
    for _, sub in tmp.groupby(["metal", "pdb_id"], sort=False):
        best = sub.loc[sub["fit_score"].idxmin()]
        hits.append(float(best["rmsd"]) <= threshold)
    return float(np.mean(hits)) if hits else 0.0


def grid_search(frame: pd.DataFrame, threshold: float) -> tuple[dict[str, float], float]:
    best_w = {"coord": 0.35, "geo": 1.0, "cpen": 0.0, "contact": 0.0}
    best_s = top1_success(frame, best_w, threshold)
    for w_coord in (0.0, 0.35, 0.7):
        for w_geo in (0.0, 0.5, 1.0, 1.5):
            for w_cpen in (0.0, 0.2, 0.5):
                for w_contact in (0.0, 0.25, 0.5, 1.0):
                    w = {"coord": w_coord, "geo": w_geo, "cpen": w_cpen, "contact": w_contact}
                    s = top1_success(frame, w, threshold)
                    if s > best_s + 1.0e-12:
                        best_s, best_w = s, w
    return best_w, best_s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--per-metal", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-spin", type=int, default=12)
    p.add_argument("--max-poses", type=int, default=20)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out-csv", type=Path, default=Path("processed/ff/rank_weight_poses.csv"))
    p.add_argument("--out-json", type=Path, default=Path("processed/ff/rank_weights.json"))
    p.add_argument("--params-json", type=Path, default=Path("processed/ff/metalgeo_params.json"))
    p.add_argument("--log", type=Path, default=Path("logs/fit_ff_rank_weights.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    split = pd.read_csv(args.split_csv)
    inv = stratified_train(split, args.per_metal, args.seed)
    LOG.info("train sample n=%s", len(inv))
    poses = collect_poses(inv, args.n_spin, args.max_poses)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    poses.to_csv(args.out_csv, index=False)
    weights, train_top1 = grid_search(poses, args.threshold)
    sampled = float(poses.groupby(["metal", "pdb_id"])["rmsd"].min().le(args.threshold).mean()) if len(poses) else 0.0
    blob = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_complexes": int(poses.groupby(["metal", "pdb_id"]).ngroups) if len(poses) else 0,
        "n_poses": int(len(poses)),
        "seed": args.seed,
        "per_metal": args.per_metal,
        "train_sampled_success": sampled,
        "train_top1_success": train_top1,
        "rank_weights": weights,
        "threshold_angstrom": args.threshold,
        "split": "time_split==train",
    }
    args.out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    params = json.loads(args.params_json.read_text(encoding="utf-8"))
    params["rank_weights"] = weights
    args.params_json.write_text(json.dumps(params, indent=2), encoding="utf-8")
    load_params.cache_clear()
    LOG.info("wrote %s train_top1=%s sampled=%s weights=%s", args.out_json, train_top1, sampled, weights)


if __name__ == "__main__":
    main()
