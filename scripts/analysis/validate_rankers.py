#!/usr/bin/env python3
"""Sanity-check ranker training artifacts and public-10 holdout.

Public-10 PDB IDs must not appear in the ranker training CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("processed/ml/metgeo_rank_features.csv"))
    p.add_argument("--holdout", type=Path, default=Path("processed/metallodock_ref/dataset/test_pdbid.csv"))
    p.add_argument("--compare", type=Path, default=Path("processed/ml/ranker_compare.json"))
    p.add_argument("--best", type=Path, default=Path("processed/ml/ranker_best.pt"))
    p.add_argument("--public10", type=Path, default=Path("results/ranker_public10.json"))
    p.add_argument("--out", type=Path, default=Path("results/validate_rankers.json"))
    args = p.parse_args()
    hold = set(pd.read_csv(args.holdout)["pdb_id"].astype(str).str.lower())
    leak = []
    n_rows = 0
    if args.features.exists():
        feats = pd.read_csv(args.features)
        n_rows = int(len(feats))
        train_ids = set(feats["pdb_id"].astype(str).str.lower())
        leak = sorted(train_ids & hold)
    compare = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare.exists() else {}
    pub = json.loads(args.public10.read_text(encoding="utf-8")) if args.public10.exists() else {}
    checks = {
        "features_exist": bool(args.features.exists()),
        "n_rows": n_rows,
        "no_public10_leak": leak == [],
        "leaked": leak,
        "best_ckpt": bool(args.best.exists()),
        "sklearn_forbidden": compare.get("sklearn") is False if compare else True,
        "best_name": compare.get("best_name"),
        "oof_beats_metalgeo": compare.get("best_beats_metalgeo_oof_top1"),
        "public10_ranker_beats_top1": (pub.get("ranker_beats_metgeo_top1") if pub else None),
        "public10_ranker_beats_top3": (pub.get("ranker_beats_metgeo_top3") if pub else None),
    }
    failed = [k for k, v in checks.items() if v is False]
    out = {"ok": "no_public10_leak" not in failed and "features_exist" not in failed, "failed": failed, "checks": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if "no_public10_leak" in failed or "features_exist" in failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
