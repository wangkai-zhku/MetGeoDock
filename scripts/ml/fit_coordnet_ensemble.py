#!/usr/bin/env python3
"""Fit ExtraTrees on saved CoordNet features (same split protocol).

MLP is already in coordnet.joblib. This writes coordnet_et.joblib so the
ML virtual-screening scheme can use the stronger grouped-CV model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from metgeodock.nnscore import coordnet_matrix, save_nn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("processed/ml/coordnet_features.csv"))
    p.add_argument("--out", type=Path, default=Path("processed/ml/coordnet_et.joblib"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    feats = pd.read_csv(args.features)
    x = coordnet_matrix(feats)
    y = feats["y"].to_numpy(dtype=int)
    model = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, random_state=args.seed, n_jobs=1
    )
    model.fit(x, y)
    save_nn(model, args.out, extra={"kind": "extratrees", "seed": args.seed})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
