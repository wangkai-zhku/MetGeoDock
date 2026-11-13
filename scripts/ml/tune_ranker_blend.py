#!/usr/bin/env python3
"""Tune within-list z-score blend of MetGeo and a trained ranker.

s = z(hybrid_total) + alpha * z(ranker). Lower better. Picks the largest
alpha that does not fall below MetGeo listwise top-1 on the train CSV
(public-10 held out). Writes processed/ml/ranker_blend.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from metgeodock.rankers import load_ranker, predict_rank  # noqa: E402


def _listwise(frame: pd.DataFrame, col: str) -> dict:
    h1, h3 = [], []
    for _, sub in frame.groupby("group", sort=False):
        if len(sub) < 2 or int(sub["y"].sum()) == 0:
            continue
        order = np.argsort(sub[col].to_numpy(dtype=float), kind="mergesort")
        y = sub["y"].to_numpy(dtype=int)[order]
        h1.append(int(y[0] == 1))
        h3.append(int(y[:3].max() == 1))
    return {"n": int(len(h1)), "top1": float(np.mean(h1)) if h1 else None, "top3": float(np.mean(h3)) if h3 else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("processed/ml/metgeo_rank_features.csv"))
    p.add_argument("--model", type=Path, default=Path("processed/ml/ranker_best.pt"))
    p.add_argument("--out", type=Path, default=Path("processed/ml/ranker_blend.json"))
    args = p.parse_args()
    feats = pd.read_csv(args.features)
    if "group" not in feats.columns:
        feats["group"] = feats["pdb_id"].astype(str) + "_" + feats["rec_id"].astype(str)
    bundle = load_ranker(args.model)
    if bundle is None:
        raise SystemExit("ranker checkpoint missing")
    scores = []
    cols = list(bundle.get("columns") or [])
    for rec in feats.to_dict(orient="records"):
        feats_row = {c: float(rec.get(c, 0.0)) for c in cols}
        scores.append(predict_rank(bundle, feats_row))
    feats = feats.copy()
    feats["ranker"] = np.asarray(scores, dtype=float)
    feats["hybrid"] = feats["hybrid_total"].astype(float)
    base = _listwise(feats, "hybrid")
    grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]
    rows = []
    best = {"alpha": 0.0, "top1": base.get("top1"), "top3": base.get("top3")}
    for alpha in grid:
        tmp = feats.copy()
        tmp["blend"] = np.nan
        for _, sub in tmp.groupby("group", sort=False):
            h = sub["hybrid"].to_numpy(dtype=float)
            r = sub["ranker"].to_numpy(dtype=float)
            zh = (h - h.mean()) / (float(h.std()) + 1.0e-6)
            zr = (r - r.mean()) / (float(r.std()) + 1.0e-6)
            tmp.loc[sub.index, "blend"] = zh + float(alpha) * zr
        rates = _listwise(tmp, "blend")
        rec = {"alpha": alpha, **rates}
        rows.append(rec)
        t1 = float(rates.get("top1") or -1)
        t3 = float(rates.get("top3") or -1)
        b1 = float(best.get("top1") or -1)
        b3 = float(best.get("top3") or -1)
        floor = float(base.get("top1") or 0)
        if t1 + 1.0e-9 >= floor and (t1 > b1 + 1.0e-6 or (abs(t1 - b1) <= 1.0e-6 and t3 > b3)):
            best = {"alpha": alpha, "top1": t1, "top3": t3}
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "metalgeo": base,
        "grid": rows,
        "alpha": best["alpha"],
        "blend": {"top1": best["top1"], "top3": best["top3"]},
        "note": "s = z(hybrid) + alpha z(ranker); lower better; public-10 not in this CSV",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
