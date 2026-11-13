#!/usr/bin/env python3
"""Grid-search linear MetGeoDock weights on pose-rank features.

Lower score is better for native recovery. Searches a small grid over
geo/gp/clash/dist/cn/slef_tanh and reports top-1 native recovery and AP.
Does not invent weights: writes the best grid point to results/.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def score_row(row: pd.Series, w: dict[str, float]) -> float:
    return (
        w["geo"] * float(row.get("geo_rmsd_or", 0.0))
        + w["gp"] * float(row.get("gp_nll", 0.0))
        + w["clash"] * float(row.get("clash", 0.0))
        + w["dist"] * float(row.get("dist_harm", 0.0))
        + w["cn"] * float(row.get("cn_pen", 0.0))
        + w["slef"] * float(np.tanh(float(row.get("slef", 0.0)) / 50.0)) * 10.0
    )


def native_top1(frame: pd.DataFrame, scores: np.ndarray) -> float:
    tmp = frame.assign(_s=scores)
    hits = []
    for _, sub in tmp.groupby("target_id"):
        order = sub.sort_values("_s")
        hits.append(int(order.iloc[0]["is_native"] == 1))
    return float(np.mean(hits)) if hits else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=Path, default=Path("processed/ml/pose_rank_features.csv"))
    p.add_argument("--out", type=Path, default=Path("results/metgeo_weight_search.json"))
    args = p.parse_args()
    frame = pd.read_csv(args.rank).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = frame["is_native"].to_numpy(dtype=int)
    grid = {
        "geo": (2.0, 6.0),
        "gp": (0.8, 1.8),
        "clash": (2.0, 8.0, 16.0),
        "dist": (1.0, 3.5),
        "cn": (0.0, 1.2),
        "slef": (0.0, 0.12),
    }
    keys = list(grid)
    best = None
    table = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        w = dict(zip(keys, vals))
        s = (
            w["geo"] * frame["geo_rmsd_or"].to_numpy(dtype=float)
            + w["gp"] * frame["gp_nll"].to_numpy(dtype=float)
            + w["clash"] * frame["clash"].to_numpy(dtype=float)
            + w["dist"] * frame.get("dist_harm", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
            + w["cn"] * frame.get("cn_pen", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
            + w["slef"] * np.tanh(frame["slef"].to_numpy(dtype=float) / 50.0) * 10.0
        )
        ap = float(average_precision_score(y, -s))
        top1 = native_top1(frame, s)
        rec = {"weights": w, "ap": ap, "native_top1": top1}
        table.append(rec)
        if best is None or (top1, ap) > (best["native_top1"], best["ap"]):
            best = rec
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": int(len(frame)),
        "n_native": int(y.sum()),
        "best": best,
        "n_grid": len(table),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
