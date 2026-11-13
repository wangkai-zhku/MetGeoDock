"""Fit (a, b, c) for score = a*z(E) + b*z(H_free) - c*z(n_clusters).

Training data comes only from crystal complexes (see structure_triplet.py): for each
receptor the ligand observed in the crystal is label 1, other crystal ligands are 0.
Nothing here touches activity values.

Fitting uses pairwise differences (native minus each decoy) with a logistic loss, so
the three coefficients come out on a common scale. Validation is grouped by complex
(GroupKFold), reporting how often the native ligand ranks first.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")


def zsc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def load_groups(pattern="results/calib3/part_*.json"):
    groups = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        try:
            groups += json.loads(Path(f).read_text())
        except Exception:
            continue
    return groups


def native_rank(groups, w):
    """Mean normalised rank of the native ligand (0 = first, 1 = last)."""
    ranks = []
    for g in groups:
        rows = [r for r in g["ligands"]
                if np.isfinite(r["E"]) and np.isfinite(r["H_free"]) and np.isfinite(r["n_clusters"])]
        if len(rows) < 4 or sum(r["is_native"] for r in rows) != 1:
            continue
        E, H, N = zsc([r["E"] for r in rows]), zsc([r["H_free"] for r in rows]), zsc([r["n_clusters"] for r in rows])
        s = w[0] * E + w[1] * H - w[2] * N
        order = np.argsort(s)
        pos = [i for i, r in enumerate(rows) if r["is_native"] == 1][0]
        ranks.append(float(np.where(order == pos)[0][0]) / (len(rows) - 1))
    return (float(np.mean(ranks)) if ranks else float("nan")), len(ranks)


def top1_rate(groups, w):
    hits = tot = 0
    for g in groups:
        rows = [r for r in g["ligands"]
                if np.isfinite(r["E"]) and np.isfinite(r["H_free"]) and np.isfinite(r["n_clusters"])]
        if len(rows) < 4 or sum(r["is_native"] for r in rows) != 1:
            continue
        E, H, N = zsc([r["E"] for r in rows]), zsc([r["H_free"] for r in rows]), zsc([r["n_clusters"] for r in rows])
        s = w[0] * E + w[1] * H - w[2] * N
        pos = [i for i, r in enumerate(rows) if r["is_native"] == 1][0]
        hits += int(int(np.argmin(s)) == pos)
        tot += 1
    return (hits / tot if tot else float("nan")), tot


def pointwise_dataset(groups):
    """One row per ligand, z-scored inside its receptor; label = is it the crystal ligand.

    Pointwise weighting keeps every complex contributing equally, unlike pairwise
    differences where a complex that fails produces 6 negative rows and dominates.
    """
    X, y, gid = [], [], []
    for gi, g in enumerate(groups):
        rows = [r for r in g["ligands"]
                if np.isfinite(r["E"]) and np.isfinite(r["H_free"]) and np.isfinite(r["n_clusters"])]
        if len(rows) < 4 or sum(r["is_native"] for r in rows) != 1:
            continue
        E = zsc([r["E"] for r in rows])
        H = zsc([r["H_free"] for r in rows])
        N = zsc([r["n_clusters"] for r in rows])
        for i, r in enumerate(rows):
            X.append([E[i], H[i], -N[i]])
            y.append(int(r["is_native"]))
            gid.append(gi)
    return np.array(X, float), np.array(y), np.array(gid)


def main():
    groups = load_groups()
    print(f"groups loaded: {len(groups)}")
    if not groups:
        print("no data yet"); return

    base_rank, n0 = native_rank(groups, [1.0, 0.0, 0.0])
    base_top1, _ = top1_rate(groups, [1.0, 0.0, 0.0])
    print(f"\nE only:        mean native rank {base_rank:.3f}  top-1 {base_top1:.3f}  (n={n0})")

    X, y, gid = pointwise_dataset(groups)
    print(f"rows: {len(y)}  (native {int(y.sum())})")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    # grouped CV estimate of generalisation
    ranks_cv, top1_cv = [], []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups=gid):
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        w = -clf.coef_[0]          # score lower = native, so flip sign
        sub = [groups[g] for g in np.unique(gid[te])]
        r, n = native_rank(sub, w)
        t, _ = top1_rate(sub, w)
        if np.isfinite(r):
            ranks_cv.append(r); top1_cv.append(t)

    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    w = -clf.coef_[0]
    if w[0] < 0:
        w = -w
    w = w / w[0]                      # normalise so a = 1
    print(f"\nfitted coefficients (a normalised to 1): a=1.000  b={w[1]:.3f}  c={w[2]:.3f}")
    print(f"  raw logistic coefs: {clf.coef_[0]}")

    full_rank, nf = native_rank(groups, w)
    full_top1, _ = top1_rate(groups, w)
    print(f"\ncalibrated (all data): mean native rank {full_rank:.3f}  top-1 {full_top1:.3f}  (n={nf})")
    print(f"grouped 5-fold CV:     mean native rank {np.mean(ranks_cv):.3f}  top-1 {np.mean(top1_cv):.3f}")

    out = {
        "n_groups": len(groups), "n_pairwise": int(len(y)),
        "coefficients": {"a_E": 1.0, "b_H_free": round(float(w[1]), 4),
                         "c_n_clusters": round(float(w[2]), 4)},
        "raw_logistic": [round(float(v), 4) for v in clf.coef_[0]],
        "baseline_E_only": {"mean_native_rank": round(base_rank, 4), "top1": round(base_top1, 4)},
        "calibrated_all": {"mean_native_rank": round(full_rank, 4), "top1": round(full_top1, 4)},
        "grouped_cv": {"mean_native_rank": round(float(np.mean(ranks_cv)), 4),
                       "top1": round(float(np.mean(top1_cv)), 4), "n_folds": len(ranks_cv)},
        "note": "coefficients fitted on crystal-structure ligand recovery ONLY; no activity data used",
    }
    (ROOT / "results/calib3_coefficients.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE results/calib3_coefficients.json")


if __name__ == "__main__":
    main()
