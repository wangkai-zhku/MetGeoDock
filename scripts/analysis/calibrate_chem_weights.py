"""Do the Vina-style chemistry terms improve library ranking? Calibrate and test honestly.

Protocol
  * features: the score terms recorded by screen_with_chem.py (coord, geo, steric, contact,
    hbond, hydrophobic, desolvation, map_overlap, centroid_pen)
  * ranking: score = w . z(features), with z() a percentile transform (the raw terms span
    many orders of magnitude and the score saturates)
  * fitting: logistic regression on is_active, protein-grouped 5-fold CV so no ligand is in
    both train and test
  * baseline: the same procedure with the chemistry terms removed

If the chemistry-augmented model does not beat both the geometric-only model and AutoDock
Vina, that is the answer and it gets reported as such.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]

GEOM = ["coord", "geo", "steric", "contact", "map_overlap", "centroid_pen"]
CHEM = ["hbond", "hydrophobic", "desolvation"]


def auroc(y, s):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float(1.0 - (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ef(y, s, frac):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n, na = len(y), int(y.sum())
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(r)[:k]].sum() / k) / (na / n)) if na else np.nan


def pct(v):
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def load(targets):
    rows = []
    for t in targets:
        for f in sorted((ROOT / "results/chem_scores").glob(f"{t}_p*.csv")):
            for r in csv.DictReader(open(f)):
                rec = {"target": t, "label": int(r["label"]), "ligand_id": r["ligand_id"]}
                ok = True
                for k in GEOM + CHEM:
                    try:
                        rec[k] = float(r[k])
                    except Exception:
                        ok = False
                        break
                if ok:
                    rows.append(rec)
    return rows


def fit_and_eval(rows, features, n_splits=5, seed=7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.column_stack([pct([r[f] for r in rows]) for f in features])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["ligand_id"] for r in rows])
    oof = np.full(len(rows), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    mask = np.isfinite(oof)
    # higher probability of active = better, so negate for the lower-is-better convention
    return auroc(y[mask], -oof[mask]), ef(y[mask], -oof[mask], 0.01), mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["HDAC1"])
    a = ap.parse_args()

    rows = load(a.targets)
    if not rows:
        print("no rows in results/chem_scores yet"); return
    n_act = sum(r["label"] for r in rows)
    print(f"{len(rows)} ligands ({n_act} actives, {len(rows)-n_act} decoys) from {a.targets}\n")

    print(f"{'model':30s} {'AUROC':>7s} {'EF1%':>6s} {'n':>5s}")
    for name, feats in (("geometry only", GEOM),
                        ("geometry + chemistry", GEOM + CHEM),
                        ("chemistry only", CHEM)):
        au, e1, n = fit_and_eval(rows, feats)
        print(f"{name:30s} {au:7.3f} {e1:6.2f} {n:5d}")

    # single-term diagnostics: which chemistry term carries signal on its own
    print()
    for f in CHEM:
        au = auroc([r["label"] for r in rows], [r[f] for r in rows])
        print(f"  single term {f:14s} AUROC {au:.3f}")


if __name__ == "__main__":
    main()
