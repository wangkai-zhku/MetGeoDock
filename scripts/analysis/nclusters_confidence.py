"""Is n_clusters a sampling-confidence indicator rather than a score term?

Hypothesis (option 3): a ligand whose docking converges to few binding modes has a
trustworthy E; one that scatters over many modes does not. If true, E's enrichment
should degrade as n_clusters grows, and filtering/weighting by n_clusters should
improve the ranking without ever using entropy as an additive term.

Diagnostics:
  A. AUROC / EF of E stratified by n_clusters bin
  B. ranking after dropping high-n_clusters ligands
  C. soft weighting: score = E + lambda * n_clusters (lambda small, acts as a
     confidence penalty rather than a fitted entropy coefficient)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
HDAC = Path("/home/wangkai/cursor-task/HDAC")
TARGETS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def load():
    nc_a, nc_d = {}, {}
    for t in TARGETS:
        for name, store in (("entropy_parts", nc_a), ("entropy_decoys", nc_d)):
            f = ROOT / "results" / name / f"{t}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                try:
                    store[r["ligand_id"]] = float(r["n_clusters"])
                except Exception:
                    continue
    data = {}
    for t in TARGETS:
        f = HDAC / "docking/results" / f"{t}.csv"
        if not f.exists():
            continue
        rows = []
        for r in csv.DictReader(open(f)):
            lid = r["ligand_id"]
            if r.get("score") in (None, "", "nan"):
                continue
            nc = nc_a.get(lid, nc_d.get(lid))
            if nc is None:
                continue
            rows.append({"y": int(lid in nc_a), "E": float(r["score"]), "N": nc})
        if sum(x["y"] for x in rows) >= 10:
            data[t] = rows
    return data


def main():
    data = load()
    print(f"targets with E+n_clusters: {len(data)}")

    # ---- A. AUROC of E stratified by n_clusters -------------------------
    print("\n=== A. E-only AUROC stratified by n_clusters ===")
    print(f"{'bin':>10s} {'n':>6s} {'actives':>8s} {'AUROC(E)':>9s} {'EF1%':>7s}")
    bins = defaultdict(list)
    for t, rows in data.items():
        for r in rows:
            k = int(min(r["N"], 4))
            bins[f"{k}" if k < 4 else "4+"].append(r)
    for key in ("1", "2", "3", "4+"):
        sub = bins.get(key, [])
        if len(sub) < 20 or len({r["y"] for r in sub}) < 2:
            continue
        y = [r["y"] for r in sub]; E = [r["E"] for r in sub]
        print(f"{key:>10s} {len(sub):6d} {sum(y):8d} {auroc(y, -np.array(E)):9.3f} {ef(y, E, 0.01):7.2f}")

    # pooled by target to remove pooling artefacts
    print("\n  per-target AUROC of E, split at n_clusters median:")
    print(f"{'target':8s} {'low-median':>11s} {'high-median':>12s} {'n_low':>6s} {'n_high':>7s}")
    for t, rows in data.items():
        med = float(np.median([r["N"] for r in rows]))
        lo = [r for r in rows if r["N"] <= med]
        hi = [r for r in rows if r["N"] > med]
        if len(lo) < 10 or len(hi) < 10:
            continue
        a_lo = auroc([r["y"] for r in lo], -np.array([r["E"] for r in lo]))
        a_hi = auroc([r["y"] for r in hi], -np.array([r["E"] for r in hi]))
        print(f"{t:8s} {a_lo:11.3f} {a_hi:12.3f} {len(lo):6d} {len(hi):7d}")

    # ---- B. drop high-n_clusters ligands --------------------------------
    print("\n=== B. drop ligands with n_clusters above a cutoff ===")
    print(f"{'cutoff':>8s} {'targets':>8s} {'mean AUROC(E)':>14s} {'mean EF1%':>10s}")
    for cut in (2, 3, 4, 5, 99):
        aucs, efs = [], []
        for t, rows in data.items():
            keep = [r for r in rows if r["N"] <= cut]
            if len(keep) < 20 or len({r["y"] for r in keep}) < 2:
                continue
            aucs.append(auroc([r["y"] for r in keep], -np.array([r["E"] for r in keep])))
            efs.append(ef([r["y"] for r in keep], [r["E"] for r in keep], 0.01))
        if aucs:
            print(f"{cut:>8d} {len(aucs):8d} {np.mean(aucs):14.3f} {np.mean(efs):10.2f}")

    # ---- C. soft confidence penalty -------------------------------------
    print("\n=== C. score = z(E) + lambda * z(n_clusters) (penalise uncertain poses) ===")
    print(f"{'lambda':>7s} {'mean AUROC':>11s} {'mean EF1%':>10s} {'mean EF5%':>10s}")
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        aucs, e1, e5 = [], [], []
        for t, rows in data.items():
            E = np.array([r["E"] for r in rows], float)
            N = np.array([r["N"] for r in rows], float)
            y = np.array([r["y"] for r in rows], int)
            def z(v):
                return (v - v.mean()) / (v.std() + 1e-9)
            s = z(E) + lam * z(N)
            aucs.append(auroc(y, -s)); e1.append(ef(y, s, 0.01)); e5.append(ef(y, s, 0.05))
        print(f"{lam:7.1f} {np.mean(aucs):11.3f} {np.mean(e1):10.2f} {np.mean(e5):10.2f}")


if __name__ == "__main__":
    main()
