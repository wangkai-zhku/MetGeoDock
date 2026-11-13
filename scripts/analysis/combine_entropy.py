"""Combine the potential with a conformational-entropy term: score = z(E) + lam * z(S).

Physics: dG = dH - T dS. E (MetGeo) approximates dH, S (ensemble spread)
approximates the entropy lost on binding. Both are z-scored inside each target so
they are on the same scale before mixing; lam plays the role of T.

Reports AUROC / EF1% / EF5% as a function of lam, with the same score direction
as everywhere else (LOWER = better binder).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
HDAC = Path("/home/wangkai/cursor-task/HDAC")
T = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]
LAMS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
S_FEATURES = ["n_clusters", "mean_pair_rmsd"]


def load_rows():
    out = {}
    for label, d in (("act", ROOT / "results/entropy_parts"), ("dec", ROOT / "results/entropy_decoys")):
        for t in T:
            f = d / f"{t}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                rec = {"target": t, "ligand_id": r["ligand_id"], "label": int(r["label"])}
                for k in ("n_clusters", "mean_pair_rmsd", "n_poses", "E_best"):
                    try:
                        rec[k] = float(r[k])
                    except Exception:
                        rec[k] = np.nan
                out[(t, r["ligand_id"])] = rec
    return out


def load_metgeo():
    m = {}
    for t in T:
        f = HDAC / "docking/results" / f"{t}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            try:
                m[(t, r["ligand_id"])] = float(r["score"])
            except Exception:
                continue
    return m


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def main():
    ent, score = load_rows(), load_metgeo()
    keys = [k for k in ent if k in score]
    print(f"joined: {len(keys)} molecules "
          f"(actives {sum(1 for k in keys if ent[k]['label']==1)})")
    if not keys:
        print("no join yet"); return

    for sfeat in S_FEATURES:
        desc = []
        for t in T:
            sub = [ent[k] for k in keys if k[0] == t]
            if len({r["label"] for r in sub}) < 2:
                continue
            E = np.array([score[(t, r["ligand_id"])] for r in sub], float)
            S = np.array([r[sfeat] for r in sub], float)
            ok = np.isfinite(E) & np.isfinite(S)
            if ok.sum() < 20:
                continue
            E, S = E[ok], S[ok]
            y = np.array([r["label"] for r in sub], int)[ok]
            Ez = (E - E.mean()) / (E.std() + 1e-9)
            Sz = (S - S.mean()) / (S.std() + 1e-9)
            desc.append((t, y, Ez, Sz))
        if not desc:
            continue
        print(f"\n=== S = {sfeat} (higher entropy -> higher score -> ranked worse) ===")
        print(f"{'lambda':>7s} {'AUROC':>7s} {'EF1%':>6s} {'EF5%':>6s}")
        best = None
        for lam in LAMS:
            aucs, e1s, e5s = [], [], []
            for t, y, Ez, Sz in desc:
                s = Ez + lam * Sz
                aucs.append(roc_auc_score(y, -s))
                e1s.append(ef(y, s, 0.01)); e5s.append(ef(y, s, 0.05))
            a, e1, e5 = np.mean(aucs), np.mean(e1s), np.mean(e5s)
            flag = ""
            if lam == 0.0:
                base = a
            elif a > base + 0.01:
                flag = "  <-- better than lam=0"
            if best is None or a > best[1]:
                best = (lam, a)
            print(f"{lam:7.2f} {a:7.3f} {e1:6.2f} {e5:6.2f}{flag}")
        print(f"best lambda = {best[0]:.2f} (AUROC {best[1]:.3f}); targets used = {len(desc)}")


if __name__ == "__main__":
    main()
