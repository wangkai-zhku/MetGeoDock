"""Three-term score: E_metalgeo + alpha*H_free + beta*n_clusters.

Physics: dG = dH - T dS_conf. H_free (free-ligand conformational entropy) enters
with a positive weight (more entropy to lose = worse), while n_clusters empirically
correlates positively with potency, so it is allowed a negative weight.

Validation is leave-one-target-out on the real pChEMBL values of the actives, which
is the only unbiased way to pick alpha/beta on this data.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
TARGETS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/itpe/*.csv"))):
        t = Path(f).stem
        for r in csv.DictReader(open(f)):
            rec = {"target": t}
            ok = True
            for k in ("E_best", "H_free", "n_clusters"):
                try:
                    rec[k] = float(r[k])
                except Exception:
                    ok = False
                    break
            try:
                rec["pchembl"] = float(r["pchembl"])
            except Exception:
                ok = False
            if ok:
                rows.append(rec)
    return rows


def zscore(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s + 1e-9)


def prep(rows):
    """z-score every term inside each target so weights are comparable."""
    out = []
    for t in TARGETS:
        sub = [r for r in rows if r["target"] == t]
        if len(sub) < 10:
            continue
        E = zscore([r["E_best"] for r in sub])
        H = zscore([r["H_free"] for r in sub])
        N = zscore([r["n_clusters"] for r in sub])
        P = np.array([r["pchembl"] for r in sub], float)
        for i in range(len(sub)):
            out.append({"target": t, "E": E[i], "H": H[i], "N": N[i], "p": P[i]})
    return out


def rho_of(data, wE, wH, wN):
    s = np.array([wE * d["E"] + wH * d["H"] + wN * d["N"] for d in data])
    p = np.array([d["p"] for d in data])
    if len(np.unique(s)) < 2:
        return np.nan
    return float(stats.spearmanr(s, p)[0])


def main():
    rows = load()
    data = prep(rows)
    targets = sorted({d["target"] for d in data})
    print(f"actives: {len(data)}  targets: {targets}")

    # sign convention: score should go DOWN as potency goes UP
    # so we report rho between (-score) and pChEMBL; equivalently use -rho here.
    base_E = rho_of(data, 1.0, 0.0, 0.0)
    base_H = rho_of(data, 0.0, 1.0, 0.0)
    base_N = rho_of(data, 0.0, 0.0, 1.0)
    print(f"\nsingle-term Spearman (E / H_free / n_clusters): "
          f"{base_E:+.3f} / {base_H:+.3f} / {base_N:+.3f}")

    grid = np.arange(-2.0, 2.01, 0.2)
    best = None
    for a in grid:
        for b in grid:
            r = rho_of(data, 1.0, a, b)
            if not np.isfinite(r):
                continue
            if best is None or abs(r) > abs(best[0]):
                best = (r, a, b)
    r, a, b = best
    print(f"\nbest on ALL data: rho={r:+.3f} at alpha={a:+.1f}, beta={b:+.1f}")

    # leave-one-target-out: pick alpha,beta on the rest, evaluate on the held-out one
    print("\n=== leave-one-target-out ===")
    print(f"{'held out':10s} {'baseline E':>11s} {'tuned':>8s} {'alpha':>7s} {'beta':>6s}")
    tuned_all, base_all = [], []
    for t in targets:
        train = [d for d in data if d["target"] != t]
        test = [d for d in data if d["target"] == t]
        bb = None
        for aa in grid:
            for b2 in grid:
                rr = rho_of(train, 1.0, aa, b2)
                if not np.isfinite(rr):
                    continue
                if bb is None or abs(rr) > abs(bb[0]):
                    bb = (rr, aa, b2)
        _, aa, b2 = bb
        r_test = rho_of(test, 1.0, aa, b2)
        r_base = rho_of(test, 1.0, 0.0, 0.0)
        tuned_all.append(r_test)
        base_all.append(r_base)
        print(f"{t:10s} {r_base:+11.3f} {r_test:+8.3f} {aa:+7.1f} {b2:+6.1f}")
    print(f"\nheld-out mean: baseline {np.mean(base_all):+.3f}  ->  tuned {np.mean(tuned_all):+.3f}")


if __name__ == "__main__":
    main()
