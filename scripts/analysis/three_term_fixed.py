"""Fixed-sign three-term score: E + alpha*H_free - beta*n_clusters (alpha, beta >= 0).

Rationale: the physical signs are not free parameters.
  * E (MetGeo) is already "lower = better"
  * H_free large means more conformational entropy to lose -> penalise (alpha >= 0)
  * n_clusters empirically correlates positively with potency -> reward (minus beta)

Only the magnitudes are fitted, which is what a small dataset can support.
Validation is leave-one-target-out on the actives' measured pChEMBL.
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


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def prep(rows):
    out = []
    for t in TARGETS:
        sub = [r for r in rows if r["target"] == t]
        if len(sub) < 10:
            continue
        E, H, N = z([r["E_best"] for r in sub]), z([r["H_free"] for r in sub]), z([r["n_clusters"] for r in sub])
        P = np.array([r["pchembl"] for r in sub], float)
        for i in range(len(sub)):
            out.append({"target": t, "E": E[i], "H": H[i], "N": N[i], "p": P[i]})
    return out


def rho(data, alpha, beta):
    """Spearman between the score and pChEMBL. Score lower = better, so a
    *negative* rho means the score ranks potency correctly."""
    s = np.array([d["E"] + alpha * d["H"] - beta * d["N"] for d in data])
    p = np.array([d["p"] for d in data])
    if len(np.unique(s)) < 2:
        return np.nan
    return float(stats.spearmanr(s, p)[0])


def main():
    data = prep(load())
    targets = sorted({d["target"] for d in data})
    print(f"actives {len(data)}, targets {targets}")

    # pure E baseline (alpha=beta=0)
    print("\n=== single terms (rho vs pChEMBL; negative = correct direction) ===")
    print(f"  E only            {rho(data, 0.0, 0.0):+.3f}")
    for a in (0.5, 1.0):
        print(f"  E + {a}*H_free      {rho(data, a, 0.0):+.3f}")
    for b in (0.5, 1.0):
        print(f"  E - {b}*n_clusters  {rho(data, 0.0, b):+.3f}")

    grid = np.arange(0.0, 6.01, 0.5)
    print("\n=== fixed-sign grid on all data (want most negative rho) ===")
    best = None
    for a in grid:
        for b in grid:
            r = rho(data, a, b)
            if np.isfinite(r) and (best is None or r < best[0]):
                best = (r, a, b)
    r, a, b = best
    print(f"  best: alpha={a:.2f}, beta={b:.2f}, rho={r:+.3f}")

    print("\n=== leave-one-target-out, signs fixed, only magnitudes fitted ===")
    print(f"{'held out':10s} {'E only':>9s} {'+H_free':>9s} {'+nclust':>9s} {'both':>9s} {'alpha':>6s} {'beta':>5s}")
    cols = {"E": [], "H": [], "N": [], "both": []}
    for t in targets:
        train = [d for d in data if d["target"] != t]
        test = [d for d in data if d["target"] == t]
        bb = None
        for aa in grid:
            for b2 in grid:
                rr = rho(train, aa, b2)
                if np.isfinite(rr) and (bb is None or rr < bb[0]):
                    bb = (rr, aa, b2)
        _, aa, b2 = bb
        r_E = rho(test, 0.0, 0.0)
        r_H = rho(test, aa, 0.0)
        r_N = rho(test, 0.0, b2)
        r_b = rho(test, aa, b2)
        cols["E"].append(r_E); cols["H"].append(r_H); cols["N"].append(r_N); cols["both"].append(r_b)
        print(f"{t:10s} {r_E:+9.3f} {r_H:+9.3f} {r_N:+9.3f} {r_b:+9.3f} {aa:6.2f} {b2:5.2f}")
    print("\nheld-out mean:")
    for k, v in cols.items():
        print(f"  {k:6s} {np.mean(v):+.3f}")
    print("\n(fitted alpha/beta from the training folds:")
    print(f"  chosen on all data: alpha={a:.2f}, beta={b:.2f})")


if __name__ == "__main__":
    main()
