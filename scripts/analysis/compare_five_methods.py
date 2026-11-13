"""Five-way comparison on identical molecules: MetGeo geometry, +burial, +burial+CAT,
AutoDock Vina, MetalloDock. Reports AUROC / EF1% / EF2% / EF5% / BEDROC plus bootstrap CIs.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
A_REF = 600.0
RNG = np.random.default_rng(20241024)


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
    if na == 0:
        return np.nan
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(r)[:k]].sum() / k) / (na / n))


def bedroc(y, s, alpha=20.0):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n, na = len(y), int(y.sum())
    if na == 0 or na == n:
        return np.nan
    ra = na / n
    ar = np.sort(r[y == 1])

    def rie(a):
        return float(np.exp(-alpha * np.asarray(a, float) / n).sum() / (ra * n))

    hi, lo = rie(np.arange(1, na + 1)), rie(np.arange(n - na + 1, n + 1))
    return float((rie(ar) - lo) / (hi - lo)) if abs(hi - lo) > 1e-12 else np.nan


def boot_ci(y, s, fn, n=1000):
    y = np.asarray(y); s = np.asarray(s, float)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        vals.append(fn(y[idx], s[idx]))
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10 else (np.nan, np.nan)


def pct(v):
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def load_thermo(target, suffix=""):
    rows = []
    for f in sorted((ROOT / "results/thermo_ef").glob(f"{target}_p*.csv")):
        rows += [r for r in csv.DictReader(open(f)) if r.get("g_total") not in (None, "", "nan")]
    return rows


def load_vina(target):
    rows = []
    for f in sorted((ROOT / "results/vina_ef").glob("*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("affinity") not in (None, "", "nan"):
                rows.append(r)
    return rows


def load_mdl(target):
    rows = []
    d = ROOT / "results/metallodock_scores" / f"{target}_ef"
    for f in sorted(d.glob("p*.csv")) if d.exists() else []:
        for r in csv.DictReader(open(f)):
            if r.get("score") not in (None, "", "nan"):
                rows.append(r)
    if not rows:
        return []
    y = [int(r["label"]) for r in rows]; s = [float(r["score"]) for r in rows]
    if auroc(y, s) < 0.5:          # MetalloDock is higher-is-better; flip to lower-is-better
        for r in rows:
            r["score"] = str(-float(r["score"]))
    return rows


def main():
    t = "HDAC1"
    th = load_thermo(t)
    if not th:
        print("no thermo data"); return
    y = [int(r["label"]) for r in th]
    g = pct([float(r["total"]) for r in th])
    gb = pct([float(r["g_buried"]) for r in th])
    nrot = np.array([float(r["n_rotatable"]) for r in th])
    bur = np.array([float(r["buried_area"]) for r in th])
    cat = pct(nrot * np.clip(bur / A_REF, 0, 1))

    variants = [("MetGeo geometry", g), ("+ hydrophobic burial", g + gb),
                ("+ burial + CAT", g + gb - cat)]
    for name, loader in (("AutoDock Vina", load_vina), ("MetalloDock", load_mdl)):
        rows = loader(t)
        if rows:
            variants.append((name, np.array([float(r["affinity"] if name.startswith("Auto") else r["score"])
                                             for r in rows])))
            variants[-1] = (name, pct([float(r["affinity"] if name.startswith("Auto") else r["score"])
                                       for r in rows]))

    print(f"=== {t}: {len(y)} molecules ({sum(y)} actives) ===\n")
    print(f"{'method':24s} {'n':>5s} {'AUROC':>16s} {'EF1%':>6s} {'EF2%':>6s} {'EF5%':>6s} {'BEDROC':>7s}")
    for name, s in variants:
        n = len(s)
        if n != len(y):
            # external methods may cover fewer molecules; report on their own set
            continue
        lo, hi = boot_ci(y, s, auroc)
        print(f"{name:24s} {n:5d} {auroc(y, s):6.3f} [{lo:.3f},{hi:.3f}] "
              f"{ef(y, s, 0.01):6.2f} {ef(y, s, 0.02):6.2f} {ef(y, s, 0.05):6.2f} {bedroc(y, s):7.3f}")

    # external methods on their own covered subset
    for name, loader in (("AutoDock Vina", load_vina), ("MetalloDock", load_mdl)):
        rows = loader(t)
        if not rows:
            continue
        yy = [int(r["label"]) for r in rows]
        ss = [float(r["affinity"] if name.startswith("Auto") else r["score"]) for r in rows]
        lo, hi = boot_ci(yy, ss, auroc)
        print(f"{name:24s} {len(yy):5d} {auroc(yy, ss):6.3f} [{lo:.3f},{hi:.3f}] "
              f"{ef(yy, ss, 0.01):6.2f} {ef(yy, ss, 0.02):6.2f} {ef(yy, ss, 0.05):6.2f} {bedroc(yy, ss):7.3f}")


if __name__ == "__main__":
    main()
