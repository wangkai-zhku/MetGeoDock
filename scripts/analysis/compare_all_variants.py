"""Compare every scoring variant across targets, with EF, BEDROC and a per-target diagnosis of
the conformational-entropy direction.

Answers three questions at once:
  1. does the buried-area term help beyond MetGeo, on every target?
  2. which sign does the g_conf term want on each target, and does that sign follow the pocket
     geometry reported by metgeodock.pocket_shape?
  3. how do MetalGeoDock, Vina and MetalloDock compare at equal EF definitions?
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]


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


def pct(v):
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def load_thermo(target):
    rows = []
    for f in sorted((ROOT / "results/thermo_scores").glob(f"{target}_p*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("g_total") in (None, "", "nan"):
                continue
            rows.append(r)
    return rows


def load_metgeo(target):
    p = ROOT / f"results/common_set/{target}_metgeo.csv"
    if not p.exists():
        return []
    return [r for r in csv.DictReader(open(p)) if r.get("score") not in (None, "", "nan")]


def load_vina(target):
    rows = []
    for f in sorted((ROOT / "results/vina_baseline").glob(f"{target}_p*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("affinity") not in (None, "", "nan"):
                rows.append(r)
    return rows


def load_mdl(target):
    p = ROOT / f"results/metallodock_scores/{target}.csv"
    if not p.exists():
        return []
    rows = [r for r in csv.DictReader(open(p)) if r.get("score") not in (None, "", "nan")]
    if not rows:
        return []
    # MetalloDock scores are higher-is-better; flip so every variant follows lower-is-better
    y = [int(r["label"]) for r in rows]
    s = [float(r["score"]) for r in rows]
    if auroc(y, s) < 0.5:
        for r in rows:
            r["score"] = str(-float(r["score"]))
    return rows


def variant_scores(rows, name):
    t = pct([float(r["total"]) for r in rows])
    gb = pct([float(r["g_buried"]) for r in rows])
    gd = pct([float(r["g_desolv"]) for r in rows])
    gh = pct([float(r["g_hbond"]) for r in rows])
    gc = pct([float(r["g_conf"]) for r in rows])
    return {
        "MetGeo": t,
        "MetGeo + buried": t + gb,
        "MetGeo + buried + desolv": t + gb + gd,
        "buried + desolv": gb + gd,
        "g_conf as physics": gc,
        "g_conf reversed": 1.0 - gc,
        "MetGeo + all (physics)": t + gb + gd + gh + gc,
        "MetGeo + all (reversed)": t + gb + gd + gh + (1.0 - gc),
    }[name]


ALL = ["MetGeo", "MetGeo + buried", "MetGeo + buried + desolv", "buried + desolv",
       "g_conf as physics", "g_conf reversed", "MetGeo + all (physics)",
       "MetGeo + all (reversed)"]


def main():
    targets = [t for t in ("HDAC1", "HDAC2", "CA2", "CAPN1")
               if (ROOT / f"results/thermo_scores/{t}_p0.csv").exists()]
    if not targets:
        print("no thermo_scores yet"); return

    print("=== 每个靶点：各打分变体的 AUROC / EF1% / BEDROC20 ===\n")
    for t in targets:
        rows = load_thermo(t)
        y = [int(r["label"]) for r in rows]
        if len({*y}) < 2:
            print(f"{t}: 只有一类标签，跳过\n"); continue
        print(f"--- {t}  (n={len(rows)}, actives={sum(y)}) ---")
        print(f"{'variant':30s} {'AUROC':>7s} {'EF1%':>6s} {'EF5%':>6s} {'BEDROC':>7s}")
        for v in ALL:
            s = variant_scores(rows, v)
            print(f"{v:30s} {auroc(y, s):7.3f} {ef(y, s, 0.01):6.2f} "
                  f"{ef(y, s, 0.05):6.2f} {bedroc(y, s):7.3f}")
        # g_conf diagnosis: which sign wins on this target
        au_p = auroc(y, variant_scores(rows, "g_conf as physics"))
        au_r = auroc(y, variant_scores(rows, "g_conf reversed"))
        print(f"  -> g_conf 方向: physics {au_p:.3f} vs reversed {au_r:.3f} "
              f"=> {'physics' if au_p > au_r else 'REVERSED'}\n")

    print("=== 与外部方法对比（同一批分子）===\n")
    print(f"{'target':8s} {'method':16s} {'n':>4s} {'AUROC':>7s} {'EF1%':>6s} {'BEDROC':>7s}")
    for t in targets:
        for name, loader, key in (("MetalGeoDock", load_metgeo, "score"),
                                  ("AutoDock Vina", load_vina, "affinity"),
                                  ("MetalloDock", load_mdl, "score")):
            rows = loader(t)
            if not rows:
                continue
            y = [int(r["label"]) for r in rows]
            if len({*y}) < 2:
                continue
            s = [float(r[key]) for r in rows]
            print(f"{t:8s} {name:16s} {len(y):4d} {auroc(y, s):7.3f} "
                  f"{ef(y, s, 0.01):6.2f} {bedroc(y, s):7.3f}")
        print()


if __name__ == "__main__":
    main()
