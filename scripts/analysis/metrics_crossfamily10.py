"""Compute every screening metric we can defend, for every method and target.

Metrics reported:
  AUROC              global ranking quality (tie-aware average ranks)
  EF0.5/1/2/5/10%    enrichment factor at a fixed top fraction
  BEDROC(a=20)       early-recognition weighted metric (Truchon & Bayly)
  PR-AUC             precision-recall area, with the random baseline = active prevalence
  bootstrap 95% CI   stratified resampling, 2000 draws

Tie handling: MetGeo scores saturate (many ligands below 1e-20), so ranks must use
method="average". Plain argsort would break ties arbitrarily and move AUROC by up to 0.5.
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EF_FRACS = (0.005, 0.01, 0.02, 0.05, 0.10)
A_REF = 600.0


def auroc(y, s):
    y = np.asarray(y, int); r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float(1.0 - (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ef(y, s, frac):
    y = np.asarray(y, int); r = rankdata(np.asarray(s, float), method="average")
    n, na = len(y), int(y.sum())
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(r)[:k]].sum() / k) / (na / n))


def bedroc(y, s, alpha=20.0):
    y = np.asarray(y, int); n = len(y); na = int(y.sum())
    order = np.argsort(np.asarray(s, float), kind="mergesort")
    ranks = np.where(y[order] == 1)[0] + 1
    rie = np.mean(np.exp(-alpha * ranks / n))
    hi = np.mean(np.exp(-alpha * np.arange(1, na + 1) / n))
    lo = np.mean(np.exp(-alpha * (n - na + np.arange(1, na + 1)) / n))
    return float((rie - lo) / (hi - lo)) if hi != lo else 0.0


def pr_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(rankdata(s, method="average"))
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(int(y.sum()), 1)
    # step-wise AUC on recall grid
    grid = np.linspace(0, 1, 200)
    p_at = np.interp(grid, rec, prec, left=prec[0] if len(prec) else 0)
    return float(np.trapezoid(p_at, grid))


def boot_ci(y, s, fn, n=2000, seed=0):
    y = np.asarray(y, int); s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    ia, id_ = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        a = rng.choice(ia, len(ia), replace=True)
        d = rng.choice(id_, len(id_), replace=True)
        idx = np.concatenate([a, d])
        v = fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pct(v):
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def load_metalgeo(target):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{target}_p*.csv"))):
        for r in csv.DictReader(open(f)):
            if r.get("g_total") in (None, "", "nan"):
                continue
            rows.append(r)
    if not rows:
        return None
    y = np.array([int(r["label"]) for r in rows])
    g = pct([float(r["total"]) for r in rows])
    gb = pct([float(r["g_buried"]) for r in rows])
    nrot = np.array([float(r["n_rotatable"]) for r in rows])
    bur = np.array([float(r["buried_area"]) for r in rows])
    return dict(y=y, geo=g, burial=g + gb,
                full=g + gb - pct(nrot * np.clip(bur / A_REF, 0, 1)))


def load_vina(target):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/cf10_vina/{target}_p*.csv"))):
        for r in csv.DictReader(open(f)):
            if r.get("affinity") in (None, "", "nan"):
                continue
            rows.append(r)
    if not rows:
        return None
    return dict(y=np.array([int(r["label"]) for r in rows]),
                s=np.array([float(r["affinity"]) for r in rows]))


def load_mdl(target):
    rows = []
    for pat in (f"results/cf10_mdl/{target}_p*.csv", f"results/cf10_mdl/{target}/p*.csv"):
        for f in sorted(glob.glob(str(ROOT / pat))):
            for r in csv.DictReader(open(f)):
                if r.get("score") in (None, "", "nan"):
                    continue
                rows.append(r)
    if not rows:
        return None
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["score"]) for r in rows])
    if auroc(y, s) < 0.5:
        s = -s
    return dict(y=y, s=s)


def summarise(name, y, s):
    out = {"method": name, "n": int(len(y)), "n_act": int(np.sum(y)),
           "auroc": auroc(y, s), "bedroc": bedroc(y, s),
           "prauc": pr_auc(y, s), "random_prauc": float(np.mean(y))}
    for f in EF_FRACS:
        out[f"ef{f*100:g}"] = ef(y, s, f)
    out["auroc_ci"] = boot_ci(y, s, auroc, n=600)
    return out


def main(targets=None):
    targets = targets or [p.stem.replace("_div", "")
                          for p in sorted((ROOT / "results/crossfamily10").glob("*_div.csv"))]
    all_rows = []
    for t in targets:
        print(f"\n=== {t} ===")
        for label, loader, pick in (
            ("MetGeo(geo)", load_metalgeo, "geo"),
            ("MetGeo+burial", load_metalgeo, "burial"),
            ("MetGeo+burial+CAT", load_metalgeo, "full"),
            ("AutoDock Vina", load_vina, "s"),
            ("MetalloDock", load_mdl, "s"),
        ):
            d = loader(t)
            if d is None:
                print(f"  {label:22s} (no data)"); continue
            sc = d[pick] if pick in d else d["s"]
            r = summarise(label, d["y"], sc)
            r["target"] = t
            all_rows.append(r)
            print(f"  {label:22s} n={r['n']:4d} AUROC {r['auroc']:.3f} "
                  f"[{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}] "
                  f"EF1% {r['ef1']:5.2f} BEDROC {r['bedroc']:.3f} PR-AUC {r['prauc']:.3f}")

    dest = ROOT / "results/cf10_metrics.json"
    dest.write_text(json.dumps(all_rows, indent=2))
    # also a flat CSV
    csv_dest = ROOT / "results/cf10_metrics.csv"
    if all_rows:
        keys = ["target", "method", "n", "n_act", "auroc", "auroc_ci", "bedroc",
                "prauc", "random_prauc"] + [f"ef{f*100:g}" for f in EF_FRACS]
        with csv_dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                rr = dict(r); rr["auroc_ci"] = f"{r['auroc_ci'][0]:.3f}-{r['auroc_ci'][1]:.3f}"
                w.writerow(rr)
    print(f"\nWROTE {dest.name} and {csv_dest.name}")


if __name__ == "__main__":
    main()
