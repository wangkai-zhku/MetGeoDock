"""Large-library screening benchmark for HDAC subtypes.

Actives come from the curated ChEMBL table via docking/results/{target}.csv;
decoys from the per-subtype 5,690-molecule libraries under results/biglib_scores/.
Metrics use tie-aware average ranks, which matters because MetGeo saturates.

Outputs results/biglib_metrics.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")


def auroc(y, s):
    """P(score_active < score_decoy); average ranks keep saturated ties neutral."""
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    u = r[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(1.0 - u / (n1 * n0))


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


def boot_ci(y, s, fn, n_boot=2000, seed=20241024):
    """Stratified bootstrap CI: resample actives and decoys independently."""
    y = np.asarray(y); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pi = rng.integers(0, len(pos), len(pos))
        ni = rng.integers(0, len(neg), len(neg))
        ss = np.concatenate([pos[pi], neg[ni]])
        yy = np.concatenate([np.ones(len(pi), int), np.zeros(len(ni), int)])
        vals.append(fn(yy, ss))
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) < 10:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def pr_auc(y, s):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    h = y[np.argsort(r)]
    tp = np.cumsum(h)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * h).sum() / int(y.sum()))


def load(target):
    dec = []
    for f in sorted(glob.glob(str(ROOT / f"results/biglib_scores/{target}_part*.csv"))):
        for r in csv.DictReader(open(f)):
            if r.get("score") not in (None, "", "nan"):
                dec.append(float(r["score"]))
    act_ids = []
    p = HDAC / "docking/results" / f"{target}.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("score") not in (None, "", "nan") and "_A" in r["ligand_id"]:
                act_ids.append((r["ligand_id"], float(r["score"])))
    return [x[1] for x in act_ids], dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*",
                    default=["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/biglib_metrics.csv")
    a = ap.parse_args()

    print(f"{'target':8s} {'n_act':>5s} {'n_dec':>6s} {'AUROC':>7s} {'EF1%':>6s} "
          f"{'EF5%':>6s} {'BEDROC20':>9s} {'PR-AUC':>7s} {'PRrand':>7s}")
    rows = []
    for t in a.targets:
        act, dec = load(t)
        if len(act) < 10 or len(dec) < 100:
            print(f"{t:8s} {len(act):5d} {len(dec):6d}   (no big-library screen)")
            continue
        y = [1] * len(act) + [0] * len(dec)
        s = act + dec
        lo, hi = boot_ci(y, s, auroc)
        r = dict(target=t, n_act=len(act), n_dec=len(dec), auroc=auroc(y, s),
                 auroc_ci_lo=lo, auroc_ci_hi=hi,
                 ef1=ef(y, s, 0.01), ef5=ef(y, s, 0.05), ef10=ef(y, s, 0.10),
                 bedroc=bedroc(y, s, 20), bedroc80=bedroc(y, s, 80.0),
                 prauc=pr_auc(y, s), prand=float(np.mean(y)))
        rows.append(r)
        sig = "*" if lo > 0.5 else " "
        print(f"{t:8s} {len(act):5d} {len(dec):6d} {r['auroc']:7.3f} "
              f"[{lo:.3f},{hi:.3f}]{sig} {r['ef1']:6.2f} {r['ef5']:6.2f} "
              f"{r['bedroc']:9.3f} {r['prauc']:7.3f} {r['prand']:7.3f}")
    if rows:
        print(f"\nmean AUROC over {len(rows)} subtypes: "
              f"{np.mean([r['auroc'] for r in rows]):.3f}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
