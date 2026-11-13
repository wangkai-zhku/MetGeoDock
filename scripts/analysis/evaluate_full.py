"""Unified virtual-screening evaluation for the HDAC benchmark.

Metrics (following the review recommendations):
  AUROC with stratified bootstrap 95% CI (molecules resampled within each target,
      targets resampled at the outer level)
  EF1% / EF5% / EF10%   (with CI)
  BEDROC alpha = 20 and 80 (Truchon & Bayly 2007)
  Top-N recall (N = 50, 100, 200 - directly interpretable as assay throughput)
  PR-AUC with the correct random baseline (actives fraction, NOT 0.5)

MCC is deliberately NOT reported: it needs a binarisation threshold that is
arbitrary for a continuous docking score.

Usage:
  evaluate_full.py --actives <dir> --decoys <glob-dir> --out <prefix> [--tag name]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

TARGETS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]
ALPHAS = (20.0, 80.0)
TOPN = (50, 100, 200)
N_BOOT = 2000


# ----------------------------------------------------------------- primitives
def auroc(y, s):
    """AUROC where LOWER score = better binder."""
    y = np.asarray(y)
    s = -np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0 or n == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def topn_recall(y, s, k):
    y = np.asarray(y)
    s = np.asarray(s, float)
    na = int(y.sum())
    if na == 0:
        return float("nan")
    k = min(k, len(y))
    return float(y[np.argsort(s)[:k]].sum() / na)


def bedroc(y, s, alpha):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0 or n == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    r = ranks[y == 1]
    num = np.exp(-alpha * r / n).sum()
    den = na * (1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0)
    return float(num / den)


def pr_auc(y, s):
    """Average precision; the random baseline equals the actives fraction."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    na = int(y.sum())
    if na == 0:
        return float("nan")
    order = np.argsort(s)
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / na)


# ------------------------------------------------------------------- bundling
def load_target(actives_dir: Path, decoy_dir: Path, t: str, labels: dict):
    act = []
    f = actives_dir / f"{t}.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            if labels.get(r["ligand_id"]) != 1:
                continue
            try:
                act.append(float(r["score"]))
            except Exception:
                pass
    dec = []
    for f in sorted(glob.glob(str(decoy_dir / f"{t}_part*.csv"))):
        for r in csv.DictReader(open(f)):
            if r.get("score") in (None, "", "nan"):
                continue
            try:
                dec.append(float(r["score"]))
            except Exception:
                pass
    return act, dec


def boot_ci(values_by_target, fn, n=N_BOOT, seed=7):
    """Stratified bootstrap: resample molecules inside targets, targets outside."""
    rng = np.random.default_rng(seed)
    names = list(values_by_target)
    out = []
    for _ in range(n):
        picked = rng.choice(names, size=len(names), replace=True)
        vals = []
        for name in picked:
            y, s = values_by_target[name]
            idx = rng.integers(0, len(y), len(y))
            yy, ss = y[idx], s[idx]
            if len(np.unique(yy)) < 2:
                continue
            vals.append(fn(yy, ss))
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out.append(float(np.mean(vals)))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actives", type=Path, required=True, help="dir with <TARGET>.csv actives scores")
    ap.add_argument("--decoys", type=Path, required=True, help="dir with <TARGET>_part*.csv decoys scores")
    ap.add_argument("--labels", type=Path, required=True, help="hdac_dataset.csv for label mapping")
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--tag", default="metgeo")
    a = ap.parse_args()

    labels = {r["ligand_id"]: int(r["label"]) for r in csv.DictReader(open(a.labels))}
    per_target = {}
    for t in TARGETS:
        act, dec = load_target(a.actives, a.decoys, t, labels)
        if len(act) < 10 or len(dec) < 100:
            print(f"{t}: skipped (actives={len(act)}, decoys={len(dec)})")
            continue
        y = np.array([1] * len(act) + [0] * len(dec), int)
        s = np.array(act + dec, float)
        per_target[t] = (y, s)
        print(f"{t}: {len(act)} actives + {len(dec)} decoys", flush=True)

    if not per_target:
        print("no target had enough data")
        return

    rows = []
    summary = {"tag": a.tag, "n_targets": len(per_target), "metrics": {}}
    metric_fns = {
        "auroc": auroc,
        "ef1": lambda y, s: ef(y, s, 0.01),
        "ef5": lambda y, s: ef(y, s, 0.05),
        "ef10": lambda y, s: ef(y, s, 0.10),
        "bedroc20": lambda y, s: bedroc(y, s, 20.0),
        "bedroc80": lambda y, s: bedroc(y, s, 80.0),
        "top50_recall": lambda y, s: topn_recall(y, s, 50),
        "top100_recall": lambda y, s: topn_recall(y, s, 100),
        "top200_recall": lambda y, s: topn_recall(y, s, 200),
        "prauc": pr_auc,
    }

    for t, (y, s) in per_target.items():
        rec = {"target": t, "n": len(y), "n_actives": int(y.sum()),
               "random_prauc_baseline": round(float(y.mean()), 4)}
        for name, fn in metric_fns.items():
            rec[name] = round(fn(y, s), 4)
        rows.append(rec)

    print(f"\n=== per-target ({a.tag}) ===")
    keys = ["auroc", "ef1", "ef5", "ef10", "bedroc20", "bedroc80", "top100_recall", "prauc"]
    print(f"{'target':8s} " + " ".join(f"{k:>13s}" for k in keys))
    for r in rows:
        print(f"{r['target']:8s} " + " ".join(f"{r[k]:13.3f}" for k in keys))

    print("\n=== cross-target mean with stratified bootstrap 95% CI ===")
    for name, fn in metric_fns.items():
        vals = [fn(y, s) for y, s in per_target.values()]
        mean = float(np.nanmean(vals))
        lo, hi = boot_ci(per_target, fn)
        summary["metrics"][name] = {"mean": round(mean, 4),
                                    "ci95": [round(lo, 4), round(hi, 4)],
                                    "n_targets": int(np.isfinite(vals).sum())}
        print(f"  {name:14s} {mean:7.3f}  [{lo:7.3f}, {hi:7.3f}]")

    summary["per_target"] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with (a.out.parent / f"{a.out.name}_per_target.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (a.out.parent / f"{a.out.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWROTE {a.out.name}_per_target.csv and _summary.json")


if __name__ == "__main__":
    main()
