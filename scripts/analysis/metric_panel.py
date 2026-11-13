"""Unified metric comparison across every scoring variant.

Computes the full metric panel requested for the manuscript - AUROC, EF1%, EF5%,
EF10%, BEDROC (alpha=20 and 80), PR-AUC with its random baseline, top-N recall,
plus stratified bootstrap 95% CIs - for each scoring variant on the same ligands:

  E                     the geometric term alone
  E + H                 geometric term plus conformational entropy (b = 1)
  E - N                 geometric term minus binding-mode count (c = 1)
  E + b*H - c*N (LOO)   chemistry-class-calibrated entropy coefficients

MCC is deliberately NOT reported: it needs a decision threshold, and a virtual
screening ranking has no natural one.

Outputs results/metric_panel.csv (one row per variant x scope) and prints a table.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
RNG = np.random.default_rng(20241024)
N_BOOT = 2000

ZBG = {
    "CA2": "sulfonamide", "CA12": "sulfonamide", "CA1": "sulfonamide", "CA9": "sulfonamide",
    "DBH": "sulfonamide", "METAP2": "sulfonamide", "ARG1": "sulfonamide", "ARG2": "sulfonamide",
    "CDK2": "sulfonamide", "urease_note": "sulfonamide", "MMP2": "sulfonamide", "GLO1": "sulfonamide",
    "HDAC1": "hydroxamate", "HDAC2": "hydroxamate", "HDAC3": "hydroxamate", "HDAC4": "hydroxamate",
    "HDAC6": "hydroxamate", "HDAC7": "hydroxamate", "HDAC8": "hydroxamate",
    "MMP9": "hydroxamate", "MMP3": "hydroxamate", "MMP13": "hydroxamate", "ALOX5": "hydroxamate",
    "MMP1": "carboxylate", "ADAM17": "carboxylate", "ACE": "carboxylate",
    "CAPN1": "carboxylate", "CAPN2": "carboxylate", "TYR": "carboxylate",
}


# ---------------------------------------------------------------- metrics
def auroc(y, s):
    """AUROC: P(score_active < score_decoy), i.e. LOWER score = better.

    Average ranks keep saturated ties neutral (each tied pair contributes 0.5).
    Note the 1 - U/(n1*n0): the raw Mann-Whitney statistic measures
    P(score_active > score_decoy), which is the opposite of what a
    lower-is-better score needs.
    """
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    u = r[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(1.0 - u / (n1 * n0))


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return np.nan
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s, kind="mergesort")[:k]].sum() / k) / (na / n))


def bedroc(y, s, alpha=20.0):
    """Truchon-Bayly BEDROC, min-max normalised to [0, 1].

    RIE = (1/(Ra*N)) * sum_over_actives exp(-alpha * rank / N), rank 1 = best.
    Normalised between the best possible (all actives at the top) and the worst
    possible (all actives at the bottom) ordering, so 1.0 = perfect, 0.0 = fully
    inverted and a random ranking falls near the ratio of actives.
    """
    y = np.asarray(y); s = np.asarray(s, float)
    n = len(y); na = int(y.sum())
    if na == 0 or na == n:
        return np.nan
    ra = na / n
    order = np.argsort(s, kind="mergesort")   # lower score = better
    rank_of = np.empty(n, dtype=int)
    rank_of[order] = np.arange(1, n + 1)      # rank_of[i] = rank of item i
    active_ranks = rank_of[y == 1]

    def rie(ar):
        return float(np.exp(-alpha * np.asarray(ar, float) / n).sum() / (ra * n))

    rie_obs = rie(active_ranks)
    rie_max = rie(np.arange(1, na + 1))
    rie_min = rie(np.arange(n - na + 1, n + 1))
    if abs(rie_max - rie_min) < 1e-12:
        return np.nan
    return float((rie_obs - rie_min) / (rie_max - rie_min))


def pr_auc(y, s):
    """Average precision (area under precision-recall)."""
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")   # lower score = better
    hits = y[order]
    tp = np.cumsum(hits)
    prec = tp / np.arange(1, n + 1)
    return float((prec * hits).sum() / na)


def topn_recall(y, s, n):
    y = np.asarray(y); s = np.asarray(s, float)
    na = int(y.sum())
    if na == 0:
        return np.nan
    k = min(n, len(y))
    return float(y[np.argsort(s, kind="mergesort")[:k]].sum() / na)


def boot_ci(y, s, fn, n_boot=N_BOOT):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(y) < 10 or len({*y}) < 2:
        return (np.nan, np.nan)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(fn(yy, ss))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ---------------------------------------------------------------- data
def z(v):
    """Percentile-rank normalisation in [0,1].

    z-scores break down here: E spans 30 orders of magnitude with many exact ties,
    and the arithmetic reshuffles them through floating-point noise. Rank
    normalisation keeps ties tied and is scale free.
    """
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def load():
    per = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "results/entropy_big/*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                e = float(r["E_best"]); h = float(r["h_free"])
                nn = float(r["n_clusters"]); lab = int(r["label"])
            except Exception:
                continue
            if not np.isfinite(e) or not np.isfinite(h):
                continue
            per[r["target"]].append({"E": e, "H": h, "N": nn, "y": lab,
                                     "metal": r.get("metal", "")})
    # Shuffle within each target: the CSVs list actives before decoys, and saturated
    # scores produce many ties. A stable sort would then hand every tie to the actives
    # and inflate EF/BEDROC far above AUROC. Fixed seed keeps this reproducible.
    rng = np.random.default_rng(7)
    out = {}
    for t, v in per.items():
        if len(v) < 20 or len({x["y"] for x in v}) != 2:
            continue
        idx = rng.permutation(len(v))
        out[t] = [v[i] for i in idx]
    return out


def fit_bc(rows):
    E = z([r["E"] for r in rows]); H = z([r["H"] for r in rows]); N = z([r["N"] for r in rows])
    y = np.array([r["y"] for r in rows], float)
    X = np.column_stack([H, -N])
    tgt = -(y - y.mean())
    beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(2), X.T @ tgt)
    return float(beta[0]), float(beta[1])


def main():
    data = load()
    classes = defaultdict(list)
    for t in data:
        classes[ZBG.get(t, "other")].append(t)
    print(f"targets: {len(data)}  classes: { {k: len(v) for k, v in classes.items()} }")

    panel = []
    per_target = []

    for t in sorted(data):
        rows = data[t]
        y = [r["y"] for r in rows]
        E = z([r["E"] for r in rows]); H = z([r["H"] for r in rows]); N = z([r["N"] for r in rows])
        # LOO within class
        cls = ZBG.get(t, "other")
        train = [r for u in classes[cls] if u != t for r in data[u]]
        if len(train) >= 20:
            b, c = fit_bc(train)
        else:
            b, c = 1.0, 0.0
        variants = {
            "E": E,
            "E + H": E + H,
            "E - N": E - N,
            "E + b·H − c·N (LOO)": E + b * H - c * N,
        }
        for name, s in variants.items():
            per_target.append({"target": t, "class": cls, "variant": name,
                               "n": len(rows), "n_act": int(sum(y)),
                               "auroc": auroc(y, s), "ef1": ef(y, s, 0.01),
                               "ef5": ef(y, s, 0.05), "ef10": ef(y, s, 0.10),
                               "bedroc20": bedroc(y, s, 20.0), "bedroc80": bedroc(y, s, 80.0),
                               "prauc": pr_auc(y, s), "top50": topn_recall(y, s, 50)})

    # aggregate per variant per scope
    for scope, targets in [("ALL", sorted(data))] + [(f"class:{c}", sorted(v)) for c, v in classes.items()]:
        for variant in ("E", "E + H", "E - N", "E + b·H − c·N (LOO)"):
            rows = [r for r in per_target if r["target"] in targets and r["variant"] == variant]
            if not rows:
                continue
            # pooled ligands for CI-bearing metrics
            yy, ss = [], []
            for t in targets:
                tr = data[t]
                if variant == "E":
                    s = z([r["E"] for r in tr])
                elif variant == "E + H":
                    s = z([r["E"] for r in tr]) + z([r["H"] for r in tr])
                elif variant == "E - N":
                    s = z([r["E"] for r in tr]) - z([r["N"] for r in tr])
                else:
                    cls = ZBG.get(t, "other")
                    train = [r for u in classes[cls] if u != t for r in data[u]]
                    b, c = fit_bc(train) if len(train) >= 20 else (1.0, 0.0)
                    s = z([r["E"] for r in tr]) + b * z([r["H"] for r in tr]) - c * z([r["N"] for r in tr])
                yy += [r["y"] for r in tr]; ss += list(s)
            au = float(np.nanmean([r["auroc"] for r in rows]))
            lo, hi = boot_ci(yy, ss, auroc)
            panel.append({
                "scope": scope, "variant": variant, "n_targets": len(targets),
                "n_ligands": len(yy), "n_act": int(sum(yy)),
                "auroc_mean": round(au, 3), "auroc_ci_lo": round(lo, 3), "auroc_ci_hi": round(hi, 3),
                "ef1": round(float(np.nanmean([r["ef1"] for r in rows])), 2),
                "ef5": round(float(np.nanmean([r["ef5"] for r in rows])), 2),
                "ef10": round(float(np.nanmean([r["ef10"] for r in rows])), 2),
                "bedroc20": round(float(np.nanmean([r["bedroc20"] for r in rows])), 3),
                "bedroc80": round(float(np.nanmean([r["bedroc80"] for r in rows])), 3),
                "prauc": round(float(np.nanmean([r["prauc"] for r in rows])), 3),
                "prauc_random": round(float(np.mean(yy)), 3),
                "top50": round(float(np.nanmean([r["top50"] for r in rows])), 3),
            })

    keys = ["scope", "variant", "n_targets", "n_ligands", "n_act", "auroc_mean",
            "auroc_ci_lo", "auroc_ci_hi", "ef1", "ef5", "ef10", "bedroc20", "bedroc80",
            "prauc", "prauc_random", "top50"]
    out = ROOT / "results/metric_panel.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(panel)

    out2 = ROOT / "results/metric_per_target.csv"
    keys2 = ["target", "class", "variant", "n", "n_act", "auroc", "ef1", "ef5", "ef10",
             "bedroc20", "bedroc80", "prauc", "top50"]
    with out2.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys2, extrasaction="ignore")
        w.writeheader(); w.writerows(per_target)

    # print ALL scope
    print(f"\n{'variant':24s} {'AUROC':>14s} {'EF1%':>6s} {'EF5%':>6s} {'BEDROC20':>9s} "
          f"{'PR-AUC':>7s} {'PRrand':>7s} {'Top50':>6s}")
    for r in panel:
        if r["scope"] != "ALL":
            continue
        print(f"{r['variant']:24s} {r['auroc_mean']:.3f} [{r['auroc_ci_lo']:.3f},{r['auroc_ci_hi']:.3f}] "
              f"{r['ef1']:6.2f} {r['ef5']:6.2f} {r['bedroc20']:9.3f} {r['prauc']:7.3f} "
              f"{r['prauc_random']:7.3f} {r['top50']:6.3f}")
    print(f"\nWROTE {out}\nWROTE {out2}")


if __name__ == "__main__":
    main()
