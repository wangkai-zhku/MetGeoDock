"""Screening metrics for the cross-family-10 benchmark: MetalGeoDock (3 variants),
AutoDock Vina and MetalloDock, all on the same ligand sets and receptors.

Conventions
-----------
* Every score is normalised to "lower is better" before ranking:
    MetalGeoDock  total            (energy-like, lower = better)
    MetalGeoDock  g_buried         (burial free energy, lower = better)
    MetalGeoDock  CAT              n_rot * clip(buried/600)  (penalty, lower = better)
    Vina          affinity         (kcal/mol, more negative = better)
    MetalloDock   -score           (MDN score, higher = better -> negated)
  No per-target sign flipping is applied anywhere; that would inflate AUROC.
* Rows from corrupt CSV lines (two processes writing the same file), 0-byte
  files and rows with a missing score are dropped, and reported as counts.
* A target is marked COMPLETE only when all 4 parts finished with a DONE line
  and no partial file is present; incomplete targets are still scored but
  flagged so they never end up in a headline number by accident.

Metrics: AUROC, EF1/5/10%, BEDROC(alpha=20), PR-AUC, bootstrap 95% CI on AUROC
(stratified resampling of actives and decoys, 2000 draws).

Usage
-----
    python scripts/analysis/cf10_metrics_all.py                    # all methods
    python scripts/analysis/cf10_metrics_all.py --methods mg,vina  # subset
    python scripts/analysis/cf10_metrics_all.py --bootstrap 500    # faster
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
A_REF = 600.0
EF_FRACS = (0.01, 0.05, 0.10)
TARGETS = ["MMP2", "CA2", "HDAC1", "CAPN1", "METAP2", "CDK2", "urease",
           "ALOX5", "DBH", "ARG2", "HDAC2"]


# ---------------------------------------------------------------- metrics
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """P(score_active < score_decoy); average ranks keep saturated ties neutral."""
    r = rankdata(s, method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(1.0 - (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ef(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    r = rankdata(s, method="average")
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(r)[:k]].sum() / k) / (na / n))


def bedroc(y: np.ndarray, s: np.ndarray, alpha: float = 20.0) -> float:
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    order = np.argsort(np.asarray(s, float), kind="mergesort")
    ranks = np.where(y[order] == 1)[0] + 1
    rie = np.mean(np.exp(-alpha * ranks / n))
    hi = np.mean(np.exp(-alpha * np.arange(1, na + 1) / n))
    lo = np.mean(np.exp(-alpha * (n - na + np.arange(1, na + 1)) / n))
    return float((rie - lo) / (hi - lo)) if hi != lo else 0.0


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(rankdata(s, method="average"))
    yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    grid = np.linspace(0, 1, 200)
    return float(np.trapezoid(np.interp(grid, rec, prec, left=prec[0]), grid))


def boot_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Stratified bootstrap CI for AUROC."""
    rng = np.random.default_rng(seed)
    ia, id_ = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ia, len(ia), replace=True),
                              rng.choice(id_, len(id_), replace=True)])
        v = auroc(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pct(v) -> np.ndarray:
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def summarise(name: str, target: str, y: np.ndarray, s: np.ndarray, n_boot: int) -> dict:
    out = {"method": name, "target": target, "n": int(len(y)), "n_act": int(y.sum()),
           "auroc": auroc(y, s), "bedroc": bedroc(y, s), "prauc": pr_auc(y, s),
           "prand": float(np.mean(y))}
    for f in EF_FRACS:
        out[f"ef{f * 100:g}"] = ef(y, s, f)
    lo, hi = boot_ci(y, s, n_boot)
    out["ci_lo"], out["ci_hi"] = lo, hi
    return out


# ---------------------------------------------------------------- loaders
def read_rows(files, need: str, extra=()):
    """Read CSV rows, dropping corrupt lines and rows without a usable score."""
    rows, corrupt, missing = [], 0, 0
    for f in files:
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames:
                continue                      # 0-byte file from an interrupted run
            nf = len(rd.fieldnames)
            for r in rd:
                if None in r or len(r) != nf:
                    corrupt += 1
                    continue
                v = (r.get(need) or "").strip()
                if v in ("", "nan"):
                    missing += 1
                    continue
                try:
                    float(v)
                    for k in extra:
                        float(r[k])
                except (ValueError, TypeError):
                    missing += 1
                    continue
                rows.append(r)
    return rows, corrupt, missing


def done_parts(logdir: Path, target: str) -> int:
    n = 0
    for f in sorted(glob.glob(str(logdir / f"{target}_p*.log"))):
        try:
            if "DONE" in Path(f).read_text(errors="ignore"):
                n += 1
        except OSError:
            pass
    return n


def load_metalgeo(target: str):
    files = sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{target}_p*.csv")))
    rows, corrupt, missing = read_rows(
        files, "g_total", ("total", "g_buried", "n_rotatable", "buried_area"))
    if not rows:
        return None
    y = np.array([int(r["label"]) for r in rows])
    g = pct([float(r["total"]) for r in rows])
    gb = pct([float(r["g_buried"]) for r in rows])
    nrot = np.array([float(r["n_rotatable"]) for r in rows])
    bur = np.array([float(r["buried_area"]) for r in rows])
    cat = pct(nrot * np.clip(bur / A_REF, 0, 1))
    return {"y": y, "geo": g, "burial": g + gb, "full": g + gb - cat,
            "corrupt": corrupt, "missing": missing, "nfiles": len(files)}


def load_vina(target: str):
    files = sorted(glob.glob(str(ROOT / f"results/cf10_vina/{target}_p*.csv")))
    rows, corrupt, missing = read_rows(files, "affinity")
    if not rows:
        return None
    return {"y": np.array([int(r["label"]) for r in rows]),
            "s": np.array([float(r["affinity"]) for r in rows]),
            "corrupt": corrupt, "missing": missing, "nfiles": len(files)}


def load_mdl(target: str):
    files = (sorted(glob.glob(str(ROOT / f"results/cf10_mdl/{target}_p*.csv")))
             or sorted(glob.glob(str(ROOT / f"results/cf10_mdl/{target}/p*.csv"))))
    rows, corrupt, missing = read_rows(files, "score")
    if not rows:
        return None
    # MetalloDock MDN score: higher = better (actives 54.0 vs decoys 43.6 on HDAC1)
    return {"y": np.array([int(r["label"]) for r in rows]),
            "s": -np.array([float(r["score"]) for r in rows]),
            "corrupt": corrupt, "missing": missing, "nfiles": len(files)}


METHODS = {
    "mg":   ("MetalGeoDock(geo)", load_metalgeo, "geo",
             ROOT / "logs/cf10"),
    "mgb":  ("MetalGeoDock+burial", load_metalgeo, "burial",
             ROOT / "logs/cf10"),
    "mgc":  ("MetGeoDock+burial+CAT", load_metalgeo, "full",
             ROOT / "logs/cf10"),
    "vina": ("AutoDock Vina", load_vina, "s",
             ROOT / "logs/cf10vina"),
    "mdl":  ("MetalloDock", load_mdl, "s",
             ROOT / "logs/cf10mdl"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="mg,mgb,mgc,vina,mdl")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out-prefix", default="results/cf10_metrics_all")
    a = ap.parse_args()

    wanted = [m.strip() for m in a.methods.split(",") if m.strip() in METHODS]
    all_rows, completeness = [], []

    for key in wanted:
        label, loader, pick, logdir = METHODS[key]
        print()
        print("=" * 104)
        print(f"{label}")
        print("=" * 104)
        print(f"{'target':<8}{'part':>6}{'n':>6}{'AUROC':>7}{'95% CI':>16}"
              f"{'EF1%':>7}{'EF5%':>7}{'EF10%':>8}{'BEDROC':>8}{'PR-AUC':>8}{'':>4}")
        per = []
        for t in TARGETS:
            d = loader(t)
            if d is None:
                continue
            y, s = d["y"], d[pick]
            if len(set(y.tolist())) < 2:
                continue
            r = summarise(label, t, y, s, a.bootstrap)
            dp = done_parts(logdir, t)
            r["parts_done"] = dp
            r["complete"] = bool(dp == 4 and d["nfiles"] == 4 and d["corrupt"] == 0)
            r["corrupt_rows"] = d["corrupt"]
            r["missing_rows"] = d["missing"]
            all_rows.append(r)
            per.append(r)
            flag = "" if r["complete"] else " *"
            ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            print(f"{t:<8}{dp:>4}/4{r['n']:>6}{r['auroc']:>7.3f}{ci:>16}"
                  f"{r['ef1']:>7.2f}{r['ef5']:>7.2f}{r['ef10']:>8.2f}"
                  f"{r['bedroc']:>8.3f}{r['prauc']:>8.3f}{flag:>4}")
        good = [r for r in per if r["complete"]]
        for tag, grp in (("全部靶点", per), ("仅完整靶点", good)):
            if not grp:
                continue
            f = lambda k: float(np.mean([r[k] for r in grp]))
            med = lambda k: float(np.median([r[k] for r in grp]))
            print(f"{tag:<8}{len(grp):>4}个{f('auroc'):>10.3f}{'':>16}"
                  f"{f('ef1'):>7.2f}{f('ef5'):>7.2f}{f('ef10'):>8.2f}"
                  f"{f('bedroc'):>8.3f}{f('prauc'):>8.3f}  (宏平均)")
            print(f"{'':<8}{'':>5}{'':>5}{med('auroc'):>7.3f}{'':>16}"
                  f"{med('ef1'):>7.2f}{med('ef5'):>7.2f}{med('ef10'):>8.2f}"
                  f"{med('bedroc'):>8.3f}{med('prauc'):>8.3f}  (中位数)")
        completeness.append((label, len(per), len(good)))

    out = Path(ROOT / a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_rows, open(out.with_suffix(".json"), "w"), indent=2)
    keys = ["method", "target", "n", "n_act", "auroc", "ci_lo", "ci_hi",
            "ef1", "ef5", "ef10", "bedroc", "prauc", "prand",
            "parts_done", "complete", "corrupt_rows", "missing_rows"]
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print()
    print("=" * 104)
    print("完成度概览")
    print("=" * 104)
    for label, n_all, n_ok in completeness:
        print(f"  {label:<26} 有数据 {n_all:>2} 靶点，其中 4/4 完整 {n_ok:>2} 个")
    print(f"\nWROTE {out.with_suffix('.csv').name} and {out.with_suffix('.json').name}")
    print("  (标 * 的靶点 part 未跑齐或含坏行，不要直接进正文)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
