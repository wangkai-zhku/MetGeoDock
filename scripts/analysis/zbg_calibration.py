"""Per-ZBG-class calibration of the entropy coefficients b and c.

Model (fixed a=1 as requested):
    score = E + b * H_free - c * n_clusters      (lower = better)
b and c are fitted inside each ligand-chemistry class, never globally, because the
cross-family fit produced a sign flip (b=-4.6) that does not transfer.

Why classes are defined by the zinc-binding group rather than by pocket shape:
the pocket descriptors do not separate the targets (CA2 and MMP1 have identical
coordination number and donor composition) while the ZBG does (sulfonamide 0.65 vs
hydroxamate 0.38 mean AUROC). Within a class the pocket-depth variation is then
absorbed by that class's own coefficients.

Protocol: leave-one-target-out. For each held-out target the coefficients come only
from the other targets of the same class, so no activity data of the test target is
ever used.

Also reports a "class baseline": the class mean AUROC of plain E, which is the
reference number to quote when a class has too few targets to calibrate.
"""
from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
ZBG = {
    "CA2": "sulfonamide", "CA12": "sulfonamide", "CA1": "sulfonamide", "CA9": "sulfonamide",
    "DBH": "sulfonamide", "METAP2": "sulfonamide", "ARG1": "sulfonamide", "ARG2": "sulfonamide",
    "CDK2": "sulfonamide", "urease_note": "sulfonamide", "MMP2": "sulfonamide",
    "HDAC1": "hydroxamate", "HDAC2": "hydroxamate", "HDAC3": "hydroxamate",
    "HDAC4": "hydroxamate", "HDAC6": "hydroxamate", "HDAC7": "hydroxamate",
    "HDAC8": "hydroxamate", "MMP9": "hydroxamate", "MMP3": "hydroxamate",
    "MMP13": "hydroxamate", "ALOX5": "hydroxamate",
    "MMP1": "carboxylate", "ADAM17": "carboxylate", "ACE": "carboxylate", "CAPN1": "carboxylate",
    "CAPN2": "carboxylate", "PLA2G1B": "other", "ASS1": "other", "CYP19A1": "other",
    "CYP1A2": "other", "CYP2B6": "other", "CYP2D6": "other", "CYP2E1": "other", "CYP3A4": "other",
    "GLO1": "sulfonamide", "TYR": "carboxylate", "HIV1_integrase": "other",
}


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
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def z(v):
    """Percentile-rank normalisation (robust to the score's huge dynamic range)."""
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def load():
    per = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "results/entropy_big/*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                e = float(r["E_best"]); h = float(r["h_free"]); n = float(r["n_clusters"])
                lab = int(r["label"])
            except Exception:
                continue
            if not np.isfinite(e) or not np.isfinite(h) or n <= 0:
                continue
            per[r["target"]].append({"E": e, "H": h, "N": n, "y": lab})
    return {t: v for t, v in per.items() if len(v) >= 20 and len({x["y"] for x in v}) == 2}


def fit_coeffs(rows, ridge=1e-3):
    """Least squares: y in {0,1} ~ 1 + z(E)*(-1) + z(H)*b + z(N)*c  (sign convention below)."""
    E = z([r["E"] for r in rows]); H = z([r["H"] for r in rows]); N = z([r["N"] for r in rows])
    y = np.array([r["y"] for r in rows], float)
    # score = E + b*H - c*N ; we want score LOW for y=1 -> fit y ~ -(E + bH - cN)
    X = np.column_stack([H, -N])
    target = -(y - y.mean())
    A = X.T @ X + ridge * np.eye(2)
    beta = np.linalg.solve(A, X.T @ target)
    return float(beta[0]), float(beta[1])


def apply_coeffs(rows, b, c):
    E = z([r["E"] for r in rows]); H = z([r["H"] for r in rows]); N = z([r["N"] for r in rows])
    return E + b * H - c * N


def main():
    data = load()
    print(f"targets with E+H_free+n_clusters: {len(data)}")
    classes = defaultdict(list)
    for t in data:
        classes[ZBG.get(t, "other")].append(t)

    print(f"\n{'class':13s} {'n_t':>4s} {'E only':>8s} {'+H only':>9s} {'+N only':>9s} {'+H+N (LOO)':>11s}")
    summary = {}
    for cls, targets in sorted(classes.items()):
        targets = sorted(targets)
        if len(targets) < 2:
            # single target: no calibration possible -> baseline only
            t = targets[0]; rows = data[t]
            a = auroc([r["y"] for r in rows], [r["E"] for r in rows])
            summary[cls] = {"n_targets": 1, "E": a, "calibrated": None, "targets": {t: a}}
            print(f"{cls:13s} {1:4d} {a:8.3f} {'-':>9s} {'-':>9s} {'baseline only':>11s}")
            continue
        a_e, a_h, a_n, a_cal, per_t = [], [], [], [], {}
        for t in targets:
            rows = data[t]
            y = [r["y"] for r in rows]
            E = z([r["E"] for r in rows])
            a_e.append(auroc(y, E))
            a_h.append(auroc(y, E + 1.0 * z([r["H"] for r in rows])))
            a_n.append(auroc(y, E - 1.0 * z([r["N"] for r in rows])))
            train = [r for u in targets if u != t for r in data[u]]
            b, c = fit_coeffs(train)
            s = apply_coeffs(rows, b, c)
            a_cal.append(auroc(y, s))
            per_t[t] = {"E": round(auroc(y, E), 3), "cal": round(auroc(y, s), 3), "b": round(b, 2), "c": round(c, 2)}
        summary[cls] = {"n_targets": len(targets), "E": float(np.mean(a_e)),
                        "H": float(np.mean(a_h)), "N": float(np.mean(a_n)),
                        "calibrated": float(np.mean(a_cal)), "targets": per_t}
        print(f"{cls:13s} {len(targets):4d} {np.mean(a_e):8.3f} {np.mean(a_h):9.3f} "
              f"{np.mean(a_n):9.3f} {np.mean(a_cal):11.3f}")

    print("\n=== per-target (LOO calibration within class) ===")
    for cls, s in sorted(summary.items()):
        print(f"\n[{cls}]  E={s['E']:.3f}  calibrated={s['calibrated'] if s['calibrated'] is None else round(s['calibrated'],3)}")
        for t, v in sorted(s["targets"].items(), key=lambda kv: -(kv[1]["E"] if isinstance(kv[1], dict) else kv[1])):
            if isinstance(v, dict):
                print(f"   {t:14s} E={v['E']:.3f}  calibrated={v['cal']:.3f}  (b={v['b']:+.2f}, c={v['c']:+.2f})")
            else:
                print(f"   {t:14s} E={v:.3f}   (baseline only)")

    import json
    out = ROOT / "results/zbg_calibration.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
