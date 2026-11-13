"""Fit b and c from the cross-family metalloenzyme set (HDAC excluded), then test on HDAC.

    score = 1*E + b*H_free - c*n_clusters

Training data: 666 ligands over 12 metalloenzyme targets spanning 8 metals, none of
which is an HDAC. Labels are real activities of those targets (pactivity >= 6 active,
<= 5 decoy).

The model is fitted twice for comparison:
  linear  : two interpretable coefficients (b, c)
  MLP     : small network that can capture non-linear interactions

Both are then applied unchanged to the HDAC benchmark; any gain there is out-of-family
transfer, since HDAC contributed no training data.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
HDAC = Path("/home/wangkai/cursor-task/HDAC")
TARGETS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]


def load_train():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/crossfamily/*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                E, H, N = float(r["E"]), float(r["H_free"]), float(r["n_clusters"])
                lab = int(r["label"])
            except Exception:
                continue
            if np.isfinite(E) and np.isfinite(H) and np.isfinite(N):
                rows.append({"E": E, "H": H, "N": N, "y": lab,
                             "target": r["target_name"], "metal": r["metal"]})
    return rows


def zcols(rows):
    E = np.array([r["E"] for r in rows]); H = np.array([r["H"] for r in rows]); N = np.array([r["n_clusters"] if "n_clusters" in r else r["N"] for r in rows])
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    return np.column_stack([z(E), z(H), -z(N)])


def fit_linear(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    w = -clf.coef_[0]              # lower score = active
    if w[0] < 0:
        w = -w
    return w / w[0], clf


def fit_mlp(X, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                                      random_state=42, alpha=1e-3, early_stopping=True))
    clf.fit(X, y)
    return clf


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def load_hdac():
    """HDAC small benchmark: E from docking, n_clusters from entropy runs, H_free from hfree."""
    hf_act, hf_dec = {}, {}
    for t in TARGETS:
        f = ROOT / "results/hfree_big" / f"{t}_actives.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                hf_act[r["ligand_id"]] = float(r["H_free"]) if r["H_free"] else np.nan
    nc_act, nc_dec = {}, {}
    for t in TARGETS:
        f = ROOT / "results/entropy_parts" / f"{t}.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                nc_act[r["ligand_id"]] = float(r["n_clusters"])
        f = ROOT / "results/entropy_decoys" / f"{t}.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                nc_dec[r["ligand_id"]] = float(r["n_clusters"])
    out = {}
    for t in TARGETS:
        f = HDAC / "docking/results" / f"{t}.csv"
        if not f.exists():
            continue
        rows = []
        for r in csv.DictReader(open(f)):
            lid = r["ligand_id"]
            if r.get("score") in (None, "", "nan"):
                continue
            is_act = lid in nc_act
            nc = nc_act.get(lid, nc_dec.get(lid, np.nan))
            hf = hf_act.get(lid, np.nan)
            rows.append({"ligand": lid, "y": int(is_act), "E": float(r["score"]),
                         "H": hf, "N": nc})
        if sum(x["y"] for x in rows) >= 10:
            out[t] = rows
    return out


def main():
    tr = load_train()
    print(f"training ligands: {len(tr)}  (actives {sum(r['y'] for r in tr)}) "
          f"targets {len({r['target'] for r in tr})} metals {len({r['metal'] for r in tr})}")
    X, y = zcols(tr), np.array([r["y"] for r in tr])

    w, _ = fit_linear(X, y)
    print(f"\n[linear]  a=1.000  b={w[1]:+.3f}  c={w[2]:+.3f}")
    print(f"  training AUROC (E only)      {auroc(y, X[:, 0]):.3f}")
    print(f"  training AUROC (E+bH-cN)     {auroc(y, X @ w):.3f}")

    mlp = fit_mlp(X, y)
    print(f"  training AUROC (MLP)         {auroc(y, mlp.predict_proba(X)[:, 1]):.3f}")

    hd = load_hdac()
    print(f"\n=== HDAC transfer test ({len(hd)} subtypes) ===")
    print(f"{'target':8s} {'n':>4s} {'A only':>8s} {'E only':>8s} {'E+bH-cN':>9s} {'MLP':>7s} {'EF1 E':>7s} {'EF1 lin':>8s}")
    aggs = {"E": [], "lin": [], "mlp": []}
    for t, rows in hd.items():
        keep = [r for r in rows if np.isfinite(r["E"]) and np.isfinite(r["H"]) and np.isfinite(r["N"])]
        if len(keep) < 20 or len({r["y"] for r in keep}) < 2:
            continue
        E = np.array([r["E"] for r in keep]); H = np.array([r["H"] for r in keep])
        N = np.array([r["N"] for r in keep]); yy = np.array([r["y"] for r in keep])
        def z(v):
            return (v - v.mean()) / (v.std() + 1e-9)
        Z = np.column_stack([z(E), z(H), -z(N)])
        s_lin = Z @ w
        s_mlp = -mlp.predict_proba(Z)[:, 1]
        a_E, a_lin, a_mlp = auroc(yy, -E), auroc(yy, -s_lin), auroc(yy, -s_mlp)
        aggs["E"].append(a_E); aggs["lin"].append(a_lin); aggs["mlp"].append(a_mlp)
        print(f"{t:8s} {len(keep):4d} {'-':>8s} {a_E:8.3f} {a_lin:9.3f} {a_mlp:7.3f} "
              f"{ef(yy, E, 0.01):7.2f} {ef(yy, s_lin, 0.01):8.2f}")
    print(f"\nmean AUROC: E only {np.mean(aggs['E']):.3f} | linear {np.mean(aggs['lin']):.3f} "
          f"| MLP {np.mean(aggs['mlp']):.3f}")

    out = {"n_train_ligands": len(tr), "n_train_targets": len({r["target"] for r in tr}),
           "coefficients_linear": {"a": 1.0, "b_H_free": round(float(w[1]), 4),
                                   "c_n_clusters": round(float(w[2]), 4)},
           "hdac_mean_auroc": {"E_only": round(float(np.mean(aggs["E"])), 4),
                               "linear": round(float(np.mean(aggs["lin"])), 4),
                               "mlp": round(float(np.mean(aggs["mlp"])), 4)}}
    (ROOT / "results/crossfamily_coefficients.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE results/crossfamily_coefficients.json")


if __name__ == "__main__":
    main()
