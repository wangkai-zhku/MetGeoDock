"""Can interaction-fingerprint features rank an HDAC library?

geom   - the pose-level terms MetGeo/hybrid already use (+ ligand properties)
ifp    - residue x interaction-type fingerprint from the docked pose
Two schemes: within-target 5-fold CV, and leave-one-subtype-out transfer.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")
TARGETS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]
GEOM = ["slef", "gp_nll", "geo_rmsd_or", "clash", "dist_harm", "cn_pen", "n_coord_lig", "n_coord_pro"]
PROPS = ["mw", "clogp", "tpsa", "hbd", "hba", "rotb", "heavy", "rings", "qed", "fsp3",
         "nhoh", "benzamide", "carboxamide"]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    if len(set(y)) < 2:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


ACTIVES_ONLY = False


def load(ifp_path: Path):
    ifp = {r["ligand_id"]: r for r in csv.DictReader(open(ifp_path))}
    meta = {r["ligand_id"]: r for r in csv.DictReader(open(HDAC / "data/processed/hdac_dataset.csv"))}
    geom = {}
    for t in TARGETS:
        f = HDAC / "docking/results" / f"{t}.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                geom[r["ligand_id"]] = r
    rows = []
    for lid, f in ifp.items():
        m, g = meta.get(lid), geom.get(lid)
        if not m or not g:
            continue
        try:
            pch = float(m["pchembl"])
        except Exception:
            pch = float("nan")
        rec = {"ligand": lid, "target": m["target"], "label": int(m["label"]), "pchembl": pch}
        for k in GEOM:
            try: rec[k] = float(g[k])
            except Exception: rec[k] = 0.0
        for k in PROPS:
            try: rec[k] = float(m[k])
            except Exception: rec[k] = 0.0
        for k, v in f.items():
            if k.startswith("ifp_"):
                try: rec[k] = float(v)
                except Exception: rec[k] = 0.0
        if np.isfinite(rec["pchembl"]) and (not ACTIVES_ONLY or rec["label"] == 1):
            rows.append(rec)
    return rows


def matrix(rows, feats):
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], float)
    y = np.array([r["label"] for r in rows], int)
    return X, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y


def spearman(a, b):
    from scipy import stats
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok])[0])


def evaluate(rows, feats, scheme, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    rhos, aucs = [], []
    def model():
        return make_pipeline(StandardScaler(),
                             HistGradientBoostingRegressor(max_iter=300, random_state=seed))
    if scheme == "within":
        for t in TARGETS:
            sub = [r for r in rows if r["target"] == t]
            if len(sub) < 20:
                continue
            if (not ACTIVES_ONLY) and len({r["label"] for r in sub}) < 2:
                continue
            _, X, y = matrix(sub, feats)
            yreg = np.array([r["pchembl"] for r in sub], float)
            idx = rng.permutation(len(y))
            folds = np.array_split(idx, 5)
            pred = np.zeros(len(y))
            for k in range(5):
                te = folds[k]
                tr = np.concatenate([folds[j] for j in range(5) if j != k])
                m = model(); m.fit(X[tr], yreg[tr])
                pred[te] = m.predict(X[te])
            rhos.append(spearman(pred, yreg)); aucs.append(auroc(y, -pred))
    else:
        for t in TARGETS:
            tr = [r for r in rows if r["target"] != t]
            te = [r for r in rows if r["target"] == t]
            if len(tr) < 50 or len(te) < 20:
                continue
            if (not ACTIVES_ONLY) and (len({r["label"] for r in te}) < 2 or len({r["label"] for r in tr}) < 2):
                continue
            _, Xtr, ytr = matrix(tr, feats)
            _, Xte, yte = matrix(te, feats)
            ytr_reg = np.array([r["pchembl"] for r in tr], float)
            yte_reg = np.array([r["pchembl"] for r in te], float)
            m = model(); m.fit(Xtr, ytr_reg)
            pred = m.predict(Xte)
            rhos.append(spearman(pred, yte_reg)); aucs.append(auroc(yte, -pred))
    fr = [v for v in rhos if np.isfinite(v)]; fa = [v for v in aucs if np.isfinite(v)]
    return (float(np.mean(fr)) if fr else float("nan")), (float(np.mean(fa)) if fa else float("nan")), len(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifp", type=Path, default=ROOT / "results" / "hdac_ifp.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ifp_ranker.json")
    ap.add_argument("--actives-only", action="store_true",
                    help="restrict to label==1 rows (pChEMBL measured on THIS target)")
    a = ap.parse_args()
    global ACTIVES_ONLY
    ACTIVES_ONLY = bool(a.actives_only)
    rows = load(a.ifp)
    print(f"joined rows: {len(rows)}  actives={sum(r['label'] for r in rows)}")
    if not rows:
        print("no rows yet"); return
    ifp_feats = [k for k in rows[0] if k.startswith("ifp_")]
    sets = {"geom_only": GEOM, "props_only": PROPS, "ifp_only": ifp_feats,
            "geom+ifp": GEOM + ifp_feats, "all": GEOM + PROPS + ifp_feats}
    print(f"IFP features: {len(ifp_feats)}")
    print(f"{'feature set':14s} {'Spearman within CV':>19s} {'Spearman leave-1-target':>24s}")
    out = {}
    for name, feats in sets.items():
        w_rho, w_auc, nw = evaluate(rows, feats, "within")
        l_rho, l_auc, nl = evaluate(rows, feats, "loto")
        out[name] = {"within_spearman": round(w_rho, 4), "within_auroc": round(w_auc, 4), "n_within": nw,
                     "loto_spearman": round(l_rho, 4), "loto_auroc": round(l_auc, 4), "n_loto": nl,
                     "n_features": len(feats)}
        print(f"{name:14s} {w_rho:19.3f} {l_rho:24.3f}   AUROC {w_auc:.3f}/{l_auc:.3f}")
    a.out.write_text(json.dumps(out, indent=2))
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
