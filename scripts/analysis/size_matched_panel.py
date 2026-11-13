"""Does any metal-specific signal survive once actives and decoys are size-matched?

The 21-target cross-family panel is dominated by a size bias (Pearson r = -0.86 between
Delta MW and AUROC), so its per-target AUROC values cannot be read as a statement about a
metal. This script removes that confound: for every target, decoys are drawn from the
existing scored pool by stratified sampling on molecular weight, so that the decoy MW
distribution matches the actives' distribution. AUROC is then recomputed.

If a target's AUROC moves toward 0.5 after matching, its original value was a size artefact.
If it stays high or low, the residual reflects chemistry the scorer does or does not capture.

Outputs results/size_matched_metrics.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import rankdata

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(20241024)
N_BINS = 10
TOL = 1.0          # keep ratio actives:decoys close to 1 for precision


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


def mw_table():
    t = {}
    p = ROOT / "processed/chembl/curated/all_targets_molecules.csv"
    for r in csv.DictReader(open(p)):
        smi = r.get("canonical_smiles")
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            t[r["molecule_chembl_id"]] = Descriptors.MolWt(m)
    return t


def match_decoys(act_mw, dec_mw, dec_score, n_bins=N_BINS):
    """Stratified draw of decoys whose MW distribution follows the actives'."""
    a = np.asarray(act_mw, float); d = np.asarray(dec_mw, float)
    if len(a) < 5 or len(d) < 5:
        return None
    lo, hi = np.percentile(a, [2, 98])
    edges = np.linspace(lo, hi, n_bins + 1)
    picked = []
    for i in range(n_bins):
        in_a = (a >= edges[i]) & (a < edges[i + 1] if i < n_bins - 1 else a <= edges[i + 1])
        k = int(in_a.sum())
        if k == 0:
            continue
        in_d = np.where((d >= edges[i]) & (d < edges[i + 1] if i < n_bins - 1 else d <= edges[i + 1]))[0]
        if len(in_d) == 0:
            continue
        take = RNG.choice(in_d, size=min(k, len(in_d)), replace=False)
        picked.extend(take.tolist())
    return np.array(sorted(set(picked)), int)


def main():
    mw = mw_table()
    auroc_map = {r["target"]: float(r["auroc"])
                 for r in csv.DictReader(open(ROOT / "results/metric_per_target.csv"))
                 if r["variant"] == "E"}
    rows = []
    for f in sorted((ROOT / "results/entropy_big").glob("*.csv")):
        recs = list(csv.DictReader(open(f)))
        if not recs:
            continue
        t, metal = recs[0]["target"], recs[0]["metal"]
        if t not in auroc_map:
            continue
        act = [(r["ligand_id"], float(r["E_best"])) for r in recs
               if r["label"] == "1" and r["E_best"] not in (None, "", "nan")]
        dec = [(r["ligand_id"], float(r["E_best"])) for r in recs
               if r["label"] == "0" and r["E_best"] not in (None, "", "nan")]
        act = [(lid, s) for lid, s in act if lid in mw]
        dec = [(lid, s) for lid, s in dec if lid in mw]
        if len(act) < 10 or len(dec) < 20:
            continue
        a_mw = [mw[lid] for lid, _ in act]
        d_mw = [mw[lid] for lid, _ in dec]
        idx = match_decoys(a_mw, d_mw, None)
        if idx is None or len(idx) < 10:
            continue
        d_sel = [dec[i] for i in idx]
        y = [1] * len(act) + [0] * len(d_sel)
        s = [x[1] for x in act] + [x[1] for x in d_sel]
        rows.append({
            "target": t, "metal": metal,
            "n_act": len(act), "n_dec_before": len(dec), "n_dec_matched": len(d_sel),
            "auroc_before": auroc_map[t], "auroc_matched": auroc(y, s),
            "ef1_matched": ef(y, s, 0.01),
            "dMW_before": float(np.mean(a_mw) - np.mean(d_mw)),
            "dMW_matched": float(np.mean(a_mw) - np.mean([mw[d_sel[i][0]] for i in range(len(d_sel))])),
        })

    out = ROOT / "results/size_matched_metrics.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'target':14s} {'metal':5s} {'n_dec':>6s} {'->':>5s} {'dMW_before':>10s} {'dMW_after':>9s} "
          f"{'AUROC_before':>12s} {'AUROC_matched':>13s}")
    for r in sorted(rows, key=lambda x: -x["auroc_matched"]):
        print(f"{r['target']:14s} {r['metal']:5s} {r['n_dec_before']:6d} {r['n_dec_matched']:5d} "
              f"{r['dMW_before']:+10.0f} {r['dMW_matched']:+9.0f} "
              f"{r['auroc_before']:12.3f} {r['auroc_matched']:13.3f}")
    b = np.array([r["auroc_before"] for r in rows])
    a = np.array([r["auroc_matched"] for r in rows])
    print(f"\nmean AUROC: before {b.mean():.3f} -> matched {a.mean():.3f}  (n={len(rows)})")
    print(f"|AUROC-0.5| before {np.mean(np.abs(b-0.5)):.3f} -> matched {np.mean(np.abs(a-0.5)):.3f}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
