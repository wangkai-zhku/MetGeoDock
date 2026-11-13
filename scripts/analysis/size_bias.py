"""Size-bias analysis of the cross-family small-library benchmark (§3.9).

For each of the 21 targets, compare the mean molecular weight of the actives with that of the
decoys and pair it with the measured AUROC. The point is that the panel's AUROC spread is
explained by how much heavier the actives are, not by which metal the enzyme uses.

Outputs results/size_bias.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]


def mean_mw(ids, table):
    vals = []
    for lid in ids:
        rec = table.get(lid)
        if not rec:
            continue
        m = Chem.MolFromSmiles(rec.get("canonical_smiles", "")) if rec.get("canonical_smiles") else None
        if m is not None:
            vals.append(Descriptors.MolWt(m))
    return float(np.mean(vals)) if vals else np.nan


def main():
    table = {r["molecule_chembl_id"]: r
             for r in csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv"))}
    auroc = {r["target"]: float(r["auroc"])
             for r in csv.DictReader(open(ROOT / "results/metric_per_target.csv"))
             if r["variant"] == "E"}
    rows = []
    for f in sorted((ROOT / "results/entropy_big").glob("*.csv")):
        recs = list(csv.DictReader(open(f)))
        if not recs:
            continue
        t, metal = recs[0]["target"], recs[0]["metal"]
        if t not in auroc:
            continue
        a_ids = [r["ligand_id"] for r in recs if r["label"] == "1"]
        d_ids = [r["ligand_id"] for r in recs if r["label"] == "0"]
        am, dm = mean_mw(a_ids, table), mean_mw(d_ids, table)
        if not np.isfinite(am) or not np.isfinite(dm):
            continue
        rows.append({"target": t, "metal": metal, "auroc": auroc[t],
                     "n_act": len(a_ids), "n_dec": len(d_ids),
                     "act_mw": round(am, 1), "dec_mw": round(dm, 1),
                     "delta_mw": round(am - dm, 1)})

    out = ROOT / "results/size_bias.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    dmw = np.array([r["delta_mw"] for r in rows])
    au = np.array([r["auroc"] for r in rows])
    rho, p = stats.spearmanr(dmw, au)
    pr_r, pr_p = stats.pearsonr(dmw, au)
    print(f"{'target':14s} {'metal':5s} {'AUROC':>6s} {'dMW':>7s}")
    for r in sorted(rows, key=lambda x: x["delta_mw"]):
        print(f"{r['target']:14s} {r['metal']:5s} {r['auroc']:6.3f} {r['delta_mw']:+7.1f}")
    print(f"\nSpearman(dMW, AUROC) = {rho:+.3f} (p={p:.2e})")
    print(f"Pearson  (dMW, AUROC) = {pr_r:+.3f} (p={pr_p:.2e}), n={len(rows)}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
