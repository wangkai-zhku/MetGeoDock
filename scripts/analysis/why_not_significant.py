"""Per-subtype analysis behind "why two subtypes are not significant" (§3.8).

For every HDAC subtype with a large-library screen, compare the actives and the decoys on the
physicochemical properties that a metal-donor scorer can exploit, and pair the difference with
the measured AUROC. TPSA is the key quantity: every metal donor is a polar atom, so a positive
Delta TPSA means the actives offer more coordination opportunities than their decoys.

Outputs results/why_not_significant.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")


def mean_props(smiles):
    vals = {k: [] for k in ("MW", "RotB", "LogP", "TPSA", "HA")}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        vals["MW"].append(Descriptors.MolWt(m))
        vals["RotB"].append(rdMolDescriptors.CalcNumRotatableBonds(m))
        vals["LogP"].append(Crippen.MolLogP(m))
        vals["TPSA"].append(rdMolDescriptors.CalcTPSA(m))
        vals["HA"].append(m.GetNumHeavyAtoms())
    return {k: float(np.mean(v)) if v else np.nan for k, v in vals.items()}


def main():
    big = {r["target"]: r for r in csv.DictReader(open(ROOT / "results/biglib_metrics.csv"))}
    rows = []
    for t in sorted(big):
        smi = {}
        ds = HDAC / "data/processed/hdac_dataset.csv"
        if not ds.exists():
            continue
        for r in csv.DictReader(open(ds)):
            if r.get("target") == t and r.get("smiles"):
                smi[r["ligand_id"]] = r["smiles"]
        p = HDAC / "docking/results" / f"{t}.csv"
        if not p.exists():
            continue
        a_smi = [smi[r["ligand_id"]] for r in csv.DictReader(open(p))
                 if "_A" in r.get("ligand_id", "") and r["ligand_id"] in smi]
        d_smi = []
        sdf = ROOT / f"docking/biglib/{t}.sdf"
        if sdf.exists():
            sup = Chem.SDMolSupplier(str(sdf), sanitize=False)
            for i, m in enumerate(sup):
                if i >= 300:          # 300 decoys is enough for a mean property
                    break
                if m is None:
                    continue
                try:
                    d_smi.append(Chem.MolToSmiles(m))
                except Exception:
                    pass
        if not a_smi or not d_smi:
            continue
        pa, pd = mean_props(a_smi), mean_props(d_smi)
        row = {"target": t, "auroc": float(big[t]["auroc"]),
               "ci_lo": float(big[t]["auroc_ci_lo"]), "ci_hi": float(big[t]["auroc_ci_hi"]),
               "n_dec": int(big[t]["n_dec"]),
               "significant": int(float(big[t]["auroc_ci_lo"]) > 0.5)}
        for k in pa:
            row[f"act_{k}"] = round(pa[k], 2)
            row[f"dec_{k}"] = round(pd[k], 2)
            row[f"d_{k}"] = round(pa[k] - pd[k], 2)
        rows.append(row)

    out = ROOT / "results/why_not_significant.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{'target':8s} {'AUROC':>6s} {'sig':>4s} {'dTPSA':>7s} {'dRotB':>6s}")
    for r in rows:
        print(f"{r['target']:8s} {r['auroc']:6.3f} {r['significant']:4d} "
              f"{r['d_TPSA']:+7.1f} {r['d_RotB']:+6.2f}")
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
