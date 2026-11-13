"""Build ONE fixed ligand set per target, used by every docking method.

Comparisons between MetalGeoDock, AutoDock Vina and MetalloDock are only meaningful on
identical molecules with identical labels. This script defines that set once:

  actives : the first N actives of the target from the curated ChEMBL table (pactivity >= 6)
  decoys  : N decoys drawn with a fixed seed from the big-library build for that target

Outputs results/common_set/{target}.csv with columns ligand_id,label,smiles,source.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")
SEED = 20241024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--n-act", type=int, default=100)
    ap.add_argument("--n-dec", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/common_set")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    # ---- actives
    smi = {}
    ds = HDAC / "data/processed/hdac_dataset.csv"
    if not ds.exists():
        print("hdac_dataset.csv missing"); return
    for r in csv.DictReader(open(ds)):
        if r.get("target") == a.target and r.get("label") == "1" and r.get("smiles"):
            smi[r["ligand_id"]] = r["smiles"]
    if not smi:
        # non-HDAC target: actives come from the curated ChEMBL table instead
        for r in csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")):
            if r.get("target_name") != a.target or not r.get("canonical_smiles"):
                continue
            try:
                if float(r["pactivity"]) < 6:
                    continue
            except Exception:
                continue
            smi.setdefault(r["molecule_chembl_id"], r["canonical_smiles"])
    act_ids = sorted(smi)[: a.n_act]
    rows = [{"ligand_id": i, "label": 1, "smiles": smi[i], "source": "chembl_active"} for i in act_ids]

    # ---- decoys from the big-library build (same molecules MetalGeoDock scored)
    scores = {}
    for f in sorted((ROOT / "results/biglib_scores").glob(f"{a.target}_part*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("score") not in (None, "", "nan"):
                scores[r["ligand_id"]] = True
    sdf = ROOT / f"docking/biglib/{a.target}.sdf"
    pool = []
    if sdf.exists():
        suppl = Chem.SDMolSupplier(str(sdf), sanitize=False)
        for i, m in enumerate(suppl):
            lid = f"{a.target}_BL{i:05d}"
            if lid not in scores or m is None:
                continue
            try:
                pool.append((lid, Chem.MolToSmiles(m)))
            except Exception:
                continue
    random.seed(SEED)
    if pool:
        picked = random.sample(pool, min(a.n_dec, len(pool)))
        rows += [{"ligand_id": i, "label": 0, "smiles": s, "source": "biglib_decoy"} for i, s in picked]
    else:
        # no big-library build for this target: draw size-diverse decoys from the global pool
        glob_pool = [p for p in csv.DictReader(open(ROOT / "results/decoy_pool_sized.csv"))
                     if p["target"] != a.target]
        picked = random.sample(glob_pool, min(a.n_dec, len(glob_pool)))
        rows += [{"ligand_id": f"{a.target}_SZ{i:04d}", "label": 0, "smiles": p["smiles"],
                  "source": "sized_pool_decoy"} for i, p in enumerate(picked)]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dest = a.out_dir / f"{a.target}.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand_id", "label", "smiles", "source"])
        w.writeheader(); w.writerows(rows)
    print(f"{a.target}: {sum(r['label'] for r in rows)} actives + "
          f"{len(rows)-sum(r['label'] for r in rows)} decoys -> {dest}")


if __name__ == "__main__":
    main()
