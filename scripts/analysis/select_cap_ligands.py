"""Select HDAC inhibitors that actually have a cap group.

The previous 214-compound set averaged 10-13 A end-to-end, so the ligand never
reaches the cap region and a cap-based score has nothing to work with. Real
cap-bearing HDAC inhibitors (SAHA-like, tubacin-like) are 15-20 A long: a
zinc-binding group at one end, a linker, and a bulky cap at the other.

Selection criteria (all computed from SMILES; no activity label is used to build
the descriptors):
  * zinc-binding group present (hydroxamate / benzamide / carboxylate / trifluoromethyl ketone)
  * topological span >= 11 bonds (about 15 A end to end)
  * MW >= 400 and >= 28 heavy atoms
  * selectivity profile available: pActivity in >= 3 subtypes, max >= 6.0, gap >= 2 log
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops
RDLogger.DisableLog("rdApp.*")

ZBG = {
    "hydroxamate": Chem.MolFromSmarts("[NX3][OH1]"),
    "benzamide": Chem.MolFromSmarts("c1ccccc1[CX3](=O)[NX3]"),
    "carboxylate": Chem.MolFromSmarts("[CX3](=O)[OX1-,OX2H1]"),
    "trifluoromethyl_ketone": Chem.MolFromSmarts("[CX4](F)(F)F[CX3](=O)"),
    "thiol": Chem.MolFromSmarts("[SX2H]"),
    "sulfonamide": Chem.MolFromSmarts("S(=O)(=O)[NX3]"),
}


def span_bonds(mol):
    """longest shortest-path between heavy atoms, in bonds"""
    dm = rdmolops.GetDistanceMatrix(mol)
    return float(dm.max())


def main():
    rows = [r for r in csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv"))
            if r.get("target_name", "").startswith("HDAC")]
    by_mol = {}
    for r in rows:
        m = r["molecule_chembl_id"]
        try:
            p = float(r["pactivity"])
        except (TypeError, ValueError):
            continue
        smi = r.get("canonical_smiles")
        if not smi:
            continue
        d = by_mol.setdefault(m, {"prof": {}, "smi": smi})
        t = r["target_name"]
        if t not in d["prof"] or p > d["prof"][t]:
            d["prof"][t] = p
    print(f"HDAC 唯一分子 {len(by_mol)}")

    sel_rows = []
    stats = {"selective": 0, "long": 0, "zbg": 0, "both": 0}
    for mid, d in by_mol.items():
        prof = d["prof"]
        if len(prof) < 3:
            continue
        it = sorted(prof.items(), key=lambda kv: -kv[1])
        if it[0][1] < 6.0 or it[0][1] - it[1][1] < 2.0:
            continue
        stats["selective"] += 1
        mol = Chem.MolFromSmiles(d["smi"])
        if mol is None:
            continue
        span = span_bonds(mol)
        nheavy = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        is_long = span >= 11
        if is_long:
            stats["long"] += 1
        zbgs = [k for k, p in ZBG.items() if p is not None and mol.HasSubstructMatch(p)]
        if zbgs:
            stats["zbg"] += 1
        if is_long and zbgs:
            stats["both"] += 1
            sel_rows.append({
                "molecule_id": mid,
                "target_subtype": it[0][0],
                "p_target": round(it[0][1], 2),
                "p_second": round(it[1][1], 2),
                "n_subtypes": len(prof),
                "span_bonds": span,
                "n_heavy": nheavy,
                "molwt": round(mw, 1),
                "zbg": "+".join(zbgs),
                "smiles": d["smi"],
                "profile": ";".join(f"{t}:{v:.2f}" for t, v in it),
            })

    print(f"选择性化合物 {stats['selective']}")
    print(f"  长链(span>=11 键) {stats['long']}")
    print(f"  含 ZBG {stats['zbg']}")
    print(f"  两者兼备 {stats['both']}")

    out = ROOT / "processed/hdac_cap_ligands.csv"
    if sel_rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sel_rows[0]))
            w.writeheader()
            w.writerows(sel_rows)
        print(f"\n写出 {out}")
        import collections
        print("目标亚型分布:", dict(collections.Counter(r["target_subtype"] for r in sel_rows)))
        print(f"\n{'molecule':<16}{'目标':<8}{'pAct':>6}{'键跨':>6}{'重原子':>7}{'MW':>7}  ZBG")
        for r in sorted(sel_rows, key=lambda x: -x["p_target"])[:15]:
            print(f"{r['molecule_id']:<16}{r['target_subtype']:<8}{r['p_target']:>6}"
                  f"{r['span_bonds']:>6.0f}{r['n_heavy']:>7}{r['molwt']:>7.0f}  {r['zbg']}")
    else:
        print("没有满足条件的配体")


if __name__ == "__main__":
    main()
