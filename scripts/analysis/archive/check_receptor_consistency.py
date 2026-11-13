"""Check that the rebuilt receptor reproduces already-finished cf10 scores.

If the newly prepared /tmp/receptors10/*_h.pdb differ from the (now lost) ones
used before the crash, the resumed runs would not be comparable with the 28
parts already in results/cf10_thermo.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from metgeodock.dock import dock_ligand
from metgeodock.io import molecule_from_rdkit, read_pdb

TARGET = sys.argv[1] if len(sys.argv) > 1 else "CDK2"
PDB = {"CDK2": "1HCK", "CA2": "3CAJ", "HDAC1": "5ICN"}[TARGET]
METAL = {"CDK2": "mg", "CA2": "zn", "HDAC1": "zn"}[TARGET]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

csv_path = ROOT / f"results/cf10_thermo/{TARGET}_p0.csv"
rows = [r for r in csv.DictReader(open(csv_path)) if (r.get("total") or "").strip()]
common = {r["ligand_id"]: r for r in csv.DictReader(
    open(ROOT / f"results/crossfamily10/{TARGET}_div.csv"))}

protein = read_pdb(f"/tmp/receptors10/{PDB}_h.pdb")
print(f"{TARGET}: receptor 1HCK->{PDB}_h.pdb, comparing {N} ligands against {csv_path.name}")
print(f"{'ligand_id':<18}{'CSV total':>14}{'rebuilt total':>16}{'abs diff':>12}")
diffs = []
for r in rows[:N]:
    smi = common[r["ligand_id"]]["smiles"]
    mol = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(m, randomSeed=0xF00D) == 0
    m = Chem.RemoveHs(m)
    res = dock_ligand(protein, molecule_from_rdkit(m, r["ligand_id"]),
                      metal=METAL, n_spin=6, max_poses=8)
    if not res.poses:
        print(f"{r['ligand_id']:<18}{float(r['total']):>14.4f}{'no pose':>16}")
        continue
    best = min(res.poses, key=lambda p: p.score.total)
    new = best.score.total
    old = float(r["total"])
    d = abs(new - old)
    diffs.append(d)
    print(f"{r['ligand_id']:<18}{old:>14.4f}{new:>16.4f}{d:>12.2e}")
if diffs:
    print(f"\nmax |diff| = {max(diffs):.3e}   mean |diff| = {sum(diffs)/len(diffs):.3e}")
    print("判定：" + ("可复现 —— 续跑数据与已有 28 个 part 可比"
                     if max(diffs) < 1e-6 else
                     "不一致 —— 新受体与旧受体不同，续跑结果不能直接拼接"))
