"""Diagnose why known selective inhibitors fail to coordinate.

These 214 compounds are literature HDAC-selective inhibitors; they must be able to
chelate the catalytic zinc. If the scorer reports a saturated score for them, the
false negative is ours, not theirs.

Prints the internal terms that screen_thermo_ef.py does not save, most importantly
min_donor_dist (closest ligand donor to metal after docking) and n_poses.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from metgeodock.dock import dock_ligand
from metgeodock.io import molecule_from_rdkit, read_pdb

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}

SAMPLES = [
    ("HDAC1", "CHEMBL1836142"), ("HDAC1", "CHEMBL3218562"),
    ("HDAC8", "CHEMBL2018302"), ("HDAC6", "CHEMBL113537"),
    ("HDAC2", "CHEMBL314389"),
]

meta = {r["molecule_id"]: r for r in csv.DictReader(
    open(ROOT / "processed/hdac_selectivity_set.csv"))}

print("失败样本内部诊断（coord_water 保持默认 2 = 预测水，不使用晶体水）")
print(f"{'化合物':<16}{'亚型':<7}{'total':>11}{'coord':>11}{'min_donor':>10}"
      f"{'mean_donor':>11}{'steric':>9}{'n_pose':>7}")
for sub, mid in SAMPLES:
    smi = meta[mid]["smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"{mid:<16}{sub:<7} SMILES 解析失败")
        continue
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
        print(f"{mid:<16}{sub:<7} 3D 嵌入失败")
        continue
    m = Chem.RemoveHs(m)
    prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{PDB[sub]}_h.pdb"))
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, mid), metal="zn",
                          n_spin=6, max_poses=8)
    except Exception as exc:
        print(f"{mid:<16}{sub:<7} 异常 {type(exc).__name__}: {exc}")
        continue
    if not res.poses:
        print(f"{mid:<16}{sub:<7} 无 pose 返回")
        continue
    best = min(res.poses, key=lambda p: p.score.total)
    t = best.score.terms
    print(f"{mid:<16}{sub:<7}{t.get('total', float('nan')):>11.2e}"
          f"{t.get('coord', float('nan')):>11.2e}"
          f"{t.get('min_donor_dist', float('nan')):>10.2f}"
          f"{t.get('mean_donor_dist', float('nan')):>11.2f}"
          f"{t.get('steric', float('nan')):>9.2f}{len(res.poses):>7}")

print("\n判定标准：coord_energy 在 min_donor_dist > dmax+0.15 (~3.05 A) 时丢弃该供体；")
print("            全部被丢弃则返回 w['uncoord']=12。若 min_donor_dist 明显 > 3.05，")
print("            说明配体根本没能被放到金属近旁（构象或位阻），而非逻辑错误。")
