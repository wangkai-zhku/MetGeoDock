"""Closed (1T64) vs open (2V5W) HDAC8: does L1 gating explain the earlier failures?

1T64 and 2V5W differ by 3.9 A RMSD on the L1 loop (max 6.2 A) while the core is
identical (0.41 A). If the L1 lid is what blocks HDAC8-selective inhibitors, the
open conformer should give markedly lower steric scores for the same ligands.
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

CONF = {"1T64_closed": ROOT / "processed/hdac_receptors/1T64_h.pdb",
        "2V5W_open": ROOT / "processed/hdac_receptors/2V5W_A_h.pdb"}


def main():
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    picks = [m for m, r in meta.items() if r["target_subtype"] == "HDAC8"][:6]
    prot = {k: read_pdb(str(v)) for k, v in CONF.items()}
    print(f"HDAC8 选择性抑制剂 {len(picks)} 个：关闭构象 vs 开放构象", flush=True)
    print(f"{'化合物':<16}{'构象':<14}{'total':>11}{'steric':>9}{'coord':>11}", flush=True)
    rows = []
    for mid in picks:
        m = Chem.AddHs(Chem.MolFromSmiles(meta[mid]["smiles"]))
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        rec = {"molecule": mid}
        for name, p in prot.items():
            try:
                res = dock_ligand(p, molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception as exc:
                print(f"{mid:<16}{name:<14} 异常 {type(exc).__name__}", flush=True)
                continue
            if not res.poses:
                continue
            best = min(res.poses, key=lambda x: x.score.total)
            t = best.score.terms
            rec[name] = {"total": float(t.get("total", np.nan)),
                         "steric": float(t.get("steric", np.nan)),
                         "coord": float(t.get("coord", np.nan))}
            print(f"{mid:<16}{name:<14}{t.get('total', float('nan')):>11.2e}"
                  f"{t.get('steric', float('nan')):>9.2f}"
                  f"{t.get('coord', float('nan')):>11.2e}", flush=True)
        rows.append(rec)

    print("\n汇总")
    for name in CONF:
        st = [r[name]["steric"] for r in rows if name in r]
        tt = [r[name]["total"] for r in rows if name in r]
        if st:
            print(f"  {name:<14} steric 中位={np.median(st):>9.2f}  "
                  f"total 中位={np.median(tt):>10.2e}  n={len(st)}")
    pairs = [(r["1T64_closed"]["steric"], r["2V5W_open"]["steric"])
             for r in rows if "1T64_closed" in r and "2V5W_open" in r]
    if pairs:
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        print(f"\nsteric 变化: 开放构象更低的有 {int((b < a).sum())}/{len(pairs)} 个")
        print(f"  平均下降 {np.mean(a - b):.2f}  (关闭 {np.mean(a):.2f} -> 开放 {np.mean(b):.2f})")


if __name__ == "__main__":
    main()
