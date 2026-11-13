"""三方法单配体耗时公平对比：同时记录墙钟与 CPU-秒。

为什么要记 CPU-秒：三个方法的并行方式不同 ——
  MetalGeoDock 单线程、Vina 用 --cpu 4、MetalloDock 批处理时 torch 可能多核，
  只比墙钟会误导。CPU-秒（user+sys）才是"总算力成本"。

配置（各方法在本质 work 里的实际用法）：
  MetalGeoDock : dock_ligand(n_spin=8, max_poses=8, include_chem=True)
  Vina         : --exhaustiveness 4 --cpu {1,4}（受体 pdbqt 转换单独计时）
  MetalloDock  : 官方 VS 前向 ligand_docking_coordination（batch 1 与 30）

用法：python scripts/analysis/benchmark_runtime.py --n 3
"""
from __future__ import annotations

import argparse
import csv
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
_CONDA = "/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin"
if _CONDA not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CONDA + ":" + os.environ.get("PATH", "")

REC = ROOT / "processed/receptors10/3CAJ_h.pdb"      # CA2
LIG_CSV = ROOT / "results/bench22_1to10/common/CA2.csv"


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def lig_mols(n):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for i, r in enumerate(csv.DictReader(open(LIG_CSV))):
        if i >= n:
            break
        out.append((r["ligand_id"], r["smiles"]))
    return out


def time_metalgeo(mols):
    from _embed import embed_3d
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit
    prot = read_pdb(str(REC))
    res = []
    for lid, smi in mols:
        m = embed_3d(Chem.MolFromSmiles(smi))
        t0, c0 = time.perf_counter(), cpu_seconds()
        r = dock_ligand(prot, molecule_from_rdkit(m, lid), metal="zn",
                        n_spin=8, max_poses=8, include_chem=True)
        res.append((lid, time.perf_counter() - t0, cpu_seconds() - c0, len(r.poses)))
    return res


def time_vina(mols, cpu):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    xyz = (-6.72, -1.76, 15.22)
    res = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dry = td / "rec_dry.pdb"
        dry.write_text("\n".join(
            ln for ln in REC.read_text(errors="ignore").splitlines()
            if ln.startswith(("ATOM", "HETATM")) and not (
                ln.startswith("HETATM") and ln[17:20].strip().upper() in {"HOH", "WAT"})) + "\nEND\n")
        rec = td / "rec.pdbqt"
        subprocess.run(["obabel", str(dry), "-O", str(rec), "-xr"], capture_output=True, timeout=300)
        for lid, smi in mols:
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(m, randomSeed=0xF00D)
            lig = td / "l.pdb"; Chem.MolToPDBFile(m, str(lig))
            lq = td / "l.pdbqt"
            subprocess.run(["obabel", str(lig), "-O", str(lq), "-xh"], capture_output=True, timeout=120)
            out = td / "o.pdbqt"
            cmd = ["vina", "--receptor", str(rec), "--ligand", str(lq),
                   "--center_x", f"{xyz[0]}", "--center_y", f"{xyz[1]}", "--center_z", f"{xyz[2]}",
                   "--size_x", "22", "--size_y", "22", "--size_z", "22",
                   "--exhaustiveness", "4", "--num_modes", "1", "--cpu", str(cpu), "--out", str(out)]
            ch0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            c0 = ch0.ru_utime + ch0.ru_stime
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            wall = time.perf_counter() - t0
            ch1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            res.append((lid, wall, (ch1.ru_utime + ch1.ru_stime) - c0, 1 if r.returncode == 0 else 0))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    mols = lig_mols(a.n)
    print(f"测速配体 {len(mols)} 个（CA2，{REC.name}）\n")

    print("=== MetalGeoDock（单线程）===")
    for lid, wall, cpu, npose in time_metalgeo(mols):
        print(f"  {lid}: 墙钟 {wall:6.1f}s  CPU {cpu:6.1f}s  位姿 {npose}")

    print("\n=== Vina（--cpu 4，实际用法）===")
    for lid, wall, cpu, ok in time_vina(mols, 4):
        print(f"  {lid}: 墙钟 {wall:6.1f}s  子进程CPU {cpu:6.1f}s")

    print("\n注：MetalloDock 的耗时用官方批处理实测（mdl_official 日志，9.98 s/配体、"
          "batch 30、OMP_NUM_THREADS=1 → 约 10 CPU-秒/配体）")


if __name__ == "__main__":
    main()
