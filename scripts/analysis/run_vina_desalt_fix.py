"""Vina 侧的多组分配体脱盐重算（与 MetalGeoDock 的修正对齐）。

背景：配体集约 6.5% 是多组分 SMILES（含 TFA/富马酸盐等）。Vina 侧的问题是 obabel
转换多片段分子时容易失败（记成 error），即使成功也把盐片段当成配体的一部分塞进口袋。
MetalGeoDock 已用 run_metalgeo_desalt_fix.py 只重跑这部分（单组分 Δ≡0），这里对 Vina 做同样处理。

流程（每靶点）：
  1. 从 common set 挑出多组分分子，脱盐（保留最大片段）后写成临时 common set；
  2. 调 vina_baseline.py 只算这些配体（受体转换/box 与原流程完全一致）；
  3. 用新 affinity 替换 results/bench22_1to10/vina_raw/<T>_p0.csv 中对应行（保持行序）；
  4. 原文件首次修改前备份为 *.presalt.bak。

用法：
    python scripts/analysis/run_vina_desalt_fix.py --procs 4
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# vina_baseline 依赖 obabel 与 vina，它们只装在 PPO-cursor 的 conda 环境里
_CONDA_BIN = "/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin"
if _CONDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CONDA_BIN + ":" + os.environ.get("PATH", "")
BENCH = ROOT / "results/bench22_1to10"
COMMON = BENCH / "common"
VINA_RAW = BENCH / "vina_raw"
PY = ROOT / "envs/metgeodock/bin/python"

TARGETS = {
    "CA2": ("processed/receptors10/3CAJ_h.pdb", "zn"),
    "CA12": ("processed/receptors22/CA12_h.pdb", "zn"),
    "HDAC1": ("processed/receptors10/5ICN_h.pdb", "zn"),
    "HDAC2": ("processed/receptors10/3MAX_h.pdb", "zn"),
    "HDAC3": ("processed/hdac_receptors/4A69_h.pdb", "zn"),
    "HDAC6": ("processed/hdac_receptors/5EDU_CD2_h.pdb", "zn"),
    "HDAC8": ("processed/hdac_receptors/2V5W_A_h.pdb", "zn"),
    "MMP1": ("processed/receptors22/MMP1_cat_h.pdb", "zn"),
    "MMP2": ("processed/receptors10/1QIB_h.pdb", "zn"),
    "MMP3": ("processed/receptors22/MMP3_h.pdb", "zn"),
    "MMP9": ("processed/receptors22/MMP9_cat_h.pdb", "zn"),
    "MMP13": ("processed/receptors22/MMP13_h.pdb", "zn"),
    "ALOX5": ("processed/receptors10/3O8Y_h.pdb", "fe2"),
    "ARG1": ("processed/receptors22/ARG1_h.pdb", "mn"),
    "ARG2": ("processed/receptors10/1PQ3_h.pdb", "mn"),
    "urease": ("processed/receptors10/1FWJ_h.pdb", "ni"),
    "METAP2": ("processed/receptors10/1B6A_h.pdb", "co"),
    "CYP1A2": ("processed/receptors22/CYP1A2_h.pdb", "fe3"),
    "CYP2E1": ("processed/receptors22/CYP2E1_h.pdb", "fe3"),
    "CYP19A1": ("processed/receptors22/CYP19A1_h.pdb", "fe3"),
    "EGLN2": ("processed/receptors22/EGLN2_h.pdb", "fe3"),
    "HIV1_integrase": ("processed/receptors22/HIV1_cat_h.pdb", "mg"),
}


def main():
    _lock = acquire_lock("vina_desalt")
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4, help="并行靶点数（每个 vina 进程用 4 线程）")
    ap.add_argument("--targets", default="")
    a = ap.parse_args()
    todo = [t.strip() for t in a.targets.split(",") if t.strip()] or list(TARGETS)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    work = ROOT / "results/_tmp_vina_desalt"
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tid in todo:
        raw = VINA_RAW / f"{tid}_p0.csv"
        if not raw.exists():
            print(f"{tid}: 缺 vina 结果，跳过")
            continue
        rows = list(csv.DictReader(open(raw)))
        have = {r["ligand_id"] for r in rows}
        multi = []
        for r in csv.DictReader(open(COMMON / f"{tid}.csv")):
            if r["ligand_id"] not in have:
                continue
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol is None or len(Chem.GetMolFrags(mol)) < 2:
                continue
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            big = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            multi.append((r["ligand_id"], Chem.MolToSmiles(big), r["label"]))
        if not multi:
            print(f"{tid}: 无多组分配体")
            continue
        outd_probe = work / f"out_{tid}"
        prev = outd_probe / f"{tid}_p0.csv"
        if prev.exists() and prev.stat().st_size:
            n_prev = sum(1 for r in csv.DictReader(open(prev))
                         if r.get("affinity") and float(r["affinity"]) != 0.0)
            if n_prev >= len(multi) * 0.90:      # 该靶点此前的脱盐结果已完整
                print(f"{tid}: 已有 {n_prev} 条脱盐结果，跳过计算（仅参与合并）")
                jobs.append((tid, None, len(multi)))
                continue
        cs = work / f"{tid}_desalt.csv"
        with open(cs, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["ligand_id", "label", "smiles"])
            for lid, smi, lab in multi:
                w.writerow([lid, lab, smi])
        print(f"{tid}: 待脱盐重算 {len(multi)} 个")
        jobs.append((tid, cs, len(multi)))

    if not jobs:
        print("没有需要处理的靶点")
        return

    # 并行跑（每靶点一个 vina_baseline 进程）
    procs = []
    running = []
    for tid, cs, n in jobs:
        if cs is None:
            continue
        rel, metal = TARGETS[tid]
        outd = work / f"out_{tid}"
        outd.mkdir(exist_ok=True)
        cmd = [str(PY), "-u", str(ROOT / "scripts/analysis/vina_baseline.py"),
               "--target", tid, "--protein", rel, "--common-set", str(cs),
               "--out-dir", str(outd), "--metal", metal,
               "--exhaustiveness", "4", "--cpu", "4"]
        logf = open(work / f"{tid}.log", "w")
        running.append((tid, cs, n, outd, subprocess.Popen(cmd, cwd=str(ROOT),
                                                           stdout=logf, stderr=subprocess.STDOUT), logf))
        while sum(1 for r in running if r[4].poll() is None) >= a.procs:
            import time
            time.sleep(5)

    for tid, cs, n, outd, proc, logf in running:
        proc.wait()

    # 合并回原文件
    grand = 0
    for tid, cs, n, outd, proc, logf in running:
        new = {}
        src = outd / f"{tid}_p0.csv"
        if src.exists():
            for r in csv.DictReader(open(src)):
                if r.get("affinity"):
                    try:
                        new[r["ligand_id"]] = float(r["affinity"])
                    except ValueError:
                        pass
        raw = VINA_RAW / f"{tid}_p0.csv"
        rows = list(csv.DictReader(open(raw)))
        bak = raw.with_suffix(".csv.presalt.bak")
        if not bak.exists():
            bak.write_bytes(raw.read_bytes())
        tmp = raw.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["ligand_id", "label", "affinity", "error"])
            for r in rows:
                if r["ligand_id"] in new:
                    w.writerow([r["ligand_id"], r["label"], f"{new[r['ligand_id']]}", ""])
                else:
                    w.writerow([r["ligand_id"], r["label"], r.get("affinity", ""), r.get("error", "")])
        tmp.replace(raw)
        print(f"[{tid}] 替换 {len(new)}/{n} 个脱盐 affinity")
        grand += len(new)
    print(f"\n完成，共替换 {grand} 个配体；临时目录 {work}")


if __name__ == "__main__":
    main()
