"""按 MetalloDock 官方虚拟筛选口径跑 22 靶点 benchmark（1:5 与 1:10）。

审计后修正的三处偏离（对比此前 screen_metallodock.py 的做法）：
  1. **前向**：用 `ligand_docking_coordination`（官方 virtual_screening.py 用的那个：
     模型预测配位原子作自回归起点 + `cal_pair_dist_and_select_nearest_residue_4vs` 打分），
     而不是 `ligand_docking` + `--coordination_idx 0`。
  2. **口袋**：按 `utils/pre_process.py:generate_pocket` 的口径 —— 以催化金属为代理中心、
     `around 10.0` 取残基、保留这些残基的**全部**原子。官方代码里虽然先删了氢，但随后
     `pocket_residue.atoms` 又把氢取回来（实测 8elk_pro_pocket.pdb 含 462 个 H），故含氢。
     此前我们用 18 Å、~2000-2200 原子，明显偏离官方 948-1520 原子的口袋规模。
  3. **配体**：去氢 SDF（模型期望重原子），与官方 dataset 一致。
  4. `--refine` 不做：refine 只用于生成修正构象和 ff_score，mdn_score 是前向直接给出的；
     且 UFF 优化在 CPU 上极慢（30 配体一批 >20 分钟），会拖垮整批。

用法：
    python scripts/analysis/mdl_official_bench22.py --set 1to5  --targets CA2 --batch 30
    python scripts/analysis/mdl_official_bench22.py --set 1to10 --targets CA2,HDAC1
    python scripts/analysis/mdl_official_bench22.py --set 1to5  --all --batch 30

输出：results/mdl_official/<set>/<TARGET>.csv   (ligand_id,label,score,error)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MDL = ROOT / "processed/metallodock_ref"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(MDL))
sys.path.insert(0, str(MDL / "utils"))       # uff_build 等内部脚本在 utils/ 下

from _embed import embed_3d  # noqa: E402

# 靶点 -> (受体, 金属, 配体集名)
TARGETS = {
    "CA2":   ("processed/receptors10/3CAJ_h.pdb", "zn", "CA2"),
    "CA12":  ("processed/receptors22/CA12_h.pdb", "zn", "CA12"),
    "HDAC1": ("processed/receptors10/5ICN_h.pdb", "zn", "HDAC1"),
    "HDAC2": ("processed/receptors10/3MAX_h.pdb", "zn", "HDAC2"),
    "HDAC3": ("processed/hdac_receptors/4A69_h.pdb", "zn", "HDAC3"),
    "HDAC6": ("processed/hdac_receptors/5EDU_CD2_h.pdb", "zn", "HDAC6"),
    "HDAC8": ("processed/hdac_receptors/2V5W_A_h.pdb", "zn", "HDAC8"),
    "MMP1":  ("processed/receptors22/MMP1_cat_h.pdb", "zn", "MMP1"),
    "MMP2":  ("processed/receptors10/1QIB_h.pdb", "zn", "MMP2"),
    "MMP3":  ("processed/receptors22/MMP3_h.pdb", "zn", "MMP3"),
    "MMP9":  ("processed/receptors22/MMP9_cat_h.pdb", "zn", "MMP9"),
    "MMP13": ("processed/receptors22/MMP13_h.pdb", "zn", "MMP13"),
    "ALOX5": ("processed/receptors10/3O8Y_h.pdb", "fe2", "ALOX5"),
    "ARG1":  ("processed/receptors22/ARG1_h.pdb", "mn", "ARG1"),
    "ARG2":  ("processed/receptors10/1PQ3_h.pdb", "mn", "ARG2"),
    "urease": ("processed/receptors10/1FWJ_h.pdb", "ni", "urease"),
    "METAP2": ("processed/receptors10/1B6A_h.pdb", "co", "METAP2"),
    "CYP1A2": ("processed/receptors22/CYP1A2_h.pdb", "fe3", "CYP1A2"),
    "CYP2E1": ("processed/receptors22/CYP2E1_h.pdb", "fe3", "CYP2E1"),
    "CYP19A1": ("processed/receptors22/CYP19A1_h.pdb", "fe3", "CYP19A1"),
    "EGLN2": ("processed/receptors22/EGLN2_h.pdb", "fe3", "EGLN2"),
    "HIV1_integrase": ("processed/receptors22/HIV1_cat_h.pdb", "mg", "HIV1_integrase"),
}
SETS = {
    "1to5": (ROOT / "processed/ligand_sets22", 50),
    "1to10": (ROOT / "processed/ligand_sets22_1to10", 1000),
}
METALS = {"ZN", "FE", "FE2", "FE3", "MN", "MN2", "MG", "CA", "CO", "CU", "NI", "CD"}


def make_pocket_official(protein: Path, metal: str, out: Path, radius: float = 10.0):
    """按官方口径裁口袋：金属 10 Å 内出现的残基，保留其全部原子（含氢）。"""
    import numpy as np

    from metgeodock.dock import detect_coordination
    from metgeodock.io import read_pdb

    prot = read_pdb(str(protein))
    sph = detect_coordination(prot, metal=metal)
    center = np.array(sph.metal_xyz, dtype=float)

    lines = protein.read_text(errors="ignore").splitlines()
    # 1) 先按原子距离找出落在 10 Å 内的残基集合
    keep_res = set()
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if np.linalg.norm(xyz - center) <= radius:
            keep_res.add((ln[21], ln[22:27]))          # (链, 残基号+插入码)
    # 2) 取这些残基的全部原子（官方 pocket_residue.atoms 的做法）
    kept = [ln for ln in lines
            if ln.startswith(("ATOM", "HETATM")) and (ln[21], ln[22:27]) in keep_res]
    out.write_text("\n".join(kept) + "\nEND\n")
    return len(kept), center


def largest_fragment(mol):
    """脱盐：只保留最大片段。

    配体集里约 6.5% 的条目是多组分 SMILES（带 TFA/富马酸盐等）。MetalloDock 从配位原子
    做 BFS 时只会遍历起点所在的连通分量，其余片段被漏掉，seq_order 长度与原子数不符，
    在 construct_interaction_graph 里抛
        RuntimeError: The expanded size of the tensor (947) must match the existing size (967)
    导致**整批**（batch=30）失败。批内含多组分的概率 1-(1-0.065)^30 ≈ 86%，
    正是此前 85% 失败率的来源。
    """
    from rdkit import Chem
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)   # 必须 sanitize，否则 RingInfo 未初始化，NumRotatableBonds 会崩
    if len(frags) <= 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def load_set(setdir: Path, name: str, dec_cap: int):
    rows = []
    for fn, lab, cap in ((f"{name}_actives.csv", 1, 100), (f"{name}_decoys.csv", 0, dec_cap)):
        p = setdir / fn
        if not p.exists():
            continue
        for i, r in enumerate(csv.DictReader(open(p))):
            if i >= cap:
                break
            rows.append((r["ligand_id"], r["smiles"], lab))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--targets", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--limit", type=int, default=0, help="调试用：每靶点只跑前 N 个配体")
    a = ap.parse_args()

    setdir, dec_cap = SETS[a.set]
    outdir = Path(a.out_dir) if a.out_dir else ROOT / "results/mdl_official" / a.set
    outdir.mkdir(parents=True, exist_ok=True)
    todo = list(TARGETS) if a.all else [t.strip() for t in a.targets.split(",") if t.strip()]
    if not todo:
        ap.error("需要 --targets 或 --all")

    os.chdir(MDL)                                  # 官方脚本依赖相对路径
    import torch
    import torch.nn as nn
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdmolops
    RDLogger.DisableLog("rdApp.*")

    from utils.fns import Early_stopper, set_random_seed
    from dataset.graph_obj import VSTestGraphDataset_Fly_SDFMOL2_Refined
    from dataset.dataloader_obj import PassNoneDataLoader
    from architecture.MetalloDock import MetalloDock

    set_random_seed(2024)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = MetalloDock()
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(device)
    stopper = Early_stopper(model_file=str(MDL / "trained_models/docking_metallodock_time.pkl"),
                            mode="lower", patience=10)
    stopper.load_model(model_obj=model, my_device=device, strict=False)
    model.eval()
    print(f"[model] 加载完成 {time.time()-t0:.1f}s, device={device}", flush=True)

    vocab = str(MDL / "dataset/chembl_fragments_voc_dict.pkl")
    work = ROOT / "results/mdl_official" / "_work" / a.set
    work.mkdir(parents=True, exist_ok=True)

    for T in todo:
        rel, metal, setname = TARGETS[T]
        dst = outdir / f"{T}.csv"
        rows = load_set(setdir, setname, dec_cap)
        if a.limit:
            rows = rows[:a.limit]
        done = list(csv.DictReader(open(dst))) if dst.exists() and dst.stat().st_size else []
        have = {r["ligand_id"] for r in done if r.get("score")}   # 失败行不算完成，需重算
        todo_rows = [r for r in rows if r[0] not in have]
        if not todo_rows:
            print(f"{T}: 已完成 {len(done)}，跳过", flush=True)
            continue

        # pocket（官方口径）
        pocket = work / f"{T}_pocket10.pdb"
        n_at, center = make_pocket_official(ROOT / rel, metal, pocket, a.radius)

        # 配体 SDF（去氢）
        ligdir = work / f"{T}_ligands"
        ligdir.mkdir(parents=True, exist_ok=True)
        names, seen = [], set()
        n_desalt = 0
        for lid, smi, lab in todo_rows:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            raw = mol
            mol = largest_fragment(mol)
            if mol is not raw and mol.GetNumHeavyAtoms() != raw.GetNumHeavyAtoms():
                n_desalt += 1
            m = embed_3d(mol)
            if m is None:
                continue
            sw = Chem.SDWriter(str(ligdir / f"{lid}.sdf")); sw.write(m); sw.close()
            names.append(lid); seen.add(lid)
        print(f"\n[{T}] pocket {n_at} 原子 @ {center.round(2)}；"
              f"待算配体 {len(names)}/{len(rows)}（其中 {n_desalt} 个做了脱盐）", flush=True)

        ds = VSTestGraphDataset_Fly_SDFMOL2_Refined(
            protein_file=str(pocket), ligand_path=str(ligdir),
            ligand_files=names, voc_file=vocab, my_device=device)
        loader = PassNoneDataLoader(dataset=ds, batch_size=a.batch, shuffle=False,
                                    num_workers=0, follow_batch=["atom2res"], pin_memory=False)
        lab_of = {lid: lab for lid, _, lab in rows}
        tmp = dst.with_suffix(".csv.tmp")
        n_ok = 0
        tstart = time.time()
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["ligand_id", "label", "score", "error"])
            for r in done:                       # 只保留已成功的记录
                if r.get("score"):
                    w.writerow([r["ligand_id"], r["label"], r["score"], ""])
            fh.flush()
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    try:
                        _, score, *_ = model.module.ligand_docking_coordination(data)
                        sc = score.detach().cpu().numpy().reshape(-1)
                    except Exception as exc:
                        for pid in data.pdb_id:
                            w.writerow([str(pid), lab_of.get(str(pid), 0), "", type(exc).__name__])
                        fh.flush()
                        continue
                    for pid, s in zip(data.pdb_id, sc):
                        pid = str(pid)
                        w.writerow([pid, lab_of.get(pid, 0), f"{float(s):.6e}", ""])
                        n_ok += 1
                    fh.flush()
                    el = time.time() - tstart
                    print(f"  {T}: {n_ok}/{len(names)}  {el:.0f}s "
                          f"({el/max(1,n_ok):.2f}s/配体)", flush=True)
        tmp.replace(dst)
        print(f"[{T}] 完成 {n_ok} 条 -> {dst.name}（{time.time()-tstart:.0f}s）", flush=True)

    print("\n全部完成")


if __name__ == "__main__":
    main()
