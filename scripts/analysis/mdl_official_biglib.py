"""验证性测试：在 biglib HDAC1（100 actives + 5690 decoys，1:57）上跑 MetalloDock 官方口径。

目的：论文 SI 声称 EF_0.5% 可达 20+，但那类指标只有在 decoy 上千（actives 比例 ~1-2%）时
才有意义 —— 我们 1:10 数据集（1100 配体，0.5% 只对应 6 个位置）的 EF0.5% 数学上限仅 11。
这里用项目现成的大库做一次交叉验证：看该软件在合适规模的数据集上能到什么水平。

配置（严格按官方 VS 口径）：
  * 前向 ligand_docking_coordination（模型自选配位原子 + _4vs 打分）
  * pocket = 催化金属 10 Å 内残基的全部原子（pre_process.py 口径）
  * 配体去氢 SDF；actives 由 SMILES embed，decoys 直接来自既有 SDF
  * 不启用 refine（只取前向给出的 mdn_score）

actives 来源：/home/wangkai/cursor-task/HDAC/data/processed/hdac_dataset.csv（label==1）
decoys 来源：docking/biglib/<TARGET>.sdf（5690 个）
受体：/home/wangkai/cursor-task/HDAC/structures/receptors_h/<TARGET>.pdb

用法：
    python scripts/analysis/mdl_official_biglib.py --target HDAC1 --batch 30
输出：results/mdl_official/biglib_<TARGET>.csv (ligand_id,label,score,error)
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
HDAC = Path("/home/wangkai/cursor-task/HDAC")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(MDL))
sys.path.insert(0, str(MDL / "utils"))

from _embed import embed_3d          # noqa: E402
from mdl_official_bench22 import make_pocket_official, largest_fragment  # noqa: E402

METALS = {"HDAC1": "zn", "HDAC2": "zn", "HDAC3": "zn", "HDAC4": "zn",
          "HDAC6": "zn", "HDAC7": "zn", "HDAC8": "zn"}


def prepare_ligands(target: str, work: Path):
    """把 actives(CSV) 与 decoys(SDF) 统一写成单分子 SDF 目录，返回 [(id,label)]。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    ligdir = work / f"{target}_ligands"
    ligdir.mkdir(parents=True, exist_ok=True)
    entries = []

    # 1) actives
    act_csv = HDAC / "data/processed/hdac_dataset.csv"
    n_act = 0
    for r in csv.DictReader(open(act_csv)):
        if r.get("target") != target or str(r.get("label")) != "1":
            continue
        smi = (r.get("smiles") or "").strip()
        lid = r["ligand_id"]
        if not smi or (ligdir / f"{lid}.sdf").exists():
            entries.append((lid, 1)); n_act += 1; continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        m = embed_3d(mol)
        if m is None:
            continue
        sw = Chem.SDWriter(str(ligdir / f"{lid}.sdf")); sw.write(m); sw.close()
        entries.append((lid, 1)); n_act += 1

    # 2) decoys（既有 SDF，逐分子落盘）
    sdf = ROOT / "docking/biglib" / f"{target}.sdf"
    n_dec = 0
    if sdf.exists():
        supp = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)
        for mol in supp:
            if mol is None:
                continue
            lid = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{target}_BL{n_dec:05d}"
            dst = ligdir / f"{lid}.sdf"
            if not dst.exists():
                sw = Chem.SDWriter(str(dst)); sw.write(mol); sw.close()
            entries.append((lid, 0)); n_dec += 1
    print(f"[{target}] actives {n_act} + decoys {n_dec} = {len(entries)}", flush=True)
    return entries, ligdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="HDAC1")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", type=int, default=0, help="分片编号（多进程并行用）")
    ap.add_argument("--nshards", type=int, default=1, help="总分片数")
    a = ap.parse_args()

    os.chdir(MDL)
    import torch
    import torch.nn as nn
    from utils.fns import Early_stopper, set_random_seed
    from dataset.graph_obj import VSTestGraphDataset_Fly_SDFMOL2_Refined
    from dataset.dataloader_obj import PassNoneDataLoader
    from architecture.MetalloDock import MetalloDock

    set_random_seed(2024)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    work = ROOT / "results/mdl_official" / "_biglib_work" / a.target
    work.mkdir(parents=True, exist_ok=True)

    receptor = HDAC / "structures/receptors_h" / f"{a.target}.pdb"
    pocket = work / f"{a.target}_pocket10.pdb"
    n_at, center = make_pocket_official(receptor, METALS.get(a.target, "zn"), pocket, 10.0)
    print(f"[{a.target}] pocket {n_at} 原子 @ {center.round(2)}", flush=True)

    entries, ligdir = prepare_ligands(a.target, work)
    lab_of = dict(entries)

    model = MetalloDock()
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(device)
    stopper = Early_stopper(model_file=str(MDL / "trained_models/docking_metallodock_time.pkl"),
                            mode="lower", patience=10)
    stopper.load_model(model_obj=model, my_device=device, strict=False)
    model.eval()

    names = [lid for lid, _ in entries][a.shard::a.nshards]   # 分片
    ds = VSTestGraphDataset_Fly_SDFMOL2_Refined(
        protein_file=str(pocket), ligand_path=str(ligdir), ligand_files=names,
        voc_file=str(MDL / "dataset/chembl_fragments_voc_dict.pkl"), my_device=device)
    loader = PassNoneDataLoader(dataset=ds, batch_size=a.batch, shuffle=False,
                                num_workers=0, follow_batch=["atom2res"], pin_memory=False)

    out = (Path(a.out) if a.out else
           ROOT / "results/mdl_official" / f"biglib_{a.target}_s{a.shard}.csv")
    n_ok = 0
    t0 = time.time()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["ligand_id", "label", "score", "error"])
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                try:
                    _, score, *_ = model.module.ligand_docking_coordination(data)
                    sc = score.detach().cpu().numpy().reshape(-1)
                except Exception as exc:
                    for pid in data.pdb_id:
                        w.writerow([str(pid), lab_of.get(str(pid), 0), "", type(exc).__name__])
                    fh.flush(); continue
                for pid, s in zip(data.pdb_id, sc):
                    pid = str(pid)
                    w.writerow([pid, lab_of.get(pid, 0), f"{float(s):.6e}", ""])
                    n_ok += 1
                fh.flush()
                el = time.time() - t0
                print(f"  {n_ok}/{len(names)}  {el:.0f}s ({el/max(1,n_ok):.2f}s/配体)", flush=True)
    print(f"[{a.target}] 完成 {n_ok} 条 -> {out}（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
