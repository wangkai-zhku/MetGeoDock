"""审计用：用官方 VS 的前向（ligand_docking_coordination）重算分数，与现有结果对比。

跳过 virtual_screening.py 里的 refine 步骤（UFF 优化很慢，且 mdn_score 是前向直接给出的），
只为回答一个问题：我们此前用 ligand_docking + coordination_idx=0 得到的分数，
与官方 VS 路径（自动预测配位原子 + _4vs 打分）差多少。

用法（在 processed/metallodock_ref 下执行）：
    python ../../scripts/analysis/mdl_vs_score.py --ligands <sdf目录> --pocket <pocket.pdb> \
        --out <csv输出> --batch 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.realpath(__file__))
MDL = os.path.abspath(os.path.join(HERE, "..", "..", "processed", "metallodock_ref"))
sys.path.insert(0, MDL)
sys.path.insert(0, os.path.join(MDL, "utils"))   # uff_build 等内部脚本都在 utils/ 下

from utils.fns import Early_stopper, set_random_seed          # noqa: E402
from dataset.graph_obj import VSTestGraphDataset_Fly_SDFMOL2_Refined  # noqa: E402
from dataset.dataloader_obj import PassNoneDataLoader         # noqa: E402
from architecture.MetalloDock import MetalloDock              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligands", required=True)
    ap.add_argument("--pocket", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--model-file", default=os.path.join(MDL, "trained_models/docking_metallodock_time.pkl"))
    ap.add_argument("--voc-file", default=os.path.join(MDL, "dataset/chembl_fragments_voc_dict.pkl"))
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()

    os.chdir(MDL)                      # 官方脚本依赖相对路径（dataset/、trained_models/）
    set_random_seed(a.seed)
    my_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = MetalloDock()
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(my_device)
    stopper = Early_stopper(model_file=a.model_file, mode="lower", patience=10)
    stopper.load_model(model_obj=model, my_device=my_device, strict=False)
    model.eval()

    names = [i.split(".")[0] for i in sorted(os.listdir(a.ligands))]
    ds = VSTestGraphDataset_Fly_SDFMOL2_Refined(
        protein_file=a.pocket, ligand_path=a.ligands, ligand_files=names,
        voc_file=a.voc_file, my_device=my_device)
    loader = PassNoneDataLoader(dataset=ds, batch_size=a.batch, shuffle=False,
                                num_workers=0, follow_batch=["atom2res"], pin_memory=False)

    rows = []
    with torch.no_grad():
        for data in loader:
            data = data.to(my_device)
            lg, score, *_ = model.module.ligand_docking_coordination(data)
            sc = score.detach().cpu().numpy().reshape(-1)
            for pid, s in zip(data.pdb_id, sc):
                rows.append((str(pid), float(s)))
    with open(a.out, "w") as fh:
        fh.write("ligand_id,score_official_vs\n")
        for pid, s in rows:
            fh.write(f"{pid},{s:.6f}\n")
    print(f"写出 {a.out}（{len(rows)} 条）")


if __name__ == "__main__":
    main()
