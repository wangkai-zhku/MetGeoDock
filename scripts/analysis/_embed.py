"""统一的 3D 构象生成，供 MetalGeoDock / Vina / MetalloDock 三条线共用。

历史行为必须保持：第一颗种子（0xF00D）的调用形式与旧脚本一模一样，
这样已跑完的靶点结果能逐位复现。

新增两道闸门，专治配体集里混进来的「不可对接分子」：
  * 寡聚物闸门：MW > 1200 或可旋转键 > 40。EGLN2 的配体集里有 3 个 3.5-3.9 kDa、
    113 个可旋转键的寡聚物，ETKDG 单次耗时 12-19 s 且必然失败。
  * 复杂分子闸门：首颗种子失败后，若可旋转键 > 8 或 MW > 600 就放弃。
    稠合多环分子换种子、尤其是 useRandomCoords 兜底，动辄几分钟且几乎不可能成功，
    会把 worker 卡死（2026-10-29 阶段1 就卡在这里）。
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

MW_MAX = 1200.0        # 寡聚物闸门
ROT_MAX = 40
RETRY_ROT_MAX = 8      # 首颗种子失败后是否还值得换种子
RETRY_MW_MAX = 600.0


def embed_3d(mol, keep_h: bool = False):
    """返回带 3D 坐标的分子；判定为不可处理时返回 None。

    keep_h=True 时保留显式氢（Vina 需要）；否则去掉氢（MetalGeoDock / MetalloDock 用重原子）。
    """
    mw = Descriptors.MolWt(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    if mw > MW_MAX or rot > ROT_MAX:
        return None
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=0xF00D) == 0:
        return m if keep_h else Chem.RemoveHs(m)
    if rot > RETRY_ROT_MAX or mw > RETRY_MW_MAX:
        return None
    for seed in (0xF00D + 1, 0xF00D + 2):
        if AllChem.EmbedMolecule(m, randomSeed=seed) == 0:
            return m if keep_h else Chem.RemoveHs(m)
    if AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=0xF00D) == 0:
        return m if keep_h else Chem.RemoveHs(m)
    return None
