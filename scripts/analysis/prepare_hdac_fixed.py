"""重建 HDAC 受体：保留正确的链与辅因子、去掉结晶水和多余拷贝。

背景（2026-10-29）：bench22 的 HDAC 受体是用 `metgeodock.prepare.split_pdb_records`
处理的，它只保留「蛋白 ATOM + 金属 + 水」，其余 HETATM 一律丢弃，于是：
  * HDAC3 (4A69) 丢了 SMRT DAD 肽段（链 C/D）和 4 个 Ins(1,4,5,6)P4 —— InsP4 是这个
    复合物的"分子胶"，HDAC3 的催化活性依赖 DAD 结合；
  * HDAC1 (5ICN) 丢了 MTA1 片段（链 A）和 IP6；
  * HDAC6 用了 5EDU 的 CD2 单域（且文件里留着两个 CD2 拷贝互相遮挡）；
  * 结晶水按项目约定在对接前会被 strip_waters 去掉，这里直接在准备阶段就不保留。

本脚本因此：
  1. 按链白名单裁剪（只取与目标分子结合的拷贝，剔除晶体中多余的对称拷贝）；
  2. 蛋白部分交给 metgeodock.prepare.prepare_receptor 加氢，keep_waters=False；
  3. 把白名单里的非标准辅因子（InsP4 / IP6 等）以重原子形式追加回受体；
  4. 写一份 report，记录链、辅因子、金属配位球。

注：辅因子以重原子追加（PDBFixer 无法给非标准残基加氢），fractional charge 记 0，
    对以范德华/接触为主项的打分影响有限，但报告里需注明。

用法：
    python scripts/analysis/prepare_hdac_fixed.py            # 全部 6 个版本
    python scripts/analysis/prepare_hdac_fixed.py --only HDAC3_fix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "processed/hdac_receptors"
OUT = ROOT / "processed/hdac_fixed"

WATER = {"HOH", "WAT", "H2O", "DOD"}
METALS = {"ZN", "FE", "FE2", "FE3", "MN", "MN2", "MG", "CA", "CO", "CU", "NI", "CD", "NA", "K"}

# 名字 -> (源 PDB, 保留链, 保留的非标准辅因子残基名, 说明)
SPECS = {
    "HDAC1_fix": ("5ICN.pdb", ["B", "A"], ["IHP"],
                  "HDAC1 催化域(链B) + MTA1 片段(链A) + IP6；剔除抑制剂 6A0"),
    "HDAC2_fix": ("3MAX.pdb", ["B"], [],
                  "HDAC2 单拷贝(链B)；剔除抑制剂 LLX 与晶体三拷贝"),
    "HDAC3_fix": ("4A69.pdb", ["A", "C"], ["I0P"],
                  "HDAC3(链A) + 就近的 SMRT DAD(链C) + Ins(1,4,5,6)P4；剔除链D/链B 拷贝"),
    "HDAC6_fix": ("5G0J.pdb", ["A"], [],
                  "HDAC6 CD1+CD2 串联域(链A)；剔除共晶抑制剂 N4R"),
    "HDAC8_2V5W": ("2V5W.pdb", ["A"], [],
                   "HDAC8 构象甲(2V5W 链A)；剔除 ALY/MCM 与晶体二拷贝"),
    "HDAC8_2V5X": ("2V5X.pdb", ["A"], [],
                   "HDAC8 构象乙(2V5X 链A)；剔除 V5X 与晶体二拷贝"),
}


def read_lines(path: Path) -> list[str]:
    return path.read_text(errors="ignore").splitlines()


def select(src: Path, chains: list[str], hets: list[str]):
    """返回 (蛋白行, 辅因子行, 金属行, 丢弃统计)"""
    prot, cof, metal, dropped = [], [], [], {}
    keep_ch = set(chains)
    for ln in read_lines(src):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        resn = ln[17:20].strip().upper()
        ch = ln[21]
        if ln.startswith("ATOM"):
            if ch in keep_ch:
                prot.append(ln)
            continue
        if resn in WATER:
            dropped["water"] = dropped.get("water", 0) + 1
        elif resn in METALS:
            if ch in keep_ch:
                metal.append(ln)
            else:
                dropped["metal_in_dropped_chain"] = dropped.get("metal_in_dropped_chain", 0) + 1
        elif resn in {h.upper() for h in hets}:
            if ch in keep_ch:
                cof.append(ln)
            else:
                dropped[f"{resn}_in_dropped_chain"] = dropped.get(f"{resn}_in_dropped_chain", 0) + 1
        else:
            dropped[resn] = dropped.get(resn, 0) + 1
    return prot, cof, metal, dropped


def write_prot(prot: list[str], dest: Path):
    """按链写蛋白行，链之间插 TER，便于 PDBFixer 正确切链。"""
    out, last = [], None
    for ln in prot:
        ch = ln[21]
        if last is not None and ch != last:
            out.append("TER")
        out.append(ln)
        last = ch
    out.append("TER")
    dest.write_text("\n".join(out) + "\nEND\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--ph", type=float, default=7.4)
    a = ap.parse_args()
    todo = [a.only] if a.only else list(SPECS)
    OUT.mkdir(parents=True, exist_ok=True)

    from metgeodock.prepare import prepare_receptor

    for name in todo:
        pdb, chains, hets, note = SPECS[name]
        src = SRC / pdb
        if not src.exists():
            print(f"[{name}] 缺源文件 {src}")
            continue
        prot, cof, metal, dropped = select(src, chains, hets)
        print(f"\n[{name}] {pdb} 链={chains} 蛋白行={len(prot)} 辅因子={len(cof)} 金属={len(metal)}")
        print(f"  去除: " + ", ".join(f"{k}×{v}" for k, v in sorted(dropped.items()) if k != "water")
              + f" | 结晶水×{dropped.get('water', 0)}")
        if not prot:
            print(f"[{name}] 没有蛋白原子，跳过")
            continue

        work = OUT / f"{name}_prot.pdb"
        write_prot(prot, work)
        with_metal = OUT / f"{name}_prot_metal.pdb"
        with_metal.write_text("\n".join(prot + metal + ["TER"]) + "\nEND\n")

        # 加氢：keep_waters=False —— 结晶水在对接前本来就会被 strip_waters 去掉
        rep = prepare_receptor(
            str(with_metal),
            out=str(OUT / f"{name}_h_step1.pdb"),
            ph=a.ph, metal="zn", keep_waters=False,
        )

        # 把非标准辅因子以重原子形式追加回受体
        final = OUT / f"{name}_h.pdb"
        body = read_lines(OUT / f"{name}_h_step1.pdb")
        body = [ln for ln in body if not ln.startswith("END")]
        if cof:
            body.append("TER")
            body.extend(cof)
        body.append("TER")
        final.write_text("\n".join(body) + "\nEND\n")

        report = {
            "name": name, "source": str(src), "chains_kept": chains,
            "cofactors_kept": {h: sum(1 for ln in cof if ln[17:20].strip().upper() == h.upper())
                               for h in hets},
            "n_cofactor_atoms": len(cof),
            "waters_removed": dropped.get("water", 0),
            "dropped": {k: v for k, v in dropped.items() if k != "water"},
            "prepare": rep,
            "note": note,
            "caveat": "辅因子为晶体重原子，未加氢；电荷列记 0。",
        }
        (OUT / f"{name}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  -> {final.name}  原子行={sum(1 for ln in body if ln.startswith(('ATOM','HETATM')))}")

    print("\n完成。输出目录:", OUT)


if __name__ == "__main__":
    main()
