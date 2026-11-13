"""Physiologically relevant assembly states.

1. 5G0J  HDAC6 CD1+CD2 tandem (zebrafish, 2.88 A). Which CD2 cap-region residues
         are buried at the CD1-CD2 interface? If interface residues fall inside the
         8-16 A cap shell, modelling CD2 alone (as our 5EDU entry does) reshapes
         the very region we score.
2. 5ZOP  HDAC4 + SMRT SRD3c peptides. Where does the SMRT beta-hairpin sit relative
         to the catalytic Zn, and how much of the cap shell does it occupy?
3. Table of the assembly state each isoform actually functions in.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
A3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
      "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}
WATER = {"HOH", "WAT", "DOD"}


def parse(pid):
    prot, het = [], []
    for ln in (ROOT / f"processed/hdac_receptors/{pid}.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        if rn in WATER:
            continue
        rec = dict(het=ln.startswith("HETATM"), rn=rn, ch=ln[21], num=int(ln[22:26]),
                   an=ln[12:16].strip(),
                   xyz=np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
        (het if rec["het"] else prot).append(rec)
    return prot, het


def zns(het):
    return [(h["ch"], h["xyz"]) for h in het if h["rn"].startswith("ZN")]


def cap_shell(prot, metal, ch, rin=8.0, rout=16.0):
    per = {}
    for a in prot:
        if a["ch"] != ch:
            continue
        per.setdefault(a["num"], {"rn": a["rn"], "xyz": []})
        per[a["num"]]["xyz"].append(a["xyz"])
    out = {}
    for k, v in per.items():
        d = float(np.linalg.norm(np.array(v["xyz"]) - metal, axis=1).min())
        if rin <= d <= rout:
            out[k] = v
    return out


def main():
    print("=" * 90)
    print("1) 5G0J  HDAC6 CD1+CD2 串联结构（zebrafish）")
    print("=" * 90)
    prot, het = parse("5G0J")
    zl = zns(het)
    print(f"  蛋白残基 {len({(a['ch'], a['num']) for a in prot})} 个，锌 {len(zl)} 个")
    for i, (ch, p) in enumerate(zl):
        near = sorted(((float(np.linalg.norm(a["xyz"] - p)), f"{a['rn']}{a['num']}")
                       for a in prot if a["ch"] == ch), key=lambda x: x[0])[:4]
        print(f"  ZN{i} 链{ch}: " + ", ".join(f"{lbl}({d:.2f})" for d, lbl in near))
    # 取残基数更多的亚基作为 CD2（CD2 通常更完整/含选择性口袋）
    bych = {}
    for a in prot:
        bych.setdefault(a["ch"], set()).add(a["num"])
    for ch in bych:
        print(f"    链{ch}: {len(bych[ch])} 残基, {min(bych[ch])}-{max(bych[ch])}")
    # 两个锌各自所在的链 -> 界面残基
    if len(zl) >= 2:
        (c1, z1), (c2, z2) = zl[0], zl[1]
        print(f"\n  两个催化域: 链{c1} / 链{c2}")
        # 界面：链1的残基与链2原子距离 < 4.5 A
        p2 = [a["xyz"] for a in prot if a["ch"] == c2]
        iface = set()
        per1 = {}
        for a in prot:
            if a["ch"] != c1:
                continue
            per1.setdefault(a["num"], {"rn": a["rn"], "xyz": []})
            per1[a["num"]]["xyz"].append(a["xyz"])
        arr2 = np.array(p2)
        for k, v in per1.items():
            arr1 = np.array(v["xyz"])
            d = np.sqrt(((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)).min()
            if d < 4.5:
                iface.add(k)
        print(f"  链{c1} 上处于 CD1-CD2 界面的残基: {len(iface)} 个")
        for ch, metal in ((c1, z1), (c2, z2)):
            cap = cap_shell(prot, metal, ch)
            ov = sorted(set(cap) & iface) if ch == c1 else []
            print(f"    链{ch}: cap 区 {len(cap)} 残基", end="")
            if ch == c1:
                print(f"，其中 {len(ov)} 个位于界面: {ov if ov else '无'}")
            else:
                print()

    print("\n" + "=" * 90)
    print("2) 5ZOP  HDAC4 + SMRT（SRD3c 肽）")
    print("=" * 90)
    prot, het = parse("5ZOP")
    zl = zns(het)
    print(f"  蛋白残基 {len({(a['ch'], a['num']) for a in prot})} 个，锌 {len(zl)} 个")
    bych = {}
    for a in prot:
        bych.setdefault(a["ch"], set()).add(a["num"])
    for ch, nums in sorted(bych.items()):
        print(f"    链{ch}: {len(nums)} 残基, {min(nums)}-{max(nums)}")
    # HDAC4 链（残基号覆盖 650-1033 的）
    hdac_ch = max(bych, key=lambda c: max(bych[c]) - min(bych[c]))
    smrt_ch = [c for c in bych if c != hdac_ch]
    # 催化锌 = 靠近 ASP/DAsp 的那个；用配位组氨酸判断
    cat_zn = None
    for ch, p in zl:
        near = [a for a in prot if a["ch"] == ch and float(np.linalg.norm(a["xyz"] - p)) < 2.6]
        if any(a["rn"] == "ASP" for a in near):
            cat_zn = (ch, p)
            break
    if cat_zn is None and zl:
        cat_zn = zl[0]
    print(f"\n  HDAC4 链 = {hdac_ch};  SMRT 链 = {smrt_ch}")
    if smrt_ch:
        for sc in smrt_ch:
            sm = [a for a in prot if a["ch"] == sc]
            arr = np.array([a["xyz"] for a in sm])
            d = float(np.linalg.norm(arr - cat_zn[1], axis=1).min())
            print(f"  SMRT 链{sc}: {len({a['num'] for a in sm})} 残基，"
                  f"距催化锌最近 {d:.1f} A")
            # 有多少 SMRT 原子落在 cap 壳内 / 催化裂隙内
            cap = cap_shell(prot, cat_zn[1], hdac_ch)
            cap_xyz = np.array([x for v in cap.values() for x in v["xyz"]])
            dc = np.sqrt(((arr[:, None, :] - cap_xyz[None, :, :]) ** 2).sum(-1))
            print(f"     SMRT 原子与 HDAC4 cap 区最小距离 {dc.min():.1f} A，"
                  f"紧贴(<4.5A)的 SMRT 原子 {int((dc.min(1) < 4.5).sum())} 个")
            # 占据口袋的位置：SMRT 残基里距催化锌 8-16 A 的
            per = {}
            for a in sm:
                per.setdefault(a["num"], {"rn": a["rn"], "xyz": []})
                per[a["num"]]["xyz"].append(a["xyz"])
            in_cap = {k: v for k, v in per.items()
                      if 8.0 <= float(np.linalg.norm(np.array(v["xyz"]) - cat_zn[1], axis=1).min()) <= 16.0}
            print(f"     SMRT 中落在 8-16 A 壳层的残基 {len(in_cap)} 个: "
                  f"{sorted((v['rn'] + str(k)) for k, v in in_cap.items())}")

    print("\n" + "=" * 90)
    print("3) 六个亚型的生理相关组装状态")
    print("=" * 90)
    rows = [
        ("HDAC1", "5ICN", "单体催化域 (MTA1/HDAC1 复合物中的 HDAC1)",
         "Sin3/NuRD/CoREST 复合物的催化核心；S263 磷模拟失活", "单体可用"),
        ("HDAC2", "3MAX", "单体催化域 (HDAC2 + 抑制剂)",
         "与 HDAC1 同源 93.7%；S394 磷酸化调控", "单体可用"),
        ("HDAC3", "4A69", "HDAC3 + SMRT DAD + Ins(1,4,5,6)P4",
         "唯一被核心阻遏物 DAD 激活的 class I；InsP4 作分子胶", "需 DAD+InsP4"),
        ("HDAC4", "2VQM/6FYZ", "催化域 + sZBD（两种构象）",
         "pseudo-enzyme (His976 朝外 7.8A)；SMRT SRD3c β-hairpin 阻断裂隙", "需考虑 SMRT 阻断"),
        ("HDAC6", "5EDU (CD2 单域+MBP)", "应用 5G0J (CD1+CD2 串联)",
         "CD1-CD2 椭圆复合物，界面变构耦合；CD2 为主活性域", "单域不完整"),
        ("HDAC8", "1T64/2V5W", "单体催化域",
         "无复合物依赖；L1 lid 门控 (open/closed 差 6.2A)", "单体可用+需开放构象"),
    ]
    print(f"{'亚型':<8}{'我们用的结构':<22}{'生理组装':<34}{'关键机制':<44}评估")
    for r in rows:
        print(f"{r[0]:<8}{r[1]:<22}{r[2]:<34}{r[3]:<44}{r[4]}")
    json.dump({"note": "assembly state audit", "rows": rows},
              open(ROOT / "results/hdac_assembly_audit.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
