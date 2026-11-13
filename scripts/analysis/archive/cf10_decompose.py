"""Decompose the three scoring variants into their terms.

full = pct(total) + pct(g_buried) - pct(n_rotatable * clip(buried_area/600, 0, 1))
        A            B                    C
All AUROCs use the project convention "lower score = better".
"""
from __future__ import annotations
import csv, glob, importlib.util, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
spec = importlib.util.spec_from_file_location(
    "mcf", ROOT / "scripts/analysis/metrics_crossfamily10.py")
mcf = importlib.util.module_from_spec(spec)
sys.modules["mcf"] = mcf
spec.loader.exec_module(mcf)

TARGETS = ["MMP2", "CAPN1", "HDAC1", "CA2", "METAP2", "CDK2", "urease"]


def load(t):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{t}_p*.csv"))):
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames:
                continue
            nf = len(rd.fieldnames)
            for r in rd:
                if None in r or len(r) != nf:
                    continue
                if (r.get("g_total") or "").strip() in ("", "nan"):
                    continue
                try:
                    for k in ("total", "g_buried", "n_rotatable", "buried_area"):
                        float(r[k])
                except ValueError:
                    continue
                rows.append(r)
    return rows


print("=" * 96)
print("分项单独排序能力（AUROC，越低越好口径；0.5 = 无信息）")
print("  A = pct(total) 纯几何   B = pct(g_buried) 埋藏自由能   C = pct(n_rot*clip(bur/600)) CAT 惩罚")
print("=" * 96)
print(f"{'target':<8}{'A 单独':>9}{'B 单独':>9}{'C 单独':>9}"
      f"{'A+B':>8}{'A-B':>8}{'A+B-C':>9}{'A-C':>8}")
per = {}
for t in TARGETS:
    rows = load(t)
    if not rows:
        continue
    y = np.array([int(r["label"]) for r in rows])
    tot = np.array([float(r["total"]) for r in rows])
    gb = np.array([float(r["g_buried"]) for r in rows])
    bur = np.array([float(r["buried_area"]) for r in rows])
    nrot = np.array([float(r["n_rotatable"]) for r in rows])
    A = mcf.pct(tot)
    B = mcf.pct(gb)
    C = mcf.pct(nrot * np.clip(bur / mcf.A_REF, 0, 1))
    per[t] = dict(y=y, A=A, B=B, C=C, tot=tot, gb=gb, bur=bur, nrot=nrot)
    au = lambda s: mcf.auroc(y, s)
    print(f"{t:<8}{au(A):>9.3f}{au(B):>9.3f}{au(C):>9.3f}"
          f"{au(A+B):>8.3f}{au(A-B):>8.3f}{au(A+B-C):>9.3f}{au(A-C):>8.3f}")
print(f"{'宏平均':<8}" + "".join(
    f"{np.mean([mcf.auroc(d['y'], s) for d in per.values()]):>9.3f}" for s in ()))

print()
print("=" * 96)
print("urease vs CDK2 vs 两个受益靶点：active 与 decoy 的中位数（看方向）")
print("=" * 96)
for t in ("urease", "CDK2", "MMP2", "CAPN1"):
    d = per[t]
    y = d["y"]
    print(f"\n--- {t}  (active={int(y.sum())}, decoy={len(y)-int(y.sum())})")
    print(f"    {'项':<16}{'active 中位':>14}{'decoy 中位':>14}{'方向':>10}")
    for name, v in (("total 几何分", d["tot"]), ("g_buried", d["gb"]),
                    ("buried_area", d["bur"]), ("n_rotatable", d["nrot"]),
                    ("CAT 原始量", d["nrot"] * np.clip(d["bur"] / mcf.A_REF, 0, 1))):
        ma, md = np.median(v[y == 1]), np.median(v[y == 0])
        # total 越低越好；g_buried 越低越好；其余是惩罚项，越低越好
        good = "active 更优" if ma < md else "decoy 更优"
        print(f"    {name:<16}{ma:>14.4g}{md:>14.4g}{good:>10}")

print()
print("=" * 96)
print("CAT 项是否被 clip 饱和（buried_area / 600 被截到 1 的比例）")
print("=" * 96)
print(f"{'target':<8}{'buried_area 中位':>18}{'>=600 占比':>12}")
for t in TARGETS:
    d = per[t]
    print(f"{t:<8}{np.median(d['bur']):>18.1f}{(d['bur'] >= mcf.A_REF).mean()*100:>11.1f}%")
