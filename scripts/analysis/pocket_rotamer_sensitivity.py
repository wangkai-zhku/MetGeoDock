"""Sensitivity sweep for the structure-restricted rotamer entropy.

If the HDAC1 vs HDAC2 gap only appears at one particular clash threshold and one
particular shell, it is an artefact. Here we sweep both and ask:
  * does the ordering HDAC1 < HDAC2 hold everywhere,
  * how large is the gap relative to the spread across subtypes,
  * which residues drive the gap (stable across settings or not).
"""
from __future__ import annotations
import importlib.util, itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
spec = importlib.util.spec_from_file_location(
    "pre", ROOT / "scripts/analysis/pocket_rotamer_entropy.py")
pre = importlib.util.module_from_spec(spec)
sys.modules["pre"] = pre
spec.loader.exec_module(pre)

SUB = list(pre.PDB)
PROBES = [2.2, 2.4, 2.5, 2.6, 2.8]
SHELLS = [(6, 12), (8, 14), (10, 16), (6, 14), (8, 16)]

# read each structure once
STRUCT = {}
for name, pid in pre.PDB.items():
    atoms, metal = pre.read_structure(pid)
    by_res = {}
    for rn, key, an, xyz in atoms:
        by_res.setdefault(key, {"rn": rn, "atoms": []})["atoms"].append((rn, key, an, xyz))
    dmin = {k: min(np.linalg.norm(x[3] - metal) for x in v["atoms"]) for k, v in by_res.items()}
    STRUCT[name] = {"atoms": atoms, "by_res": by_res, "dmin": dmin, "metal": metal}
    print(f"读入 {name} ({len(atoms)} 原子, {len(by_res)} 残基)")

print("\n" + "=" * 108)
print("结构受限总熵：碰撞阈值 × 壳层 扫描")
print("=" * 108)
table = {}
for probe in PROBES:
    for lo, hi in SHELLS:
        vals = {}
        for name, S in STRUCT.items():
            pocket = [k for k, d in S["dmin"].items() if lo < d <= hi]
            tot = 0.0
            n = 0
            for key in pocket:
                v = S["by_res"][key]
                others = [x for x in S["atoms"] if x[1] != key]
                r = pre.residue_entropy(v["rn"], v["atoms"], others, probe, S["metal"])
                if r is not None:
                    tot += r["S"]; n += 1
            vals[name] = (round(tot, 3), n)
        table[(probe, lo, hi)] = vals
        gap = vals["HDAC2"][0] - vals["HDAC1"][0]
        rel = gap / max(vals["HDAC1"][0], 1e-9)
        order = "HDAC1<HDAC2" if gap > 0 else "HDAC1>=HDAC2"
        print(f"probe={probe}  shell={lo}-{hi:<3} "
              f"HDAC1={vals['HDAC1'][0]:6.2f}({vals['HDAC1'][1]:2d}) "
              f"HDAC2={vals['HDAC2'][0]:6.2f}({vals['HDAC2'][1]:2d}) "
              f"HDAC3={vals['HDAC3'][0]:6.2f}({vals['HDAC3'][1]:2d}) "
              f"gap={gap:+6.2f} ({rel:+6.1%})  {order}")

gaps = [v["HDAC2"][0] - v["HDAC1"][0] for v in table.values()]
print(f"\nHDAC2-HDAC1 差值统计（{len(gaps)} 组参数）:")
print(f"  中位={np.median(gaps):+.2f}  均值={np.mean(gaps):+.2f}  "
      f"min={min(gaps):+.2f}  max={max(gaps):+.2f}")
print(f"  HDAC2 > HDAC1 的比例: {np.mean(np.array(gaps) > 0):.0%}")

# which residues drive the gap, under the default setting
print("\n" + "=" * 108)
print("贡献残基定位（probe=2.5, shell=8-14）：谁的锁定状态在两亚型间不同")
print("=" * 108)
probe, lo, hi = 2.5, 8.0, 14.0
detail = {}
for name in SUB:
    S = STRUCT[name]
    pocket = [k for k, d in S["dmin"].items() if lo < d <= hi]
    rows = {}
    for key in pocket:
        v = S["by_res"][key]
        others = [x for x in S["atoms"] if x[1] != key]
        r = pre.residue_entropy(v["rn"], v["atoms"], others, probe, S["metal"])
        if r is not None:
            rows[key] = (v["rn"], r["S"], r["n_allowed"])
    detail[name] = rows

# compare by residue TYPE (sequence correspondence is not resolved yet, so group by type)
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))
for name, rows in detail.items():
    for key, (rn, S, n) in rows.items():
        agg[rn][name].append(S)
print(f"{'残基类型':<8}" + "".join(f"{s[-1]:>9}" for s in SUB) + f"{'H1→H2 变化':>12}")
rows_out = []
for rn, per in agg.items():
    means = {s: float(np.mean(per.get(s, [np.nan]))) for s in SUB}
    if not np.isfinite(means["HDAC1"]) or not np.isfinite(means["HDAC2"]):
        continue
    delta = means["HDAC2"] - means["HDAC1"]
    rows_out.append((abs(delta), rn, means, delta))
for ad, rn, means, delta in sorted(rows_out, reverse=True):
    print(f"{rn:<8}" + "".join(f"{means[s]:>9.2f}" for s in SUB) + f"{delta:>+12.2f}")

json.dump({f"{k[0]}|{k[1]}-{k[2]}": v for k, v in table.items()},
          open(ROOT / "results/hdac_rotamer_sensitivity.json", "w"), indent=2)
print("\n写出 results/hdac_rotamer_sensitivity.json")
