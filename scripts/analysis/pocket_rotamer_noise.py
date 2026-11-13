"""Intra-subtype noise vs inter-subtype difference for the continuous rotamer entropy.

A descriptor is only usable if the difference between subtypes exceeds the spread
*within* a subtype (different crystal structures of the same enzyme). This script
measures both.

Structures are taken from the 16k corpus (ORI_PDB_LIGAND), where several isoforms
have 2-5 independent crystal structures.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
CORPUS = Path("/home/wangkai/cursor-task/metalloprotein/ORI_PDB_LIGAND")
spec = importlib.util.spec_from_file_location(
    "pm", ROOT / "scripts/analysis/pocket_rotamer_metal.py")
soft = importlib.util.module_from_spec(spec)
sys.modules["pm"] = soft
spec.loader.exec_module(soft)

GROUPS = {
    "HDAC1": ["5ICN", "4BKX"],
    "HDAC2": ["3MAX", "4LXZ"],
    "HDAC3": ["4A69"],
    "HDAC6": ["5EDU", "5EF7", "5EF8"],
    "HDAC8": ["1T64", "1T67", "1T69", "2V5X", "3SFF"],
}
RT = 0.593
SHELL = (8.0, 14.0)


def find_file(pid):
    hits = sorted(CORPUS.glob(f"*/zn_{pid}_pro_*.pdb")) or sorted(CORPUS.glob(f"*/*_{pid}_pro_*.pdb"))
    return hits[0] if hits else None


def read_any(path):
    atoms, metal = [], []
    for ln in Path(path).read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        an = ln[12:16].strip()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            atoms.append((rn, ln[21] + ln[22:26].strip(), an, xyz, el))
        elif rn == "ZN":
            metal.append(xyz)
    return atoms, metal


def entropy_of(pid):
    f = find_file(pid)
    if f is None:
        return pid, None
    atoms, metals = read_any(f)
    if not atoms or not metals:
        return pid, None
    metal = metals[0]
    by = {}
    for rn, key, an, xyz, el in atoms:
        by.setdefault(key, {"rn": rn, "atoms": []})["atoms"].append((rn, key, an, xyz, el))
    dmin = {k: min(np.linalg.norm(x[3] - metal) for x in v["atoms"]) for k, v in by.items()}
    pocket = [k for k, d in dmin.items() if SHELL[0] < d <= SHELL[1]]
    all_xyz = np.array([x[3] for x in atoms])
    all_el = [x[4] for x in atoms]
    vals = []
    for key in pocket:
        v = by[key]
        keep = np.array([x[1] != key for x in atoms])
        # metal steric ON, electrostatics OFF (eps=1e6)
        r = soft.residue_entropy(v["rn"], v["atoms"], all_xyz[keep],
                                 [e for e, kk in zip(all_el, keep) if kk], metal, RT, 1e6)
        if r is not None:
            vals.append(r["S"])
    if not vals:
        return pid, None
    return pid, {"n": len(vals), "total": float(np.sum(vals)), "mean": float(np.mean(vals)),
                 "file": f.name}


if __name__ == "__main__":
    pids = [p for v in GROUPS.values() for p in v]
    print(f"计算 {len(pids)} 个结构的 rotamer 熵（LJ+金属位阻，静电关闭；壳层 {SHELL[0]}-{SHELL[1]} A, RT={RT}）", flush=True)
    with Pool(min(8, len(pids))) as pool:
        res = dict(pool.map(entropy_of, pids))

    print(f"\n{'亚型':<8}{'PDB':<7}{'残基':>6}{'总熵':>9}{'每残基':>9}")
    per_group = {}
    for g, ids in GROUPS.items():
        for p in ids:
            r = res.get(p)
            if not r:
                print(f"{g:<8}{p:<7}  失败")
                continue
            print(f"{g:<8}{p:<7}{r['n']:>6}{r['total']:>9.2f}{r['mean']:>9.3f}")
            per_group.setdefault(g, []).append(r["mean"])

    print("\n" + "=" * 78)
    print("信号 vs 噪声")
    print("=" * 78)
    print(f"{'亚型':<8}{'结构数':>6}{'每残基熵 均值':>14}{'范围':>16}{'组内极差':>10}")
    for g, v in per_group.items():
        if not v:
            continue
        print(f"{g:<8}{len(v):>6}{np.mean(v):>14.4f}"
              f"{f'{min(v):.3f}-{max(v):.3f}':>16}{max(v)-min(v):>10.4f}")

    intra = [max(v) - min(v) for v in per_group.values() if len(v) > 1]
    if intra:
        noise = float(np.mean(intra))
        print(f"\n平均组内极差（噪声）: {noise:.4f}")
        h1 = np.mean(per_group["HDAC1"]) if per_group.get("HDAC1") else np.nan
        h2 = np.mean(per_group["HDAC2"]) if per_group.get("HDAC2") else np.nan
        if np.isfinite(h1) and np.isfinite(h2):
            sig = abs(h1 - h2)
            print(f"HDAC1 vs HDAC2 组间差（信号）: {sig:.4f}")
            print(f"信号/噪声 = {sig / max(noise, 1e-9):.2f}   "
                  f"({'可用' if sig > noise else '不可用：噪声大于信号'})")

    json.dump({k: v for k, v in res.items() if v},
              open(ROOT / "results/hdac_rotamer_intra_noise.json", "w"), indent=2)
    print("\n写出 results/hdac_rotamer_intra_noise.json")
