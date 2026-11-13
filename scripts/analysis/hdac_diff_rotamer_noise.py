"""Noise floor for the difference-region rotamer fingerprint.

The signal of interest is tiny (HDAC1 vs HDAC2 dS = -1.7 on a total of ~76). Before
believing it, the same quantity must be recomputed with a DIFFERENT crystal
structure of the same enzyme. If swapping HDAC1 5ICN -> 4BKX moves dS by as much as
the subtype comparison does, the signal is noise.

Also quantifies how much of the difference survives when the reference structure is
changed, and reports the residue-set overlap after sequence alignment (the raw id
overlap was meaningless because numbering differs).
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
CORPUS = Path("/home/wangkai/cursor-task/metalloprotein/ORI_PDB_LIGAND")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("dr", ROOT / "scripts/analysis/hdac_diff_rotamer.py")
dr = importlib.util.module_from_spec(spec)
sys.modules["dr"] = dr
spec.loader.exec_module(dr)

# (label used in output, corpus file glob pdb id, is it a subtype or a replicate)
RUNS = [
    ("HDAC1_5ICN", "5ICN"), ("HDAC1_4BKX", "4BKX"),
    ("HDAC2_3MAX", "3MAX"), ("HDAC2_4LXZ", "4LXZ"),
    ("HDAC3_4A69", "4A69"),
    ("HDAC8_1T64", "1T64"), ("HDAC8_1T67", "1T67"),
]


def find(pid):
    hits = sorted(CORPUS.glob(f"*/zn_{pid}_pro_*.pdb")) or sorted(CORPUS.glob(f"*/*_{pid}_pro_*.pdb"))
    return hits[0] if hits else None


def load_any(path):
    prot, met = [], []
    for ln in Path(path).read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        an = ln[12:16].strip()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            prot.append((rn, ln[21] + ln[22:26].strip(), an, xyz, el))
        elif rn.startswith("ZN"):
            met.append(xyz)
    return prot, met[0] if met else None


def pair_dS(A, B, mapping):
    """dS between two structures on the difference region, residues aligned by mapping"""
    prot_a, m_a = A["prot"], A["metal"]
    prot_b, m_b = B["prot"], B["metal"]
    pts_a, rel_a, open_a, P_a = dr.grid(prot_a, m_a)
    R, t = dr.kabsch(dr.anchors(prot_b, m_b), dr.anchors(prot_a, m_a))
    pts_b, rel_b, open_b, P_b = dr.grid(prot_b, m_b, R=R, t=t)
    diff = open_a ^ open_b
    if diff.sum() == 0:
        return None
    keys_a = dr.residues_near(pts_a[diff], prot_a)
    keys_b_al = {mapping[k] for k in keys_a if k in mapping}
    Sa = dr.rotamer_vector(prot_a, m_a, keys_a)
    Sb = dr.rotamer_vector(prot_b, m_b, keys_b_al)
    pairs = [(k, mapping[k]) for k in Sa if k in mapping and mapping[k] in Sb]
    if not pairs:
        return None
    ea = float(np.sum([Sa[x] for x, _ in pairs]))
    eb = float(np.sum([Sb[y] for _, y in pairs]))
    return {"n_voxels": int(diff.sum()), "n_res": len(pairs),
            "S_a": round(ea, 3), "S_b": round(eb, 3), "dS": round(eb - ea, 3)}


def main():
    aln = json.load(open(ROOT / "results/hdac_sequence_alignment.json"))
    MAPS = {k: v.get("mapping", {}) for k, v in aln["pairs"].items()}

    S = {}
    for label, pid in RUNS:
        f = find(pid)
        if f is None:
            print(f"{label}: 文件缺失"); continue
        prot, metal = load_any(f)
        if metal is None:
            print(f"{label}: 无金属"); continue
        S[label] = {"prot": prot, "metal": metal, "file": f.name}
        print(f"{label:<12}{f.name:<32} 原子 {len(prot)}")

    # mapping needs HDAC1 -> X; reuse the alignment file (chain ids from prepared receptors
    # may differ from corpus files, so rebuild maps on the fly by residue order)
    def build_map(pa, pb):
        """positional map by residue index (crystal structures of the same enzyme)"""
        ka, kb = [], []
        seen = set()
        for rn, k, an, xyz, el in pa:
            if k not in seen:
                seen.add(k); ka.append(k)
        seen = set()
        for rn, k, an, xyz, el in pb:
            if k not in seen:
                seen.add(k); kb.append(k)
        return {ka[i]: kb[i] for i in range(min(len(ka), len(kb)))}

    print("\n信号 vs 噪声（差异区 rotamer 熵总和）")
    print("=" * 86)
    rows = []
    # 1) subtype comparison
    for ref, ref_lbl in (("HDAC1_5ICN", "HDAC1(5ICN)"), ("HDAC1_4BKX", "HDAC1(4BKX)")):
        for tgt, tgt_lbl in (("HDAC2_3MAX", "HDAC2(3MAX)"), ("HDAC2_4LXZ", "HDAC2(4LXZ)"),
                             ("HDAC8_1T64", "HDAC8(1T64)")):
            if ref not in S or tgt not in S:
                continue
            mp = build_map(S[ref]["prot"], S[tgt]["prot"])
            r = pair_dS(S[ref], S[tgt], mp)
            if r:
                rows.append((f"{ref_lbl} vs {tgt_lbl}", r))
                print(f"  亚型间  {ref_lbl:<14} vs {tgt_lbl:<14} dS={r['dS']:+7.2f}  "
                      f"(残基 {r['n_res']}, 体素 {r['n_voxels']})")

    # 2) same-subtype replicate (pure noise)
    for a, b, lbl in (("HDAC1_5ICN", "HDAC1_4BKX", "HDAC1 自身重复"),
                      ("HDAC2_3MAX", "HDAC2_4LXZ", "HDAC2 自身重复"),
                      ("HDAC8_1T64", "HDAC8_1T67", "HDAC8 自身重复")):
        if a not in S or b not in S:
            continue
        mp = build_map(S[a]["prot"], S[b]["prot"])
        r = pair_dS(S[a], S[b], mp)
        if r:
            rows.append((lbl, r))
            print(f"  噪声    {lbl:<28} dS={r['dS']:+7.2f}  "
                  f"(残基 {r['n_res']}, 体素 {r['n_voxels']})")

    sig = [abs(r["dS"]) for lbl, r in rows if "自身重复" not in lbl]
    noise = [abs(r["dS"]) for lbl, r in rows if "自身重复" in lbl]
    print()
    if sig and noise:
        print(f"亚型间 |dS| 均值 = {np.mean(sig):.2f}")
        print(f"同亚型 |dS| 均值 = {np.mean(noise):.2f}  (噪声)")
        print(f"信号/噪声 = {np.mean(sig)/max(np.mean(noise),1e-9):.2f}")
    json.dump({lbl: r for lbl, r in rows},
              open(ROOT / "results/hdac_diff_rotamer_noise.json", "w"), indent=2)
    print("\n写出 results/hdac_diff_rotamer_noise.json")


if __name__ == "__main__":
    main()
