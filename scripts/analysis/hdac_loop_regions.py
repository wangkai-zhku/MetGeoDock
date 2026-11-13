"""Loop-based subtype descriptors: L1, L2, L6.

Literature basis (HDAC8 numbering):
  L1 loop   Ser30-Lys36   mobile lid; Lys33 forms a salt bridge with the L2
                           triple-Asp tract (Asp87-89) that locks the binding rail
                           "out"; in HDAC1/2/3/6 this loop is longer and carries a
                           Pro/Ile tip that packs over L6 -- the "L1-L6 lock" that
                           sterically blocks L-shaped (HDAC8-selective) inhibitors
  L2 loop   Pro91-Thr105  contains the "binding rail" Tyr100/Asp101 that H-bonds
                           the substrate backbone; HDAC6 carries a unique Ser here
                           (S568 human CD2) where other isoforms have Asp
  L6 loop   around Met274  forms one wall of the HDAC8-selective pocket together
                           with the catalytic Tyr306; Met274 is proposed as the
                           switch controlling the transient L1-L6 sub-pocket

This script extracts those three regions from every isoform via sequence
alignment and summarises length and key-residue differences.
"""
from __future__ import annotations
import json
from pathlib import Path
from Bio import Align

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def chain_of_metal(pid):
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if ln.startswith("HETATM") and ln[17:20].strip().upper().startswith("ZN"):
            return ln[21]
    return None


def seq_of(pid):
    ch = chain_of_metal(pid)
    seen, order = {}, []
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith("ATOM"):
            continue
        rn = ln[17:20].strip().upper()
        if rn not in A3:
            continue
        if ch is not None and ln[21] != ch:
            continue
        k = int(ln[22:26]) if ln[22:26].strip().lstrip("-").isdigit() else 0
        if k not in seen:
            seen[k] = rn
            order.append(k)
    order.sort()
    return "".join(A3[seen[k]] for k in order), order, seen


def main():
    data = {}
    for name, pid in PDB.items():
        seq, nums, resn = seq_of(pid)
        data[name] = {"seq": seq, "nums": nums, "resn": resn, "pdb": pid}
        print(f"{name:<8}{pid:<7} {len(seq)} 残基  编号 {nums[0]}-{nums[-1]}")

    ref = "HDAC8"
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -1

    # HDAC8 numbering -> index in its own sequence
    h8 = data[ref]
    idx_of = {n: i for i, n in enumerate(h8["nums"])}
    LOOPS = {"L1": (30, 36), "L2": (87, 105), "L6": (268, 280), "catalytic_Y": (306, 306)}

    print("\n以 HDAC8 编号定义的 loop 区域")
    for k, (a, b) in LOOPS.items():
        print(f"  {k:<12}{a}-{b}   HDAC8 序列: "
              f"{''.join(A3.get(h8['resn'].get(n, ''), '-') for n in range(a, b + 1) if n in h8['resn'])}")

    print("\n各亚型在这些区域的对应序列（通过 pairwise 比对映射）")
    out = {}
    for name in PDB:
        if name == ref:
            continue
        al = aligner.align(h8["seq"], data[name]["seq"])[0]
        # ref index -> other index
        m = {}
        for (a0, a1), (b0, b1) in zip(al.aligned[0], al.aligned[1]):
            for k in range(a1 - a0):
                m[a0 + k] = b0 + k
        rec = {}
        for loop, (a, b) in LOOPS.items():
            chars = []
            missing = 0
            for n in range(a, b + 1):
                i = idx_of.get(n)
                if i is None:
                    missing += 1
                    continue
                j = m.get(i)
                chars.append(data[name]["seq"][j] if j is not None and j < len(data[name]["seq"]) else "-")
            rec[loop] = "".join(chars)
        out[name] = rec
        print(f"\n  {name}:")
        for loop in LOOPS:
            print(f"    {loop:<12}{out[name][loop]}")

    # structural interpretation
    print("\n关键判据")
    print(f"{'亚型':<8}{'L1 长度':>8}{'L1 有 Pro/Ile':>14}{'L2 关键位':>10}{'L6 长度':>8}")
    for name in PDB:
        rec = out.get(name) or {"L1": "", "L2": "", "L6": ""}
        l1 = rec["L1"]
        has_pi = ("P" in l1) or ("I" in l1)
        l2key = rec["L2"][-6:] if rec["L2"] else ""
        print(f"{name:<8}{len(l1):>8}{'是' if has_pi else '否':>14}{l2key:>10}{len(rec['L6']):>8}")

    json.dump({"loops": LOOPS, "sequences": out,
               "hdac8_reference": {k: v for k, v in h8["resn"].items()
                                   if any(a <= k <= b for a, b in LOOPS.values())}},
              open(ROOT / "results/hdac_loop_regions.json", "w"), indent=2)
    print("\n写出 results/hdac_loop_regions.json")


if __name__ == "__main__":
    main()
