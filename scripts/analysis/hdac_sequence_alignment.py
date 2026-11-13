"""Sequence alignment across the six HDAC catalytic domains, to map residue ids.

Different PDB entries number their residues differently (HDAC1 Asp176 vs HDAC3
Asp170), so residue-keyed comparisons between subtypes return an empty intersection.
This builds an explicit correspondence from a pairwise alignment of the extracted
sequences, which every cross-subtype analysis needs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from Bio import Align

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
      "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
      "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
      "TYR": "Y", "VAL": "V"}


def chain_of_metal(pid):
    """chain that carries the catalytic zinc"""
    # read the PREPARED receptor, so chain ids match what analyses use
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if ln.startswith("HETATM") and ln[17:20].strip().upper().startswith("ZN"):
            return ln[21]
    return None


def sequence(pid, chain=None):
    seen, order = {}, []
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith("ATOM"):
            continue
        rn = ln[17:20].strip().upper()
        if rn not in A3:
            continue
        if chain is not None and ln[21] != chain:
            continue
        key = (ln[21], ln[22:26].strip())
        if key not in seen:
            seen[key] = rn
            order.append(key)
    order.sort(key=lambda k: (k[0], int(k[1]) if k[1].lstrip('-').isdigit() else 0))
    return "".join(A3[seen[k]] for k in order), order, seen


def main():
    data = {}
    for name, pid in PDB.items():
        ch = chain_of_metal(pid)
        seq, keys, resn = sequence(pid, chain=ch)
        data[name] = {"pdb": pid, "chain": ch, "seq": seq, "keys": keys, "resn": resn}
        print(f"{name:<8}{pid:<7} 链={ch}  残基 {len(seq)}  "
              f"范围 {keys[0][1]}-{keys[-1][1]}" if keys else f"{name}: 空序列")

    ref = "HDAC1"
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -1

    out = {"ref": ref, "pairs": {}}
    print(f"\n以 {ref} 为参照的序列比对")
    print(f"{'亚型':<8}{'比对得分':>9}{'一致度':>8}{'对齐残基':>9}{'对应关系'}")
    for name in PDB:
        if name == ref:
            continue
        A, B = data[ref], data[name]
        al = aligner.align(A["seq"], B["seq"])[0]
        # walk the alignment to build key -> key
        mapping = {}
        ai = bi = 0
        ident = 0
        aligned = 0
        for (a0, a1), (b0, b1) in zip(al.aligned[0], al.aligned[1]):
            for k in range(a1 - a0):
                ka, kb = A["keys"][a0 + k], B["keys"][b0 + k]
                mapping[ka] = kb
                aligned += 1
                if A["resn"][ka] == B["resn"][kb]:
                    ident += 1
        out["pairs"][name] = {"score": float(al.score), "n_aligned": aligned,
                              "identity": round(ident / max(aligned, 1), 3),
                              "mapping": {f"{k[0]}{k[1]}": f"{v[0]}{v[1]}"
                                          for k, v in mapping.items()}}
        print(f"{name:<8}{al.score:>9.0f}{ident/max(aligned,1):>8.1%}{aligned:>9}"
              f"  {len(mapping)} 对")

    print("\n对照文献已知的对应关系（用于验证比对正确性）")
    checks = [("HDAC2", "Series 263"), ("HDAC3", "Tyr107 vs Ser113/114")]
    # HDAC1 S263 -> HDAC2 A264
    m = out["pairs"]["HDAC2"]["mapping"]
    k = f"{data['HDAC1']['keys'][0][0]}263"
    print(f"  HDAC1 263 ({data['HDAC1']['resn'].get(('B','263'), '?')})  ->  "
          f"HDAC2 {m.get('B263', m.get('A263', '未映射'))}")
    for kk, v in data["HDAC1"]["resn"].items():
        if kk[1] == "263":
            print(f"    实际 HDAC1 263 = {v}, 链 {kk[0]}")
    if "HDAC3" in out["pairs"]:
        m3 = out["pairs"]["HDAC3"]["mapping"]
        for cand in ("B113", "B114", "A113", "A114"):
            if cand in m3:
                print(f"  HDAC1 {cand} -> HDAC3 {m3[cand]}")

    json.dump(out, open(ROOT / "results/hdac_sequence_alignment.json", "w"), indent=2)
    print("\n写出 results/hdac_sequence_alignment.json")


if __name__ == "__main__":
    main()
