"""Extract the catalytic Fe from a heme (HEM) cofactor into a standalone FE3 ion.

metgeodock only recognises metal ions written as their own HETATM residue (FE/FE3),
but in CYP450 structures the iron sits inside the HEM residue. This converts
    HETATM ... FE   HEM A 508
into
    HETATM ... FE   FE3 A 508
and removes the rest of the porphyrin, which would otherwise act as a blocking
ligand inside the pocket.

Usage:
  heme_to_fe3.py --pdb 1TQN.pdb --out /tmp/xtal_train/1TQN_fe3.pdb
"""
from __future__ import annotations

import argparse
from pathlib import Path


def convert(src: Path, dest: Path, heme_resname: str = "HEM", keep_porphyrin: bool = False) -> dict:
    lines = src.read_text(errors="ignore").splitlines()
    out, n_fe, n_dropped = [], 0, 0
    fe_xyz = None
    for ln in lines:
        if ln.startswith("HETATM") and ln[17:20].strip().upper() == heme_resname:
            atom = ln[12:16].strip().upper()
            if atom.startswith("FE"):
                # rewrite residue name to FE3 so metgeodock treats it as a金属离子
                new = ln[:17] + "FE3" + ln[20:]
                out.append(new)
                n_fe += 1
                fe_xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            elif keep_porphyrin:
                out.append(ln)
            else:
                n_dropped += 1
        else:
            out.append(ln)
    out.append("END")
    dest.write_text("\n".join(out) + "\n")
    return {"n_fe": n_fe, "n_dropped": n_dropped, "fe_xyz": fe_xyz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heme", default="HEM")
    ap.add_argument("--keep-porphyrin", action="store_true")
    a = ap.parse_args()
    info = convert(a.pdb, a.out, a.heme, a.keep_porphyrin)
    status = "OK" if info["n_fe"] else "NO_FE_FOUND"
    print(f"{a.pdb.name}: {status} fe={info['n_fe']} porphyrin_atoms_dropped={info['n_dropped']}")


if __name__ == "__main__":
    main()
