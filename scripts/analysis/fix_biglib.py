"""Recreate the decoy index and generate any missing per-target 3D libraries."""
import csv, signal, sys
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
LFC = rdMolStandardize.LargestFragmentChooser()
METALS = {"Li","Na","K","Mg","Ca","Mn","Fe","Co","Ni","Cu","Zn","Cd","Hg","Pt","Au"}
T = ["HDAC1","HDAC2","HDAC3","HDAC4","HDAC6","HDAC7","HDAC8"]
PER = 6000


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def filtered():
    raw = list(csv.DictReader(open(ROOT/"results/biglib/raw_molecules.csv")))
    ok, seen = [], set()
    for r in raw:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None: continue
        m = LFC.choose(m)
        if m is None or m.GetNumAtoms() == 0: continue
        if any(a.GetSymbol() in METALS for a in m.GetAtoms()): continue
        if not (10 <= m.GetNumHeavyAtoms() <= 50): continue
        if not (150 <= Descriptors.MolWt(m) <= 700): continue
        smi = Chem.MolToSmiles(m)
        if smi in seen: continue
        seen.add(smi); ok.append((r["chembl_id"], smi))
    return ok


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    ok = filtered()
    print(f"filtered {len(ok)}", flush=True)
    lib = ROOT/"docking/biglib"
    meta = []
    for i, t in enumerate(T):
        chunk = ok[i*PER:(i+1)*PER]
        dest = lib/f"{t}.sdf"
        if dest.exists() and dest.stat().st_size > 1e6:
            ids = []
            for m in Chem.SDMolSupplier(str(dest), removeHs=False, sanitize=False):
                if m is not None and m.HasProp("_Name"):
                    ids.append(m.GetProp("_Name"))
            for lid in ids:
                meta.append([t, lid, "", 0])
            print(f"  {t}: existing {len(ids)}", flush=True)
            continue
        if only != "ALL" and only != t:
            print(f"  {t}: skipped", flush=True)
            continue
        writer = Chem.SDWriter(str(dest))
        n = skip = 0
        for j, (cid, smi) in enumerate(chunk):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: skip += 1; continue
            mol = Chem.AddHs(mol)
            ps = AllChem.ETKDGv3(); ps.randomSeed = 0xf00d
            got = False
            for _ in range(2):
                try:
                    signal.alarm(3)
                    if AllChem.EmbedMolecule(mol, ps) == 0:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=150)
                        got = True
                    signal.alarm(0)
                    if got: break
                except Timeout:
                    signal.alarm(0); ps.useRandomCoords = True
                except Exception:
                    signal.alarm(0); break
            if not got: skip += 1; continue
            mol = Chem.RemoveHs(mol)
            lid = f"{t}_BL{j:05d}"
            mol.SetProp("_Name", lid)
            writer.write(mol)
            meta.append([t, lid, cid, 0])
            n += 1
        writer.close()
        print(f"  {t}: wrote {n}, skipped {skip}", flush=True)
    out = ROOT/"results/biglib/decoys.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["target","ligand_id","chembl_id","label"]); w.writerows(meta)
    print(f"WROTE {out.name} n={len(meta)}")


if __name__ == "__main__":
    main()
