"""Build a 6000-decoy-per-target library for a statistically meaningful EF curve."""
import csv
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
print(f"filtered {len(ok)} / {len(raw)}", flush=True)
need = PER * len(T)
if len(ok) < need:
    PER = len(ok) // len(T); need = PER * len(T)
    print(f"adjusted PER={PER}")
picked = ok[:need]
lib = ROOT/"docking/biglib"; lib.mkdir(parents=True, exist_ok=True)
meta = []
for i, t in enumerate(T):
    chunk = picked[i*PER:(i+1)*PER]
    writer = Chem.SDWriter(str(lib/f"{t}.sdf"))
    n = 0
    for j, (cid, smi) in enumerate(chunk):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        ps = AllChem.ETKDGv3(); ps.randomSeed = 0xf00d
        if AllChem.EmbedMolecule(mol, ps) != 0:
            ps.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, ps) != 0: continue
        try: AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception: pass
        mol = Chem.RemoveHs(mol)
        lid = f"{t}_BL{j:05d}"
        mol.SetProp("_Name", lid)
        writer.write(mol)
        meta.append([t, lid, cid, 0])
        n += 1
    writer.close()
    print(f"  {t}: {n} -> {t}.sdf", flush=True)
with (ROOT/"results/biglib/decoys.csv").open("w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["target","ligand_id","chembl_id","label"]); w.writerows(meta)
print(f"total decoys: {len(meta)}")
