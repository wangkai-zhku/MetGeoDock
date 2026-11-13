"""Generate 3D for one target's decoys with a per-molecule timeout (ETKDG can hang)."""
import csv, signal, sys
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def main(target: str):
    lib = ROOT / "docking/biglib"
    lib.mkdir(parents=True, exist_ok=True)
    raw = {r["chembl_id"]: r["smiles"] for r in csv.DictReader(open(ROOT / "results/biglib/raw_molecules.csv"))}
    rows = [r for r in csv.DictReader(open(ROOT / "results/biglib/decoys.csv")) if r["target"] == target]
    writer = Chem.SDWriter(str(lib / f"{target}.sdf"))
    n_ok = n_skip = 0
    for r in rows:
        smi = raw.get(r["chembl_id"])
        if not smi:
            n_skip += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skip += 1
            continue
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xf00d
        ok = False
        for attempt in range(2):
            try:
                signal.alarm(3)
                if AllChem.EmbedMolecule(mol, ps) == 0:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=150)
                    ok = True
                signal.alarm(0)
                if ok:
                    break
            except Timeout:
                signal.alarm(0)
                ps.useRandomCoords = True
            except Exception:
                signal.alarm(0)
                break
        if not ok:
            n_skip += 1
            continue
        mol = Chem.RemoveHs(mol)
        mol.SetProp("_Name", r["ligand_id"])
        writer.write(mol)
        n_ok += 1
    writer.close()
    print(f"{target}: wrote {n_ok}, skipped {n_skip}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
