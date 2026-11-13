"""Head-to-head: AutoDock Vina on the same actives and decoys as MetalGeoDock.

The point is a like-for-like comparison on identical ligands with identical labels, so the
enrichment numbers can be put side by side. Vina treats the metal as a plain atom, which is
exactly the baseline MetalGeoDock claims to improve on.

Receptors are converted to PDBQT with Open Babel; the box is centred on the catalytic metal.
Ligands come from the same ChEMBL actives and the same decoy pool used for the MetalGeoDock runs.

Usage:
  python scripts/analysis/vina_baseline.py --target CA2 --protein /tmp/xtal_train/3CAJ_h.pdb \
      --n-act 30 --n-dec 200 --parts 4 --part 0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from _embed import embed_3d  # noqa: E402


def metal_xyz(pdb: Path, metal: str | None = None) -> tuple[float, float, float] | None:
    """Coordinate of the *catalytic* metal, chosen the same way MetalGeoDock chooses it.

    Taking the first metal in file order is wrong for multi-metal structures: MMP2 has 2 Zn and
    3 Ca, and the first Zn need not be the catalytic one. Using detect_coordination keeps the
    comparison fair - all three methods must be centred on the same site.
    """
    if metal:
        try:
            from metgeodock.dock import detect_coordination
            from metgeodock.io import read_pdb
            prot = read_pdb(pdb)
            sphere = detect_coordination(prot, metal=metal)
            xyz = sphere.metal_xyz
            return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except Exception:
            pass
    for ln in pdb.read_text(errors="ignore").splitlines():
        if ln.startswith(("HETATM", "ATOM")) and ln[17:20].strip().upper() in {
                "ZN", "FE", "FE2", "FE3", "MN", "MN2", "MG", "CA", "CO", "CU", "NI", "CD"}:
            return (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return None


def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float(1.0 - (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ef(y, s, frac):
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n, na = len(y), int(y.sum())
    if na == 0:
        return np.nan
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(r)[:k]].sum() / k) / (na / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, required=True)
    ap.add_argument("--n-act", type=int, default=30)
    ap.add_argument("--n-dec", type=int, default=200)
    ap.add_argument("--metal", default=None,
                    help="metal element (zn/ca/fe2/...) so the catalytic site is chosen as in MetalGeoDock")
    ap.add_argument("--cpu", type=int, default=1,
                    help="threads per vina process; default 1 because we run many in parallel")
    ap.add_argument("--box", type=float, default=22.0)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--common-set", type=Path, default=None,
                    help="use this fixed ligand set (results/common_set/{target}.csv) so every "
                         "method is scored on identical molecules")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/vina_baseline")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")
    import random

    xyz = metal_xyz(a.protein, getattr(a, 'metal', None))
    if xyz is None:
        print("no metal found in receptor"); return
    cx, cy, cz = xyz

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rec = td / "rec.pdbqt"
        # Strip crystallographic waters before conversion: obabel writes every
        # HETATM O into the pdbqt as an OA acceptor, so retained pocket waters
        # occupy space and reject ligands. Metal ions are kept. MetalGeoDock
        # already docks against a water-free receptor (dock_ligand(keep_water
        # =False) calls strip_waters), so both methods now see the same site.
        _water = {"HOH", "WAT", "H2O", "DOD"}
        dry = td / "rec_dry.pdb"
        # 只保留结构记录行：MMP1_cat / MMP9_cat / HIV1_cat 三个受体文件每个原子行后面
        # 多了一个空行，obabel 会判定 "not a valid PDB file" 并输出 0 字节 pdbqt，
        # 之后 vina 对着空受体仍返回 rc=0、打出 "1 0.0"，导致整靶点亲和值全是 0.0。
        kept = [ln for ln in Path(a.protein).read_text(errors="ignore").splitlines()
                if ln.startswith(("ATOM", "HETATM"))
                and not (ln.startswith("HETATM")
                         and ln[17:20].strip().upper() in _water)]
        dry.write_text("\n".join(kept) + "\nEND\n")
        subprocess.run(["obabel", str(dry), "-O", str(rec), "-xr"],
                       capture_output=True, timeout=300)
        if not rec.exists():
            print("receptor conversion failed"); return

        # ligands: fixed common set when provided, so MetalGeoDock / Vina / MetalloDock
        # are all scored on identical molecules
        cs = a.common_set or (ROOT / f"results/common_set/{a.target}.csv")
        if Path(cs).exists():
            rows = list(csv.DictReader(open(cs)))
            jobs = [(int(r["label"]), r["smiles"], r["ligand_id"]) for r in rows]
            print(f"using common set {cs} ({sum(j[0] for j in jobs)} actives + "
                  f"{sum(1 for j in jobs if j[0]==0)} decoys)", flush=True)
        else:
            table = list(csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")))
            acts = [r for r in table if r.get("target_name") == a.target and r.get("canonical_smiles")
                    and r.get("pactivity") and float(r["pactivity"]) >= 6][: a.n_act]
            pool = [p for p in csv.DictReader(open(ROOT / "results/decoy_pool_sized.csv"))
                    if p["target"] != a.target]
            random.seed(20241024)
            decs = random.sample(pool, min(a.n_dec, len(pool)))
            jobs = [(1, r["canonical_smiles"], r["molecule_chembl_id"]) for r in acts]
            jobs += [(0, p["smiles"], f"{a.target}_VD{i:05d}") for i, p in enumerate(decs)]
        if a.parts > 1:
            jobs = [j for i, j in enumerate(jobs) if i % a.parts == a.part]

        dest = a.out_dir / f"{a.target}_p{a.part}.csv"
        print(f"{a.target} part {a.part}/{a.parts}: {len(jobs)} ligands, box {a.box} A at metal", flush=True)
        with dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ligand_id", "label", "affinity", "error"])
            w.writeheader()
            for k, (label, smi, lid) in enumerate(jobs, 1):
                rec_row = {"ligand_id": lid, "label": label}
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        raise ValueError("parse")
                    mol = embed_3d(mol, keep_h=True)
                    if mol is None:
                        raise ValueError("embed")
                    lig_pdb = td / "lig.pdb"
                    Chem.MolToPDBFile(mol, str(lig_pdb))
                    lig_pdbqt = td / "lig.pdbqt"
                    subprocess.run(["obabel", str(lig_pdb), "-O", str(lig_pdbqt), "-xh"],
                                   capture_output=True, timeout=120)
                    out_pdbqt = td / "out.pdbqt"
                    cmd = ["vina", "--receptor", str(rec), "--ligand", str(lig_pdbqt),
                           "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
                           "--size_x", f"{a.box}", "--size_y", f"{a.box}", "--size_z", f"{a.box}",
                           "--exhaustiveness", str(a.exhaustiveness), "--num_modes", "1",
                           "--cpu", str(a.cpu), "--out", str(out_pdbqt)]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    aff = None
                    for line in r.stdout.splitlines():
                        if line.strip().startswith("1 "):
                            aff = float(line.split()[1])
                            break
                    if aff is None:
                        raise ValueError("no score")
                    rec_row["affinity"] = aff
                except Exception as exc:
                    rec_row["error"] = type(exc).__name__
                w.writerow(rec_row)
                fh.flush()   # 不 flush 的话进程被中断会整批丢失（此前 3 个 0 字节结果就是这么来的）
                if k % 20 == 0:
                    print(f"  {a.target}/{a.part}: {k}/{len(jobs)}", flush=True)
        print(f"DONE {dest}", flush=True)


if __name__ == "__main__":
    main()
