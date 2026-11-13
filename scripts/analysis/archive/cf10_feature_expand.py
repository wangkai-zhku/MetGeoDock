"""Systematic feature expansion for the cross-family benchmark.

Every feature gets a direction fixed by physics/chemistry, NOT by the activity
labels:
  * energy-like terms (total, steric, g_*)      lower is better
  * contact / burial / hydrophobic contact      higher is better
  * penalties (n_rot, rim_overlap, TPSA)        lower is better
No weight is fitted to labels anywhere. Combinations use equal-weight z-scores,
which is a prior (all terms independent, unit importance), not a fit.

Read-only: results/cf10_thermo/*.csv + results/crossfamily10/*_div.csv.
"""
from __future__ import annotations
import csv, glob
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from scipy.stats import rankdata

RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
TARGETS = ["MMP2", "CAPN1", "CA2", "METAP2", "CDK2", "urease"]

# ---------------------------------------------------------------- SMARTS
SMARTS = {
    "carboxylate":  "[CX3](=O)[OX1-,OX2H1]",
    "hydroxamate":  "[NX3][OH1]",
    "sulfonamide":  "S(=O)(=O)[NX3]",
    "imidazole":    "c1cnc[nH]1",
    "pyridine":     "c1ccccn1",
    "thiol":        "[SX2H]",
    "phosphate":    "[PX4](=O)([OX2])[OX2]",
    "catechol":     "c1cc([OX2H])c([OX2H])cc1",
    "phenol":       "c[OX2H]",
    "hydroxyl":     "[OX2H]",
    "amine":        "[NX3;!$(N=*)]",
    "amide":        "[NX3][CX3](=[OX1])",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
DONOR = [Chem.MolFromSmarts(s) for s in ("[$([OX1-,OX2H1,OX2])]", "[$([NX2,NX3;!$(N=O)])]", "[$([SX2,SX1])]")]


def ligand_features(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {}
    f = {}
    f["molwt"] = Descriptors.MolWt(m)
    f["nheavy"] = m.GetNumHeavyAtoms()
    f["nrotb"] = rdMolDescriptors.CalcNumRotatableBonds(m)
    f["tpsa"] = Descriptors.TPSA(m)
    f["logp"] = Crippen.MolLogP(m)
    f["nhbd"] = rdMolDescriptors.CalcNumHBD(m)
    f["nhba"] = rdMolDescriptors.CalcNumHBA(m)
    f["nhetero"] = rdMolDescriptors.CalcNumHeteroatoms(m)
    f["nring"] = rdMolDescriptors.CalcNumRings(m)
    f["narom"] = rdMolDescriptors.CalcNumAromaticRings(m)
    f["naliph"] = rdMolDescriptors.CalcNumAliphaticRings(m)
    f["fsp3"] = rdMolDescriptors.CalcFractionCSP3(m)
    f["nN"] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
    f["nO"] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O")
    f["nS"] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "S")
    f["nhal"] = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    f["polar_frac"] = (f["nN"] + f["nO"]) / max(f["nheavy"], 1)
    f["hydroph_frac"] = sum(1 for a in m.GetAtoms()
                            if a.GetSymbol() in ("C", "F", "Cl", "Br", "I")) / max(f["nheavy"], 1)

    # metal-facing chemistry
    ndon = 0
    for p in DONOR:
        ndon += len(m.GetSubstructMatches(p))
    f["n_donor"] = ndon
    for k, p in PATS.items():
        f["z_" + k] = len(m.GetSubstructMatches(p)) if p is not None else 0

    # chelation capability: donor pairs at topological distance 2-3 (5-/6-membered ring)
    dm = Chem.GetDistanceMatrix(m)
    donors = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() in ("N", "O", "S")]
    pair5 = pair6 = 0
    for i in range(len(donors)):
        for j in range(i + 1, len(donors)):
            d = dm[donors[i], donors[j]]
            if d == 2:
                pair5 += 1
            elif d == 3:
                pair6 += 1
    f["chelate5"] = pair5
    f["chelate6"] = pair6
    f["chelate_tot"] = pair5 + pair6

    # Gasteiger charge spread
    try:
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(m)
        q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()])
        q = np.nan_to_num(q)
        f["q_maxpos"] = float(q.max())
        f["q_maxneg"] = float(-q.min())
        f["q_abs"] = float(np.abs(q).sum())
    except Exception:
        f["q_maxpos"] = f["q_maxneg"] = f["q_abs"] = np.nan
    return f


# CSV terms and their physics direction (+1 = higher is better)
CSV_TERMS = {
    "total": -1, "coord": -1, "geo": -1, "steric": -1, "contact": +1,
    "map_overlap": +1, "n_hbond": +1, "n_hydrophobic": +1, "buried_area": +1,
    "rim_overlap": -1, "g_buried": -1, "g_hbond": -1, "g_desolv": -1,
    "g_conf": -1, "g_total": -1, "n_rotatable": -1,
}
# derived (density) terms: same direction as their numerator
DERIVED = {
    "d_total_per_heavy": ("total", -1), "d_steric_per_heavy": ("steric", -1),
    "d_contact_per_heavy": ("contact", +1), "d_hbond_per_heavy": ("n_hbond", +1),
    "d_hydroph_per_heavy": ("n_hydrophobic", +1), "d_buried_per_heavy": ("buried_area", +1),
    "d_map_per_heavy": ("map_overlap", +1), "d_gburied_per_heavy": ("g_buried", -1),
    "d_buried_frac": ("buried_area_frac", +1),
}
LIG_DIR = {"molwt": -1, "nheavy": -1, "nrotb": -1, "tpsa": -1, "logp": +1,
           "nhbd": -1, "nhba": -1, "nhetero": -1, "nring": +1, "narom": +1,
           "naliph": +1, "fsp3": +1, "nN": -1, "nO": -1, "nS": +1, "nhal": +1,
           "polar_frac": -1, "hydroph_frac": +1, "n_donor": +1,
           "chelate5": +1, "chelate6": +1, "chelate_tot": +1,
           "q_maxpos": -1, "q_maxneg": -1, "q_abs": -1}
for k in ("carboxylate", "hydroxamate", "sulfonamide", "imidazole", "pyridine",
          "thiol", "phosphate", "catechol", "phenol", "hydroxyl", "amine", "amide"):
    LIG_DIR["z_" + k] = +1


def auroc(y, s):
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(1.0 - (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_target(t):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{t}_p*.csv"))):
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames:
                continue
            nf = len(rd.fieldnames)
            for r in rd:
                if None in r or len(r) != nf or (r.get("g_total") or "").strip() in ("", "nan"):
                    continue
                rows.append(r)
    smi = {r["ligand_id"]: r["smiles"]
           for r in csv.DictReader(open(ROOT / f"results/crossfamily10/{t}_div.csv"))}
    out, cache = [], {}
    for r in rows:
        s = smi.get(r["ligand_id"])
        if s is None:
            continue
        if s not in cache:
            cache[s] = ligand_features(s)
        lf = cache[s]
        if not lf:
            continue
        rec = {"y": int(r["label"]), "id": r["ligand_id"]}
        for k in CSV_TERMS:
            v = (r.get(k) or "").strip()
            rec[k] = float(v) if v else np.nan
        rec.update(lf)
        out.append(rec)
    return out


def build():
    data = {}
    for t in TARGETS:
        d = load_target(t)
        if len(d) > 100:
            data[t] = d
    # derived
    for t, d in data.items():
        for r in d:
            h = max(r["nheavy"], 1)
            for name, (src, _) in DERIVED.items():
                if src == "buried_area_frac":
                    r[name] = r["buried_area"] / (r["buried_area"] + r["contact"] * 10.0 + 1e-9)
                else:
                    r[name] = r[src] / h
    return data


if __name__ == "__main__":
    data = build()
    dirs = {}
    dirs.update(CSV_TERMS)
    dirs.update({k: v[1] for k, v in DERIVED.items()})
    dirs.update(LIG_DIR)

    results = []
    print(f"{'feature':<24}" + "".join(f"{t[:7]:>9}" for t in data) + f"{'均值':>8}{'一致':>6}{'方向':>5}")
    for k in dirs:
        vals = []
        for t, d in data.items():
            v = np.array([r.get(k, np.nan) for r in d])
            ok = np.isfinite(v)
            if ok.sum() < 100:
                vals.append(np.nan); continue
            y = np.array([d[i]["y"] for i in np.where(ok)[0]])
            if len(set(y.tolist())) < 2 or np.allclose(v[ok], v[ok][0]):
                vals.append(np.nan); continue
            sig = dirs[k]
            # auroc() expects lower-is-better, dirs[k] says higher-is-better for +1
            vals.append(auroc(y, -sig * v[ok]))
        arr = np.array(vals)
        if not np.isfinite(arr).any():
            continue
        n_ok = int(np.nansum(arr > 0.5))
        results.append((k, arr, n_ok))

    results.sort(key=lambda x: -np.nanmean(x[1]))
    for k, arr, n_ok in results:
        print(f"{k:<24}" + "".join(f"{x:>9.3f}" if np.isfinite(x) else f"{'-':>9}" for x in arr)
              + f"{np.nanmean(arr):>8.3f}{n_ok:>6}{'+' if dirs[k] > 0 else '-':>5}")

    np.save("/tmp/cf10_feat_results.npy",
            np.array([(k, np.nanmean(a), n) for k, a, n in results], dtype=object),
            allow_pickle=True)
    print(f"\n候选特征数: {len(results)}")
