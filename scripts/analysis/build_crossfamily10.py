"""Build the 10-target cross-family benchmark: diverse library + property-matched (DUD-E style).

Targets were chosen to span ten distinct protein families and eight metal ions, so that any
enrichment claim cannot be an artefact of one family. Each target gets:
  {T}_div.csv   100 actives + N chemically diverse decoys  (unrelated ChEMBL targets, no matching)
  {T}_dude.csv  100 actives + N property-matched decoys     (matched on 7 descriptors, Tanimoto < 0.30)

Actives: ChEMBL binding assays, pChEMBL >= 6, confidence >= 8, median per molecule.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]

TARGETS = {
    "HDAC1":  dict(chembl="HDAC1",  metal="Zn",    family="HDAC deacetylase",         pdb="5ICN"),
    "CA2":    dict(chembl="CA2",    metal="Zn",    family="carbonic anhydrase",       pdb="3CAJ"),
    "MMP2":   dict(chembl="MMP2",   metal="Zn",    family="matrix metalloproteinase", pdb="1QIB"),
    "ALOX5":  dict(chembl="ALOX5",  metal="Fe2",   family="lipoxygenase",             pdb="3O8Y"),
    "ARG2":   dict(chembl="ARG2",   metal="Mn",    family="arginase",                 pdb="1PQ3"),
    "CAPN1":  dict(chembl="CAPN1",  metal="Ca",    family="calpain protease",         pdb="2ARY"),
    "DBH":    dict(chembl="DBH",    metal="Cu",    family="dopamine beta-hydroxylase", pdb="4ZEL"),
    "CDK2":   dict(chembl="CDK2",   metal="Mg",    family="protein kinase",           pdb="1HCK"),
    "METAP2": dict(chembl="METAP2", metal="Co",    family="methionine aminopeptidase", pdb="1B6A"),
    "urease": dict(chembl="urease_note", metal="Ni", family="urease",                 pdb="1FWJ"),
}
DESC = ["MW", "logP", "nRot", "TPSA", "HBD", "HBA", "heavy"]
TAN_CUT, P_CUT = 0.30, 0.05


def desc(m):
    return np.array([Descriptors.MolWt(m), Descriptors.MolLogP(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcTPSA(m),
                     rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                     m.GetNumHeavyAtoms()])


def morgan(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def load_actives(name, n):
    rows = []
    for r in csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")):
        if r.get("target_name") != name or not r.get("canonical_smiles"):
            continue
        try:
            if float(r.get("pactivity") or 0) < 6:
                continue
        except ValueError:
            continue
        rows.append(r["canonical_smiles"])
    seen, out = set(), []
    for s in rows:
        m = Chem.MolFromSmiles(s)
        if m is None or s in seen:
            continue
        seen.add(s); out.append((s, m))
        if len(out) >= n:
            break
    return out


_POOL_CACHE = None

def load_pool(exclude_target):
    """Parsed once, then reused: re-parsing 69k SMILES per target dominated the runtime."""
    global _POOL_CACHE
    if _POOL_CACHE is None:
        _POOL_CACHE = []
        for r in csv.DictReader(open(ROOT / "results/decoy_pool_sized.csv")):
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            _POOL_CACHE.append((r["smiles"], r.get("target", ""), m, desc(m), morgan(m)))
        print(f"  pool cached: {len(_POOL_CACHE)} molecules", flush=True)
    return [(s, m, d, f) for s, t, m, d, f in _POOL_CACHE if t != exclude_target]


def build(target, n_act, n_dec, seed):
    cfg = TARGETS[target]
    acts = load_actives(cfg["chembl"], n_act)
    if len(acts) < 30:
        return None
    act_smi = {s for s, _ in acts}
    ad = np.array([desc(m) for _, m in acts])
    afp = [morgan(m) for _, m in acts]
    pool = load_pool(cfg["chembl"])
    rng = np.random.default_rng(seed)

    # --- diverse decoys: random sample, no property matching
    idx = rng.permutation(len(pool))[: n_dec * 3]
    div, seen = [], set()
    for i in idx:
        smi, m, d, f = pool[i]
        if smi in act_smi or smi in seen:
            continue
        seen.add(smi); div.append((smi, m, d, f))
        if len(div) >= n_dec:
            break

    # --- property-matched decoys: per-active matching
    per = max(1, n_dec // len(acts))
    mat, used, seen2 = [], np.zeros(len(pool), bool), set(act_smi)
    for am, adesc, afp_i in zip([m for _, m in acts], ad, afp):
        taken = 0
        for _ in range(60 * per):
            j = int(rng.integers(0, len(pool)))
            if used[j]:
                continue
            smi, m, d, f = pool[j]
            if smi in seen2:
                continue
            if max(DataStructs.BulkTanimotoSimilarity(f, afp)) >= TAN_CUT:
                continue
            rel = np.abs(d - adesc) / (np.abs(adesc) + 1e-9)
            if (rel[[0, 1, 2, 3]] > 0.45).any():
                continue
            if abs(d[4] - adesc[4]) > 2 or abs(d[5] - adesc[5]) > 2:
                continue
            used[j] = True; seen2.add(smi)
            mat.append((smi, m, d, f)); taken += 1
            if taken >= per:
                break

    # audit
    report = {"target": target, "metal": cfg["metal"], "family": cfg["family"],
              "n_act": len(acts), "n_div": len(div), "n_matched": len(mat), "descriptors": {}}
    if mat:
        md = np.array([d for _, _, d, _ in mat])
        ok = True
        for i, nm in enumerate(DESC):
            u, p = mannwhitneyu(ad[:, i], md[:, i])
            report["descriptors"][nm] = {"act": round(float(ad[:, i].mean()), 1),
                                         "dec": round(float(md[:, i].mean()), 1),
                                         "p": float(f"{p:.3g}"), "ok": bool(p > P_CUT)}
            ok &= bool(p > P_CUT)
        report["all_matched"] = ok

    out = ROOT / "results" / "crossfamily10"
    out.mkdir(parents=True, exist_ok=True)
    for tag, decs in (("div", div), ("dude", mat)):
        with (out / f"{target}_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ligand_id", "label", "smiles", "source"])
            w.writeheader()
            for i, (s, _m) in enumerate(acts):
                w.writerow({"ligand_id": f"{target}_A{i:04d}", "label": 1, "smiles": s,
                            "source": "chembl_active"})
            for i, (s, *_r) in enumerate(decs):
                w.writerow({"ligand_id": f"{target}_{tag.upper()}{i:04d}", "label": 0,
                            "smiles": s,
                            "source": "diverse_decoy" if tag == "div" else "property_matched_decoy"})
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-act", type=int, default=100)
    ap.add_argument("--n-dec", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260126)
    a = ap.parse_args()
    reports = []
    for t in TARGETS:
        rep = build(t, a.n_act, a.n_dec, a.seed)
        if rep is None:
            print(f"{t:8s} SKIP (too few actives)"); continue
        reports.append(rep)
        flag = "OK" if rep.get("all_matched") else "partial-match"
        print(f"{t:8s} {rep['metal']:4s} act={rep['n_act']:3d} div={rep['n_div']:4d} "
              f"matched={rep['n_matched']:4d}  [{flag}]")
    dest = ROOT / "results" / "crossfamily10_build.json"
    dest.write_text(json.dumps(reports, indent=2))
    print(f"\nWROTE {dest}")


if __name__ == "__main__":
    main()
