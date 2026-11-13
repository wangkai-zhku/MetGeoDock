"""Diagnose the coordination false-negative rate: is single-conformer embedding
the reason 84% of selective inhibitors fail to coordinate at their own subtype?

For each compound we generate N ETKDG conformers and dock every one of them into
the same receptor, then report how many conformers yield a non-saturated score.
A single-conformer run (randomSeed 0xF00D, as in screen_thermo_ef.py) is the
baseline for comparison.

Usage:
    python scripts/analysis/diagnose_conformer_failure.py --limit 6 --n-conf 20
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))

SUBS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC8"]
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
SAT = 1e-20          # below this the score is at the floating-point floor


def load_scores():
    S = {s: {} for s in SUBS}
    for s in SUBS:
        for f in sorted(glob.glob(str(ROOT / f"results/hdac_sel_mg/{s}_p*.csv"))):
            with open(f, newline="") as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames:
                    continue
                nf = len(rd.fieldnames)
                for r in rd:
                    if None in r or len(r) != nf:
                        continue
                    if (r.get("g_total") or "").strip() in ("", "nan"):
                        continue
                    try:
                        S[s][r["ligand_id"]] = float(r["total"])
                    except ValueError:
                        pass
    return S


import glob  # noqa: E402  (used by load_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6, help="how many failing compounds")
    ap.add_argument("--n-conf", type=int, default=20)
    ap.add_argument("--out", default="results/hdac_conformer_diagnosis.json")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit, read_pdb

    S = load_scores()
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}

    # compounds whose own subtype came out saturated (the false negatives)
    fails = [c for c in meta
             if c in S[meta[c]["target_subtype"]]
             and S[meta[c]["target_subtype"]][c] <= SAT
             and all(c in S[s] for s in SUBS)]
    # spread over subtypes
    picks, seen = [], set()
    for c in fails:
        t = meta[c]["target_subtype"]
        if t in seen and len(picks) < a.limit:
            continue
        picks.append(c); seen.add(t)
        if len(picks) >= a.limit:
            break
    picks = picks[: a.limit]
    print(f"目标亚型配位失败的化合物共 {len(fails)} 个，取 {len(picks)} 个做多构象诊断")
    for c in picks:
        print(f"  {c}  目标={meta[c]['target_subtype']}  pAct={meta[c]['p_target']}")

    receptors = {}
    rows = []
    for c in picks:
        t = meta[c]["target_subtype"]
        sub = t
        if sub not in receptors:
            receptors[sub] = read_pdb(str(ROOT / f"processed/hdac_receptors/{PDB[sub]}_h.pdb"))
        prot = receptors[sub]
        smi = meta[c]["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mh = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xF00D
        ps.pruneRmsThresh = 1.5
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=a.n_conf, params=ps))
        rec = {"molecule": c, "target": t, "n_conformers": len(cids),
               "single_conf_total": S[t][c], "conf_totals": []}
        with open(f"/tmp/_conf_{c}.txt", "w") as log:
            log.write("")
        for cid in cids:
            mc = Chem.Mol(mh, confId=cid)
            mc = Chem.RemoveHs(mc)
            try:
                res = dock_ligand(prot, molecule_from_rdkit(mc, f"{c}_{cid}"),
                                  metal="zn", n_spin=6, max_poses=8)
                tot = min((p.score.total for p in res.poses), default=float("nan"))
            except Exception as exc:
                tot = float("nan")
            rec["conf_totals"].append(tot)
            ok = np.isfinite(tot) and tot > SAT
            print(f"  {c} conf {cid}: total={tot:.3e}  {'配位成功' if ok else '饱和（失败）'}", flush=True)

        arr = np.array([x for x in rec["conf_totals"] if np.isfinite(x)])
        rec["n_non_saturated"] = int((arr > SAT).sum()) if len(arr) else 0
        rec["best_total"] = float(arr.min()) if len(arr) else float("nan")
        rec["any_success"] = bool(rec["n_non_saturated"] > 0)
        rows.append(rec)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    print("\n" + "=" * 78)
    print("多构象 vs 单构象")
    print("=" * 78)
    print(f"{'molecule':<16}{'目标':<8}{'构象数':>7}{'非饱和构象':>11}{'单构象结果':>14}{'多构象救回?':>12}")
    for r in rows:
        print(f"{r['molecule']:<16}{r['target']:<8}{r['n_conformers']:>7}"
              f"{r['n_non_saturated']:>11}{r['single_conf_total']:>14.2e}"
              f"{'是' if r['any_success'] else '否':>12}")
    n_ok = sum(1 for r in rows if r["any_success"])
    print(f"\n单构象非饱和率: 0/{len(rows)}")
    print(f"多构象非饱和率: {n_ok}/{len(rows)}")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
