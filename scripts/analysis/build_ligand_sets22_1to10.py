"""构建 actives:decoys = 1:10 的 22 靶点配体集（旧的 1:5 集完全不动）。

设计要点：
  * 新集 decoys = **旧的 500 个（保持原顺序）+ 额外 500 个**，用与旧集相同的
    MW/重原子数分层配额抽样补齐。这样 results/bench22/common/<T>.csv 里已算过的
    150 个配体（100 actives + 前 50 decoys）的对接分数可以直接复用，只需补算 950 个。
  * actives 与旧集完全相同（100 个，pactivity>=6，按活性降序）。
  * 输出到 processed/ligand_sets22_1to10/，并生成新的 common set
    results/bench22_1to10/common/<T>.csv（100 actives + 1000 decoys = 1100 个）。

用法：
    python scripts/analysis/build_ligand_sets22_1to10.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
OLD = ROOT / "processed/ligand_sets22"
OUT = ROOT / "processed/ligand_sets22_1to10"
COMMON = ROOT / "results/bench22_1to10/common"
OUT.mkdir(parents=True, exist_ok=True)
COMMON.mkdir(parents=True, exist_ok=True)

TARGETS = ["CA2", "CA12", "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8",
           "MMP1", "MMP2", "MMP3", "MMP9", "MMP13",
           "ALOX5", "ARG1", "ARG2", "urease", "METAP2",
           "CYP1A2", "CYP2E1", "CYP19A1", "EGLN2", "HIV1_integrase"]
N_ACT = 100
N_DEC = 1000          # 1:10
MW_BIN = 50.0
HA_BIN = 5
SEED = 20261030


def norm_target(name: str) -> str:
    return name.replace("_cd_proxy", "").replace("_note", "").strip()


def rdkit_props(smiles: str):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return float(Descriptors.MolWt(m)), int(m.GetNumHeavyAtoms())
    except Exception:
        return None


def main():
    rows = list(csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")))
    # 全局 decoy 池（非目标靶点、pactivity>=6、去重）
    pool, seen = [], set()
    for r in rows:
        if norm_target(r.get("target_name", "")) in TARGETS:
            continue
        smi = (r.get("canonical_smiles") or "").strip()
        if not smi or smi in seen:
            continue
        try:
            pa = float(r.get("pactivity") or 0)
        except ValueError:
            continue
        if pa < 6.0:
            continue
        seen.add(smi)
        pool.append({"id": r.get("molecule_chembl_id", ""), "smiles": smi,
                     "src": norm_target(r.get("target_name", "")), "pactivity": pa})
    print(f"全局 decoy 池 {len(pool)} 个", flush=True)

    print("计算池性质并分层…", flush=True)
    strata = defaultdict(list)
    for i, p in enumerate(pool):
        pr = rdkit_props(p["smiles"])
        if pr is None:
            continue
        mw, ha = pr
        strata[(round(mw / MW_BIN), round(ha / HA_BIN))].append(p)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(pool)}", flush=True)
    print(f"分成 {len(strata)} 个性质层", flush=True)

    rng = random.Random(SEED)
    manifest = {}
    for t in TARGETS:
        # actives：与旧集完全一致
        old_act = list(csv.DictReader(open(OLD / f"{t}_actives.csv")))
        old_dec = list(csv.DictReader(open(OLD / f"{t}_decoys.csv")))
        with open(OUT / f"{t}_actives.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ligand_id", "smiles", "pactivity", "metal"])
            for a in old_act:
                w.writerow([a["ligand_id"], a["smiles"], a["pactivity"], a["metal"]])

        used = {d["smiles"] for d in old_dec}
        pick = list(old_dec)                      # 旧 500 个保持原顺序在前
        need = N_DEC - len(pick)
        for a in old_act:
            if need <= 0:
                break
            pr = rdkit_props(a["smiles"])
            if pr is None:
                continue
            mw, ha = pr
            key = (round(mw / MW_BIN), round(ha / HA_BIN))
            cand = [d for d in strata.get(key, []) if d["smiles"] not in used]
            if len(cand) < need // max(1, len(old_act)):
                cand = [d for s, lst in strata.items() if s[0] == key[0] for d in lst
                        if d["smiles"] not in used]
            rng.shuffle(cand)
            take = min(max(1, need // max(1, len(old_act))), need, len(cand))
            for d in cand[:take]:
                used.add(d["smiles"]); pick.append({"ligand_id": d["id"], "smiles": d["smiles"]})
                need -= 1
        # 若分层配额不足，用全池补齐
        if need > 0:
            rest = [d for d in pool if d["smiles"] not in used]
            rng.shuffle(rest)
            for d in rest[:need]:
                used.add(d["smiles"]); pick.append({"ligand_id": d["id"], "smiles": d["smiles"]})

        with open(OUT / f"{t}_decoys.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ligand_id", "smiles", "source", "pactivity"])
            for d in pick:
                w.writerow([d.get("ligand_id", ""), d["smiles"],
                            d.get("source", d.get("src", "")), d.get("pactivity", "")])

        # 新 common set：100 actives + 1000 decoys（顺序与旧 common 前 150 个一致）
        with open(COMMON / f"{t}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ligand_id", "label", "smiles"])
            for a in old_act:
                w.writerow([a["ligand_id"], 1, a["smiles"]])
            for d in pick:
                w.writerow([d.get("ligand_id", ""), 0, d["smiles"]])

        manifest[t] = {"actives": len(old_act), "decoys": len(pick),
                       "ratio": round(len(pick) / max(1, len(old_act)), 1)}
        print(f"  {t:<16} actives={len(old_act):>3}  decoys={len(pick):>5}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n合计 actives {sum(v['actives'] for v in manifest.values())}，"
          f"decoys {sum(v['decoys'] for v in manifest.values())}")
    print(f"配体集 -> {OUT}/\ncommon set -> {COMMON}/")


if __name__ == "__main__":
    sys.exit(main())
