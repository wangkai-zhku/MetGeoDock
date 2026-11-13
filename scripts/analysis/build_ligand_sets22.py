"""为 22 个金属蛋白靶点生成统一配体集。

沿用 build_crossfamily10.py 的口径，保证与已有 11 靶点 benchmark 可比：
  actives : pactivity >= 6, 每靶点最多 100 个（按 pactivity 降序）
  decoys  : 从「无关靶点」的高活性分子里抽，按重原子数分层，每个 active 配 5 个
            （DUD-E 式 property matching：多样性 decoy，不与 active 共享骨架）

输出：
  processed/ligand_sets22/<TARGET>_actives.csv
  processed/ligand_sets22/<TARGET>_decoys.csv
  processed/ligand_sets22/manifest.json
"""
from __future__ import annotations
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
OUT = ROOT / "processed/ligand_sets22"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["CA2", "CA12", "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8",
           "MMP1", "MMP2", "MMP3", "MMP9", "MMP13",
           "ALOX5", "ARG1", "ARG2", "urease", "METAP2",
           "CYP1A2", "CYP2E1", "CYP19A1", "EGLN2", "HIV1_integrase"]
N_ACT = 100
N_DEC_PER_ACT = 5
MW_BIN = 50.0
HA_BIN = 5


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
    by_target = defaultdict(list)
    for r in rows:
        t = norm_target(r.get("target_name", ""))
        if t not in TARGETS:
            continue
        smi = (r.get("canonical_smiles") or "").strip()
        if not smi:
            continue
        try:
            pa = float(r.get("pactivity") or 0)
        except ValueError:
            continue
        by_target[t].append({"id": r.get("molecule_chembl_id", ""), "smiles": smi,
                             "pactivity": pa, "metal": r.get("metal", "")})

    # 全局 decoy 池：所有非目标靶点的高活性分子
    pool = []
    for r in rows:
        t = norm_target(r.get("target_name", ""))
        if t in TARGETS:
            continue
        smi = (r.get("canonical_smiles") or "").strip()
        if not smi:
            continue
        try:
            pa = float(r.get("pactivity") or 0)
        except ValueError:
            continue
        if pa < 6.0:
            continue
        pool.append({"id": r.get("molecule_chembl_id", ""), "smiles": smi,
                     "src": t, "pactivity": pa})
    # 去重
    seen, upool = set(), []
    for p in pool:
        if p["smiles"] in seen:
            continue
        seen.add(p["smiles"]); upool.append(p)
    print(f"decoy 池 {len(upool)} 个（来自非目标靶点，pactivity>=6）", flush=True)

    # 预计算池的性质并分层
    print("计算 decoy 池性质…", flush=True)
    strata = defaultdict(list)
    for i, p in enumerate(upool):
        pr = rdkit_props(p["smiles"])
        if pr is None:
            continue
        mw, ha = pr
        strata[(round(mw / MW_BIN), round(ha / HA_BIN))].append(p)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(upool)}", flush=True)
    print(f"分成 {len(strata)} 个性质层", flush=True)

    rng = random.Random(20261027)
    manifest = {}
    for t in TARGETS:
        acts = [a for a in by_target.get(t, []) if a["pactivity"] >= 6.0]
        # 去重 + 按活性降序
        seen_a, ua = set(), []
        for a in sorted(acts, key=lambda x: -x["pactivity"]):
            if a["smiles"] in seen_a:
                continue
            seen_a.add(a["smiles"]); ua.append(a)
        ua = ua[:N_ACT]

        pick_decoys = []
        used = set()
        for a in ua:
            pr = rdkit_props(a["smiles"])
            if pr is None:
                continue
            mw, ha = pr
            key = (round(mw / MW_BIN), round(ha / HA_BIN))
            cand = [d for d in strata.get(key, []) if d["smiles"] not in used]
            if len(cand) < N_DEC_PER_ACT:
                # 邻层补
                cand = [d for s, lst in strata.items() if s[0] == key[0] for d in lst
                        if d["smiles"] not in used]
            rng.shuffle(cand)
            for d in cand[:N_DEC_PER_ACT]:
                used.add(d["smiles"]); pick_decoys.append(d)

        with open(OUT / f"{t}_actives.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ligand_id", "smiles", "pactivity", "metal"])
            for a in ua:
                w.writerow([a["id"], a["smiles"], a["pactivity"], a["metal"]])
        with open(OUT / f"{t}_decoys.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ligand_id", "smiles", "source", "pactivity"])
            for d in pick_decoys:
                w.writerow([d["id"], d["smiles"], d["src"], d["pactivity"]])

        manifest[t] = {"actives": len(ua), "decoys": len(pick_decoys),
                       "metal": (ua[0]["metal"] if ua else "")}
        print(f"  {t:<16} actives={len(ua):>3}  decoys={len(pick_decoys):>4}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    ta = sum(v["actives"] for v in manifest.values())
    td = sum(v["decoys"] for v in manifest.values())
    print(f"\n合计 actives {ta}，decoys {td}，靶点 {len(manifest)}")
    print(f"写出 {OUT}/")


if __name__ == "__main__":
    main()
