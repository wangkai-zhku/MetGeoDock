"""Validate the stacked terms against overfitting and decoy bias.

Two independent checks:

1. Held-out targets: HDAC1 / HDAC2 were NOT used when the term groups were
   chosen (they were incomplete at the time). If the gain shows up there too,
   it is not a subset-selection artefact.

2. Size-matched decoys: keep 1 active : 3 decoys matched on MW (+-25%) and
   nRotB (+-1). If the gain survives, the terms carry binding-relevant signal;
   if it collapses to 0.5, the gain was decoy size bias again.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
spec = importlib.util.spec_from_file_location("fe", "/tmp/cf10_feature_expand.py")
fe = importlib.util.module_from_spec(spec)
sys.modules["fe"] = fe
spec.loader.exec_module(fe)

DATA = fe.build()
DEV = list(DATA)                      # 6 targets used while choosing terms
HOLD = ["HDAC1", "HDAC2"]

# the two groups worth validating
GROUPS = {
    "baseline (full)": [],
    "+ 配位化学组": [("chelate_tot", +1), ("n_donor", +1), ("nS", +1)],
    "+ 配位+密度": [("chelate_tot", +1), ("n_donor", +1), ("nS", +1),
                    ("d_buried_per_heavy", +1), ("d_hbond_per_heavy", +1)],
}


def pct(v):
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def col(d, name):
    v = np.array([r.get(name, np.nan) for r in d], float)
    return np.where(np.isfinite(v), v, np.nanmedian(v)) if np.isfinite(v).any() else np.zeros(len(d))


def score(d, terms):
    s = pct(col(d, "total")) + pct(col(d, "g_buried")) \
        - pct(col(d, "n_rotatable") * np.clip(col(d, "buried_area") / 600.0, 0, 1))
    for name, sign in terms:
        s = s + (-sign) * pct(col(d, name))
    return s


print("=" * 96)
print("检查 1：留出靶点 HDAC1 / HDAC2（选择特征组时未使用）")
print("=" * 96)
hold = {}
for t in HOLD:
    try:
        d = fe.load_target(t)
    except Exception as e:
        print(f"  {t}: 读取失败 {e}")
        continue
    if len(d) < 80:
        print(f"  {t}: 数据不足 ({len(d)} 行)")
        continue
    hold[t] = d
print(f"{'组合':<22}" + "".join(f"{t:>10}" for t in hold) + f"{'均值':>9}")
for label, terms in GROUPS.items():
    au = []
    for t, d in hold.items():
        y = np.array([r["y"] for r in d])
        au.append(fe.auroc(y, score(d, terms)))
    print(f"{label:<22}" + "".join(f"{x:>10.3f}" for x in au) + f"{np.mean(au):>9.3f}")

print()
print("=" * 96)
print("检查 2：size-matched decoy（1 active : 3 decoy，MW ±25%、nRotB ±1）")
print("=" * 96)


def size_match(t, d, k=3, mw_tol=0.25, nrot_tol=1):
    smi = {r["ligand_id"]: r["smiles"]
           for r in __import__("csv").DictReader(
               open(ROOT / f"results/crossfamily10/{t}_div.csv"))}
    cache = {}

    def desc(s):
        if s in cache:
            return cache[s]
        m = Chem.MolFromSmiles(s)
        cache[s] = (Descriptors.MolWt(m), rdMolDescriptors.CalcNumRotatableBonds(m)) if m else None
        return cache[s]

    act = [(i, desc(smi.get(r["id"], ""))) for i, r in enumerate(d) if r["y"] == 1]
    dec = [(i, desc(smi.get(r["id"], ""))) for i, r in enumerate(d) if r["y"] == 0]
    act = [(i, x) for i, x in act if x]
    dec = [(i, x) for i, x in dec if x]
    pairs = []
    for ai, (mw_a, nr_a) in act:
        for di, (mw_d, nr_d) in dec:
            if abs(nr_d - nr_a) > nrot_tol:
                continue
            rel = abs(mw_d - mw_a) / max(mw_a, 1.0)
            if rel <= mw_tol:
                pairs.append((rel, ai, di))
    pairs.sort()
    used, cnt, keep = set(), {}, []
    for rel, ai, di in pairs:
        if di in used or cnt.get(ai, 0) >= k:
            continue
        used.add(di); cnt[ai] = cnt.get(ai, 0) + 1
        keep.append(ai); keep.append(di)
    return [d[i] for i in keep]


print(f"{'组合':<22}" + "".join(f"{t[:8]:>10}" for t in DEV) + f"{'宏平均':>9}{'vs基线':>8}")
base_mean = None
lines = []
for label, terms in GROUPS.items():
    au = []
    for t in DEV:
        sub = size_match(t, DATA[t])
        if len(sub) < 60:
            au.append(float("nan")); continue
        y = np.array([r["y"] for r in sub])
        au.append(fe.auroc(y, score(sub, terms)))
    m = float(np.nanmean(au))
    if base_mean is None:
        base_mean = m
    lines.append((label, au, m))
for label, au, m in lines:
    print(f"{label:<22}" + "".join(f"{x:>10.3f}" if np.isfinite(x) else f"{'-':>10}" for x in au)
          + f"{m:>9.3f}{m - base_mean:>+8.3f}")
