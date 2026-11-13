"""A) size-matched re-evaluation of the three scoring variants.
B) per-target optimum CAT sign vs. protein-side descriptors.

Read-only: consumes results/cf10_thermo/*.csv, results/crossfamily10/*_div.csv,
results/pocket_descriptors.csv. Writes nothing into the project tree.
"""
from __future__ import annotations
import csv, glob, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
spec = importlib.util.spec_from_file_location(
    "mcf", ROOT / "scripts/analysis/metrics_crossfamily10.py")
mcf = importlib.util.module_from_spec(spec)
sys.modules["mcf"] = mcf
spec.loader.exec_module(mcf)

TARGETS = ["MMP2", "CAPN1", "HDAC1", "CA2", "METAP2", "CDK2", "urease"]


def load_scores(t):
    """ligand_id -> dict of terms (only rows with valid g_total)."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{t}_p*.csv"))):
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
                    rec = {k: float(r[k]) for k in
                           ("total", "g_buried", "n_rotatable", "buried_area")}
                except ValueError:
                    continue
                rec["label"] = int(r["label"])
                out[r["ligand_id"]] = rec
    return out


def des(cache, smi):
    if smi in cache:
        return cache[smi]
    m = Chem.MolFromSmiles(smi)
    v = None
    if m is not None:
        v = (Descriptors.MolWt(m), rdMolDescriptors.CalcNumRotatableBonds(m),
             m.GetNumHeavyAtoms())
    cache[smi] = v
    return v


def size_match(div_csv, k_per_active=3, mw_tol=0.25, nrot_tol=1):
    """Greedy 1:k MW/nRotB caliper matching, decoys reused at most once."""
    rows = list(csv.DictReader(open(div_csv)))
    cache = {}
    act = [(r["ligand_id"], des(cache, r["smiles"])) for r in rows if r["label"].strip() == "1"]
    dec = [(r["ligand_id"], des(cache, r["smiles"])) for r in rows if r["label"].strip() == "0"]
    act = [(i, d) for i, d in act if d]
    dec = [(i, d) for i, d in dec if d]
    pairs = []
    for ai, (aid, (mw_a, nrot_a, _)) in enumerate(act):
        for did, (mw_d, nrot_d, _) in dec:
            if abs(nrot_d - nrot_a) > nrot_tol:
                continue
            rel = abs(mw_d - mw_a) / max(mw_a, 1.0)
            if rel > mw_tol:
                continue
            pairs.append((rel, ai, did))
    pairs.sort()
    used_dec, got = set(), {i: 0 for i in range(len(act))}
    keep_act, keep_dec = [], []
    for rel, ai, did in pairs:
        if did in used_dec or got[ai] >= k_per_active:
            continue
        used_dec.add(did)
        got[ai] += 1
        keep_act.append(act[ai][0])
        keep_dec.append(did)
    return keep_act, keep_dec


def variant_aurocs(y, tot, gb, bur, nrot):
    A = mcf.pct(tot); B = mcf.pct(gb)
    C = mcf.pct(nrot * np.clip(bur / mcf.A_REF, 0, 1))
    return dict(A=mcf.auroc(y, A), B=mcf.auroc(y, B), C=mcf.auroc(y, C),
                AB=mcf.auroc(y, A + B), full=mcf.auroc(y, A + B - C))


print("=" * 100)
print("A) size-matched 复评（1 active : 3 decoy，MW ±25%、nRotB ±1）")
print("=" * 100)
res_un, res_mt = {}, {}
for t in TARGETS:
    s = load_scores(t)
    if not s:
        continue
    div = ROOT / f"results/crossfamily10/{t}_div.csv"
    all_ids = [i for i in s]
    y = np.array([s[i]["label"] for i in all_ids])
    tot = np.array([s[i]["total"] for i in all_ids])
    gb = np.array([s[i]["g_buried"] for i in all_ids])
    bur = np.array([s[i]["buried_area"] for i in all_ids])
    nrot = np.array([s[i]["n_rotatable"] for i in all_ids])
    res_un[t] = variant_aurocs(y, tot, gb, bur, nrot)

    ka, kd = size_match(div)
    ids = [i for i in ka + kd if i in s]
    if len(ids) < 40 or len({s[i]["label"] for i in ids}) < 2:
        continue
    y2 = np.array([s[i]["label"] for i in ids])
    res_mt[t] = variant_aurocs(
        y2, np.array([s[i]["total"] for i in ids]),
        np.array([s[i]["g_buried"] for i in ids]),
        np.array([s[i]["buried_area"] for i in ids]),
        np.array([s[i]["n_rotatable"] for i in ids]))
    res_mt[t]["n"] = len(ids)
    res_mt[t]["n_act"] = int(y2.sum())

print(f"{'target':<8}{'n匹配':>7}| {'A 未匹配':>9}{'A 匹配':>8} |"
      f"{'B 未匹配':>9}{'B 匹配':>8} |{'full 未匹配':>12}{'full 匹配':>10}")
for t in TARGETS:
    if t not in res_mt:
        continue
    u, m = res_un[t], res_mt[t]
    print(f"{t:<8}{m['n']:>7}| {u['A']:>9.3f}{m['A']:>8.3f} |"
          f"{u['B']:>9.3f}{m['B']:>8.3f} |{u['full']:>12.3f}{m['full']:>10.3f}")
mu = lambda d, k: np.mean([v[k] for v in d.values()])
print(f"{'宏平均':<8}{'':>7}| {mu(res_un,'A'):>9.3f}{mu(res_mt,'A'):>8.3f} |"
      f"{mu(res_un,'B'):>9.3f}{mu(res_mt,'B'):>8.3f} |"
      f"{mu(res_un,'full'):>12.3f}{mu(res_mt,'full'):>10.3f}")

print()
print("=" * 100)
print("B) 每个靶点上 CAT 项的最优符号，与蛋白侧描述符对照")
print("=" * 100)
pocket = {r["target"]: r for r in csv.DictReader(open(ROOT / "results/pocket_descriptors.csv"))}
alias = {"urease": ("urease", "urease_note")}
print(f"{'target':<8}{'C单独':>7}{'A+B+C':>8}{'A+B-C':>8}{'最优':>6}"
      f"{'cn':>5}{'n_8A':>6}{'rg_pocket':>11}{'lig_mw':>8}{'corr(C,A)':>11}")
rows_b = []
for t in TARGETS:
    s = load_scores(t)
    if not s:
        continue
    ids = list(s)
    y = np.array([s[i]["label"] for i in ids])
    tot = np.array([s[i]["total"] for i in ids])
    gb = np.array([s[i]["g_buried"] for i in ids])
    bur = np.array([s[i]["buried_area"] for i in ids])
    nrot = np.array([s[i]["n_rotatable"] for i in ids])
    A = mcf.pct(tot); B = mcf.pct(gb)
    C = mcf.pct(nrot * np.clip(bur / mcf.A_REF, 0, 1))
    plus, minus = mcf.auroc(y, A + B + C), mcf.auroc(y, A + B - C)
    best = "+" if plus > minus else "-"
    rho = spearmanr(C, A).correlation
    p = pocket.get(t)
    if p is None:
        for alt in alias.get(t, ()):
            p = pocket.get(alt)
            if p:
                break
    cn = float(p["cn"]) if p else np.nan
    n8 = float(p["n_8A"]) if p else np.nan
    rg = float(p["rg_pocket"]) if p else np.nan
    lmw = float(p["lig_mw_mean"]) if p else np.nan
    rows_b.append(dict(t=t, C=mcf.auroc(y, C), plus=plus, minus=minus,
                       best=best, cn=cn, n8=n8, rg=rg, lmw=lmw, rho=rho))
    print(f"{t:<8}{mcf.auroc(y, C):>7.3f}{plus:>8.3f}{minus:>8.3f}{best:>6}"
          f"{cn:>5.0f}{n8:>6.0f}{rg:>11.2f}{lmw:>8.1f}{rho:>11.2f}")

print()
print("CAT 最优方向（+ 加 / − 减）与蛋白描述符的关联（Spearman，n=7，仅供参考）")
plus = np.array([1.0 if r["best"] == "+" else 0.0 for r in rows_b])
for key, name in (("cn", "配位数 cn"), ("n8", "金属 8Å 内原子数"),
                  ("rg", "口袋回旋半径"), ("lmw", "配体 MW 均值"),
                  ("C", "C 单独 AUROC")):
    v = np.array([r[key] for r in rows_b], float)
    ok = np.isfinite(v)
    if ok.sum() >= 4 and len(set(plus[ok])) > 1:
        rho = spearmanr(v[ok], plus[ok]).correlation
        print(f"  {name:<18} rho={rho:+.2f}   (n={int(ok.sum())})")
