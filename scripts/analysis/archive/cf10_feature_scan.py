"""Which scoring terms carry signal on the cross-family benchmark, and can any
of them rescue the targets where the current score fails (urease, CDK2, METAP2)?

Read-only: consumes results/cf10_thermo/*.csv and results/crossfamily10/*_div.csv.

Convention: every feature is scored as "lower is better"; AUROC > 0.5 means the
term alone ranks actives ahead of decoys. No per-target sign flipping.
"""
from __future__ import annotations
import csv, glob, sys
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import rankdata

RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
FEATS = ["total", "coord", "geo", "steric", "contact", "map_overlap", "hbond",
         "hydrophobic", "desolvation", "n_hbond", "n_hydrophobic", "buried_area",
         "rim_overlap", "g_buried", "g_hbond", "g_desolv", "g_conf", "g_total",
         "n_rotatable"]
GOOD = ["MMP2", "CAPN1", "CA2", "METAP2", "CDK2", "urease"]
A_REF = 600.0


def auroc(y, s):
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(1.0 - (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(t):
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
                def fv(rr, k):
                    v = (rr.get(k) or "").strip()
                    try:
                        return float(v)
                    except ValueError:
                        return np.nan
                rec = {k: fv(r, k) for k in FEATS}
                if not np.isfinite(rec["total"]):
                    continue
                rec["label"] = int(r["label"]); rec["id"] = r["ligand_id"]
                rows.append(rec)
    # descriptors from SMILES
    smi = {r["ligand_id"]: r["smiles"]
           for r in csv.DictReader(open(ROOT / f"results/crossfamily10/{t}_div.csv"))}
    for r in rows:
        m = Chem.MolFromSmiles(smi.get(r["id"], ""))
        if m is None:
            r["nheavy"] = np.nan; r["nrot"] = np.nan; r["mw"] = np.nan
        else:
            r["nheavy"] = m.GetNumHeavyAtoms()
            r["nrot"] = rdMolDescriptors.CalcNumRotatableBonds(m)
            r["mw"] = Descriptors.MolWt(m)
    return [r for r in rows if np.isfinite(r["nheavy"])]


def residualise(vals, base):
    """OLS residual of vals on the columns of base (size/flexibility)."""
    X = np.column_stack([np.ones(len(base))] + [b for b in base.T])
    beta, *_ = np.linalg.lstsq(X, vals, rcond=None)
    return vals - X @ beta


data = {t: load(t) for t in GOOD}
data = {t: d for t, d in data.items() if len(d) > 100}

print("=" * 118)
print("1) 单项 AUROC（越低越好口径；>0.5 = 该分项单独就能排对）")
print("=" * 118)
hdr = f"{'feature':<15}" + "".join(f"{t[:7]:>9}" for t in data) + f"{'均值':>8}{'>0.5的靶点':>11}"
print(hdr)
tbl = {}
for k in FEATS:
    vals = []
    for t, d in data.items():
        v = np.array([r[k] for r in d])
        ok = np.isfinite(v)
        if ok.sum() < 100 or len({d[i]["label"] for i in np.where(ok)[0]}) < 2:
            vals.append(np.nan); continue
        y = np.array([d[i]["label"] for i in np.where(ok)[0]])
        s = -v[ok]
        vals.append(np.nan if np.allclose(s, s[0]) else auroc(y, s))
    tbl[k] = vals
    v = np.array(vals)
    n_ok = int(np.nansum(v > 0.5))
    print(f"{k:<15}" + "".join(f"{x:>9.3f}" if np.isfinite(x) else f"{'const':>9}" for x in v)
          + f"{np.nanmean(v):>8.3f}{n_ok:>11}")

print()
print("=" * 118)
print("2) 先对分子尺寸/柔性做残差化（f ~ nHeavy + nRotB + MW），再看 AUROC")
print("   目的：把「分子大小」这根轴摘掉，只留下与尺寸无关的排序信息")
print("=" * 118)
print(hdr)
res_tbl = {}
for k in FEATS:
    vals = []
    for t, d in data.items():
        v = np.array([r[k] for r in d])
        ok = np.isfinite(v)
        if ok.sum() < 100:
            vals.append(np.nan); continue
        dd = [d[i] for i in np.where(ok)[0]]
        y = np.array([r["label"] for r in dd])
        if len(set(y.tolist())) < 2 or np.allclose(v[ok], v[ok][0]):
            vals.append(np.nan); continue
        base = np.column_stack([[r["nheavy"] for r in dd], [r["nrot"] for r in dd],
                                [r["mw"] for r in dd]])
        vals.append(auroc(y, -residualise(v[ok], base)))
    res_tbl[k] = vals
    v = np.array(vals)
    print(f"{k:<15}" + "".join(f"{x:>9.3f}" if np.isfinite(x) else f"{'const':>9}" for x in v)
          + f"{np.nanmean(v):>8.3f}{int(np.nansum(v > 0.5)):>11}")

print()
print("=" * 118)
print("3) 谁能救 urease / CDK2 / METAP2？（要求：在这三个靶点上 >0.60，且方向在其他靶点也成立）")
print("=" * 118)
HARD = [t for t in ("urease", "CDK2", "METAP2") if t in data]
print(f"{'feature':<15}" + "".join(f"{t[:8]:>10}" for t in HARD) + f"{'硬靶点均值':>12}{'易靶点均值':>12}{'泛化?':>8}")
for k in FEATS:
    hard = [tbl[k][list(data).index(t)] for t in HARD]
    easy = [tbl[k][list(data).index(t)] for t in data if t not in HARD]
    if not np.all(np.isfinite(hard)):
        continue
    if np.mean(hard) < 0.58:
        continue
    ok = np.mean(easy) > 0.55
    print(f"{k:<15}" + "".join(f"{x:>10.3f}" for x in hard)
          + f"{np.mean(hard):>12.3f}{np.mean(easy):>12.3f}{'✓' if ok else '×':>8}")

print()
print("=" * 118)
print("4) 现有口径的分项在不同靶点上的方向一致性（几何项 vs 埋藏项 vs CAT）")
print("=" * 118)
print(f"{'target':<8}{'coord':>8}{'geo':>8}{'steric':>8}{'contact':>9}{'map_ovl':>9}"
      f"{'hbond':>8}{'hydroph':>9}{'g_buried':>10}{'n_rot':>8}")
for t, d in data.items():
    y = np.array([r["label"] for r in d])
    def a(k):
        v = np.array([r[k] for r in d]); ok = np.isfinite(v)
        if ok.sum() < 100:
            return float("nan")
        yy = np.array([d[i]["label"] for i in np.where(ok)[0]])
        return auroc(yy, -v[ok]) if not np.allclose(v[ok], v[ok][0]) else float("nan")
    print(f"{t:<8}{a('coord'):>8.3f}{a('geo'):>8.3f}{a('steric'):>8.3f}{a('contact'):>9.3f}"
          f"{a('map_overlap'):>9.3f}{a('hbond'):>8.3f}{a('hydrophobic'):>9.3f}"
          f"{a('g_buried'):>10.3f}{a('n_rotatable'):>8.3f}")
