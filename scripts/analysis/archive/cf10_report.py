"""Read-only status + metrics report for the cross-family-10 benchmark.

Reuses the metric functions from scripts/analysis/metrics_crossfamily10.py but
(a) writes nothing into the project tree, (b) drops corrupt/partial CSV rows
that make the official script crash.
"""
from __future__ import annotations
import csv, glob, importlib.util, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
spec = importlib.util.spec_from_file_location(
    "mcf", ROOT / "scripts/analysis/metrics_crossfamily10.py")
mcf = importlib.util.module_from_spec(spec)
sys.modules["mcf"] = mcf
spec.loader.exec_module(mcf)

TARGETS = ["MMP2", "CA2", "HDAC1", "CAPN1", "METAP2", "CDK2", "urease",
           "ALOX5", "DBH", "ARG2", "HDAC2"]


def rows_safe(paths, need):
    """Read CSV rows, skipping interleaved/corrupt lines and empty values."""
    ok, dropped = [], 0
    for f in paths:
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames:      # 0-byte file left by an interrupted run
                continue
            nf = len(rd.fieldnames)
            for r in rd:
                if None in r or len(r) != nf:
                    dropped += 1
                    continue
                v = (r.get(need) or "").strip()
                if v in ("", "nan"):
                    continue
                try:
                    float(v)
                except ValueError:
                    dropped += 1
                    continue
                ok.append(r)
    return ok, dropped


def load_metalgeo(target):
    files = sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{target}_p*.csv")))
    rows, dropped = rows_safe(files, "g_total")
    if not rows:
        return None
    y = np.array([int(r["label"]) for r in rows])
    g = mcf.pct([float(r["total"]) for r in rows])
    gb = mcf.pct([float(r["g_buried"]) for r in rows])
    nrot = np.array([float(r["n_rotatable"]) for r in rows])
    bur = np.array([float(r["buried_area"]) for r in rows])
    return dict(y=y, geo=g, burial=g + gb,
                full=g + gb - mcf.pct(nrot * np.clip(bur / mcf.A_REF, 0, 1)),
                dropped=dropped, nfiles=len(files))


def load_vina(target):
    files = sorted(glob.glob(str(ROOT / f"results/cf10_vina/{target}_p*.csv")))
    rows, dropped = rows_safe(files, "affinity")
    if not rows:
        return None
    return dict(y=np.array([int(r["label"]) for r in rows]),
                s=np.array([float(r["affinity"]) for r in rows]),   # 越低越好（脚本统一约定）
                dropped=dropped, nfiles=len(files))


# ---------------- 1. completeness ----------------
print("=" * 80)
print("数据完整性")
print("=" * 80)
print(f"{'target':<8}{'cf10 part':>10}{'有效行':>8}{'误差行':>8}{'坏行':>6}"
      f"{'vina part':>10}{'有效行':>8}{'误差行':>8}{'坏行':>6}")
for t in TARGETS:
    tf = sorted(glob.glob(str(ROOT / f"results/cf10_thermo/{t}_p*.csv")))
    vf = sorted(glob.glob(str(ROOT / f"results/cf10_vina/{t}_p*.csv")))
    tg, tdrop = rows_safe(tf, "g_total")
    traw = sum(1 for f in tf for _ in csv.DictReader(open(f)))
    vg, vdrop = rows_safe(vf, "affinity")
    vraw = sum(1 for f in vf for _ in csv.DictReader(open(f)))
    print(f"{t:<8}{f'{len(tf)}/4':>10}{len(tg):>8}{traw-len(tg):>8}{tdrop:>6}"
          f"{f'{len(vf)}/4':>10}{len(vg):>8}{vraw-len(vg):>8}{vdrop:>6}")

# ---------------- 2. metrics ----------------
GOOD = ["MMP2", "CA2", "HDAC1", "CAPN1", "METAP2", "CDK2", "urease"]
HDR = (f"{'target':<8}{'n':>5}{'AUROC':>7}{'95% CI':>16}"
       f"{'EF1%':>7}{'EF5%':>7}{'EF10%':>8}{'BEDROC':>8}{'PR-AUC':>8}")


def block(title, loader, pick):
    rows = []
    for t in GOOD:
        d = loader(t)
        if d is None or pick not in d:
            continue
        r = mcf.summarise(title, d["y"], d[pick])
        r["target"] = t
        rows.append(r)
    print()
    print("=" * 80)
    print(f"{title}  （{len(rows)} 个已完成靶点 × 600 ligand = 100 active + 500 decoy）")
    print("=" * 80)
    print(HDR)
    for r in rows:
        ci = r["auroc_ci"]
        print(f"{r['target']:<8}{r['n']:>5}{r['auroc']:>7.3f}"
              f"{f'[{ci[0]:.3f},{ci[1]:.3f}]':>16}"
              f"{r['ef1']:>7.2f}{r['ef5']:>7.2f}{r['ef10']:>8.2f}"
              f"{r['bedroc']:>8.3f}{r['prauc']:>8.3f}")
    if rows:
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"{'宏平均':<8}{'':>5}{m('auroc'):>7.3f}{'':>16}{m('ef1'):>7.2f}"
              f"{m('ef5'):>7.2f}{m('ef10'):>8.2f}{m('bedroc'):>8.3f}{m('prauc'):>8.3f}")
        print(f"{'中位数':<8}{'':>5}{np.median([r['auroc'] for r in rows]):>7.3f}"
              f"{'':>16}{np.median([r['ef1'] for r in rows]):>7.2f}"
              f"{np.median([r['ef5'] for r in rows]):>7.2f}"
              f"{np.median([r['ef10'] for r in rows]):>8.2f}"
              f"{np.median([r['bedroc'] for r in rows]):>8.3f}"
              f"{np.median([r['prauc'] for r in rows]):>8.3f}")
    return rows


geo = block("MetGeo（纯几何）", load_metalgeo, "geo")
bur = block("MetGeo + 埋藏面积", load_metalgeo, "burial")
full = block("MetGeoDock（几何+埋藏+CAT 熵项）", load_metalgeo, "full")

# ---------------- 3. Vina (3 targets only) ----------------
vrows = []
for t in ["MMP2", "CA2", "HDAC1"]:
    d = load_vina(t)
    if d is None:
        continue
    r = mcf.summarise("AutoDock Vina", d["y"], d["s"])
    r["target"] = t
    vrows.append(r)
print()
print("=" * 80)
print("AutoDock Vina 基线（只有 3/10 靶点跑完，对比时注意不同靶点集）")
print("=" * 80)
print(HDR)
for r in vrows:
    ci = r["auroc_ci"]
    print(f"{r['target']:<8}{r['n']:>5}{r['auroc']:>7.3f}"
          f"{f'[{ci[0]:.3f},{ci[1]:.3f}]':>16}"
          f"{r['ef1']:>7.2f}{r['ef5']:>7.2f}{r['ef10']:>8.2f}"
          f"{r['bedroc']:>8.3f}{r['prauc']:>8.3f}")
print(f"{'宏平均':<8}{'':>5}{np.mean([r['auroc'] for r in vrows]):>7.3f}{'':>16}"
      f"{np.mean([r['ef1'] for r in vrows]):>7.2f}{np.mean([r['ef5'] for r in vrows]):>7.2f}"
      f"{np.mean([r['ef10'] for r in vrows]):>8.2f}{np.mean([r['bedroc'] for r in vrows]):>8.3f}"
      f"{np.mean([r['prauc'] for r in vrows]):>8.3f}")

# 同靶点对比（3 个都跑完的靶点）
print()
print("同靶点对比（MetGeoDock vs Vina，仅 3 个两方法都跑完的靶点）")
print("=" * 80)
print(f"{'target':<8}{'MetGeoDock AUROC':>18}{'Vina AUROC':>12}")
for t in ["MMP2", "CA2", "HDAC1"]:
    dm = load_metalgeo(t); dv = load_vina(t)
    if dm is None or dv is None:
        continue
    am = mcf.auroc(dm["y"], dm["full"]); av = mcf.auroc(dv["y"], dv["s"])
    print(f"{t:<8}{am:>18.3f}{av:>12.3f}")

# ---------------- 4. partial targets ----------------
print()
print("=" * 80)
print("未跑完的靶点（样本不全，仅供参考，不能进论文）")
print("=" * 80)
print(f"{'target':<8}{'有效/应有':>12}{'err率':>8}{'AUROC':>8}{'EF1%':>8}")
for t in ["ALOX5", "DBH", "ARG2"]:
    d = load_metalgeo(t)
    if d is None or len(d["y"]) < 20:
        print(f"{t:<8}{'数据不足':>12}")
        continue
    n = len(d["y"])
    r = mcf.summarise("full", d["y"], d["full"])
    errpct = 100.0 * (1 - n / 600)
    print(f"{t:<8}{f'{n}/600':>12}{errpct:>7.1f}%{r['auroc']:>8.3f}{r['ef1']:>8.2f}")
