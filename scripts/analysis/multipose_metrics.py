"""multipose 结果的全聚合方式对比：top-k（k=1..8）+ 中位数/均值/一致性。

输入 results/bench22_multipose/1to10/<T>.csv 的 energies 列（全部位姿能量，升序），
对每种聚合方式算 22 靶点的 AUC / EF1% / EF5% / EF10% / BEDROC / logAUC，并与随机基线对照。

设计意图：一次对接已经拿到 8 个位姿，就应当把所有可能的聚合方式都试一遍再决定用哪个，
而不是先射箭后画靶。脚本会把**所有**变体都列出来（含"选错了"的那些），避免选择性报告。

用法：
    python scripts/analysis/multipose_metrics.py
    python scripts/analysis/multipose_metrics.py --targets-with-fewest 12
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import numpy as np  # noqa: E402
from bench22_metrics import auroc, bedroc_tc, ef, ef_multi, proc_logauc  # noqa: E402

BENCH = ROOT / "results/bench22_multipose/1to10"


def load_target(t: str):
    """返回 (labels, {variant: scores})。score 越大越好 → 对能量取负。"""
    p = BENCH / f"{t}.csv"
    if not p.exists():
        return None
    y, energies, fallback = [], [], []
    for r in csv.DictReader(open(p)):
        try:
            lab = int(r["label"])
        except (TypeError, ValueError):
            continue
        raw = r.get("energies") or ""
        ens = [float(x) for x in raw.split(";") if x]
        if not ens:
            # 该行没有完整位姿信息（理论上不该发生）
            try:
                ens = [float(r["E_best"])]
            except (TypeError, ValueError):
                continue
            fallback.append(r["ligand_id"])
        y.append(lab)
        energies.append(sorted(ens))
    if not y:
        return None
    y = np.array(y)

    def agg(k):
        out = []
        for ens in energies:
            kk = min(k, len(ens))
            out.append(-float(np.mean(ens[:kk])))
        return np.array(out)

    variants = {f"top{k}": agg(k) for k in range(1, 9)}
    variants["median_all"] = np.array([-float(np.median(e)) for e in energies])
    variants["mean_all"] = np.array([-float(np.mean(e)) for e in energies])
    variants["worst"] = np.array([-float(e[-1]) for e in energies])
    # 一致性：gap = top3 - best（越小越一致，取负使"越小越好"变成"越大越好"）
    variants["-gap(top3-best)"] = np.array(
        [-((np.mean(e[:min(3, len(e))]) - e[0])) for e in energies])
    variants["-spread"] = np.array([-float(np.std(e)) for e in energies])
    return y, variants, len(fallback)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="")
    ap.add_argument("--out", default="results/multipose_metrics.json")
    a = ap.parse_args()

    targets = ([t.strip() for t in a.targets.split(",") if t.strip()]
               if a.targets else sorted(p.stem for p in BENCH.glob("*.csv") if not p.name.endswith(".tmp")))
    data = {}
    for t in targets:
        r = load_target(t)
        if r is None:
            continue
        y, variants, n_fb = r
        data[t] = (y, variants)
        if n_fb:
            print(f"  ! {t}: {n_fb} 个配体缺少 energies 列，回退到 E_best")
    if not data:
        print(f"没有可用数据（{BENCH}）")
        return

    names = list(next(iter(data.values()))[1])
    print(f"\n靶点 {len(data)} 个；聚合方式 {len(names)} 种\n")
    header = f"{'聚合方式':22s}{'AUC':>8s}{'EF1%':>8s}{'EF5%':>8s}{'EF10%':>8s}{'BEDROC':>8s}{'logAUC':>8s}"
    print(header)
    print("-" * len(header))
    rows = []
    for name in names:
        aucs, e1, e5, e10, bd, lg = [], [], [], [], [], []
        for t, (y, variants) in data.items():
            s = variants[name]
            aucs.append(auroc(y, s))
            e = ef_multi(y, s)
            e1.append(e[0.01]); e5.append(e[0.05]); e10.append(e[0.10])
            bd.append(bedroc_tc(y, s)); lg.append(proc_logauc(y, s))
        rows.append({"variant": name, "auroc": float(np.mean(aucs)),
                     "ef1": float(np.mean(e1)), "ef5": float(np.mean(e5)),
                     "ef10": float(np.mean(e10)), "bedroc": float(np.mean(bd)),
                     "logauc": float(np.mean(lg))})
        print(f"{name:22s}{np.mean(aucs):>8.3f}{np.mean(e1):>8.2f}{np.mean(e5):>8.2f}"
              f"{np.mean(e10):>8.2f}{np.mean(bd):>8.3f}{np.mean(lg):>8.3f}")
    print("\n（随机基线：AUC 0.500，EF 1.00，BEDROC 0.113，logAUC 0.146）")

    import json
    out = ROOT / a.out
    out.write_text(json.dumps({"n_targets": len(data), "variants": rows}, indent=2))
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
