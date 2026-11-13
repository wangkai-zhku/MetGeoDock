"""22 靶点 benchmark 全指标计算与三方法对比。

指标（每个靶点、每个方法）：
    AUROC            整体排序能力
    EF1%/EF5%/EF10%  富集因子（前 1%/5%/10%）
    BEDROC(α=20)     早期识别（Truchon 2007）
    pROC/logAUC      半对数 ROC 面积（Clark & Webster 2003）
    RIE              robust initial enhancement
    top-1%/top-5%     命中个数
    bootstrap 95% CI  AUROC 与 EF1% 的重采样区间

用法：
    python bench22_metrics.py --methods metalgeo,vina,metallodock
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
BENCH = ROOT / "results/bench22"
OUT = ROOT / "results/bench22_metrics.json"

TARGETS = ["CA2", "CA12", "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8",
           "MMP1", "MMP2", "MMP3", "MMP9", "MMP13", "ALOX5", "ARG1", "ARG2",
           "urease", "METAP2", "CYP1A2", "CYP2E1", "CYP19A1", "EGLN2", "HIV1_integrase"]


def load(path: Path):
    """返回 (labels, scores)，score 已转成「越大越好」（原为 total，越小越好）。"""
    y, s = [], []
    for r in csv.DictReader(open(path)):
        try:
            y.append(int(r["label"])); s.append(-float(r["score"]))
        except (ValueError, KeyError):
            continue
    return np.array(y), np.array(s)


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    n = len(y); na = int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    top = np.argsort(-s)[:k]
    return float((y[top].sum() / k) / (na / n))


def rie_score(y, s, alpha=20.0):
    """Robust Initial Enhancement（Truchon & Covey 2007）。"""
    n = len(y); na = int(y.sum())
    if na == 0:
        return float("nan")
    ra = na / n
    ys = y[np.argsort(-s)]
    idx = np.flatnonzero(ys)                       # actives 排位（0-based）
    return float(np.exp(-alpha * idx / n).sum()
                 / (ra * (1 - math.exp(-alpha)) / (math.exp(alpha / n) - 1)))


EF_FRACS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
EF_KEYS = {0.005: "ef05", 0.01: "ef1", 0.02: "ef2", 0.05: "ef5", 0.10: "ef10", 0.20: "ef20"}


def ef_multi(y, s, fracs=EF_FRACS):
    """一次排序算出多个 frac 的 EF（bootstrap 复用，避免每个 frac 各排一次）。"""
    n, na = len(y), int(y.sum())
    if n == 0 or na == 0:
        return {f: float("nan") for f in fracs}
    cum = np.cumsum(y[np.argsort(-s)])
    base = na / n
    out = {}
    for f in fracs:
        k = max(1, int(round(n * f)))
        out[f] = float((cum[k - 1] / k) / base)
    return out


def bedroc_tc(y, s, alpha=20.0):
    """Truchon & Covey BEDROC，直接调用 RDKit 的参考实现。

    原实现第二项写成 (1-factor)*(na/n)，对 alpha=20 而言 factor≈4e-8，
    结果恒等于 na/n（本例 0.667），BEDROC 退化成常数、完全没有区分度。
    自写公式容易在归一化上出错，这里改用 rdkit.ML.Scoring.Scoring.CalcBEDROC。
    校验：完美排序 1.000 / 反向排序 0.000；同规模（100 actives + 50 decoys、
    alpha=20）随机排序的期望约 0.657 ± 0.112，解读时须以该基线为准。
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    n = len(y); na = int(y.sum())
    if na == 0 or na == n:
        return float("nan")
    order = np.argsort(-s)
    rows = [[float(s[i]), int(y[i])] for i in order]
    return float(CalcBEDROC(rows, 1, alpha))


def proc_logauc(y, s):
    """logAUC：x 轴取对数后的 ROC 面积，再除以 log10(1/0.001) 归一化。"""
    n = len(y); na = int(y.sum()); nn = n - na
    if na == 0 or nn == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y) / na
    fp = np.cumsum(1 - y) / nn
    x = np.clip(fp, 1e-3, 1.0)
    lx = np.log10(x)
    area = np.trapezoid(tp, lx) / math.log10(1 / 1e-3)
    return float(abs(area))


def rie(y, s, alpha=20.0):  # 兼容旧名字，返回 RIE
    return rie_score(y, s, alpha)


def bootstrap_all(y, s, n_boot=500, seed=42):
    """一次重采样同时算全部指标的 95% CI（比逐指标重采样快 7 倍）。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ("auroc", "ef05", "ef1", "ef2", "ef5", "ef10", "ef20", "bedroc", "logauc", "rie")
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        pairs = [("auroc", auroc(yy, ss)), ("bedroc", bedroc_tc(yy, ss)),
                 ("logauc", proc_logauc(yy, ss)), ("rie", rie_score(yy, ss))]
        efs = ef_multi(yy, ss)
        pairs += [(EF_KEYS[f], efs[f]) for f in EF_FRACS]
        for k, v in pairs:
            if not math.isnan(v):
                vals[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in vals.items() if v}


def top_hits(y, s, frac):
    """顶部 frac 比例中命中的 actives 个数（整数，便于直接报告）。"""
    n, na = len(y), int(y.sum())
    if na == 0:
        return 0
    k = max(1, int(round(n * frac)))
    return int(y[np.argsort(-s)[:k]].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="metalgeo,vina,metallodock")
    ap.add_argument("--targets", default="")
    ap.add_argument("--bench-dir", default="", help="结果目录，默认 results/bench22")
    ap.add_argument("--out", default="", help="指标 JSON 路径")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap 次数（95% CI）")
    a = ap.parse_args()
    global BENCH, OUT
    if a.bench_dir:
        BENCH = Path(a.bench_dir)
    if a.out:
        OUT = Path(a.out)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()] or TARGETS

    report = {}
    print(f"{'靶点':<16}{'方法':<12}{'n':>5}{'AUC':>7}{'EF1%':>7}{'EF5%':>7}"
          f"{'EF10%':>7}{'BEDROC':>8}{'logAUC':>8}")
    for t in targets:
        for m in methods:
            p = BENCH / f"{m}_{t}.csv"
            if not p.exists():
                continue
            y, s = load(p)
            if len(y) == 0:
                continue
            auc = auroc(y, s)
            efs = ef_multi(y, s)
            e1, e5, e10 = efs[0.01], efs[0.05], efs[0.10]
            bd, lg, ri = bedroc_tc(y, s), proc_logauc(y, s), rie_score(y, s)
            ci = bootstrap_all(y, s, n_boot=a.boot)
            report.setdefault(t, {})[m] = {
                "n": int(len(y)), "n_active": int(y.sum()),
                "auroc": auc, "auroc_ci": ci.get("auroc", [float("nan")] * 2),
                "ef05": efs[0.005], "ef05_ci": ci.get("ef05", [float("nan")] * 2),
                "ef1": e1, "ef1_ci": ci.get("ef1", [float("nan")] * 2),
                "ef2": efs[0.02], "ef2_ci": ci.get("ef2", [float("nan")] * 2),
                "ef5": e5, "ef5_ci": ci.get("ef5", [float("nan")] * 2),
                "ef10": e10, "ef10_ci": ci.get("ef10", [float("nan")] * 2),
                "ef20": efs[0.20], "ef20_ci": ci.get("ef20", [float("nan")] * 2),
                "bedroc": bd, "bedroc_ci": ci.get("bedroc", [float("nan")] * 2),
                "logauc": lg, "logauc_ci": ci.get("logauc", [float("nan")] * 2),
                "rie": ri, "rie_ci": ci.get("rie", [float("nan")] * 2),
                "top1pct_hits": top_hits(y, s, 0.01),
                "top5pct_hits": top_hits(y, s, 0.05)}
            print(f"{t:<16}{m:<12}{len(y):>5}{auc:>7.3f}{e1:>7.2f}{e5:>7.2f}"
                  f"{e10:>7.2f}{bd:>8.3f}{lg:>8.3f}"
                  f"   AUC CI [{ci.get('auroc',[float('nan')]*2)[0]:.3f},{ci.get('auroc',[float('nan')]*2)[1]:.3f}]")
        print()

    if report:
        OUT.write_text(json.dumps(report, indent=2))
        print(f"写出 {OUT}")
        # 汇总
        print(f"\n{'方法':<12}{'靶点数':>7}{'平均AUC':>9}{'平均EF1%':>9}"
              f"{'平均BEDROC':>11}{'平均logAUC':>11}")
        for m in methods:
            rows = [v[m] for v in report.values() if m in v]
            if not rows:
                continue
            print(f"{m:<12}{len(rows):>7}{np.mean([r['auroc'] for r in rows]):>9.3f}"
                  f"{np.mean([r['ef1'] for r in rows]):>9.2f}"
                  f"{np.mean([r['bedroc'] for r in rows]):>11.3f}"
                  f"{np.mean([r['logauc'] for r in rows]):>11.3f}")


if __name__ == "__main__":
    main()
