"""生成 1:5 与 1:10 的完整对比报告（Markdown）。

要点：
  * 两个比例都限制到**同一批靶点**（默认取两套数据都齐备的靶点），口径严格可比；
  * 单靶点 CI 来自 bench22_metrics 的配体级 bootstrap；
  * 方法层面均值额外给**靶点级 bootstrap 95% CI**（重采样靶点，n_boot=10000），
    这才是"方法整体表现"该报的区间（配体级 CI 会因同靶点配体相关而偏窄）；
  * 配对比较给出 MetGeoDock 相对另两方法的 ΔAUC、胜场数与 Wilcoxon p。

用法：
    python scripts/analysis/bench22_report.py
    python scripts/analysis/bench22_report.py --out reports/bench22_1to5_vs_1to10.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
METRICS = [("auroc", "AUROC"), ("ef05", "EF0.5%"), ("ef1", "EF1%"), ("ef2", "EF2%"),
           ("ef5", "EF5%"), ("ef10", "EF10%"), ("ef20", "EF20%"),
           ("bedroc", "BEDROC"), ("logauc", "logAUC"), ("rie", "RIE")]
METHODS = [("metalgeo_full", "MetGeoDock"), ("vina", "Vina"), ("metallodock", "MetalloDock")]


def load(p: Path) -> dict:
    return json.load(open(p)) if p.exists() else {}


def boot_ci(vals, n_boot=10000, seed=42):
    """靶点级 bootstrap：重采样靶点，返回均值的 95% CI。"""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def collect(d: dict, targets: list[str], method: str, key: str):
    out = []
    for t in targets:
        v = d.get(t, {}).get(method, {}).get(key)
        if v is not None:
            out.append(float(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d5", default="results/bench22_metrics_1to5_full.json")
    ap.add_argument("--d10", default="results/bench22_metrics_1to10_full.json")
    ap.add_argument("--out", default="reports/bench22_1to5_vs_1to10.md")
    a = ap.parse_args()

    d5 = load(ROOT / a.d5)
    d10 = load(ROOT / a.d10)
    # 两套都齐备的靶点才算可比集
    common = sorted(set(d5) & set(d10))
    common = [t for t in common if all(m in d10.get(t, {}) and m in d5.get(t, {}) for m, _ in METHODS)]
    lines = []
    P = lines.append
    P("# 22 靶点金属蛋白酶虚拟筛选 benchmark：1:5 vs 1:10\n")
    P(f"可比靶点：**{len(common)}** 个（两套数据三方法齐备者）\n")
    P("指标方向：AUROC / EF / BEDROC / logAUC / RIE 均为**越大越好**；"
      "BEDROC(α=20) 与 logAUC 的随机基线随 actives 比例变化，见表下注。\n")

    # ---------- 表 1：方法层面均值 + 靶点级 bootstrap CI ----------
    for tag, d in (("1:5（100 actives + 500 decoys，取前 50 为 decoy 集）", d5),
                   ("1:10（100 actives + 1000 decoys）", d10)):
        P(f"\n## 表 1. {tag}\n")
        head = "| 方法 | " + " | ".join(name for _, name in METRICS) + " |"
        sep = "|---" * (len(METRICS) + 1) + "|"
        P(head); P(sep)
        for m, label in METHODS:
            cells = []
            for key, _ in METRICS:
                vals = collect(d, common, m, key)
                if not vals:
                    cells.append("—"); continue
                mu = np.mean(vals)
                lo, hi = boot_ci(vals)
                cells.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}]")
            P(f"| **{label}** | " + " | ".join(cells) + " |")
        # 随机基线
        P("\n*随机基线（decoys 与 actives 不可区分的期望值）*：" +
          ("AUROC 0.500、BEDROC 0.677、logAUC 0.154" if "1:5" in tag
           else "AUROC 0.500、BEDROC 0.113、logAUC 0.146") + "\n")

    # ---------- 表 2：配对比较 ----------
    P("\n## 表 2. 配对比较（同靶点，1:10）\n")
    P("| 对比 | ΔAUROC [95% CI] | 胜/负 | Wilcoxon p |")
    P("|---|---|---|---|")
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        mi, li = METHODS[i]; mj, lj = METHODS[j]
        x = collect(d10, common, mi, "auroc"); y = collect(d10, common, mj, "auroc")
        if len(x) != len(y) or len(x) < 3:
            continue
        diff = np.array(x) - np.array(y)
        rng = np.random.default_rng(42)
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        wins = int((diff > 0).sum()); loss = int((diff < 0).sum())
        p = wilcoxon(x, y).pvalue
        P(f"| {li} vs {lj} | {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {wins}/{loss} | {p:.3f} |")

    # ---------- 表 3：逐靶点 AUROC，两比例并列 ----------
    P("\n## 表 3. 逐靶点 AUROC（1:5 → 1:10）\n")
    P("| 靶点 | MetGeoDock | Vina | MetalloDock | 1:10 最佳 |")
    P("|---|---|---|---|---|")
    for t in common:
        cells, best_v, best_m = [], -1, ""
        for m, label in METHODS:
            v5 = d5[t][m]["auroc"]; v10 = d10[t][m]["auroc"]
            cells.append(f"{v5:.3f} → **{v10:.3f}**")
            if v10 > best_v:
                best_v, best_m = v10, label
        P(f"| {t} | " + " | ".join(cells) + f" | {best_m} |")

    # ---------- 表 4：1:10 各方法胜出次数 ----------
    P("\n## 表 4. 1:10 下各方法逐靶点胜出次数\n")
    cnt = {label: 0 for _, label in METHODS}
    for t in common:
        aucs = [d10[t][m]["auroc"] for m, _ in METHODS]
        cnt[METHODS[int(np.argmax(aucs))][1]] += 1
    P("| 方法 | 最优靶点数 |")
    P("|---|---|")
    for _, label in METHODS:
        P(f"| {label} | {cnt[label]} / {len(common)} |")

    text = "\n".join(lines) + "\n"
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
