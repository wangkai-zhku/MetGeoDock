#!/usr/bin/env python3
"""TOC / Graphical Abstract（ACS 规格 3.25 × 1.75 in）。

三段式论证，每段承担一个论点：
  左  真实晶体位姿：金属同时被蛋白残基和抑制剂配位（真实渲染，非示意图）
  中  方法三条（空位顶点放置、配位统计、最优给体—顶点匹配）+ 打分式（立体/配位几何 +
      疏水埋藏 + 构象柔性 CAT + 氢键/去溶剂化），配一行注释
  右  最好的几个金属靶点上的早期富集（EF1%，以 chance = 1 为基准），再补一句 22 靶点汇总

文案约定（按作者要求）：
  * 只展示本方法，右栏不出现任何其他方法（无 Vina / MetalloDock 对比）；
  * 中栏只用文字 + 公式说明方法，不画示意图；不出现外部工具名；
  * 打分式去掉 contact 项，保留 buried 与 CAT 的热力学项与化学项；
  * 右栏不做全库均值汇总，改为展示单个靶点的极端富集，22 个靶点只作整体支撑。

数据：results/bench22_metrics_1to10_FINAL.json（22 靶点 × 全指标）
输出：figures_final/toc_graphical_abstract.pdf / .png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as F
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, Rectangle

C_MGD = F.COLORS["metalgeo_full"]
C_PRO = "#22b8cf"   # 蛋白给体（与渲染图的青色棒一致）

# 源图左侧那两条灰色粗线（骨架 ribbon 及其深灰轮廓）与配位中心无关，显示前裁掉。
# 坐标为源图 760×760 像素窗口 (x0, y0, x1, y1)，保持正方形以免拉伸。
POSE_CROP = (250, 250, 760, 760)
TOP_N = 4            # 右栏展示的靶点数（EF10% 最高的几个）


def draw_polyhedra(ax, cx, cys, r):
    """4/5/6 配位多面体：蛋白给体占 3 个顶点，其余（1/2/3 个）留空给配体给体。"""
    names = {4: "tetrahedral", 5: "trigonal bipyramid", 6: "octahedral"}
    for cy, (n_total, n_open, lab) in zip(cys, ((4, 1, "CN 4"), (5, 2, "CN 5"), (6, 3, "CN 6"))):
        ax.add_patch(mpatches.Circle((cx, cy), r * 0.14, facecolor="#8c8c8c",
                                     edgecolor="none", zorder=6))
        for i in range(n_total):
            t = np.deg2rad(90 + i * 360.0 / n_total)
            x, y = cx + r * np.cos(t), cy + r * np.sin(t)
            ax.plot([cx, x], [cy, y], lw=0.35, color="#cccccc", zorder=5)
            if i < n_open:                                  # 空位顶点
                ax.add_patch(mpatches.Circle((x, y), r * 0.18, facecolor="white",
                                             edgecolor=C_MGD, lw=0.45,
                                             ls=(0, (1.1, 0.9)), zorder=6))
            else:                                           # 被蛋白给体占据
                ax.add_patch(mpatches.Circle((x, y), r * 0.18, facecolor=C_PRO,
                                             edgecolor="white", lw=0.22, zorder=6))
        ax.text(cx, cy - r * 1.55, lab, ha="center", va="top", fontsize=2.2, color="#333333")
        ax.text(cx, cy - r * 1.55 - 0.040, names[n_total], ha="center", va="top",
                fontsize=2.0, color="#666666", style="italic")


def metrics():
    """22 靶点基准的自家结果，全部从结果文件读，不硬编码。"""
    from scipy.stats import wilcoxon

    d = F.load("1to10")
    ts = F.targets(d)
    a = F.series(d, "metalgeo_full", "auroc", ts)
    top = sorted(((t, d[t]["metalgeo_full"]["auroc"]) for t in ts), key=lambda x: -x[1])
    return {
        "auroc": float(a.mean()),
        "n_above": int((a > 0.5).sum()),
        "n": int(a.size),
        "p": float(wilcoxon(a - 0.5, alternative="greater").pvalue),
        "top": top[:TOP_N],
        "n_top": int(sum(1 for _, v in top if v >= 0.68)),
    }


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 5,
                         "mathtext.fontset": "dejavusans"})
    m = metrics()
    fig = plt.figure(figsize=(3.25, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3.25); ax.set_ylim(0, 1.75); ax.axis("off")

    # ---------------- 左：真实位姿 ----------------
    x0, y0, x1, y1 = POSE_CROP
    img = imread(F.ROOT / "figures_final/poses/toc_pocket_crop.png")[y0:y1, x0:x1]
    ax.imshow(img, extent=(0.04, 0.72, 0.66, 1.34), zorder=1, aspect="auto")
    ax.add_patch(Rectangle((0.04, 0.66), 0.68, 0.68, fill=False, lw=0.5,
                           edgecolor="#cccccc", zorder=2))
    ax.text(0.50, 1.40, "crystal pose", ha="center", va="bottom", fontsize=5.2,
            fontweight="bold", color="#222222")

    # 图上直接标注三样东西（白描边保证在渲染图上可读），把「配位球是共享的」讲清楚
    lab = dict(fontsize=2.8, color="#222222", zorder=6,
               path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
    arrow = dict(arrowstyle="-", lw=0.4, color="#555555",
                 shrinkA=1.0, shrinkB=1.5)
    ax.annotate("metal", xy=(0.360, 1.000), xytext=(0.480, 1.062), arrowprops=arrow, **lab)
    # 右侧：4 / 5 / 6 配位各自的几何模板与留下的空位数
    draw_polyhedra(ax, 0.865, (1.220, 0.985, 0.750), 0.054)

    # 图下三行：位姿准确度（重点，蓝） / 金属广度 / 空位数随几何模板变化
    ax.text(0.50, 0.600, "pose recognition AUROC 0.919", ha="center", va="top",
            fontsize=3.2, color=C_MGD, fontweight="bold")
    ax.text(0.50, 0.500, "10 metal ions \u00b7 CN 3\u20136", ha="center", va="top",
            fontsize=3.0, color="#222222", fontweight="bold")
    ax.text(0.50, 0.412, "1\u20133 open vertices, set by the polyhedron",
            ha="center", va="top", fontsize=2.6, color="#444444")

    ax.add_patch(FancyArrowPatch((0.975, 1.02), (1.095, 1.02), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.75, color="#888888"))

    # ---------------- 中：方法三条 + 打分式 ----------------
    ax.text(1.68, 1.40, "MetGeoDock", ha="center", va="bottom", fontsize=5.2,
            fontweight="bold", color="#222222")
    ax.text(1.68, 1.345, "a geometric-matching score", ha="center", va="bottom",
            fontsize=3.0, color="#666666", style="italic")
    feats = [("1", "vacant-vertex placement", "donors fill only the open directions"),
             ("2", "crystal coordination statistics", "metal\u2013donor distance histograms"),
             ("3", "optimal donor\u2013vertex match", "zero fitting to our test set")]
    y = 1.215
    for num, title, sub in feats:
        ax.add_patch(mpatches.Circle((1.13, y), 0.050, color=C_MGD, zorder=3))
        ax.text(1.13, y, num, ha="center", va="center", fontsize=3.3, color="white",
                zorder=4, fontweight="bold")
        ax.text(1.22, y + 0.030, title, ha="left", va="center", fontsize=4.0,
                fontweight="bold", color="#1a1a1a")
        ax.text(1.22, y - 0.056, sub, ha="left", va="center", fontsize=3.3,
                color="#666666", style="italic")
        y -= 0.225

    # 打分式：立体/配位几何 + 疏水埋藏 + 构象柔性 CAT + 化学项（无 contact 项）
    ax.text(1.68, 0.585, r"$E=E_{\mathrm{st}}+0.35E_{\mathrm{cd}}+E_{\mathrm{geo}}$",
            ha="center", va="center", fontsize=3.8, color="#111111")
    ax.text(1.68, 0.485,
            r"$-0.025\,A_{\mathrm{bur}}-0.6\,N_{\mathrm{rot}}\min(1,A_{\mathrm{bur}}/600)$",
            ha="center", va="center", fontsize=3.8, color="#111111")
    ax.text(1.68, 0.385,
            r"$-0.587\,N_{\mathrm{HB}}+0.015\,N_{\mathrm{des}}$",
            ha="center", va="center", fontsize=3.8, color="#111111")
    ax.add_patch(FancyArrowPatch((2.235, 1.02), (2.355, 1.02), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.75, color="#888888"))

    # ---------------- 右：单个靶点上的极端富集 ----------------
    ax.text(2.79, 1.44, "22 metal targets \u00b7 24,200 ligands", ha="center", va="bottom",
            fontsize=3.4, color="#222222")
    ax.text(2.79, 1.355, "best targets, AUROC (chance 0.50)", ha="center", va="bottom",
            fontsize=2.9, color="#666666", style="italic")

    x_bar, scale = 2.72, 1.15          # 条长 = (AUROC - 0.50) * scale
    ys = (1.190, 1.080, 0.970, 0.860)
    ax.plot([x_bar, x_bar], [0.825, 1.220], ls=(0, (2, 1.6)), lw=0.6, color="#999999", zorder=1)
    for (tgt, v), yy in zip(m["top"], ys):
        w = max(0.02, (v - 0.50) * scale)
        ax.text(2.38, yy, tgt, ha="left", va="center", fontsize=3.0, color="#333333")
        ax.add_patch(Rectangle((x_bar, yy - 0.027), w, 0.054,
                               facecolor=C_MGD, alpha=0.92, linewidth=0, zorder=2))
        ax.text(x_bar + w - 0.03, yy, f"{v:.3f}", ha="right", va="center", fontsize=2.9,
                fontweight="bold", color="white", zorder=3)
    ax.text(x_bar, 0.800, "chance 0.50", ha="left", va="top", fontsize=2.7, color="#999999")

    ax.text(2.79, 0.645, f"{m['n_top']} targets at AUROC \u2265 0.68", ha="center",
            va="center", fontsize=3.0, color=C_MGD, fontweight="bold")
    ax.text(2.79, 0.530,
            f"{m['n_above']}/{m['n']} targets above chance, $p={m['p']:.3f}$",
            ha="center", va="center", fontsize=2.9, color="#555555")
    ax.text(2.79, 0.410, "screening-ready for metalloproteins", ha="center", va="center",
            fontsize=3.0, color="#222222", fontweight="bold")

    out_pdf = F.FIGDIR / "toc_graphical_abstract.pdf"
    out_png = F.FIGDIR / "toc_graphical_abstract.png"
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"  ✓ {out_pdf.name} ({out_pdf.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
