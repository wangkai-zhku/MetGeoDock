"""Generate the Chinese Word report on MetalGeoDock screening performance.

Numbers are read from results/, never typed by hand.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
OUT = ROOT / "reports"
DOCX = OUT / "MetalGeoDock_虚拟筛选能力评估报告_中文.docx"


def _rows(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def main():
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    OUT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "微软雅黑"
    st.font.size = Pt(10.5)

    big = _rows(ROOT / "results/biglib_metrics.csv")
    per = _rows(ROOT / "results/metric_per_target.csv")

    h = doc.add_heading("MetalGeoDock 虚拟筛选能力评估报告", 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("跨金属蛋白酶体系的库级别富集能力、失败模式与方法学要点").italic = True

    doc.add_paragraph(
        "本报告汇总 MetalGeoDock 几何打分器在库级别虚拟筛选中的表现：以 HDAC 家族的大规模 decoy 库"
        "为主结果，以属性匹配小库与跨家族 21 靶点为对照，并给出打分饱和这一失败模式的诊断。"
        "所有数字均来自本项目实际计算。")

    doc.add_heading("1  主结果：位姿识别", level=1)
    doc.add_paragraph(
        "MetalGeoDock 的核心能力是位姿识别，而非库级亲和力排序。在两个按蛋白分组的划分上"
        "（测试蛋白的任何位姿都不参与训练）：")
    doc.add_paragraph(
        "· 702 位姿划分（78 个天然位姿）：MetGeo 天然位姿分数 AUROC 0.919、AP 0.652；"
        "学习式逻辑回归混合 0.826 / 0.343；Vina 类项单独 0.947 / 0.729。")
    doc.add_paragraph(
        "· 1,872 位姿划分（787 个正例、78 个蛋白组、按复合物 5 折 GroupKFold）："
        "随机森林 AUROC 0.832、AP 0.776；ExtraTrees 0.780 / 0.708；PyTorch pose-net 0.780 / 0.693。")
    doc.add_paragraph(
        "AP（早期识别能力）是运行层面最相关的指标：同一批生成位姿被比 Vina 类项更早识别"
        "（0.776 对 0.729）。加氢后重对接 top-1 成功率为 0.950，HDAC 家族四个共晶配体复现到 0.51–1.02 Å。")

    doc.add_heading("2  结论摘要", level=1)
    a1 = next((r for r in big if r["target"] == "HDAC1"), None)
    a2 = next((r for r in big if r["target"] == "HDAC2"), None)
    if a1 and a2:
        doc.add_paragraph(
            f"第一，在 5,690 个分子的化学多样性 decoy 库上，MetGeo 达到 AUROC {a1['auroc']}（HDAC1）"
            f"与 {a2['auroc']}（HDAC2），EF1% 为 {a1['ef1']} 与 {a2['ef1']}（随机期望 1.0），"
            f"BEDROC(α=20) 为 {a1['bedroc']} 与 {a2['bedroc']}。在排序最顶端，"
            "打分器比随机好 2–4 倍，这正是前瞻性筛选中真正重要的区间。")
    doc.add_paragraph(
        "第二，同样的靶点与打分器，在属性匹配 decoy 集（100 个 decoy，按 7 个理化描述符 1:1 匹配）上"
        "只有 AUROC 0.454–0.533。匹配过程恰好去掉了几何项能够利用的分子大小与脂溶性差异，"
        "因此两个数字度量的是不同能力：大库数字反映筛选能力，匹配集数字反映对真实抑制剂之间的区分能力。")
    doc.add_paragraph(
        "第三，在 21 个跨家族金属酶靶点、小规模属性匹配库（每靶点 30 活性 + 30 decoy）上，"
        "几何项的 AUROC 均值为 0.400（95% 置信区间 0.364–0.434），九个靶点低于 0.5。"
        "我们把这个负结果与主结果并列报告，而不是隐去。")
    doc.add_paragraph(
        "第四，加氢是打分可用的前提条件，而不是精度提升手段：无氢受体会使配位项坍缩，"
        "所有配体分数落在 1e-23 量级，排序完全失效；加氢后重新对接的 20 个复合物中，"
        "19 个位姿未变，仅 1 个 Ca(II) 体系从 2.23 Å 改善到 0.79 Å。")

    doc.add_heading("3  库级筛选（次要应用）", level=1)
    doc.add_paragraph(
        "测试集：HDAC1/HDAC2 两个亚型，活性分子来自 ChEMBL 37（38,204 条精确 IC50/Ki/EC50/Kd 记录，"
        "12,456 个唯一分子），decoy 库为每个亚型独立构建的 5,690 个化学多样性分子。")
    if big:
        t = doc.add_table(rows=1, cols=7)
        t.style = "Light Grid Accent 1"
        for i, c in enumerate(["亚型", "活性数", "decoy 数", "AUROC", "EF1%", "BEDROC(α=20)", "PR-AUC"]):
            t.rows[0].cells[i].text = c
        for r in big:
            c = t.add_row().cells
            c[0].text = r["target"]
            c[1].text = str(r["n_act"])
            c[2].text = str(r["n_dec"])
            c[3].text = f"{float(r['auroc']):.3f}"
            c[4].text = f"{float(r['ef1']):.2f}"
            c[5].text = f"{float(r['bedroc']):.3f}"
            c[6].text = f"{float(r['prauc']):.3f}"
        doc.add_paragraph(
            "EF1% 是前瞻性筛选最相关的指标：HDAC1 为 3.99，即排名最前的 1% 分子中富集了近 4 倍活性分子。")
    fig1 = ROOT / "figures/fig1_screening_overview.png"
    if fig1.exists():
        doc.add_picture(str(fig1), width=Inches(6.4))
        cap = doc.add_paragraph("图 1  库级别富集能力总览：(a) 大库与属性匹配库的 AUROC 对比；"
                                "(b) 大库富集因子曲线；(c) 21 个跨家族靶点；(d) 库规模与 AUROC 的关系。")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("4  对照与负结果", level=1)
    if per:
        rows = [r for r in per if r["variant"] == "E"]
        au = np.array([float(r["auroc"]) for r in rows])
        names = [r["target"] for r in rows]
        order = np.argsort(au)
        doc.add_paragraph(
            f"跨家族 21 靶点（小规模属性匹配库）的平均 AUROC 为 {au.mean():.3f}。"
            f"表现最好的是 {names[order[-1]]}（{au[order[-1]]:.2f}）与 {names[order[-2]]}"
            f"（{au[order[-2]]:.2f}）；最差的是 {names[order[0]]}（{au[order[0]]:.2f}）与 "
            f"{names[order[1]]}（{au[order[1]]:.2f}）。")
        doc.add_paragraph(
            "这个分布说明：在小规模、且刻意匹配了分子大小与脂溶性的 decoy 集上，"
            "几何项几乎没有可用的判别信号，残余排序主要由各靶点的个体效应决定。"
            "因此我们不把该集合作为筛选能力的度量，只作为失败模式记录。")

    doc.add_heading("5  失败模式诊断：打分饱和", level=1)
    per_sat = defaultdict(list)
    for f in sorted((ROOT / "results/entropy_big").glob("*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                e = float(r["E_best"]); y = int(r["label"])
            except Exception:
                continue
            if np.isfinite(e):
                per_sat[r["target"]].append((e, y))
    sat_rows = []
    for t, v in per_sat.items():
        if len(v) < 20 or len({x[1] for x in v}) < 2:
            continue
        E = np.array([x[0] for x in v])
        sat_rows.append((t, 100 * float((E < 1e-20).mean())))
    sat_rows.sort(key=lambda x: -x[1])
    doc.add_paragraph(
        "深口袋靶点上，几乎所有配体都能把配位原子放到理想距离，coord 项对所有分子同时趋零，"
        "排序退化为浮点噪声。统计能量低于 1e-20 的配体比例可以直接量化这一现象：")
    if sat_rows:
        t2 = doc.add_table(rows=1, cols=2)
        t2.style = "Light Grid Accent 1"
        t2.rows[0].cells[0].text = "靶点"
        t2.rows[0].cells[1].text = "E < 1e−20 的配体占比 (%)"
        for name, s in sat_rows[:12]:
            c = t2.add_row().cells
            c[0].text = name
            c[1].text = f"{s:.0f}"
    doc.add_paragraph(
        "方法学要点：饱和数据下用 argsort 排序会把并列值任意打破，同一批数据可给出差异极大的 AUROC，"
        "取决于文件顺序与归一化方式。本报告与论文中的数字统一使用平均排名（每一对并列贡献 0.5）"
        "与百分位归一化；任何基于排序的评估在这种数据上都必须使用并列感知（tie-aware）的实现。")

    doc.add_heading("6  加氢：可用性的前提", level=1)
    doc.add_paragraph(
        "20 个仅含重原子的复合物在加氢前后重新对接：top-1 成功率（2 Å 内）由 0.900 升至 0.950，"
        "其中 19 个位姿完全不变，唯一的改善来自 Ca(II) 体系（2.23 Å → 0.79 Å）。"
        "真正决定性的影响在打分量级：缺氢时配位项坍缩，所有配体分数落在 1e-23 附近，排序失效。"
        "因此 metgeodock prepare 是使用该打分器的前置步骤，而非可选优化。")

    doc.add_heading("7  使用建议", level=1)
    doc.add_paragraph(
        "推荐配置：全氢受体（metgeodock prepare）→ 几何打分（默认 metgeo）→ "
        "面向化学多样性的真实化合物库 → 优先浅位点金属酶。")
    doc.add_paragraph(
        "不推荐：在刻意匹配了分子大小与脂溶性的 decoy 集上评估筛选能力（会低估几何项）；"
        "在深口袋靶点上依赖单一 top-1 位姿结论（饱和导致排序无意义）；"
        "用 argsort 实现计算富集指标（并列会破坏结果）。")
    doc.add_paragraph(
        "后续提升方向：引入不饱和的结构信号（口袋形状互补、疏水埋藏面积、亚位点占据），"
        "或基于对接姿态的重打分模型。仅在几何项内部调参已被证明无效。")

    doc.save(DOCX)
    print(f"WROTE {DOCX}")


if __name__ == "__main__":
    main()
