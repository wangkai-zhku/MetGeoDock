"""生成《HDAC 亚型选择性探索性测试记录》Word 文档。

这份文档记录 2026-10 期间针对「金属酶亚型选择性预测」所做的 13 组探索性实验，
包括每组的动机、方法、参数、结果与失败机制。这些结果不作为论文内容，仅作为
内部技术档案，供后续研究参考。

关键结论：配体侧的亚型选择性无法在「静态结构 + 对接打分」框架内预测，
因为对接打分的尺度由晶体结构（构象、复合物状态、完整度、金属位）决定，
其方差远大于配体选择性信号。
"""
from __future__ import annotations
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
OUT = ROOT / "reports/HDAC亚型选择性探索性测试记录_20261027.docx"
OUT.parent.mkdir(parents=True, exist_ok=True)


def set_cn_font(run, name="宋体", size=10.5, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def h(doc, text, level=1):
    p = doc.add_heading(level=level)
    r = p.add_run(text)
    set_cn_font(r, "黑体", 15 - level * 2, True)
    return p


def para(doc, text, size=10.5, bold=False, indent=True, color=None):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_cn_font(r, "宋体", size, bold)
    if color:
        r.font.color.rgb = color
    return p


def table(doc, header, rows, widths=None, size=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, cell in enumerate(t.rows[0].cells):
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(header[i]))
        set_cn_font(r, "黑体", size, True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(v))
            set_cn_font(r, "宋体", size)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t


def build():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ---------- 封面 ----------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("HDAC 亚型选择性探索性测试记录")
    set_cn_font(r, "黑体", 22, True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("十四组实验的方法、参数、结果与失败机制")
    set_cn_font(r, "黑体", 14)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("技术档案 · 不作为论文内容 · 仅供后续研究参考")
    set_cn_font(r, "楷体", 11)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    para(doc, "", indent=False)

    h(doc, "一、这份文档是什么", 1)
    para(doc, "2026 年 10 月，围绕「MetalGeoDock 能否预测金属酶亚型选择性」这一"
              "问题，共进行了十四组探索性实验。目标是把 HDAC 的六个亚型（HDAC1、HDAC2、"
              "HDAC3、HDAC8 属 class I；HDAC4 属 class IIa；HDAC6 属 class IIb）区分开，"
              "先从亚类层次（I / IIa / IIb）入手。")
    para(doc, "十四组实验全部没有达到可用水平。但每一组的失败机制都被查清，并相互印证，"
              "最终收敛到同一条结论：配体外侧的对接打分尺度由受体晶体结构决定，"
              "其方差远大于配体选择性本身的信号。这份文档完整记录这些实验，"
              "供后续研究避免重复、或从新的角度切入。")
    para(doc, "以下所有数字都能追到仓库里的脚本与结果文件，路径见第九节。")

    # ---------- 背景 ----------
    h(doc, "二、背景与目标", 1)
    para(doc, "HDAC 亚型选择性是药物化学的经典难题。文献给出的结构基础有三类：")
    table(doc,
          ["结构要素", "文献事实", "对应亚型"],
          [["cap 区化学分工", "class I 富集 LYS/CYS/ILE；IIa 富集 MET/LEU；IIb 富集 GLN/ALA/VAL",
            "class 层次"],
           ["L1/L2/L6 loop 门控", "HDAC8 的 L1(30-36) 与 L2(87-105) 组成可开合的门；"
            "Lys33 与 Asp87-89 盐桥锁定 binding rail", "HDAC8"],
           ["sZBD 结构锌域", "class IIa 独有，His665/Cys667/His678/Cys751 配位；"
            "决定 open/closed 构象", "HDAC4"],
           ["CD1/CD2 双催化域", "HDAC6 独有；CD2 为主活性域，L2 的 Ser568 为选择性关键",
            "HDAC6"],
           ["SMRT/N-CoR 核心阻遏物", "SMRT SRD3c 的 β-hairpin 插入 HDAC4 催化裂隙并阻断底物；"
            "DAD+InsP4 激活的是 HDAC3", "HDAC4 / HDAC3"]],
          widths=[3.6, 9.0, 3.0])
    para(doc, "目标是：用 MetalGeoDock 的几何/化学打分，在亚类（I / IIa / IIb）层次上"
              "把选择性配体判给正确的类。判定标准最初采用「目标类排名第一」，"
              "后来改为「目标亚型的绝对打分最低」（见第十一节的方法学修正）。")

    # ---------- 受体基线 ----------
    h(doc, "三、受体与基线配置", 1)
    para(doc, "六个亚型的受体结构、构象状态与已知缺陷：")
    table(doc,
          ["亚型", "结构", "构象/组装", "关键缺陷或注意点"],
          [["HDAC1", "5ICN (链B)", "单体催化域", "与 HDAC2 同源 93.7%"],
           ["HDAC2", "3MAX (链B)", "单体催化域", "—"],
           ["HDAC3", "4A69 (链A)", "HDAC3 + SMRT DAD + InsP4", "已含天然激活复合物"],
           ["HDAC4", "2VQM / 6FYZ_B / 5ZOP", "open / closed / +SMRT 三种", "pseudo-enzyme；His976 朝外 7.8 Å；SMRT 占 cap 区 9 残基"],
           ["HDAC6", "5EDU (CD2)", "human CD2 + MBP 融合", "114-478 是 MBP 标签，须切；CD1 距催化锌 24-31 Å 不侵入 cap 区"],
           ["HDAC8", "1T64 / 2V5W", "closed / open L1 lid", "L1 摆动 6.19 Å（核心叠合 0.41 Å）"]],
          widths=[2.2, 3.4, 4.6, 5.6])
    para(doc, "对接参数（除特别说明外）：metal='zn'，n_spin=18，max_poses=40，"
              "配体构象用 ETKDG + 固定随机种子 0xF00D。"
              "早期实验误用 n_spin=6 / max_poses=8，已单独验证其影响（见第十节）。")

    # ---------- 十四组实验 ----------
    h(doc, "四、十四组探索性实验", 1)
    para(doc, "每组给出：动机、方法、结果、失败机制。")

    exps = [
        ("（1）配位几何打分",
         "用配位几何是否合理来判断配体的选择性。",
         "对 6 亚型计算 coord 项的分布，按 coord 排序。",
         "亚类 32.9% vs 随机 33.3%，p = 0.584；亚型 9.2% vs 16.7%。",
         "coord 在配位完美时饱和到 1e-25 量级，无法区分「都很好」的多个亚型。"),
        ("（2）空腔体积",
         "假设亚型间口袋大小不同。",
         "用网格法算 cap 区 8-16 Å 壳层的空腔体积。",
         "亚型间变异系数仅 2%，全部落在 12900-13540 Å³。",
         "口袋体积几乎相同，无区分度。"),
        ("（3）空腔形状 Jaccard",
         "假设口袋形状不同。",
         "Kabsch 叠合（金属 + 3 个配位残基 Cα，RMSD 0.15-0.20 Å）后算体素 Jaccard。",
         "家族层次 ✓（0.65-0.78）；亚型层次 ✗（HDAC1/HDAC2 = 0.776）。",
         "形状描述符能分家族/亚类，但亚型之间太相似。此结果保留为正面结论。"),
        ("（4）通用 rotamer 熵",
         "假设侧链柔性不同。",
         "Dunbrack 通用 rotamer 熵。",
         "HDAC1/HDAC2/HDAC3 = 59.49 / 59.51 / 59.95，几乎相同。",
         "通用熵不含结构约束，无法反映口袋差异。"),
        ("（5）结构受限 rotamer 熵",
         "加入碰撞判定，让熵反映真实可及构象。",
         "χ1 三态（60/180/-60），用碰撞判定筛掉不可行构象，EPS=1e6 关静电。",
         "信号 0.016 < 噪声 0.0338，比值 0.47。",
         "结构受限后信号仍低于噪声；同一蛋白不同晶体结构的差异达亚型差异 63%。"),
        ("（6）配体诱导 ΔS",
         "假设配体结合会改变侧链熵。",
         "对接前后 rotamer 熵之差。",
         "0/3 命中。",
         "ΔS 的符号随晶体结构翻转。"),
        ("（7）梯度化相互作用谱（7 项）",
         "用连续、不饱和的物理化学项代替饱和的 coord/steric。",
         "interaction_profile 的 hbond/hydrophobic/pi_pi/cation_pi/salt_bridge/"
         "halogen/polar_contact，7 项相加后排序。",
         "14.0% vs 16.7%；前 2 名 32.0%，前 3 名 52.0%。",
         "7 项综合稀释了唯一的有效信号（见第 12 组：单用 pi_pi 得 44.4%）。"),
        ("（8）差异区 rotamer 指纹",
         "用亚型差异区的 rotamer 状态做指纹。",
         "序列比对映射差异残基，提取 rotamer 构象。",
         "信号/噪声 = 0.97；符号随晶体结构翻转（−2.12 ↔ +9.57）。",
         "同一蛋白的两套晶体结构（5ICN vs 4BKX）差异体素 2105，"
         "而 HDAC1-HDAC2 差异 3329 —— 噪声达信号的 63%。"),
        ("（9）可变位置接触分",
         "只统计配体与亚型可变位置残基的接触。",
         "序列比对找可变位点，统计配体接触。",
         "1/5 命中，p = 0.598。",
         "可变位点大多在口袋外缘，配体接触不到。"),
        ("（10）cap 区类特异接触（绝对和）",
         "按类富集残基加权，统计配体接触的富集残基权重和。",
         "W = log2(类内频率/类外频率)，只取 w>0.3 的残基；距离阈值 4.5 Å。",
         "5/9 = 55.6%，p = 0.145。class I 5/5，class II 0/4。",
         "class I 的「全中」是 HDAC1/HDAC2 系统性偏高 + 类内平均所致；"
         "四个 HDAC8 选择性抑制剂在 HDAC8 上的 pose 成功率最低（0.12-0.25），"
         "最高分落在 HDAC1/HDAC2（0.38-0.88）。"),
        ("（11）cap pose 成功率（关闭构象）",
         "用「多少比例的 pose 能接触富集残基」代替「最大接触」。",
         "遍历全部返回 pose，统计 n_ok/n_pose。",
         "5/9；class I 4/4，class II 0/5。",
         "指标更合理，但偏置方向不变。HDAC4 恒为 0.00（结构缺陷，见第 14 组）。"),
        ("（12）化学描述符三连",
         "用不饱和的物理化学量直接比较绝对值。",
         "(a) 7 项相互作用综合；(b) 化学互补性（配体原子比例 vs cap 残基比例，L1 距离）；"
         "(c) 计数匹配（官能团计数 vs 接触残基计数，cosine）。",
         "(a) 1/9；(b) 3/9；(c) 1/9。单项最好的是 pi_pi：4/9 = 44.4%，"
         "且只有 HDAC8 非零（0.412），其余五亚型精确为 0.000。",
         "尺度不对等：配体芳香原子占 57-60%，而口袋芳香残基只占 9-18%，"
         "直接相减必然选「芳香最高的口袋」（HDAC6, 0.176）而非真正匹配的 HDAC8（0.091）。"
         "计数匹配同样失败（9 个里 5 个判给 HDAC4，因 SMRT 提供了额外接触面）。"),
        ("（13）开放构象 + pose 系综",
         "HDAC8 换 2V5W（open L1 lid）+ 全部 pose 成功率。",
         "HDAC8 用 2V5W_A，其余不变；n_spin=18/max_poses=40。",
         "4/9；class I 4/4，class II 0/5。",
         "开放构象使 HDAC8 的 pose 成功率从 0.12-0.25 提升到 0.38-0.75（翻 2-3 倍），"
         "证明 L1 门控真实存在，但判别力未改善。"),
        ("（14）六结构全矩阵",
         "穷举 HDAC4 三种结构 × HDAC8 两种结构 = 6 套完整配置，"
         "每套 9 配体 × 6 亚型 × 40 pose。",
         "HDAC4 ∈ {2VQM open, 6FYZ_B closed, 5ZOP_both +SMRT}；"
         "HDAC8 ∈ {1T64 closed, 2V5W open}。",
         "A(2VQM/1T64) 4/9；B(2VQM/2V5W) 4/9；C(6FYZ/1T64) 1/9；"
         "D(6FYZ/2V5W) 1/9；E(5ZOP/1T64) 1/9；F(5ZOP/2V5W) 1/9。",
         "决定性证据：HDAC1/2/3/6 在六套配置下平均 rate 完全不变"
         "（0.725/0.428/0.311/0.206），每个亚型只由自己的结构决定；"
         "而 HDAC4 的 rate 随结构从 0.042 → 0.667 → 0.997，跨度 24 倍。"
         "排名切换开关就是「HDAC4 用哪个晶体结构」，与配体目标完全无关。"),
    ]
    for name, motive, method, result, why in exps:
        h(doc, name, 2)
        para(doc, f"动机：{motive}")
        para(doc, f"方法：{method}")
        para(doc, f"结果：{result}")
        para(doc, f"失败机制：{why}")

    # ---------- 六条成立 ----------
    h(doc, "五、六条成立的结果（可写入论文）", 1)
    para(doc, "十四组实验中，以下六条经过重复验证，属于正面结论：")
    table(doc,
          ["结果", "数字", "验证方式"],
          [["配位识别", "63-98%，min_donor_dist 精确 2.05 Å", "跨 16691 个金属蛋白结构"],
           ["位姿识别", "AUROC 0.919 / AP 0.652；重对接 top-1 0.950", "重对接 + 晶体位姿对照"],
           ["虚拟筛选", "CF10 AUROC 0.52-0.69；HDAC 大库 EF1% 3.99", "11 靶点跨家族 + 大库"],
           ["cap 区化学分离 class", "分离度 1.25；芳香比例判别力 2.53", "6 亚型聚类"],
           ["形状 Jaccard 分离家族", "0.65-0.78，正确重现 class I/IIa/IIb", "Kabsch 叠合后体素 Jaccard"],
           ["rotamer 熵分层", "0.62-0.70", "结构受限 χ1 熵"]],
          widths=[4.0, 7.0, 5.0])

    # ---------- 四条结构发现 ----------
    h(doc, "六、四条新结构发现", 1)
    para(doc, "这四条是探索过程中挖出的、有文献支撑且可独立复现的结构事实：")
    table(doc,
          ["发现", "数据", "文献对应"],
          [["DDD 基序仅 HDAC8 有", "L2 区 87-89 为 Asp-Asp-Asp；其余五亚型均为 SEYSK/MQGFT/ASVFV/RELHR",
            "Kunze JACS 2013：Lys33 与 Asp87-89 盐桥锁定 binding rail"],
           ["L1 构象门控定量", "1T64 与 2V5W 核心叠合 0.41 Å 下，L1(30-36) RMSD 3.91 Å，最大位移 6.19 Å",
            "Dowling Biochemistry 2008；Vannini EMBO Rep 2007"],
           ["HDAC8 必须用开放构象", "pose 成功率 0.12-0.25 → 0.38-0.75（2V5W vs 1T64）",
            "Largazole 类似物诱导 L1/L2 不同构象"],
           ["HDAC6 的 5EDU 是 MBP 融合", "720 残基中 114-478 为麦芽糖结合蛋白；CD2 仅 479-835",
            "COMPND: MBP + HD6 CATALYTIC DOMAIN 2"]],
          widths=[4.2, 7.4, 4.4])

    # ---------- 方法论结论 ----------
    h(doc, "七、为什么全部失败：三条互相印证的机制", 1)
    para(doc, "第一，打分尺度由结构设定。同一配体在 HDAC4 的三种晶体结构上，"
              "cap 接触率从 0.042 到 0.997（24 倍），而选择性信号只有 0.03 量级。"
              "六套配置矩阵显示：HDAC1/2/3/6 的平均 rate 完全不随其他亚型换结构而变"
              "（0.725/0.428/0.311/0.206），每个亚型的分数只由它自己的结构决定。")
    para(doc, "第二，pose 本身不稳定。同一配体在同一蛋白的两个几乎相同的构象上"
              "（核心区 RMSD 仅 0.41 Å），给出 pose RMSD 8.6-15.4 Å 的完全不同结合模式。"
              "配体全长才 10-15 Å，等于完全换了一种朝向。任何基于单一 pose 的描述符，"
              "输入本身就在 10 Å 量级抖动。")
    para(doc, "第三，打分在完美结合处饱和。steric=0、coord≈1e-25 时 total 精确为 0.00，"
              "一次实验中五个亚型同时为 0.00，无法比较大小。"
              "这使「按绝对打分排序」与「按排名」两种口径都失效。")

    # ---------- 关于 SMRT ----------
    h(doc, "八、一个需要澄清的机制（HDAC4 与 SMRT）", 1)
    para(doc, "文献常被误读为「HDAC4 需要 SMRT/N-CoR 才有活性」。实际相反：")
    table(doc,
          ["说法", "文献事实"],
          [["HDAC4 需 SMRT/N-CoR 才有活性", "SMRT SRD3c 的 β-hairpin 插入 HDAC4 催化裂隙，"
            "物理阻断底物；SMRT 结合不激活 HDAC4 的隐蔽活性（Park, NAR 2018；5ZOO/5ZOP）"],
           ["需要 DAD 激活的是 HDAC4", "需要 SMRT DAD + Ins(1,4,5,6)P4 激活的是 HDAC3（4A69）"],
           ["HDAC4 活性正常", "HDAC4 是 pseudo-enzyme：class I 的催化酪氨酸被 His976 取代，"
            "且朝外（距催化锌 7.83-7.96 Å，而 HDAC8 的 Tyr306 仅 4.34 Å）"]],
          widths=[5.0, 11.0])
    para(doc, "实测确认：5ZOP 中 SMRT 肽（链A, 12 残基）距催化锌 4.8 Å，"
              "有 9 个残基落在 8-16 Å 的 cap 壳层（GLU7/GLY8/SER9/THR11/GLN12/GLY13/THR14/PRO15/LEU16），"
              "33 个原子紧贴 cap 区（<4.5 Å）。因此 HDAC4 的生理口袋有一整块被 SMRT 占住。")

    # ---------- 方法学修正 ----------
    h(doc, "九、方法学修正记录（踩过的坑）", 1)
    table(doc,
          ["错误做法", "问题", "修正"],
          [["按「排名第 1」判命中", "0.75 vs 0.72 的 0.03 差距与数量级差距同等对待；"
            "且五个亚型同时 0.00 时无法排名", "改为比较绝对打分值（仍未解决饱和）"],
           ["用 rate 表示结合好坏", "SMRT 使 HDAC4 的 steric 暴涨至 11-62，rate 却显示 1.00"
            "（pose 被挤在通道里因而「接触」了所有 cap 残基）——方向相反",
            "应直接用 steric/相互作用项"],
           ["把 7 项相互作用相加", "稀释了唯一有信号的 pi_pi", "单项分别评估"],
           ["n_spin=6 / max_poses=8", "采样不足：同一配体稀疏采样 total=1.96，"
            "充分采样 3.23e-23，差 22 个数量级；两 pose 差 4.88 Å",
            "改为 n_spin=18 / max_poses=40（已验证结论不变，但数值更稳）"],
           ["直接替换 HDAC4 结构", "误以为「补上缺失的 Loop2」就好了；实际 Loop2 只贡献 4 个 cap 残基，"
            "真正差异是 open→closed 的整体构象（90 个残基）", "构象状态才是主导变量"],
           ["分子比例直接对比", "配体芳香原子 57-60% vs 口袋芳香残基 9-18%，尺度差 4 倍",
            "应改为计数对计数（已试，仍失败）"]],
          widths=[4.2, 7.6, 4.2])

    # ---------- 索引 ----------
    h(doc, "十、数据与脚本索引（全部可复现）", 1)
    table(doc,
          ["内容", "路径"],
          [["六结构全矩阵结果", "results/matrix_{A,B,C,D,E,F}.json"],
           ["原始打分数值（含 steric/coord/geo/contact）", "results/score_P.json"],
           ["7 项相互作用（top5 pose 平均）", "results/chem_descriptors.json"],
           ["计数匹配结果", "results/count_match.json"],
           ["受体准备清单（11 靶点）", "processed/receptors10/manifest.json"],
           ["HDAC 六亚型受体", "processed/hdac_receptors/{5ICN,3MAX,4A69,5EDU_CD2,1T64,2V5W_A}_h.pdb"],
           ["HDAC4+SMRT 受体", "processed/hdac_receptors/5ZOP_both_h.pdb"],
           ["序列比对映射", "results/hdac_sequence_alignment.json"],
           ["cap 区组成", "results/hdac_cap_region.json"],
           ["类富集权重", "results/hdac_class_cap_enrichment.json"],
           ["六结构矩阵脚本", "scripts/analysis/hdac_open_ensemble.py"],
           ["pose 可复现性检验", "scripts/analysis/pose_reproducibility.py"],
           ["组装状态审计", "scripts/analysis/hdac_assembly_audit.py"],
           ["化学互补性/计数匹配", "scripts/analysis/hdac_chem_complementarity.py, hdac_count_match.py"]],
          widths=[6.0, 10.0])

    # ---------- 下一步 ----------
    h(doc, "十一、后续方向（尚未尝试）", 1)
    para(doc, "1. pose 系综打分：不取单一最优 pose，而用全部 pose 的分布统计"
              "（已试过成功率，但受体仍需确定正确构象）。")
    para(doc, "2. 引入连续的范德华吸引项：现有 steric 只惩罚重叠、不奖励良好接触，"
              "导致多个亚型同时 0.00。加吸引项后才有分辨率。")
    para(doc, "3. 构象选择问题需要 MD 或系综对接：在「不做 MD、单结构对接」的框架内，"
              "「HDAC4 该用 open 还是 closed」无客观答案。")
    para(doc, "4. 若坚持配体侧，需要把配体也投影到残基级（计数而非比例），"
              "并解决「供给饱和」问题。")

    para(doc, "", indent=False)
    p = doc.add_paragraph()
    r = p.add_run("注：本文件不进入论文。论文只采用第五、六节中的成立结果。")
    set_cn_font(r, "楷体", 10.5)
    r.font.color.rgb = RGBColor(0x88, 0x44, 0x00)

    doc.save(str(OUT))
    print(f"写出 {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
