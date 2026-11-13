#!/usr/bin/env python3
"""把 manuscript/MetGeoDock_manuscript.md 转成投稿用 .docx。

按 ACS 期刊常见要求排版：
  · Times New Roman 12 pt，双倍行距，页边距 1 英寸
  · 标题层级：H1 粗体 14 pt 居中、H2 粗体 12 pt、H3 粗斜体 12 pt
  · 行内 **粗体** / *斜体* / `代码` / [n] 引用保留
  · Markdown 表格 → Word 表格（Table Grid，表头粗体）
  · $$...$$ 公式块转为居中斜体文本行（可后续在 Word 里替换为公式对象）

用法：python scripts/make_docx.py [输出路径]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "manuscript/MetGeoDock_manuscript.md"


def add_runs(par, text: str, base_bold=False, base_italic=False, size=None, mono=False):
    """解析 **粗体**、*斜体*、`代码` 并写入 runs。"""
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        bold, italic, code = base_bold, base_italic, False
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            tok, bold = tok[2:-2], True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            tok, italic = tok[1:-1], True
        elif tok.startswith("`") and tok.endswith("`"):
            tok, code = tok[1:-1], True
        run = par.add_run(tok)
        run.bold = bold
        run.italic = italic or code
        if size:
            run.font.size = Pt(size)
        if mono or code:
            run.font.name = "Courier New"


def main():
    args = [a for a in sys.argv[1:] if not a.endswith(".md")]
    src = next((Path(a) for a in sys.argv[1:] if a.endswith(".md")), SRC)
    out = Path(args[0]) if args else src.with_suffix(".docx")
    md = src.read_text().splitlines()
    doc = Document()

    # 版式
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(1)
        sec.top_margin = sec.bottom_margin = Inches(1)
    pf = st.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)

    i, n = 0, len(md)
    in_math = False
    while i < n:
        line = md[i].rstrip()
        if not line.strip():
            i += 1
            continue

        # 公式块
        if line.strip().startswith("$$"):
            block = []
            if line.strip() != "$$":
                block.append(line.strip().strip("$"))
            i += 1
            while i < n and "$$" not in md[i]:
                block.append(md[i].strip())
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.5
            add_runs(p, " ".join(block), base_italic=True, size=11)
            continue

        # 表格
        if line.lstrip().startswith("|") and i + 1 < n and set(md[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = []
            while i < n and md[i].lstrip().startswith("|"):
                cells = [c.strip() for c in md[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            if rows:
                tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
                tbl.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c, cell in enumerate(row[: len(rows[0])]):
                        par = tbl.cell(r, c).paragraphs[0]
                        par.paragraph_format.line_spacing = 1.0
                        add_runs(par, cell, base_bold=(r == 0), size=9)
                doc.add_paragraph()
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            lvl, text = len(m.group(1)), m.group(2).strip()
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(12 if lvl <= 2 else 8)
            if lvl == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, text, base_bold=True, size=14)
            elif lvl == 2:
                add_runs(p, text, base_bold=True, size=12)
            else:
                add_runs(p, text, base_bold=True, base_italic=False, size=12)
            i += 1
            continue

        # 列表
        if re.match(r"^\s*[-*]\s+", line):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.line_spacing = 1.5
            add_runs(p, re.sub(r"^\s*[-*]\s+", "", line))
            i += 1
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.line_spacing = 1.5
            add_runs(p, re.sub(r"^\s*\d+\.\s+", "", line))
            i += 1
            continue

        # 普通段落：合并连续行
        buf = [line]
        i += 1
        while i < n and md[i].strip() and not re.match(r"^(#{1,4}\s|\||\$\$|\s*[-*]\s|\s*\d+\.\s)", md[i]):
            buf.append(md[i].strip())
            i += 1
        p = doc.add_paragraph()
        add_runs(p, " ".join(buf))

    doc.save(out)
    print(f"✓ {out}  ({out.stat().st_size // 1024} KB)")
    print(f"   段落 {len(doc.paragraphs)}，表格 {len(doc.tables)}")


if __name__ == "__main__":
    main()
