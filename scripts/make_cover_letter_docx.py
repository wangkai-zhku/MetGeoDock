#!/usr/bin/env python3
"""把 cover letter 的 .txt 转成投稿用 .docx（信函格式，单倍行距）。

与 make_docx.py 的区别：那个脚本面向正文 Markdown（双倍行距、12 pt、表格解析），
本脚本面向纯文本信函（11 pt、1.15 行距、段落之间空一行）。

用法：
    python scripts/make_cover_letter_docx.py manuscript/cover_letters/cover_letter_jacs.txt
    python scripts/make_cover_letter_docx.py            # 转换目录下所有 .txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "manuscript" / "cover_letters"


def convert(src: Path) -> Path:
    blocks = [b.strip() for b in src.read_text().split("\n\n") if b.strip()]
    doc = Document()
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(1)
        sec.top_margin = sec.bottom_margin = Inches(1)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(8)
    st.paragraph_format.line_spacing = 1.15

    for block in blocks:
        text = " ".join(line.strip() for line in block.split("\n"))
        par = doc.add_paragraph()
        for tok in re.split(r"(\*\*[^*]+\*\*)", text):
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                par.add_run(tok[2:-2]).bold = True
            else:
                par.add_run(tok)

    out = src.with_suffix(".docx")
    doc.save(out)
    print(f"  {out.name}  ({out.stat().st_size // 1024} KB, {len(doc.paragraphs)} 段)")
    return out


def main() -> None:
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:] if a.endswith(".txt")]
    else:
        targets = sorted(DEFAULT_DIR.glob("*.txt"))
    for t in targets:
        if not t.exists():
            print(f"  跳过（不存在）: {t}")
            continue
        convert(t)


if __name__ == "__main__":
    main()
