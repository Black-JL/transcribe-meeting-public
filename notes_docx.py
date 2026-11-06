#!/usr/bin/env python3
"""notes_docx.py — meeting notes/minutes (markdown) -> Word document.

Handles headings, bullets (incl. checkboxes), numbered lists, **bold**,
*italic*, and markdown tables. House style: Times New Roman 10 pt body,
11 pt bold headers, margins 0.7" top/bottom and 0.8" left/right, single spacing.

Usage: notes_docx.py NOTES.md OUT.docx
"""
import re
import sys

from docx import Document
from docx.shared import Inches, Pt

FONT = "Times New Roman"


def base_doc():
    doc = Document()
    for name in ("Normal", "List Bullet", "List Bullet 2", "List Number"):
        if name in [s.name for s in doc.styles]:
            st = doc.styles[name]
            st.font.name = FONT
            st.font.size = Pt(10)
            pf = st.paragraph_format
            pf.space_before = Pt(0); pf.space_after = Pt(2); pf.line_spacing = 1.0
    for lvl in (1, 2, 3):
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = FONT
        st.font.size = Pt(11)
        st.font.bold = True
        st.font.color.rgb = None
        st.paragraph_format.space_before = Pt(8); st.paragraph_format.space_after = Pt(2)
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Inches(0.7)
        sec.left_margin = sec.right_margin = Inches(0.8)
    return doc


def add_inline(par, text):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows
             if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
    if not cells:
        return
    ncol = max(len(r) for r in cells)
    t = doc.add_table(rows=len(cells), cols=ncol)
    t.style = "Table Grid"
    for i, r in enumerate(cells):
        for j in range(ncol):
            par = t.cell(i, j).paragraphs[0]
            text = r[j] if j < len(r) else ""
            if i == 0:
                par.add_run(text.strip("*")).bold = True
            else:
                add_inline(par, text)
    doc.add_paragraph()


def write_md(md_text, out):
    doc = base_doc()
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].rstrip()
        i += 1
        if s.strip().startswith("|"):
            block = [s]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, block)
            continue
        if not s.strip() or re.match(r"^\s*[-_*]{3,}\s*$", s):
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            doc.add_heading(m.group(2).strip(), level=min(len(m.group(1)), 3))
            continue
        m = re.match(r"^\s*>\s?(.*)", s)
        if m:
            p = doc.add_paragraph(); add_inline(p, m.group(1)); continue
        m = re.match(r"^(\s*)[-*]\s+\[( |x|X)\]\s+(.*)", s)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run("☑ " if m.group(2).lower() == "x" else "☐ ")
            add_inline(p, m.group(3)); continue
        m = re.match(r"^(\s*)[-*]\s+(.*)", s)
        if m:
            style = "List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"
            add_inline(doc.add_paragraph(style=style), m.group(2)); continue
        m = re.match(r"^\s*\d+[.)]\s+(.*)", s)
        if m:
            add_inline(doc.add_paragraph(style="List Number"), m.group(1)); continue
        add_inline(doc.add_paragraph(), s.strip())
    doc.save(out)


if __name__ == "__main__":
    write_md(open(sys.argv[1], encoding="utf-8").read(), sys.argv[2])
    print(sys.argv[2])
