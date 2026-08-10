"""Render Numi Lab Paper II as a verified A4 PDF."""

from __future__ import annotations

import json
from pathlib import Path
import re

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/pqi2/PAPER_II.md"
SUMMARY = ROOT / "research/pqi2/results/study-summary.json"
OUTPUT = ROOT / "output/pdf/physically-qualified-imagination-paper-ii.pdf"

INK = colors.HexColor("#18231f")
MOSS = colors.HexColor("#6d8c7b")
PALE = colors.HexColor("#edf3ef")
WARM = colors.HexColor("#f7f4ee")
MUTED = colors.HexColor("#617069")


def _markup(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def _chart(summary: dict) -> Drawing:
    drawing = Drawing(165 * mm, 56 * mm)
    drawing.add(Rect(0, 0, 165 * mm, 56 * mm, fillColor=PALE, strokeColor=None, rx=5, ry=5))
    drawing.add(String(8 * mm, 48 * mm, "Qualified runs by task", fontName="Helvetica-Bold", fontSize=10, fillColor=INK))
    tasks = ("raise-right-hand", "raise-left-hand", "raise-both-hands")
    labels = ("Right", "Left", "Both")
    for row, (task, label) in enumerate(zip(tasks, labels)):
        y = (35 - row * 12) * mm
        drawing.add(String(8 * mm, y + 2 * mm, label, fontName="Helvetica", fontSize=8, fillColor=MUTED))
        for method, x, color in (("raw-generated", 32, colors.HexColor("#c9b8ac")),
                                 ("stable-base-composition", 91, MOSS)):
            records = [item for item in summary["records"] if item["task"] == task and item["method"] == method]
            success = sum(item["qualified_success"] for item in records)
            width = success * 10 * mm
            drawing.add(Rect(x * mm, y, 50 * mm, 6 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#d6ded9")))
            if width:
                drawing.add(Rect(x * mm, y, width, 6 * mm, fillColor=color, strokeColor=None))
            drawing.add(String((x + 52) * mm, y + 1.5 * mm, f"{success}/5", fontName="Helvetica-Bold", fontSize=8, fillColor=INK))
    drawing.add(String(32 * mm, 6 * mm, "raw generated", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    drawing.add(String(91 * mm, 6 * mm, "stable-base composition", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return drawing


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=28,
                                leading=31, textColor=INK, alignment=TA_LEFT, spaceAfter=6 * mm),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=15,
                                   leading=20, textColor=MOSS, spaceAfter=22 * mm),
        "author": ParagraphStyle("author", parent=base["Normal"], fontSize=10, leading=15, textColor=MUTED),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                             leading=19, textColor=INK, spaceBefore=5 * mm, spaceAfter=2.5 * mm, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                             leading=14, textColor=MOSS, spaceBefore=4 * mm, spaceAfter=2 * mm, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2,
                               leading=11.8, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=2.2 * mm),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=7.4, leading=10.2,
                                textColor=MUTED, alignment=TA_LEFT),
    }


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#dce4df"))
    canvas.line(22 * mm, A4[1] - 17 * mm, A4[0] - 22 * mm, A4[1] - 17 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(22 * mm, A4[1] - 13 * mm, "NUMI LAB  /  PHYSICALLY QUALIFIED IMAGINATION II")
    canvas.drawRightString(A4[0] - 22 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def _cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(WARM)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(MOSS)
    canvas.rect(0, A4[1] - 10 * mm, A4[0], 10 * mm, fill=1, stroke=0)
    canvas.restoreState()


def build() -> Path:
    summary = json.loads(SUMMARY.read_text())
    styles = _styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, title="Physically Qualified Imagination II",
        author="Numan Thabit, Numi Lab", subject="Controlled multi-seed stable-base composition study",
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=23 * mm, bottomMargin=18 * mm,
    )
    cover_frame = Frame(24 * mm, 24 * mm, A4[0] - 48 * mm, A4[1] - 55 * mm, id="cover")
    body_frame = Frame(22 * mm, 18 * mm, A4[0] - 44 * mm, A4[1] - 41 * mm, id="body")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=_cover, autoNextPageTemplate="body"),
        PageTemplate(id="body", frames=[body_frame], onPage=_header_footer, autoNextPageTemplate="body"),
    ])

    lines = SOURCE.read_text().splitlines()
    story = [Spacer(1, 33 * mm)]
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(_markup(line[2:]), styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(_markup(line[3:]), styles["subtitle"]))
        elif line.startswith("### "):
            heading = line[4:]
            if heading == "Abstract":
                story.append(Spacer(1, 18 * mm))
                story.append(Paragraph("Paper II  /  Controlled multi-seed study", styles["h2"]))
                story.append(PageBreak())
            story.append(Paragraph(_markup(heading), styles["h1"]))
            if heading == "3. Results":
                story.append(_chart(summary))
                story.append(Spacer(1, 4 * mm))
        elif line.startswith("#### "):
            story.append(Paragraph(_markup(line[5:]), styles["h2"]))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(set(cell) <= set("-: ") for cell in cells):
                    rows.append([Paragraph(_markup(cell), styles["small"]) for cell in cells])
                index += 1
            table = Table(rows, colWidths=[37 * mm, 20 * mm, 24 * mm, 21 * mm, 24 * mm, 34 * mm], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), PALE), ("GRID", (0, 0), (-1, -1), 0.35, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(table)
            story.append(Spacer(1, 4 * mm))
            continue
        elif line.startswith("Numan Thabit -"):
            story.append(Paragraph(_markup(line), styles["author"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(_markup(line), styles["small"]))
            story.append(Spacer(1, 1.5 * mm))
        else:
            paragraph = line
            while index + 1 < len(lines) and lines[index + 1].strip() and not lines[index + 1].lstrip().startswith(("#", "|")):
                index += 1
                paragraph += " " + lines[index].strip()
            story.append(Paragraph(_markup(paragraph), styles["body"]))
        index += 1
    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(build())
