"""Stakeholder report export (Markdown and PDF) built from the deterministic insights."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#6D5EF5")
INK = colors.HexColor("#14151A")
MUTED = colors.HexColor("#6B6F7B")
RULE = colors.HexColor("#E4E5EA")
SOFT = colors.HexColor("#F5F5FA")
WARN = colors.HexColor("#B4580A")


def _escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ReportService:
    @staticmethod
    def build_markdown(file_name: str, insights: dict[str, Any], recommendations: list[dict[str, Any]]) -> str:
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        health = insights["health"]
        lines = [
            f"# Data Analysis Report — {file_name}",
            "",
            f"_Generated {generated} by AI Data Analyst Agent. All figures are computed deterministically with Pandas._",
            "",
            "## Key metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        lines += [f"| {kpi['label']} | {kpi['display']} |" for kpi in insights["kpis"]]
        lines += [
            "",
            f"## Data health: {health['score']}/100 ({health['label']})",
            "",
            "| Dimension | Score |",
            "|---|---|",
        ]
        lines += [f"| {name.capitalize()} | {value}% |" for name, value in health["dimensions"].items()]
        lines += [
            "",
            f"Missing cells: {health['missing_cells']:,} · Invalid values: {health['invalid_cells']:,} · "
            f"Duplicate rows: {health['duplicate_rows']:,} · Issues detected: {health['issue_count']}",
            "",
            "## Key findings",
            "",
        ]
        if insights["findings"]:
            lines += [f"- **{finding['title']}** — {finding['detail']}" for finding in insights["findings"]]
        else:
            lines.append("- No notable patterns were detected.")
        lines += ["", "## Charts", ""]
        for chart in insights["charts"]:
            lines.append(f"### {chart['title']}")
            if chart.get("insight"):
                lines.append(f"_{chart['insight']}_")
            if chart["labels"]:
                lines += ["", f"| {chart['x_label']} | {chart['y_label']} |", "|---|---|"]
                lines += [
                    f"| {label} | {value} |"
                    for label, value in list(zip(chart["labels"], chart["datasets"][0]["data"]))[:15]
                ]
            lines.append("")
        lines += ["## Data quality recommendations", ""]
        if recommendations:
            for item in recommendations:
                column = f" `{item['column']}`" if item.get("column") else ""
                lines.append(f"- **{item['category'].replace('_', ' ').capitalize()}**{column} ({item['status']}): {item['recommendation']}")
        else:
            lines.append("- No data quality issues were detected.")
        lines += ["", "## Columns", "", "| Column | Type | Role | Missing % | Unique |", "|---|---|---|---|---|"]
        lines += [
            f"| {c['name']} | {c['dtype']} | {c['role']} | {c['missing_percentage']} | {c['unique']} |"
            for c in insights["column_summary"]
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _chart_drawing(chart: dict[str, Any], width: float, height: float = 62 * mm) -> Drawing | None:
        labels = chart["labels"][:12]
        values = [v if isinstance(v, (int, float)) else 0 for v in chart["datasets"][0]["data"][:12]]
        if chart["type"] not in ("bar", "histogram", "line") or len(values) < 2:
            return None
        drawing = Drawing(width, height)
        plot = VerticalBarChart() if chart["type"] != "line" else HorizontalLineChart()
        plot.x, plot.y = 36, 30
        plot.width, plot.height = width - 50, height - 46
        plot.data = [values]
        plot.categoryAxis.categoryNames = [str(label)[:12] for label in labels]
        plot.categoryAxis.labels.fontSize = 6.5
        plot.categoryAxis.labels.fillColor = MUTED
        plot.categoryAxis.strokeColor = RULE
        if len(labels) > 6:
            plot.categoryAxis.labels.angle = 30
            plot.categoryAxis.labels.boxAnchor = "ne"
        plot.valueAxis.labels.fontSize = 6.5
        plot.valueAxis.labels.fillColor = MUTED
        plot.valueAxis.strokeColor = RULE
        plot.valueAxis.gridStrokeColor = RULE
        plot.valueAxis.visibleGrid = True
        low = min(values)
        plot.valueAxis.valueMin = min(0, low)
        if chart["type"] == "line":
            plot.lines[0].strokeColor = ACCENT
            plot.lines[0].strokeWidth = 2
        else:
            plot.bars[0].fillColor = ACCENT
            plot.bars[0].strokeColor = None
            plot.barSpacing = 2
        drawing.add(plot)
        drawing.add(String(0, height - 10, chart["title"], fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
        return drawing

    @staticmethod
    def build_pdf(file_name: str, insights: dict[str, Any], recommendations: list[dict[str, Any]]) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
            title=f"Data Analysis Report — {file_name}", author="AI Data Analyst Agent",
        )
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=INK, alignment=TA_LEFT, spaceAfter=2)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=INK, spaceBefore=12, spaceAfter=6)
        body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK)
        muted = ParagraphStyle("muted", parent=body, textColor=MUTED, fontSize=8.5)
        width = A4[0] - 36 * mm
        story: list[Any] = []

        generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        story += [
            Paragraph("DATA ANALYSIS REPORT", ParagraphStyle("eyebrow", parent=muted, textColor=ACCENT, fontName="Helvetica-Bold", fontSize=8)),
            Paragraph(_escape(file_name), h1),
            Paragraph(f"Generated {generated} · {insights['rows']:,} rows × {insights['columns']} columns · figures computed with Pandas", muted),
            Spacer(1, 8),
        ]

        kpis = insights["kpis"][:5]
        kpi_table = Table(
            [[Paragraph(f"<font size=7.5 color='#6B6F7B'>{_escape(k['label'].upper())}</font><br/><font size=14><b>{_escape(k['display'])}</b></font>", body) for k in kpis]],
            colWidths=[width / len(kpis)] * len(kpis),
        )
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SOFT),
            ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story += [kpi_table]

        health = insights["health"]
        story.append(Paragraph(f"Data health — {health['score']}/100 ({health['label']})", h2))
        dims = health["dimensions"]
        health_table = Table(
            [[name.capitalize() for name in dims], [f"{value}%" for value in dims.values()]],
            colWidths=[width / len(dims)] * len(dims),
        )
        health_table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica", 8), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 11), ("TEXTCOLOR", (0, 1), (-1, 1), INK),
            ("LINEBELOW", (0, 1), (-1, 1), 0.6, RULE), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ]))
        story += [health_table, Spacer(1, 3), Paragraph(
            f"Missing cells {health['missing_cells']:,} · invalid values {health['invalid_cells']:,} · "
            f"duplicate rows {health['duplicate_rows']:,} · issues detected {health['issue_count']}", muted)]

        story.append(Paragraph("Key findings", h2))
        for finding in insights["findings"] or [{"title": "No notable patterns", "detail": "Nothing stood out in the automatic scan.", "severity": "info"}]:
            colour = "#B4580A" if finding.get("severity") == "warning" else "#6D5EF5"
            story.append(Paragraph(f"<font color='{colour}'>■</font>&nbsp; <b>{_escape(finding['title'])}</b> — {_escape(finding['detail'])}", body))
            story.append(Spacer(1, 3))

        drawings = [d for d in (ReportService._chart_drawing(c, width) for c in insights["charts"]) if d is not None]
        if drawings:
            story.append(Paragraph("Charts", h2))
            for chart, drawing in zip([c for c in insights["charts"] if ReportService._chart_drawing(c, width)], drawings):
                block = [drawing]
                if chart.get("insight"):
                    block.append(Paragraph(_escape(chart["insight"]), muted))
                block.append(Spacer(1, 8))
                story.append(KeepTogether(block))

        story.append(Paragraph("Data quality recommendations", h2))
        if recommendations:
            rows = [["Issue", "Column", "Recommendation", "Status"]]
            for item in recommendations[:25]:
                rows.append([
                    Paragraph(_escape(item["category"].replace("_", " ").capitalize()), body),
                    Paragraph(_escape(item.get("column") or "—"), body),
                    Paragraph(_escape(item["recommendation"]), body),
                    Paragraph(_escape(item["status"]), body),
                ])
            table = Table(rows, colWidths=[30 * mm, 28 * mm, width - 78 * mm, 20 * mm], repeatRows=1)
            table.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No data quality issues were detected.", body))

        story.append(Paragraph("Column overview", h2))
        rows = [["Column", "Type", "Role", "Missing %", "Unique"]]
        rows += [[Paragraph(_escape(c["name"]), body), c["dtype"], c["role"], f"{c['missing_percentage']}", f"{c['unique']:,}"] for c in insights["column_summary"]]
        table = Table(rows, colWidths=[width * 0.34, width * 0.16, width * 0.18, width * 0.16, width * 0.16], repeatRows=1)
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONT", (1, 1), (-1, -1), "Helvetica", 8.5),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)

        def _footer(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(18 * mm, 9 * mm, "AI Data Analyst Agent")
            canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
            canvas.restoreState()

        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return buffer.getvalue()
