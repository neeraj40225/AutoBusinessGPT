"""PDF report generation with ReportLab.

Assembles the analysis into a downloadable executive report: cover, executive
summary (the LLM narrative or template fallback), KPI table, insights, model
results, and recommendations. Returns an in-memory buffer so the app can offer
it as a download without writing to disk.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak,
)

from core.config import settings
from pipeline.runner import AnalysisState
from utils.helpers import format_compact, format_percent
from utils.logger import get_logger

logger = get_logger(__name__)

_PRIMARY = colors.HexColor("#4F46E5")
_ACCENT = colors.HexColor("#0EA5E9")
_DARK = colors.HexColor("#0F172A")
_MUTED = colors.HexColor("#64748B")
_LIGHT = colors.HexColor("#F1F5F9")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=26,
                                textColor=_PRIMARY, spaceAfter=6),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=13,
                                   textColor=_MUTED, alignment=TA_CENTER, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=15,
                             textColor=_DARK, spaceBefore=14, spaceAfter=8),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=10.5,
                               textColor=_DARK, leading=15, spaceAfter=6),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=9,
                                textColor=_MUTED),
        "bullet": ParagraphStyle("bl", parent=base["Normal"], fontSize=10.5,
                                 textColor=_DARK, leading=15, leftIndent=12, spaceAfter=4),
    }


def _kpi_table(kpis: dict[str, Any], styles) -> Table:
    label = {
        "total_revenue": "Total Revenue", "total_profit": "Total Profit",
        "profit_margin": "Profit Margin", "unique_customers": "Unique Customers",
        "total_orders": "Total Orders", "avg_transaction": "Avg Transaction",
        "rows": "Records Analysed", "latest_month_growth": "Latest MoM Growth",
    }
    rows: list[list[str]] = []
    for key, val in kpis.items():
        name = label.get(key, key.replace("_", " ").title())
        if key in ("total_revenue", "total_profit", "avg_transaction"):
            display = format_compact(val)
        elif key in ("profit_margin", "latest_month_growth"):
            display = format_percent(val)
        elif isinstance(val, float):
            display = f"{val:,.1f}"
        else:
            display = f"{val:,}"
        rows.append([name, display])

    table = Table(rows, colWidths=[8 * cm, 6 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), _DARK),
        ("TEXTCOLOR", (1, 0), (1, -1), _PRIMARY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def _severity_color(sev: str):
    return {"critical": colors.HexColor("#DC2626"), "warning": colors.HexColor("#D97706"),
            "positive": colors.HexColor("#059669")}.get(sev, _MUTED)


def build_report(state: AnalysisState) -> BytesIO:
    """Render the full PDF report and return an in-memory buffer."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm,
                            bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm)
    styles = _styles()
    story: list[Any] = []

    ins = state.insights
    mapping = state.mapping

    # -- cover ---------------------------------------------------------------
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph(settings.app_name, styles["title"]))
    story.append(Paragraph("Automated Business Intelligence Report", styles["subtitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"Detected business type: <b>{mapping.business_type}</b>", styles["subtitle"]))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["subtitle"]))
    story.append(PageBreak())

    # -- executive summary ---------------------------------------------------
    story.append(Paragraph("Executive Summary", styles["h2"]))
    summary = ins.narrative if ins and ins.narrative else "Analysis completed."
    for para in summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["body"]))
    tag = "AI-generated" if (ins and ins.ai_generated) else "Template-generated (no API key)"
    story.append(Paragraph(f"<i>{tag}</i>", styles["small"]))

    # -- KPIs ----------------------------------------------------------------
    story.append(Paragraph("Key Metrics", styles["h2"]))
    if ins and ins.kpis:
        story.append(_kpi_table(ins.kpis, styles))

    # -- data quality --------------------------------------------------------
    if state.quality_after:
        q = state.quality_after
        story.append(Paragraph("Data Quality", styles["h2"]))
        story.append(Paragraph(
            f"Quality score after cleaning: <b>{q.score:.0f}/100</b> ({q.grade}). "
            f"{state.cleaning_log.rows_removed if state.cleaning_log else 0} rows removed during cleaning.",
            styles["body"]))

    # -- insights ------------------------------------------------------------
    story.append(Paragraph("Key Insights", styles["h2"]))
    if ins:
        for i in ins.insights:
            c = _severity_color(i.severity)
            story.append(Paragraph(
                f'<font color="{c.hexval()}">●</font> <b>{i.title}</b>', styles["body"]))
            story.append(Paragraph(i.detail, styles["bullet"]))

    # -- models --------------------------------------------------------------
    story.append(Paragraph("Machine Learning Results", styles["h2"]))
    ml = state.ml
    if ml and ml.ran:
        if ml.forecast:
            story.append(Paragraph(
                f"<b>Sales Forecast:</b> best model {ml.forecast.best_name}, "
                f"projecting {len(ml.forecast.forecast)} months ahead.", styles["body"]))
        if ml.churn:
            story.append(Paragraph(
                f"<b>Churn:</b> {ml.churn.best_name}, ROC-AUC "
                f"{ml.churn.best_score.roc_auc:.3f}, ~{ml.churn.churn_rate*100:.0f}% at-risk.",
                styles["body"]))
        if ml.segmentation:
            story.append(Paragraph(
                f"<b>Segmentation:</b> {ml.segmentation.best_k} segments "
                f"(silhouette {ml.segmentation.silhouette:.2f}).", styles["body"]))
    if ml and ml.skipped:
        story.append(Paragraph(
            "Models not run: " + "; ".join(f"{k} ({v})" for k, v in ml.skipped.items()),
            styles["small"]))

    # -- recommendations -----------------------------------------------------
    story.append(Paragraph("Recommendations", styles["h2"]))
    for rec in _recommendations(state):
        story.append(Paragraph(f"• {rec}", styles["bullet"]))

    doc.build(story)
    buffer.seek(0)
    logger.info("Report built (%d bytes)", buffer.getbuffer().nbytes)
    return buffer


def _recommendations(state: AnalysisState) -> list[str]:
    recs: list[str] = []
    ml = state.ml
    if ml and ml.churn and ml.churn.churn_rate > 0.3:
        recs.append("Launch a retention campaign targeting the critical-risk customer segment.")
    if ml and ml.forecast:
        recs.append("Use the sales forecast to plan inventory and staffing for the coming months.")
    if state.features and state.features.has_product():
        prod = state.features.product
        if "profit" in prod.columns and (prod["profit"] < 0).any():
            recs.append("Review or discontinue the loss-making products identified in this report.")
    if ml and ml.segmentation:
        recs.append("Tailor marketing to the VIP segment to protect the highest-value customers.")
    if not recs:
        recs.append("Enrich the dataset with revenue, date, and customer columns to unlock deeper analysis.")
    return recs


__all__ = ["build_report"]
