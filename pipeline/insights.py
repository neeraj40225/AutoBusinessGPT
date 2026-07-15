"""Insight generation — turn computed results into written findings.

Two layers: deterministic insights (computed directly from the data — these are
always correct because they are arithmetic, not generation) and an optional
Gemini narrative that turns the numbers into prose. The LLM is handed
*conclusions with numbers attached* and asked to write, never to compute — that
is what keeps it from hallucinating figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.config import Role
from detection.schema import SchemaMapping
from ml.orchestrator import MLResults
from pipeline.features import FeatureSets
from utils import llm
from utils.helpers import format_compact, format_percent, safe_divide
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Insight:
    title: str
    detail: str
    severity: str = "info"      # info | positive | warning | critical

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "detail": self.detail, "severity": self.severity}


@dataclass
class InsightReport:
    kpis: dict[str, Any]
    insights: list[Insight] = field(default_factory=list)
    narrative: str = ""             # LLM prose, or "" if unavailable
    ai_generated: bool = False


def compute_kpis(frame: pd.DataFrame, mapping: SchemaMapping, fs: FeatureSets) -> dict[str, Any]:
    """Headline metrics, computed from whatever roles exist."""
    kpis: dict[str, Any] = {"rows": len(frame)}

    rev = mapping.role_to_column(Role.REVENUE)
    if rev and rev in frame.columns and pd.api.types.is_numeric_dtype(frame[rev]):
        kpis["total_revenue"] = float(frame[rev].sum())
        kpis["avg_transaction"] = float(frame[rev].mean())

    profit = mapping.role_to_column(Role.PROFIT)
    if profit and profit in frame.columns and pd.api.types.is_numeric_dtype(frame[profit]):
        kpis["total_profit"] = float(frame[profit].sum())
        if "total_revenue" in kpis:
            kpis["profit_margin"] = safe_divide(kpis["total_profit"], kpis["total_revenue"]) * 100

    cust = mapping.role_to_column(Role.CUSTOMER_ID)
    if cust and cust in frame.columns:
        kpis["unique_customers"] = int(frame[cust].nunique())

    order = mapping.role_to_column(Role.ORDER_ID)
    if order and order in frame.columns:
        kpis["total_orders"] = int(frame[order].nunique())

    if fs.has_monthly() and len(fs.monthly) >= 2:
        recent = fs.monthly["sales"].iloc[-1]
        prev = fs.monthly["sales"].iloc[-2]
        kpis["latest_month_growth"] = safe_divide(recent - prev, prev) * 100

    return kpis


def generate(frame: pd.DataFrame, mapping: SchemaMapping, fs: FeatureSets,
             ml: MLResults) -> InsightReport:
    """Produce KPIs, deterministic insights, and an optional LLM narrative."""
    kpis = compute_kpis(frame, mapping, fs)
    insights = _deterministic_insights(frame, mapping, fs, ml, kpis)

    report = InsightReport(kpis=kpis, insights=insights)

    if llm.is_available():
        try:
            report.narrative = _llm_narrative(mapping, kpis, insights, ml)
            report.ai_generated = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Narrative generation failed: %s", exc)
            report.narrative = _template_narrative(mapping, kpis, insights)
    else:
        report.narrative = _template_narrative(mapping, kpis, insights)

    logger.info("Generated %d insights (ai=%s)", len(insights), report.ai_generated)
    return report


def _deterministic_insights(frame, mapping, fs, ml, kpis) -> list[Insight]:
    out: list[Insight] = []

    # revenue concentration
    if fs.has_product() and "revenue" in fs.product.columns and len(fs.product) >= 5:
        top5 = fs.product["revenue"].head(5).sum()
        total = fs.product["revenue"].sum()
        share = safe_divide(top5, total) * 100
        if share > 50:
            out.append(Insight(
                "Revenue is concentrated in few products",
                f"The top 5 products account for {share:.0f}% of revenue. "
                "Concentration is a risk if any of them falter.",
                "warning" if share > 70 else "info",
            ))

    # loss-making products
    if fs.has_product() and "profit" in fs.product.columns:
        loss = fs.product[fs.product["profit"] < 0]
        if not loss.empty:
            out.append(Insight(
                "Some products lose money",
                f"{len(loss)} products have negative total profit, together "
                f"losing {format_compact(abs(loss['profit'].sum()))}.",
                "critical" if len(loss) > len(fs.product) * 0.2 else "warning",
            ))

    # growth trend
    if "latest_month_growth" in kpis:
        g = kpis["latest_month_growth"]
        if abs(g) > 5:
            out.append(Insight(
                f"Revenue {'grew' if g > 0 else 'fell'} {abs(g):.0f}% last month",
                f"Most recent month-over-month change is {g:+.1f}%.",
                "positive" if g > 0 else "warning",
            ))

    # forecast direction
    if ml.forecast is not None:
        fc = ml.forecast.forecast
        hist_avg = ml.forecast.history["sales"].tail(3).mean()
        fc_avg = fc["forecast"].mean()
        change = safe_divide(fc_avg - hist_avg, hist_avg) * 100
        out.append(Insight(
            f"Forecast points {'up' if change >= 0 else 'down'}",
            f"The {ml.forecast.best_name} model projects the next "
            f"{len(fc)} months averaging {format_compact(fc_avg)} "
            f"({change:+.0f}% vs recent).",
            "positive" if change >= 0 else "warning",
        ))

    # churn
    if ml.churn is not None:
        rate = ml.churn.churn_rate * 100
        critical = (ml.churn.scored["risk_band"] == "Critical").sum()
        out.append(Insight(
            "Churn risk detected",
            f"About {rate:.0f}% of customers match the churn profile; "
            f"{critical} are at critical risk and worth immediate outreach.",
            "critical" if rate > 40 else "warning",
        ))

    # segmentation
    if ml.segmentation is not None:
        counts = ml.segmentation.label_counts()
        vip = counts.get("VIP", 0)
        out.append(Insight(
            "Customers split into value segments",
            f"Segmentation found {ml.segmentation.best_k} groups; {vip} customers "
            "are VIPs driving disproportionate value.",
            "info",
        ))

    if not out:
        out.append(Insight(
            "Analysis complete",
            "The dataset was processed successfully. Richer insights need "
            "columns like revenue, dates, customers or products.",
            "info",
        ))
    return out


def _facts_block(mapping, kpis, insights, ml) -> str:
    lines = [f"Business type: {mapping.business_type}", "", "KPIs:"]
    label = {
        "total_revenue": "Total revenue", "total_profit": "Total profit",
        "profit_margin": "Profit margin %", "unique_customers": "Unique customers",
        "total_orders": "Total orders", "rows": "Records",
        "latest_month_growth": "Latest MoM growth %",
    }
    for key, val in kpis.items():
        name = label.get(key, key)
        if isinstance(val, float):
            lines.append(f"- {name}: {val:,.1f}")
        else:
            lines.append(f"- {name}: {val}")
    lines.append("\nKey findings:")
    for i in insights:
        lines.append(f"- [{i.severity}] {i.title}: {i.detail}")
    lines.append(f"\nModels run: {', '.join(ml.ran) or 'none'}")
    if ml.skipped:
        lines.append(f"Models skipped: {', '.join(f'{k} ({v})' for k, v in ml.skipped.items())}")
    return "\n".join(lines)


def _llm_narrative(mapping, kpis, insights, ml) -> str:
    facts = _facts_block(mapping, kpis, insights, ml)
    prompt = (
        "You are a business analyst writing the executive summary of an "
        "automated analysis. Using ONLY the facts below, write 2-3 short "
        "paragraphs: what the business looks like, what stands out, and what to "
        "do next. Do not invent numbers beyond those given. Be concrete and "
        "plain-spoken.\n\n" + facts
    )
    return llm.generate(prompt, temperature=0.4)


def _template_narrative(mapping, kpis, insights) -> str:
    parts = [
        f"This looks like a {mapping.business_type.lower()} dataset covering "
        f"{kpis.get('rows', 0):,} records."
    ]
    if "total_revenue" in kpis:
        parts.append(f"Total revenue is {format_compact(kpis['total_revenue'])}.")
    if "profit_margin" in kpis:
        parts.append(f"Profit margin sits at {format_percent(kpis['profit_margin'])}.")
    crit = [i for i in insights if i.severity in ("critical", "warning")]
    if crit:
        parts.append("Key attention areas: " + "; ".join(i.title.lower() for i in crit[:3]) + ".")
    return " ".join(parts)


__all__ = ["Insight", "InsightReport", "generate", "compute_kpis"]
