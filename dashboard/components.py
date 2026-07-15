"""Reusable Streamlit UI components and the app's CSS.

Kept separate from page logic so the visual language lives in one place. The
``dataframe`` helper merges caller kwargs over defaults (never passes duplicates
to st.dataframe) — the same collision-avoidance pattern used in charts.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.config import THEME
from utils.helpers import format_compact, format_percent

_SEVERITY = {
    "critical": ("#DC2626", "Critical"),
    "warning": ("#D97706", "Warning"),
    "positive": ("#059669", "Positive"),
    "info": ("#4F46E5", "Insight"),
}


def inject_css() -> None:
    st.markdown(f"""
    <style>
      .stApp {{ background: {THEME['surface_alt']}; }}
      .hero-title {{ font-size: 2.1rem; font-weight: 700; color: {THEME['text']};
                     letter-spacing: -0.02em; margin-bottom: 0.2rem; }}
      .hero-sub {{ font-size: 1rem; color: {THEME['text_muted']}; margin-bottom: 1.2rem; }}
      .kpi-card {{ background: white; border: 1px solid {THEME['border']};
                   border-radius: 12px; padding: 1rem 1.2rem; }}
      .kpi-label {{ font-size: 0.78rem; color: {THEME['text_muted']};
                    text-transform: uppercase; letter-spacing: 0.03em; }}
      .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {THEME['text']};
                    margin-top: 0.15rem; }}
      .insight-card {{ background: white; border-left: 4px solid {THEME['primary']};
                       border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
                       box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
      .insight-title {{ font-weight: 600; color: {THEME['text']}; font-size: 0.98rem; }}
      .insight-detail {{ color: {THEME['text_muted']}; font-size: 0.88rem; margin-top: 0.2rem; }}
      .badge {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
                font-size: 0.72rem; font-weight: 600; color: white; }}
      .stage-done {{ color: {THEME['success']}; }}
      .stage-active {{ color: {THEME['primary']}; font-weight: 600; }}
      .stage-pending {{ color: {THEME['text_muted']}; opacity: 0.5; }}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="hero-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi_cards(kpis: dict[str, Any]) -> None:
    """Render KPI cards from whatever metrics exist."""
    display: list[tuple[str, str]] = []
    order = [
        ("total_revenue", "Revenue", lambda v: format_compact(v)),
        ("total_profit", "Profit", lambda v: format_compact(v)),
        ("profit_margin", "Margin", lambda v: format_percent(v)),
        ("unique_customers", "Customers", lambda v: f"{int(v):,}"),
        ("total_orders", "Orders", lambda v: f"{int(v):,}"),
        ("avg_transaction", "Avg Txn", lambda v: format_compact(v)),
        ("rows", "Records", lambda v: f"{int(v):,}"),
        ("latest_month_growth", "MoM Growth", lambda v: format_percent(v)),
    ]
    for key, label, fmt in order:
        if key in kpis:
            display.append((label, fmt(kpis[key])))

    display = display[:6]
    if not display:
        return
    cols = st.columns(len(display))
    for col, (label, value) in zip(cols, display):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True)


def insight_card(insight: Any) -> None:
    color, badge = _SEVERITY.get(insight.severity, _SEVERITY["info"])
    st.markdown(
        f'<div class="insight-card" style="border-left-color:{color}">'
        f'<span class="badge" style="background:{color}">{badge}</span> '
        f'<span class="insight-title">{insight.title}</span>'
        f'<div class="insight-detail">{insight.detail}</div></div>',
        unsafe_allow_html=True)


def timeline(stages: list[str], current: int) -> str:
    """Render an HTML processing timeline; return the markup."""
    rows = []
    for i, stage in enumerate(stages):
        if i < current:
            rows.append(f'<div class="stage-done">✓ {stage}</div>')
        elif i == current:
            rows.append(f'<div class="stage-active">⟳ {stage}…</div>')
        else:
            rows.append(f'<div class="stage-pending">○ {stage}</div>')
    return "<div style='line-height:1.9;font-size:0.95rem'>" + "".join(rows) + "</div>"


def dataframe(frame: Any, **kwargs: Any) -> None:
    opts: dict[str, Any] = {"use_container_width": True, "hide_index": True}
    opts.update(kwargs)
    st.dataframe(frame, **opts)


def api_key_warning() -> None:
    st.info("This feature needs a Google Gemini API key. Add it in **Settings**, "
            "or set `GEMINI_API_KEY` in a `.env` file.")


__all__ = ["inject_css", "hero", "kpi_cards", "insight_card", "timeline",
           "dataframe", "api_key_warning"]
