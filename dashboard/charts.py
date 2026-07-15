"""Plotly chart builders for the dashboard.

Charts are selected dynamically from what the pipeline produced — there is no
fixed set. Each builder takes a dataframe and returns a Figure, or an empty-state
figure when the data isn't there.

Layout note (a lesson learned the hard way): ``_layout`` accepts **kwargs and
merges caller overrides *into* the base dict, so a builder can override
``showlegend`` or ``xaxis`` without passing a duplicate keyword to
``update_layout`` — which raises a TypeError.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from core.config import PALETTE, THEME

_FONT = "Inter, -apple-system, Segoe UI, sans-serif"


def _layout(title: str = "", height: int = 360, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": {"text": title, "font": {"size": 15, "color": THEME["text"], "family": _FONT},
                  "x": 0.01, "xanchor": "left"},
        "height": height,
        "margin": {"l": 12, "r": 16, "t": 44 if title else 12, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": _FONT, "size": 11, "color": THEME["text"]},
        "showlegend": False,
        "xaxis": {"showgrid": False, "linecolor": THEME["border"],
                  "tickfont": {"size": 10, "color": THEME["text_muted"]}},
        "yaxis": {"showgrid": True, "gridcolor": THEME["border"], "zeroline": False,
                  "tickfont": {"size": 10, "color": THEME["text_muted"]}},
    }
    base.update(overrides)
    return base


def empty(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font={"size": 13, "color": THEME["text_muted"]},
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(**_layout(height=280))
    return fig


def revenue_trend(monthly: pd.DataFrame) -> go.Figure:
    if monthly.empty:
        return empty("No time series available")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["sales"], mode="lines+markers",
        line={"color": THEME["primary"], "width": 2.5},
        marker={"size": 5}, name="Revenue",
        fill="tozeroy", fillcolor="rgba(79,70,229,0.08)"))
    if "roll_mean_3" in monthly.columns:
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["roll_mean_3"], mode="lines",
            line={"color": THEME["accent"], "width": 1.5, "dash": "dot"},
            name="3-mo avg"))
    fig.update_layout(**_layout("Revenue Trend", showlegend=True,
                                legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"}))
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    if history.empty or forecast.empty:
        return empty("Train the forecast to see projections")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(forecast["month"]) + list(forecast["month"])[::-1],
        y=list(forecast["upper"]) + list(forecast["lower"])[::-1],
        fill="toself", fillcolor="rgba(14,165,233,0.12)",
        line={"color": "rgba(0,0,0,0)"}, name="95% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=history["month"], y=history["sales"], mode="lines",
        line={"color": THEME["primary"], "width": 2.5}, name="Actual"))
    bridge_x = [history["month"].iloc[-1]] + list(forecast["month"])
    bridge_y = [history["sales"].iloc[-1]] + list(forecast["forecast"])
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y, mode="lines+markers",
        line={"color": THEME["accent"], "width": 2.5, "dash": "dash"},
        marker={"size": 5}, name="Forecast"))
    fig.update_layout(**_layout("Sales Forecast", showlegend=True,
                                legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"}))
    return fig


def category_bars(product_or_cat: pd.DataFrame, label_col: str, value_col: str = "revenue",
                  title: str = "Top Categories") -> go.Figure:
    if product_or_cat.empty or value_col not in product_or_cat.columns:
        return empty()
    top = product_or_cat.nlargest(10, value_col)
    fig = go.Figure(go.Bar(
        x=top[value_col], y=top[label_col].astype(str), orientation="h",
        marker={"color": THEME["primary"]}))
    fig.update_layout(**_layout(title, yaxis={"autorange": "reversed", "showgrid": False,
                                              "tickfont": {"size": 10, "color": THEME["text_muted"]}}))
    return fig


def segment_scatter(projection: pd.DataFrame) -> go.Figure:
    if projection.empty:
        return empty()
    fig = go.Figure()
    order = ["VIP", "Regular", "Occasional", "At Risk"]
    labels = [l for l in order if l in projection["segment_label"].unique()]
    labels += [l for l in projection["segment_label"].unique() if l not in labels]
    for i, label in enumerate(labels):
        sub = projection[projection["segment_label"] == label]
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers", name=str(label),
            marker={"size": 6, "color": PALETTE[i % len(PALETTE)], "opacity": 0.7}))
    fig.update_layout(**_layout("Customer Segments", showlegend=True,
                                legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"},
                                xaxis={"showgrid": False, "visible": False},
                                yaxis={"showgrid": False, "visible": False}))
    return fig


def segment_bars(profiles: pd.DataFrame) -> go.Figure:
    if profiles.empty or "count" not in profiles.columns:
        return empty()
    fig = go.Figure(go.Bar(
        x=profiles["segment_label"].astype(str), y=profiles["count"],
        marker={"color": [PALETTE[i % len(PALETTE)] for i in range(len(profiles))]}))
    fig.update_layout(**_layout("Customers per Segment"))
    return fig


def churn_risk_bars(scored: pd.DataFrame) -> go.Figure:
    if scored.empty:
        return empty()
    order = ["Low", "Medium", "High", "Critical"]
    colors = {"Low": THEME["success"], "Medium": THEME["warning"],
              "High": "#EA580C", "Critical": THEME["danger"]}
    counts = scored["risk_band"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        x=order, y=counts.values,
        marker={"color": [colors[b] for b in order]}))
    fig.update_layout(**_layout("Churn Risk Distribution"))
    return fig


def roc_chart(fpr: list[float], tpr: list[float], auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line={"color": THEME["border"], "dash": "dash"}, name="No skill"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy",
                             fillcolor="rgba(79,70,229,0.1)",
                             line={"color": THEME["primary"], "width": 2.5},
                             name=f"ROC (AUC {auc:.3f})"))
    fig.update_layout(**_layout(f"ROC Curve — AUC {auc:.3f}", showlegend=True,
                                legend={"x": 0.98, "y": 0.06, "xanchor": "right"},
                                xaxis={"title": {"text": "FPR", "font": {"size": 10}}, "range": [0, 1]},
                                yaxis={"title": {"text": "TPR", "font": {"size": 10}}, "range": [0, 1.02]}))
    return fig


def importance_bars(importance: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    if importance.empty:
        return empty()
    top = importance.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"], orientation="h",
        marker={"color": THEME["accent"]}))
    fig.update_layout(**_layout(title, yaxis={"showgrid": False,
                                             "tickfont": {"size": 10, "color": THEME["text_muted"]}}))
    return fig


def correlation_heatmap(frame: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    cols = [c for c in numeric_cols if c in frame.columns][:12]
    if len(cols) < 2:
        return empty("Not enough numeric columns for correlation")
    corr = frame[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols, colorscale="RdBu", zmid=0,
        zmin=-1, zmax=1, colorbar={"thickness": 12}))
    fig.update_layout(**_layout("Correlation Matrix", height=420,
                                xaxis={"showgrid": False}, yaxis={"showgrid": False}))
    return fig


__all__ = [
    "empty", "revenue_trend", "forecast_chart", "category_bars",
    "segment_scatter", "segment_bars", "churn_risk_bars", "roc_chart",
    "importance_bars", "correlation_heatmap",
]
