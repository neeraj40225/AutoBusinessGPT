"""Render a completed analysis as an interactive dashboard.

Every panel is conditional on what the pipeline actually produced — a dataset
with no customers shows no churn tab, and the UI says why rather than showing an
empty chart. This is the honest-degradation principle surfaced to the user.
"""

from __future__ import annotations

import streamlit as st

from core.config import Role
from dashboard import charts, components
from pipeline.runner import AnalysisState
from sql_agent import copilot
from utils import llm


def render(state: AnalysisState) -> None:
    """Render the full results dashboard for a completed analysis."""
    ins = state.insights
    components.hero(
        f"{state.mapping.business_type} Analysis",
        f"{len(state.cleaned):,} records · "
        f"{len(state.ml.ran)} models · source: {state.mapping.source}",
    )

    if ins:
        components.kpi_cards(ins.kpis)
    st.markdown("###")

    tabs = st.tabs(["Overview", "Insights", "Forecast", "Customers", "Copilot", "Data"])

    with tabs[0]:
        _overview(state)
    with tabs[1]:
        _insights(state)
    with tabs[2]:
        _forecast(state)
    with tabs[3]:
        _customers(state)
    with tabs[4]:
        _copilot(state)
    with tabs[5]:
        _data(state)


def _overview(state: AnalysisState) -> None:
    fs = state.features
    ins = state.insights

    if ins and ins.narrative:
        st.markdown("#### Executive Summary")
        st.write(ins.narrative)
        st.markdown("###")

    col1, col2 = st.columns(2)
    with col1:
        if fs and fs.has_monthly():
            st.plotly_chart(charts.revenue_trend(fs.monthly), use_container_width=True)
        else:
            st.caption("No time series — needs a date and revenue column.")
    with col2:
        if fs and fs.has_product():
            st.plotly_chart(
                charts.category_bars(fs.product, "product", "revenue", "Top Products"),
                use_container_width=True)
        else:
            st.caption("No product breakdown — needs a product column.")

    # correlation across numeric features if we have customer features
    if fs and fs.has_customer():
        numeric = ["recency_days", "frequency", "monetary", "avg_order_value", "orders_per_year"]
        st.plotly_chart(charts.correlation_heatmap(fs.customer, numeric), use_container_width=True)


def _insights(state: AnalysisState) -> None:
    ins = state.insights
    if not ins or not ins.insights:
        st.caption("No insights generated.")
        return
    for insight in ins.insights:
        components.insight_card(insight)

    # quality summary
    if state.quality_after:
        q = state.quality_after
        st.markdown("###")
        st.markdown("#### Data Quality")
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality Score", f"{q.score:.0f}/100", q.grade)
        c2.metric("Rows Removed", state.cleaning_log.rows_removed if state.cleaning_log else 0)
        c3.metric("Issues Found", len(state.quality_before.issues) if state.quality_before else 0)
        if state.cleaning_log and state.cleaning_log.actions:
            with st.expander("What cleaning did"):
                for a in state.cleaning_log.actions:
                    st.write(f"- {a}")


def _forecast(state: AnalysisState) -> None:
    ml = state.ml
    if ml.forecast is None:
        reason = ml.skipped.get("forecast", "not enough time-series data")
        components.hero("", "")
        st.info(f"Forecasting was not run: {reason}")
        return

    fc = ml.forecast
    st.success(f"Best model: **{fc.best_name}** (selected on held-out RMSE)")
    st.plotly_chart(charts.forecast_chart(fc.history, fc.forecast), use_container_width=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("##### Model comparison")
        components.dataframe(fc.scores_frame)
    with c2:
        st.markdown("##### Projected months")
        components.dataframe(fc.forecast.round(0))


def _customers(state: AnalysisState) -> None:
    ml = state.ml
    if ml.churn is None and ml.segmentation is None:
        reason = ml.skipped.get("churn") or ml.skipped.get("segmentation") or "no customer data"
        st.info(f"Customer models were not run: {reason}")
        return

    if ml.segmentation is not None:
        seg = ml.segmentation
        st.markdown(f"#### Segmentation — {seg.best_k} segments (silhouette {seg.silhouette:.2f})")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.segment_scatter(seg.projection), use_container_width=True)
        with c2:
            st.plotly_chart(charts.segment_bars(seg.profiles), use_container_width=True)
        components.dataframe(seg.profiles)

    if ml.churn is not None:
        ch = ml.churn
        st.markdown("###")
        st.markdown(f"#### Churn — {ch.best_name} (AUC {ch.best_score.roc_auc:.3f})")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                charts.roc_chart(ch.roc_points["fpr"], ch.roc_points["tpr"], ch.best_score.roc_auc),
                use_container_width=True)
        with c2:
            st.plotly_chart(charts.churn_risk_bars(ch.scored), use_container_width=True)
        if not ch.importance.empty:
            st.plotly_chart(charts.importance_bars(ch.importance, "What drives churn"),
                            use_container_width=True)
        st.markdown("##### Highest-risk customers")
        top = ch.scored.sort_values("churn_probability", ascending=False).head(50)
        components.dataframe(top.round(3))


def _copilot(state: AnalysisState) -> None:
    st.markdown("#### Business Copilot")
    st.caption("Ask a question in plain English. It writes SQL, runs it read-only, and answers.")

    if not llm.is_available():
        components.api_key_warning()
        return

    history = st.session_state.setdefault("copilot_history", [])
    for resp in history:
        with st.chat_message("user"):
            st.write(resp.question)
        with st.chat_message("assistant"):
            if resp.ok:
                st.write(resp.answer)
                with st.expander(f"SQL · {resp.row_count} rows · {resp.elapsed_ms:.0f}ms"):
                    st.code(resp.sql, language="sql")
                if not resp.frame.empty:
                    components.dataframe(resp.frame.head(50))
            else:
                st.error(resp.error)

    q = st.chat_input("e.g. Which category has the highest profit margin?")
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = copilot.ask(q, state.column_map)
            if resp.ok:
                st.write(resp.answer)
                with st.expander(f"SQL · {resp.row_count} rows · {resp.elapsed_ms:.0f}ms"):
                    st.code(resp.sql, language="sql")
                if not resp.frame.empty:
                    components.dataframe(resp.frame.head(50))
            else:
                st.error(resp.error)
        history.append(resp)


def _data(state: AnalysisState) -> None:
    st.markdown("#### Cleaned Data")
    st.caption(f"{len(state.cleaned):,} rows after cleaning")
    components.dataframe(state.cleaned.head(200))

    st.markdown("##### Detected schema")
    schema_rows = [cm.to_dict() for cm in state.mapping.columns]
    import pandas as pd
    components.dataframe(pd.DataFrame(schema_rows))


__all__ = ["render"]
