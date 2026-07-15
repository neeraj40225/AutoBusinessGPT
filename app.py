"""AutoBusinessGPT — Streamlit application entry point.

Run with:  streamlit run app.py

Flow: upload → detect schema → confirm/correct mapping → run analysis (with a
live processing timeline) → interactive results dashboard, downloads, settings.
All state is held in st.session_state so Streamlit reruns don't lose progress.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st
  

from core.config import BusinessType, settings
from dashboard import components, results
from detection.detector import detect_schema
from detection.schema import ROLE_CHOICES, SchemaMapping
from pipeline import runner
from pipeline.loader import load_dataframe
from report.report_builder import build_report
from utils import llm
from utils.helpers import AutoBusinessError, UnsupportedFileError
from utils.logger import get_logger

logger = get_logger(__name__)


def _init_state() -> None:
    st.session_state.setdefault("stage", "upload")   # upload | confirm | results
    st.session_state.setdefault("raw", None)
    st.session_state.setdefault("mapping", None)
    st.session_state.setdefault("analysis", None)


def _reset() -> None:
    for key in ("stage", "raw", "mapping", "analysis", "copilot_history"):
        st.session_state.pop(key, None)
    _init_state()


def _sidebar() -> str:
    st.sidebar.markdown(f"### {settings.app_name}")
    st.sidebar.caption(settings.tagline)
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["Analyze", "Downloads", "Settings"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    if llm.is_available():
        st.sidebar.caption(f"🟢 Gemini · `{settings.llm.model}`")
    else:
        st.sidebar.caption("🔴 Gemini not configured")

    if st.session_state.get("analysis") is not None:
        st.sidebar.divider()
        if st.sidebar.button("↺ New analysis", use_container_width=True):
            _reset()
            st.rerun()

    return str(page)


# --------------------------------------------------------------------------- #
# Stage 1: upload
# --------------------------------------------------------------------------- #
def _page_upload() -> None:
    components.hero(settings.app_name, settings.tagline)

    st.markdown("Upload a business dataset (CSV or Excel) and the app will detect "
                "its structure, clean it, model it, and explain it — automatically.")

    file = st.file_uploader("Drag & drop your dataset", type=["csv", "xlsx", "xls", "tsv"])

    if file is not None:
        if st.button("Analyze Business", type="primary", use_container_width=True):
            try:
                with st.spinner("Reading dataset…"):
                    frame = load_dataframe(file)
                st.session_state["raw"] = frame
                with st.spinner("Detecting schema (Gemini + heuristics)…"):
                    mapping = detect_schema(frame)
                st.session_state["mapping"] = mapping
                st.session_state["stage"] = "confirm"
                st.rerun()
            except UnsupportedFileError as exc:
                st.error(f"Could not read that file: {exc}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Upload failed")
                st.error(f"Something went wrong: {exc}")

    with st.expander("What happens after I click Analyze?"):
        st.markdown(
            "1. **Detect** — columns are mapped to roles (customer, revenue, date…) "
            "and a business type is proposed.\n"
            "2. **Confirm** — you review and correct the mapping.\n"
            "3. **Analyze** — validate → clean → build database → engineer features "
            "→ train the models the data supports → generate insights.\n"
            "4. **Explore** — dashboards, a SQL copilot, and a downloadable report.\n\n"
            "Models only run when the data supports them; the app tells you what it "
            "skipped and why."
        )


# --------------------------------------------------------------------------- #
# Stage 2: confirm schema
# --------------------------------------------------------------------------- #
def _page_confirm() -> None:
    mapping: SchemaMapping = st.session_state["mapping"]
    frame: pd.DataFrame = st.session_state["raw"]

    components.hero("Confirm the detected schema",
                    f"Detected via {mapping.source}. Review the mappings, then analyze.")

    # business type
    types = list(BusinessType.ALL)
    default_idx = types.index(mapping.business_type) if mapping.business_type in types else len(types) - 1
    chosen_type = st.selectbox(
        f"Business type  (detected: {mapping.business_type}, "
        f"{mapping.business_type_confidence:.0%} confident)",
        types, index=default_idx)

    st.markdown("#### Column roles")
    if mapping.low_confidence:
        st.warning(f"{len(mapping.low_confidence)} column(s) were uncertain — "
                   "please check the highlighted rows.")

    st.caption("Adjust any role below. Set to (unmapped) to exclude a column from analysis.")

    overrides: dict[str, str] = {}
    low_conf_cols = {cm.column for cm in mapping.low_confidence}

    for cm in mapping.columns:
        cols = st.columns([3, 3, 2, 4])
        flag = "⚠️ " if cm.column in low_conf_cols else ""
        cols[0].markdown(f"{flag}**{cm.column}**")
        current = cm.role or "(unmapped)"
        idx = ROLE_CHOICES.index(current) if current in ROLE_CHOICES else 0
        new_role = cols[1].selectbox(
            "role", ROLE_CHOICES, index=idx,
            key=f"role_{cm.column}", label_visibility="collapsed")
        overrides[cm.column] = new_role
        cols[2].caption(f"{cm.confidence:.0%}")
        sample = ", ".join(str(v) for v in cm.sample_values[:2])
        cols[3].caption(f"e.g. {sample}" if sample else cm.reason)

    # validate: warn on duplicate role assignments (ignore unmapped/None)
    assigned = [r for r in overrides.values() if r and r != "(unmapped)"]
    dupes = sorted({r for r in assigned if assigned.count(r) > 1})
    if dupes:
        st.error(f"Each role can map to only one column. Duplicated: {', '.join(dupes)}. "
                 "Fix before analyzing.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state["stage"] = "upload"
            st.rerun()
    with col2:
        if st.button("Confirm & Analyze", type="primary", use_container_width=True,
                     disabled=bool(dupes)):
            mapping.apply_overrides(overrides)
            mapping.business_type = chosen_type
            _run_analysis(frame, mapping)


def _run_analysis(frame: pd.DataFrame, mapping: SchemaMapping) -> None:
    """Run the pipeline with a live timeline, then move to results."""
    placeholder = st.empty()
    progress_bar = st.progress(0.0)

    def on_progress(i: int, name: str) -> None:
        placeholder.markdown(components.timeline(runner.STAGES, i), unsafe_allow_html=True)
        progress_bar.progress((i + 1) / len(runner.STAGES))
        time.sleep(0.15)  # let the UI paint each stage

    try:
        state = runner.run_analysis(frame, mapping, progress=on_progress)
        placeholder.markdown(components.timeline(runner.STAGES, len(runner.STAGES)),
                             unsafe_allow_html=True)
        progress_bar.progress(1.0)
        st.session_state["analysis"] = state
        st.session_state["stage"] = "results"
        time.sleep(0.3)
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed")
        st.error(f"Analysis failed: {exc}")


# --------------------------------------------------------------------------- #
# Stage 3: results
# --------------------------------------------------------------------------- #
def _page_results() -> None:
    state = st.session_state["analysis"]
    results.render(state)


# --------------------------------------------------------------------------- #
# Other pages
# --------------------------------------------------------------------------- #
def _page_downloads() -> None:
    components.hero("Downloads", "Export your analysis")
    state = st.session_state.get("analysis")
    if state is None:
        st.info("Run an analysis first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### PDF Report")
        if st.button("Generate report", type="primary"):
            with st.spinner("Building PDF…"):
                buf = build_report(state)
            st.session_state["report_pdf"] = buf.getvalue()
        if st.session_state.get("report_pdf"):
            st.download_button("Download PDF", st.session_state["report_pdf"],
                               file_name="autobusiness_report.pdf", mime="application/pdf")

    with col2:
        st.markdown("##### Cleaned Data")
        csv = state.cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", csv,
                           file_name="cleaned_data.csv", mime="text/csv")

    # feature exports
    if state.features:
        st.markdown("##### Engineered Features")
        fs = state.features
        fcols = st.columns(3)
        if fs.has_customer():
            fcols[0].download_button("Customer features",
                fs.customer.to_csv(index=False).encode("utf-8"),
                file_name="customer_features.csv", mime="text/csv")
        if fs.has_monthly():
            fcols[1].download_button("Monthly series",
                fs.monthly.to_csv(index=False).encode("utf-8"),
                file_name="monthly_series.csv", mime="text/csv")
        if fs.has_product():
            fcols[2].download_button("Product features",
                fs.product.to_csv(index=False).encode("utf-8"),
                file_name="product_features.csv", mime="text/csv")


def _page_settings() -> None:
    components.hero("Settings", "Configure the AI backend")

    st.markdown("#### Gemini API Key")
    st.caption("Stored for this session only. For a permanent key, use a `.env` file. "
               "Without a key the app runs on offline heuristics and template narratives.")
    key = st.text_input("API key", type="password",
                        placeholder="Configured via .env" if settings.llm.is_configured else "Paste key…")
    if st.button("Save key"):
        st.session_state["gemini_api_key"] = key.strip()
        llm.reset_client()
        st.success("Saved for this session.")
        st.rerun()

    st.metric("LLM status", "Connected" if llm.is_available() else "Not configured")
    st.metric("Model", settings.llm.model)

    st.divider()
    st.markdown("#### Detection strategy")
    st.write(f"Current: `{settings.detection.strategy}` "
             "(Gemini-first with heuristic fallback)")
    st.caption("Set DETECTION_STRATEGY in .env to `heuristic_only` or `gemini_only`.")

    st.divider()
    st.markdown("#### Optional features")
    from rag import document_chat
    st.write(f"Document chat (RAG): {'✅ available' if document_chat.is_available() else '❌ needs sentence-transformers + faiss-cpu'}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title=f"{settings.app_name}", page_icon="📊",
                       layout="wide", initial_sidebar_state="expanded")
    components.inject_css()
    _init_state()

    page = _sidebar()

    if page == "Downloads":
        _page_downloads()
    elif page == "Settings":
        _page_settings()
    else:  # Analyze
        stage = st.session_state["stage"]
        if stage == "upload":
            _page_upload()
        elif stage == "confirm":
            _page_confirm()
        elif stage == "results":
            _page_results()


if __name__ == "__main__":
    main()
