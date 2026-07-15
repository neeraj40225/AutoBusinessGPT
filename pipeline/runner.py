"""The end-to-end analysis pipeline.

One object holds the state of an analysis run so the UI can drive it stage by
stage (showing the processing timeline) and hold the results across Streamlit
reruns. The stages mirror the spec's timeline: load → detect → (confirm) →
quality → clean → database → features → ML → insights.

Detection is separated from the rest because the user confirms the schema
between them. ``run_analysis`` covers everything *after* confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from detection.schema import SchemaMapping
from pipeline import cleaner, database, features, insights, quality
from ml.orchestrator import MLResults, run_all
from utils.logger import get_logger

logger = get_logger(__name__)

# Stages shown in the UI timeline (after schema confirmation).
STAGES: list[str] = [
    "Validating data quality",
    "Cleaning data",
    "Building database",
    "Engineering features",
    "Training ML models",
    "Generating insights",
]


@dataclass
class AnalysisState:
    """Holds every artifact produced by an analysis run."""

    raw: pd.DataFrame
    mapping: SchemaMapping
    quality_before: Any = None
    cleaned: pd.DataFrame | None = None
    cleaning_log: Any = None
    quality_after: Any = None
    column_map: dict[str, str] = field(default_factory=dict)
    features: features.FeatureSets | None = None
    ml: MLResults | None = None
    insights: insights.InsightReport | None = None
    completed: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "business_type": self.mapping.business_type,
            "rows": len(self.cleaned) if self.cleaned is not None else len(self.raw),
            "models_ran": self.ml.ran if self.ml else [],
            "features_built": self.features.built if self.features else [],
        }


ProgressCallback = Callable[[int, str], None]


def run_analysis(
    raw: pd.DataFrame,
    mapping: SchemaMapping,
    progress: ProgressCallback | None = None,
) -> AnalysisState:
    """Run the full post-confirmation pipeline.

    Args:
        raw: The uploaded (untouched) dataframe.
        mapping: The user-confirmed schema mapping.
        progress: Optional callback(stage_index, stage_name) for the UI timeline.

    Returns:
        A populated :class:`AnalysisState`.
    """
    state = AnalysisState(raw=raw, mapping=mapping)

    def step(i: int) -> None:
        if progress:
            progress(i, STAGES[i])

    # 1. quality (before)
    step(0)
    state.quality_before = quality.analyze(raw, mapping)

    # 2. clean
    step(1)
    state.cleaned, state.cleaning_log = cleaner.clean(raw, mapping)
    state.quality_after = quality.analyze(state.cleaned, mapping)

    # 3. database
    step(2)
    state.column_map = database.build_database(state.cleaned, mapping)

    # 4. features
    step(3)
    state.features = features.engineer(state.cleaned, mapping)

    # 5. ML
    step(4)
    state.ml = run_all(state.features)

    # 6. insights
    step(5)
    state.insights = insights.generate(state.cleaned, mapping, state.features, state.ml)

    state.completed = True
    logger.info("Analysis complete: %s", state.summary())
    return state


__all__ = ["run_analysis", "AnalysisState", "STAGES"]
