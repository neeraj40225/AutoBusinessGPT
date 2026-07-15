"""ML orchestration — decide which models can run, then run them.

The gating rule is explicit and inspectable, not delegated to an LLM: a model
runs iff the feature family it needs exists and has enough rows. This is the
deliberate design choice — a fixed rule fails loudly and identically every run,
where an LLM deciding "should I run churn?" would be non-deterministic and
untestable.

Each model returns a typed result object or None (with a recorded reason). The
dashboard renders whatever ran and states what didn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import settings
from pipeline.features import FeatureSets
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLResults:
    """Container for every model's output plus skip reasons."""

    forecast: Any = None          # ml.forecasting.ForecastResult | None
    churn: Any = None             # ml.churn.ChurnResult | None
    segmentation: Any = None      # ml.segmentation.SegmentationResult | None
    ran: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"ran": self.ran, "skipped": self.skipped}


def _can_forecast(fs: FeatureSets) -> str | None:
    if not fs.has_monthly():
        return fs.skipped.get("monthly", "no monthly series")
    if len(fs.monthly) < 6:
        return f"only {len(fs.monthly)} months (need ≥6)"
    return None


def _can_churn(fs: FeatureSets) -> str | None:
    if not fs.has_customer():
        return fs.skipped.get("customer", "no customer features")
    if len(fs.customer) < settings.ml.min_rows_for_ml:
        return f"only {len(fs.customer)} customers (need ≥{settings.ml.min_rows_for_ml})"
    # churn needs repeat behaviour to be meaningful
    repeat = (fs.customer["frequency"] > 1).sum()
    if repeat < 20:
        return f"only {repeat} repeat customers — churn signal too weak"
    return None


def _can_segment(fs: FeatureSets) -> str | None:
    if not fs.has_customer():
        return fs.skipped.get("customer", "no customer features")
    if len(fs.customer) < settings.ml.min_rows_for_ml:
        return f"only {len(fs.customer)} customers (need ≥{settings.ml.min_rows_for_ml})"
    return None


def run_all(fs: FeatureSets) -> MLResults:
    """Train every model whose gate passes; record why the rest were skipped."""
    from ml import churn, forecasting, segmentation

    results = MLResults()

    plan: list[tuple[str, Callable[[FeatureSets], str | None], Callable[[FeatureSets], Any]]] = [
        ("forecast", _can_forecast, lambda f: forecasting.train(f.monthly)),
        ("churn", _can_churn, lambda f: churn.train(f.customer)),
        ("segmentation", _can_segment, lambda f: segmentation.train(f.customer)),
    ]

    for name, gate, runner in plan:
        reason = gate(fs)
        if reason:
            results.skipped[name] = reason
            logger.info("Skipping %s: %s", name, reason)
            continue
        try:
            out = runner(fs)
            setattr(results, name, out)
            results.ran.append(name)
            logger.info("Model %s trained.", name)
        except Exception as exc:  # noqa: BLE001 - one model failing must not kill the rest
            results.skipped[name] = f"training error: {exc}"
            logger.exception("Model %s failed", name)

    return results


__all__ = ["MLResults", "run_all"]
