"""Schema-detection result types — the contract every detector returns.

A :class:`SchemaMapping` is the single object the confirmation UI renders and
the pipeline consumes. Detectors (heuristic or Gemini) differ only in how they
populate it; downstream code never knows which ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config import Role


@dataclass
class ColumnMapping:
    """One column's detected role and the confidence behind it."""

    column: str
    role: str | None            # a Role.* value, or None if unmapped
    confidence: float           # 0.0–1.0
    reason: str = ""            # short human explanation, shown on confirm screen
    dtype: str = ""             # observed pandas dtype
    sample_values: list[Any] = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        """True when confidence is low enough to warrant a human glance."""
        from core.config import settings
        return self.role is not None and self.confidence < settings.detection.confirm_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "role": self.role or "(unmapped)",
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "dtype": self.dtype,
        }


@dataclass
class SchemaMapping:
    """The full detected schema for an uploaded dataset."""

    columns: list[ColumnMapping]
    business_type: str
    business_type_confidence: float
    business_type_reason: str = ""
    source: str = "heuristic"   # "gemini" or "heuristic" — provenance for the UI
    n_rows: int = 0

    # -- role lookups --------------------------------------------------------
    def role_to_column(self, role: str) -> str | None:
        """Return the column assigned to a role, or None. First wins on ties."""
        for cm in self.columns:
            if cm.role == role:
                return cm.column
        return None

    def column_to_role(self, column: str) -> str | None:
        for cm in self.columns:
            if cm.column == column:
                return cm.role
        return None

    def has_role(self, role: str) -> bool:
        return self.role_to_column(role) is not None

    def mapped_roles(self) -> set[str]:
        return {cm.role for cm in self.columns if cm.role}

    @property
    def low_confidence(self) -> list[ColumnMapping]:
        return [cm for cm in self.columns if cm.needs_confirmation]

    def apply_overrides(self, overrides: dict[str, str | None]) -> None:
        """Apply user corrections from the confirm screen.

        Args:
            overrides: {column_name: new_role_or_None}. A role set to None or the
                sentinel "(unmapped)" clears the mapping. Confidence is forced to
                1.0 for any human-confirmed column.
        """
        for cm in self.columns:
            if cm.column in overrides:
                new = overrides[cm.column]
                cm.role = None if new in (None, "(unmapped)", "") else new
                cm.confidence = 1.0
                cm.reason = "Confirmed by user"

    def summary(self) -> dict[str, Any]:
        return {
            "business_type": self.business_type,
            "source": self.source,
            "columns_mapped": len(self.mapped_roles()),
            "columns_total": len(self.columns),
            "low_confidence": len(self.low_confidence),
        }


# valid override targets for the UI dropdown
ROLE_CHOICES: list[str] = ["(unmapped)", *Role.ALL]

__all__ = ["ColumnMapping", "SchemaMapping", "ROLE_CHOICES"]
