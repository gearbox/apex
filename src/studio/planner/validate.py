"""A drafted plan gets exactly the checks a user-submitted plan gets (P2a-7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.studio.validation import normalize_plan, validate_built_document

if TYPE_CHECKING:
    from src.studio.documents import PlanDocument
    from src.studio.errors import DocumentIssue

__all__ = ["validate_plan"]


def validate_plan(plan: PlanDocument) -> tuple[PlanDocument | None, tuple[DocumentIssue, ...]]:
    """Validate and normalize a plan built from a draft.

    Returns:
        ``(normalized, ())`` when valid, else ``(None, issues)`` with every issue (paths and codes
        only - never plan text). Never raises for validation.
    """
    return validate_built_document(plan, normalize_plan)
