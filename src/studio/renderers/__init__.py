"""Target renderers. ``RENDERERS`` is the registry the export service dispatches on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.core.enums import StudioExportTarget
from src.studio.renderers.suno import SunoRenderer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.studio.renderers.base import TargetRenderer

__all__ = ["RENDERERS"]

RENDERERS: Final[Mapping[StudioExportTarget, TargetRenderer]] = {
    StudioExportTarget.SUNO: SunoRenderer(),
}
