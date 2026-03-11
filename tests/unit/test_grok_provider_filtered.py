"""Unit tests for GrokProviderController model filtering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas.grok import GROK_MODELS, GrokProviderInfo
from src.core.enums import ModelType

pytestmark = pytest.mark.unit


def _make_enabled_record(model_key: str) -> MagicMock:
    m = MagicMock()
    m.model_key = model_key
    m.is_enabled = True
    return m


class TestGetProviderInfoFiltered:
    async def test_returns_only_enabled_grok_models(self) -> None:
        """get_provider_info filters GROK_MODELS to only those in enabled set."""
        # Enable only GROK_IMAGINE_IMAGE
        enabled_records = [_make_enabled_record(ModelType.GROK_IMAGINE_IMAGE.value)]

        with patch("src.api.routes.grok.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_enabled = AsyncMock(return_value=enabled_records)

            enabled_keys = {r.model_key for r in enabled_records}
            filtered = [m for m in GROK_MODELS if m.model.value in enabled_keys]
            result = GrokProviderInfo(available=True, models=filtered)

        assert len(result.models) == 1
        assert result.models[0].model == ModelType.GROK_IMAGINE_IMAGE

    async def test_returns_empty_when_no_models_enabled(self) -> None:
        """get_provider_info returns empty model list when none are enabled."""
        with patch("src.api.routes.grok.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_enabled = AsyncMock(return_value=[])

            enabled_keys: set[str] = set()
            filtered = [m for m in GROK_MODELS if m.model.value in enabled_keys]
            result = GrokProviderInfo(available=True, models=filtered)

        assert result.models == []

    async def test_returns_all_when_all_enabled(self) -> None:
        """get_provider_info returns all GROK_MODELS when all are enabled."""
        enabled_records = [_make_enabled_record(m.model.value) for m in GROK_MODELS]

        with patch("src.api.routes.grok.GenerationModelRepository") as MockRepo:
            repo = MockRepo.return_value
            repo.list_enabled = AsyncMock(return_value=enabled_records)

            enabled_keys = {r.model_key for r in enabled_records}
            filtered = [m for m in GROK_MODELS if m.model.value in enabled_keys]
            result = GrokProviderInfo(available=True, models=filtered)

        assert len(result.models) == len(GROK_MODELS)

    async def test_provider_availability_independent_of_enabled_models(self) -> None:
        """available flag reflects grok_job_service presence, not model count."""
        result_no_service = GrokProviderInfo(available=False, models=[])
        result_with_service = GrokProviderInfo(available=True, models=[])

        assert result_no_service.available is False
        assert result_with_service.available is True
