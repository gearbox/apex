"""Tests for the providers discovery endpoint."""

from __future__ import annotations

import msgspec

from src.api.routes.providers import ProviderModelInfo, ProvidersResponse
from src.core.enums import GenerationType, ModelType


class TestProvidersResponseSchema:
    def test_roundtrip(self) -> None:
        resp = ProvidersResponse(
            providers=[],
            models=[],
        )
        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=ProvidersResponse)
        assert decoded.providers == []
        assert decoded.models == []

    def test_model_info_fields(self) -> None:
        info = ProviderModelInfo(
            model="grok-imagine-image",
            name="Grok Imagine",
            description="T2I and I2I",
            provider="grok",
            is_enabled=True,
            supports_t2i=True,
            supports_i2i=True,
            supports_t2v=False,
            supports_i2v=False,
            supports_v2v=False,
            supports_flf2v=False,
            max_images=10,
        )
        assert info.supports_t2i is True
        assert info.supports_t2v is False


class TestBuildModelInfoConsistency:
    """Verify that _build_model_info uses enum properties consistently."""

    def test_grok_imagine_image_capabilities(self) -> None:
        mt = ModelType.GROK_IMAGINE_IMAGE
        assert mt.supports_generation_type(GenerationType.T2I) is True
        assert mt.supports_generation_type(GenerationType.I2I) is True
        assert mt.supports_generation_type(GenerationType.T2V) is False
        assert mt.max_concurrent_outputs == 10

    def test_aisha_capabilities(self) -> None:
        mt = ModelType.AISHA
        assert mt.supports_generation_type(GenerationType.T2I) is True
        assert mt.supports_generation_type(GenerationType.I2I) is True
        assert mt.supports_generation_type(GenerationType.T2V) is False
        assert mt.max_concurrent_outputs == 4
