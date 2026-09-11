"""Tests for generation-related enums and shared schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from src.api.schemas.jobs import JobCreatedResponse
from src.core.enums import (
    GenerationType,
    JobStatus,
    MediaKind,
    ModelType,
    Provider,
)


class TestModelType:
    """Tests for ModelType enum."""

    def test_provider_property_aisha(self) -> None:
        """Test Aisha models return Aisha provider."""
        assert ModelType.AISHA_IMAGE.provider == Provider.AISHA

    def test_provider_property_grok(self) -> None:
        """Test Grok models return Grok provider."""
        assert ModelType.GROK_IMAGINE_IMAGE.provider == Provider.GROK
        assert ModelType.GROK_2_IMAGE.provider == Provider.GROK
        assert ModelType.GROK_IMAGINE_VIDEO.provider == Provider.GROK

    def test_output_media(self) -> None:
        """Test the media kinds each model can emit."""
        assert ModelType.GROK_IMAGINE_VIDEO.output_media == frozenset({MediaKind.VIDEO})
        assert ModelType.GROK_IMAGINE_IMAGE.output_media == frozenset({MediaKind.IMAGE})
        assert ModelType.GROK_2_IMAGE.output_media == frozenset({MediaKind.IMAGE})
        assert ModelType.AISHA_IMAGE.output_media == frozenset({MediaKind.IMAGE})


class TestGenerationType:
    """Tests for GenerationType enum."""

    def test_output_kind(self) -> None:
        """Test the media kind emitted by each generation type."""
        assert GenerationType.T2V.output_kind is MediaKind.VIDEO
        assert GenerationType.I2V.output_kind is MediaKind.VIDEO
        assert GenerationType.V2V.output_kind is MediaKind.VIDEO
        assert GenerationType.FLF2V.output_kind is MediaKind.VIDEO
        assert GenerationType.T2I.output_kind is MediaKind.IMAGE
        assert GenerationType.I2I.output_kind is MediaKind.IMAGE


class TestJobCreatedResponse:
    """Tests for JobCreatedResponse schema."""

    def test_valid_response(self) -> None:
        """Test valid response creation."""
        job_id = UUID("12345678-1234-1234-1234-123456789012")
        response = JobCreatedResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            name="Test Job",
            model=ModelType.GROK_IMAGINE_IMAGE.value,
            generation_type=GenerationType.T2I,
            created_at=datetime.now(UTC),
            message="Generation completed",
        )
        assert response.job_id == job_id
        assert response.status == JobStatus.COMPLETED
        assert response.message == "Generation completed"
