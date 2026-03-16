"""Tests for generation-related enums and shared schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from src.api.schemas.jobs import JobCreatedResponse
from src.core.enums import (
    GenerationType,
    JobStatus,
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

    def test_supports_image_input(self) -> None:
        """Test supports_image_input property."""
        assert ModelType.GROK_IMAGINE_IMAGE.supports_image_input is True
        assert ModelType.GROK_2_IMAGE.supports_image_input is False
        assert ModelType.GROK_IMAGINE_VIDEO.supports_image_input is True
        assert ModelType.AISHA_IMAGE.supports_image_input is True

    def test_is_video_model(self) -> None:
        """Test is_video_model property."""
        assert ModelType.GROK_IMAGINE_VIDEO.is_video_model is True
        assert ModelType.GROK_IMAGINE_IMAGE.is_video_model is False
        assert ModelType.GROK_2_IMAGE.is_video_model is False
        assert ModelType.AISHA_IMAGE.is_video_model is False


class TestGenerationType:
    """Tests for GenerationType enum."""

    def test_is_video(self) -> None:
        """Test is_video property."""
        assert GenerationType.T2V.is_video is True
        assert GenerationType.I2V.is_video is True
        assert GenerationType.V2V.is_video is True
        assert GenerationType.FLF2V.is_video is True
        assert GenerationType.T2I.is_video is False
        assert GenerationType.I2I.is_video is False

    def test_requires_image_input(self) -> None:
        """Test requires_image_input property."""
        assert GenerationType.I2I.requires_image_input is True
        assert GenerationType.I2V.requires_image_input is True
        assert GenerationType.FLF2V.requires_image_input is True
        assert GenerationType.T2I.requires_image_input is False
        assert GenerationType.T2V.requires_image_input is False
        assert GenerationType.V2V.requires_image_input is False

    def test_requires_video_input(self) -> None:
        """Test requires_video_input property."""
        assert GenerationType.V2V.requires_video_input is True
        assert GenerationType.T2I.requires_video_input is False
        assert GenerationType.I2I.requires_video_input is False
        assert GenerationType.T2V.requires_video_input is False
        assert GenerationType.I2V.requires_video_input is False


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
