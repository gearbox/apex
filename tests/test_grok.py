"""Tests for Grok client and schemas."""

from __future__ import annotations

from uuid import UUID

import msgspec
import pytest

from src.api.schemas.grok import (
    GROK_MODELS,
    GrokImageEditRequest,
    GrokImageRequest,
    GrokJobResponse,
    GrokProviderInfo,
    GrokVideoFromImageRequest,
    GrokVideoRequest,
)
from src.core.enums import (
    AspectRatio,
    GenerationType,
    JobStatus,
    ModelType,
    Provider,
    VideoResolution,
)


class TestModelType:
    """Tests for ModelType enum."""

    def test_provider_property_comfyui(self) -> None:
        """Test ComfyUI models return ComfyUI provider."""
        assert ModelType.AISHA.provider == Provider.COMFYUI

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
        assert ModelType.AISHA.supports_image_input is True

    def test_is_video_model(self) -> None:
        """Test is_video_model property."""
        assert ModelType.GROK_IMAGINE_VIDEO.is_video_model is True
        assert ModelType.GROK_IMAGINE_IMAGE.is_video_model is False
        assert ModelType.GROK_2_IMAGE.is_video_model is False
        assert ModelType.AISHA.is_video_model is False


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


class TestGrokImageRequest:
    """Tests for GrokImageRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid image request creation."""
        request = GrokImageRequest(prompt="A beautiful sunset")
        assert request.prompt == "A beautiful sunset"
        assert request.model == ModelType.GROK_IMAGINE_IMAGE
        assert request.n == 1
        assert request.aspect_ratio == AspectRatio.RATIO_1_1

    def test_custom_model(self) -> None:
        """Test request with custom model."""
        request = GrokImageRequest(
            prompt="A cat",
            model=ModelType.GROK_2_IMAGE,
            n=4,
            aspect_ratio=AspectRatio.RATIO_16_9,
        )
        assert request.model == ModelType.GROK_2_IMAGE
        assert request.n == 4
        assert request.aspect_ratio == AspectRatio.RATIO_16_9

    def test_invalid_model_raises_error(self) -> None:
        """Test that video model raises error."""
        with pytest.raises(ValueError, match="does not support image generation"):
            GrokImageRequest(prompt="test", model=ModelType.GROK_IMAGINE_VIDEO)

    def test_json_serialization(self) -> None:
        """Test JSON roundtrip."""
        request = GrokImageRequest(
            prompt="A test prompt",
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=2,
        )
        data = msgspec.json.encode(request)
        decoded = msgspec.json.decode(data, type=GrokImageRequest)
        assert decoded.prompt == request.prompt
        assert decoded.model == request.model
        assert decoded.n == request.n

    def test_prompt_validation_min_length(self) -> None:
        """Test prompt minimum length validation."""
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt": ""}',
                type=GrokImageRequest,
            )

    def test_n_validation_range(self) -> None:
        """Test n parameter validation."""
        # Too low
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt": "test", "n": 0}',
                type=GrokImageRequest,
            )

        # Too high
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt": "test", "n": 11}',
                type=GrokImageRequest,
            )

        # Valid range
        result = msgspec.json.decode(
            b'{"prompt": "test", "n": 10}',
            type=GrokImageRequest,
        )
        assert result.n == 10


class TestGrokImageEditRequest:
    """Tests for GrokImageEditRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid edit request creation."""
        image_id = UUID("12345678-1234-1234-1234-123456789012")
        request = GrokImageEditRequest(
            prompt="Change the sky to purple",
            input_image_id=image_id,
        )
        assert request.prompt == "Change the sky to purple"
        assert request.input_image_id == image_id
        assert request.model == ModelType.GROK_IMAGINE_IMAGE

    def test_invalid_model_raises_error(self) -> None:
        """Test that non-editing model raises error."""
        image_id = UUID("12345678-1234-1234-1234-123456789012")
        with pytest.raises(ValueError, match="does not support image editing"):
            GrokImageEditRequest(
                prompt="test",
                input_image_id=image_id,
                model=ModelType.GROK_2_IMAGE,
            )


class TestGrokVideoRequest:
    """Tests for GrokVideoRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid video request creation."""
        request = GrokVideoRequest(prompt="A cat playing with yarn")
        assert request.prompt == "A cat playing with yarn"
        assert request.model == ModelType.GROK_IMAGINE_VIDEO
        assert request.duration == 5
        assert request.aspect_ratio == AspectRatio.RATIO_16_9
        assert request.resolution == VideoResolution.RES_720P

    def test_custom_parameters(self) -> None:
        """Test request with custom parameters."""
        request = GrokVideoRequest(
            prompt="Ocean waves",
            duration=10,
            aspect_ratio=AspectRatio.RATIO_9_16,
            resolution=VideoResolution.RES_480P,
        )
        assert request.duration == 10
        assert request.aspect_ratio == AspectRatio.RATIO_9_16
        assert request.resolution == VideoResolution.RES_480P

    def test_invalid_model_raises_error(self) -> None:
        """Test that image model raises error."""
        with pytest.raises(ValueError, match="does not support video generation"):
            GrokVideoRequest(prompt="test", model=ModelType.GROK_IMAGINE_IMAGE)

    def test_duration_validation(self) -> None:
        """Test duration validation."""
        # Too short
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt": "test", "duration": 0}',
                type=GrokVideoRequest,
            )

        # Too long
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt": "test", "duration": 16}',
                type=GrokVideoRequest,
            )

        # Valid range
        result = msgspec.json.decode(
            b'{"prompt": "test", "duration": 15}',
            type=GrokVideoRequest,
        )
        assert result.duration == 15


class TestGrokVideoFromImageRequest:
    """Tests for GrokVideoFromImageRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid I2V request creation."""
        image_id = UUID("12345678-1234-1234-1234-123456789012")
        request = GrokVideoFromImageRequest(
            prompt="Animate the character waving",
            input_image_id=image_id,
        )
        assert request.prompt == "Animate the character waving"
        assert request.input_image_id == image_id
        assert request.model == ModelType.GROK_IMAGINE_VIDEO


class TestGrokVideoEditRequest:
    """Tests for GrokVideoEditRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid V2V request creation."""
        from src.api.schemas.grok import GrokVideoEditRequest

        request = GrokVideoEditRequest(
            prompt="Change the background to a beach",
            input_video_url="https://example.com/video.mp4",
        )
        assert request.prompt == "Change the background to a beach"
        assert request.input_video_url == "https://example.com/video.mp4"
        assert request.model == ModelType.GROK_IMAGINE_VIDEO

    def test_invalid_model_raises_error(self) -> None:
        """Test that image model raises error."""
        from src.api.schemas.grok import GrokVideoEditRequest

        with pytest.raises(ValueError, match="does not support video editing"):
            GrokVideoEditRequest(
                prompt="test",
                input_video_url="https://example.com/video.mp4",
                model=ModelType.GROK_IMAGINE_IMAGE,
            )


class TestGrokJobResponse:
    """Tests for GrokJobResponse schema."""

    def test_valid_response(self) -> None:
        """Test valid response creation."""
        from datetime import datetime, timezone

        job_id = UUID("12345678-1234-1234-1234-123456789012")
        response = GrokJobResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            name="Test Job",
            model=ModelType.GROK_IMAGINE_IMAGE,
            generation_type=GenerationType.T2I,
            created_at=datetime.now(timezone.utc),
            message="Generation completed",
        )
        assert response.job_id == job_id
        assert response.status == JobStatus.COMPLETED
        assert response.message == "Generation completed"


class TestGrokProviderInfo:
    """Tests for GrokProviderInfo schema."""

    def test_default_models(self) -> None:
        """Test default models are included."""
        info = GrokProviderInfo(available=True)
        assert info.provider == "grok"
        assert info.name == "xAI Grok"
        assert info.available is True
        assert len(info.models) == len(GROK_MODELS)

    def test_model_capabilities(self) -> None:
        """Test model info capabilities are correct."""
        info = GrokProviderInfo(available=True)

        # Find grok-imagine-image
        imagine_image = next(m for m in info.models if m.model == ModelType.GROK_IMAGINE_IMAGE)
        assert imagine_image.supports_t2i is True
        assert imagine_image.supports_i2i is True
        assert imagine_image.supports_t2v is False
        assert imagine_image.supports_v2v is False
        assert imagine_image.max_images == 10

        # Find grok-2-image
        grok2_image = next(m for m in info.models if m.model == ModelType.GROK_2_IMAGE)
        assert grok2_image.supports_t2i is True
        assert grok2_image.supports_i2i is False

        # Find grok-imagine-video
        imagine_video = next(m for m in info.models if m.model == ModelType.GROK_IMAGINE_VIDEO)
        assert imagine_video.supports_t2v is True
        assert imagine_video.supports_i2v is True
        assert imagine_video.supports_v2v is True
        assert imagine_video.supports_t2i is False
