"""Tests for the model metadata registry."""

from __future__ import annotations

import pytest

from src.core.enums import ModelType
from src.core.model_registry import MODEL_METADATA, get_model_meta


class TestModelRegistryCompleteness:
    """Every ModelType enum member must have a MODEL_METADATA entry."""

    def test_all_models_registered(self) -> None:
        for mt in ModelType:
            assert mt in MODEL_METADATA, (
                f"ModelType.{mt.name} ({mt.value}) missing from MODEL_METADATA"
            )

    def test_no_extra_keys(self) -> None:
        """MODEL_METADATA should not contain keys that aren't ModelType members."""
        for key in MODEL_METADATA:
            assert key in ModelType, f"MODEL_METADATA key {key!r} is not a ModelType member"


class TestGetModelMeta:
    def test_returns_correct_meta(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_IMAGE)
        assert meta.max_prompt_length == 4096
        assert meta.supports_negative_prompt is False

    def test_aisha_image_supports_negative_prompt(self) -> None:
        meta = get_model_meta(ModelType.AISHA_IMAGE)
        assert meta.supports_negative_prompt is True

    def test_video_model_has_video_meta(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_VIDEO)
        assert meta.video is not None
        assert meta.video.max_duration == 15
        assert len(meta.video.resolutions) > 0

    def test_image_model_has_no_video_meta(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_IMAGE)
        assert meta.video is None

    def test_grok_image_has_output_resolutions(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_IMAGE)
        assert meta.image is not None
        assert meta.image.output_resolutions is not None
        assert "1024x1024" in meta.image.output_resolutions
        # Grok does not expose user-controllable height
        assert meta.image.min_height is None
        assert meta.image.max_height is None

    def test_aisha_image_has_height_range(self) -> None:
        meta = get_model_meta(ModelType.AISHA_IMAGE)
        assert meta.image is not None
        assert meta.image.min_height == 256
        assert meta.image.max_height == 2048
        assert meta.image.default_height == 1024
        assert meta.image.output_resolutions is None

    def test_aisha_video_has_video_meta(self) -> None:
        meta = get_model_meta(ModelType.AISHA_VIDEO)
        assert meta.video is not None
        assert meta.video.max_duration == 10
        assert meta.image is None

    def test_raises_keyerror_for_unknown(self) -> None:
        with pytest.raises(KeyError):
            get_model_meta("not-a-model")  # type: ignore[arg-type]


class TestRateLimitConfig:
    """Rate limit configs must be valid when present."""

    @pytest.mark.parametrize("mt", list(ModelType))
    def test_rate_limit_values_positive(self, mt: ModelType) -> None:
        meta = get_model_meta(mt)
        if meta.rate_limit is not None:
            assert meta.rate_limit.max_requests > 0
            assert meta.rate_limit.window_seconds > 0

    def test_grok_video_has_rate_limit(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_VIDEO)
        assert meta.rate_limit is not None
        assert meta.rate_limit.max_requests == 10
        assert meta.rate_limit.window_seconds == 60
