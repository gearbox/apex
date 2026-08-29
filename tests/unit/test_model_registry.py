"""Tests for the model metadata registry."""

from __future__ import annotations

import pytest

from src.core.enums import AspectRatio, GenerationType, ModelType
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

    def test_inputs_are_present_and_first_last_frame_models_accept_two_sources(self) -> None:
        """Registry input limits must accommodate every declared generation type."""
        for model, meta in MODEL_METADATA.items():
            assert meta.inputs is not None, f"{model.value} has no input capabilities"
            supported_types = set()
            if meta.image is not None:
                supported_types.update(meta.image.supported_types)
            if meta.video is not None:
                supported_types.update(meta.video.supported_types)
            if GenerationType.FLF2V in supported_types:
                assert meta.inputs.source_media is not None
                assert meta.inputs.source_media.max >= 2


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


class TestEditAspectRatiosCapability:
    """Registry invariants for the i2i reshape-on-edit capability (edit_aspect_ratios)."""

    def test_grok_imagine_image_declares_no_edit_reshape_capability(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_IMAGE)
        assert meta.image is not None
        assert meta.image.edit_aspect_ratios == ()

    def test_grok_2_image_declares_no_edit_reshape_capability(self) -> None:
        meta = get_model_meta(ModelType.GROK_2_IMAGE)
        assert meta.image is not None
        assert meta.image.edit_aspect_ratios == ()

    def test_aisha_image_declares_full_edit_reshape_capability(self) -> None:
        meta = get_model_meta(ModelType.AISHA_IMAGE)
        assert meta.image is not None
        assert len(meta.image.edit_aspect_ratios) == len(tuple(AspectRatio.__members__.values()))


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


class TestGenerationTypeInputRequirements:
    @pytest.mark.parametrize(
        ("generation_type", "requires_image", "requires_video"),
        [
            (GenerationType.T2I, False, False),
            (GenerationType.I2I, True, False),
            (GenerationType.T2V, False, False),
            (GenerationType.I2V, True, False),
            (GenerationType.V2V, False, True),
            (GenerationType.FLF2V, True, False),
        ],
    )
    def test_input_requirement_baseline(
        self,
        generation_type: GenerationType,
        requires_image: bool,
        requires_video: bool,
    ) -> None:
        assert generation_type.requires_image_input is requires_image
        assert generation_type.requires_video_input is requires_video
