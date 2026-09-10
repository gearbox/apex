"""Tests for the model metadata registry and Alembic model imports."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import src.db.models as models
from src.core.enums import MEDIA_SLOT_KINDS, AspectRatio, GenerationType, MediaKind, ModelType
from src.core.model_registry import MODEL_METADATA, get_model_meta
from src.db.models.base import Base


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

    def test_generation_modes_match_output_sections_and_input_kinds(self) -> None:
        """Every registered mode has a matching output section and input contract."""
        for model, meta in MODEL_METADATA.items():
            assert meta.generation_modes, f"{model.value} has no generation modes"
            for generation_type, mode in meta.generation_modes.items():
                assert getattr(meta, generation_type.output_kind.value) is not None
                contract = mode.source_media
                assert (contract is None) is (not generation_type.input_kinds)
                if contract is not None:
                    assert contract.media_types <= generation_type.input_kinds
                    assert not contract.roles or (
                        len(contract.roles) == contract.min == contract.max
                        and len({MEDIA_SLOT_KINDS[role] for role in contract.roles}) == 1
                    )


class TestProvisioningDisplayHints:
    @pytest.mark.parametrize(
        "model_type",
        (ModelType.AISHA_IMAGE, ModelType.AISHA_IMAGE_LITE, ModelType.AISHA_VIDEO),
    )
    def test_aisha_models_have_configured_display_hints(self, model_type: ModelType) -> None:
        meta = get_model_meta(model_type)

        assert meta.typical_bootstrap_seconds is not None
        assert meta.typical_bootstrap_seconds > 0
        assert meta.typical_attach_seconds is not None
        assert meta.typical_attach_seconds > 0

    def test_models_without_on_demand_provisioning_keep_null_hints(self) -> None:
        meta = get_model_meta(ModelType.GROK_IMAGINE_IMAGE)

        assert meta.typical_bootstrap_seconds is None
        assert meta.typical_attach_seconds is None


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


class TestGenerationTypeInputKinds:
    @pytest.mark.parametrize(
        ("generation_type", "input_kinds"),
        [
            (GenerationType.T2I, frozenset()),
            (GenerationType.I2I, frozenset({MediaKind.IMAGE})),
            (GenerationType.T2V, frozenset()),
            (GenerationType.I2V, frozenset({MediaKind.IMAGE})),
            (GenerationType.V2V, frozenset({MediaKind.VIDEO})),
            (GenerationType.FLF2V, frozenset({MediaKind.IMAGE})),
        ],
    )
    def test_input_requirement_baseline(
        self,
        generation_type: GenerationType,
        input_kinds: frozenset[MediaKind],
    ) -> None:
        assert generation_type.input_kinds == input_kinds


def test_every_declarative_model_is_exported_from_models_registry() -> None:
    """Alembic only sees models imported by src.db.models through Base.metadata."""
    for module_info in pkgutil.iter_modules(models.__path__):
        if module_info.name == "base":
            continue
        module = importlib.import_module(f"{models.__name__}.{module_info.name}")
        for name, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate.__module__ != module.__name__ or not issubclass(candidate, Base):
                continue
            assert name in models.__all__
            assert getattr(models, name) is candidate
