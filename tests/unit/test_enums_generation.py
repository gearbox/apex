"""Tests for ModelType generation capability methods."""

from __future__ import annotations

import pytest

from src.core.enums import (
    _GENERATION_TYPE_MEDIA,
    _META_BY_MEDIA,
    GenerationType,
    MediaKind,
    ModelType,
)


@pytest.mark.parametrize("generation_type", GenerationType)
def test_every_generation_type_has_media_table_row(generation_type: GenerationType) -> None:
    """Adding a generation type requires declaring its media behavior."""
    assert generation_type in _GENERATION_TYPE_MEDIA


@pytest.mark.parametrize("media_kind", MediaKind)
def test_every_media_kind_has_model_metadata_row(media_kind: MediaKind) -> None:
    """Every media kind must map to a model-registry section."""
    assert media_kind in _META_BY_MEDIA


class TestModelTypeSupportsGenerationType:
    """Tests for ModelType.supports_generation_type()."""

    @pytest.mark.parametrize(
        ("model", "gen_type", "expected"),
        [
            # T2I: only image models (meta.image is not None)
            (ModelType.AISHA_IMAGE, GenerationType.T2I, True),
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.T2I, True),
            (ModelType.GROK_2_IMAGE, GenerationType.T2I, True),
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.T2I, False),  # video-only model
            (ModelType.AISHA_VIDEO, GenerationType.T2I, False),  # video-only model
            # I2I: image models only (meta.image is not None and meta.video is None)
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.I2I, True),
            (ModelType.AISHA_IMAGE, GenerationType.I2I, True),
            (ModelType.GROK_2_IMAGE, GenerationType.I2I, False),
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.I2I, False),
            # T2V: only video models
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.T2V, True),
            (ModelType.AISHA_VIDEO, GenerationType.T2V, True),
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.T2V, False),
            (ModelType.AISHA_IMAGE, GenerationType.T2V, False),
            # I2V: video models
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.I2V, True),
            (ModelType.AISHA_VIDEO, GenerationType.I2V, True),
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.I2V, False),
            # V2V / FLF2V: video models only
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.V2V, True),
            (ModelType.AISHA_IMAGE, GenerationType.V2V, False),
            (
                ModelType.AISHA_VIDEO,
                GenerationType.V2V,
                False,
            ),  # AISHA_VIDEO supports FLF2V but not V2V
            (
                ModelType.GROK_IMAGINE_VIDEO,
                GenerationType.FLF2V,
                False,
            ),  # Grok video model doesn't support FLF2V
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.FLF2V, False),
        ],
    )
    def test_supports_generation_type(
        self,
        model: ModelType,
        gen_type: GenerationType,
        expected: bool,
    ) -> None:
        assert model.supports_generation_type(gen_type) is expected


class TestModelTypeMaxConcurrentOutputs:
    """Tests for ModelType.max_concurrent_outputs."""

    def test_video_model_always_one(self) -> None:
        assert ModelType.GROK_IMAGINE_VIDEO.max_concurrent_outputs == 1

    def test_aisha_image_four(self) -> None:
        assert ModelType.AISHA_IMAGE.max_concurrent_outputs == 4

    def test_grok_image_ten(self) -> None:
        assert ModelType.GROK_IMAGINE_IMAGE.max_concurrent_outputs == 10
        assert ModelType.GROK_2_IMAGE.max_concurrent_outputs == 10
