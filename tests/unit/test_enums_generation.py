"""Tests for ModelType generation capability methods."""

from __future__ import annotations

import pytest

from src.core.enums import GenerationType, ModelType


class TestModelTypeSupportsGenerationType:
    """Tests for ModelType.supports_generation_type()."""

    @pytest.mark.parametrize(
        ("model", "gen_type", "expected"),
        [
            # Every model supports T2I
            (ModelType.AISHA, GenerationType.T2I, True),
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.T2I, True),
            (ModelType.GROK_2_IMAGE, GenerationType.T2I, True),
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.T2I, True),
            # I2I: needs image input AND not video model
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.I2I, True),
            (ModelType.AISHA, GenerationType.I2I, True),
            (ModelType.GROK_2_IMAGE, GenerationType.I2I, False),
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.I2I, False),
            # T2V: only video models
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.T2V, True),
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.T2V, False),
            (ModelType.AISHA, GenerationType.T2V, False),
            # I2V: video model + image input
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.I2V, True),
            (ModelType.GROK_IMAGINE_IMAGE, GenerationType.I2V, False),
            # V2V / FLF2V: video models only
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.V2V, True),
            (ModelType.AISHA, GenerationType.V2V, False),
            (ModelType.GROK_IMAGINE_VIDEO, GenerationType.FLF2V, True),
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

    def test_aisha_four(self) -> None:
        assert ModelType.AISHA.max_concurrent_outputs == 4

    def test_grok_image_ten(self) -> None:
        assert ModelType.GROK_IMAGINE_IMAGE.max_concurrent_outputs == 10
        assert ModelType.GROK_2_IMAGE.max_concurrent_outputs == 10
