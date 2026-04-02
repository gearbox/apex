"""Tests for the providers discovery endpoint (v2)."""

from __future__ import annotations

import msgspec

from src.api.schemas.providers import (
    ImageConstraints,
    ModelInfo,
    ProviderInfo,
    ProvidersResponse,
    UserContext,
    VideoConstraints,
)
from src.core.enums import GenerationType, ModelType
from src.core.model_registry import get_model_meta


class TestProvidersResponseSchema:
    def test_roundtrip_empty(self) -> None:
        resp = ProvidersResponse(providers=[])
        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=ProvidersResponse)
        assert decoded.providers == []
        assert decoded.user_context is None

    def test_roundtrip_with_user_context(self) -> None:
        resp = ProvidersResponse(
            providers=[],
            user_context=UserContext(subscription_tier="pro"),
        )
        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=ProvidersResponse)
        assert decoded.user_context is not None
        assert decoded.user_context.subscription_tier == "pro"

    def test_user_context_null_when_omitted(self) -> None:
        resp = ProvidersResponse(providers=[])
        data = msgspec.json.decode(msgspec.json.encode(resp))
        assert data["user_context"] is None


class TestModelInfoSchema:
    def test_image_model_fields(self) -> None:
        info = ModelInfo(
            model_key="grok-imagine-image",
            name="Grok Imagine Image",
            description="Flagship image gen",
            capabilities=["t2i", "i2i"],
            is_enabled=True,
            max_images=10,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1", "16:9"],
            image=ImageConstraints(output_resolutions=["1024x1024", "2048x2048"]),
            video=None,
        )
        assert info.video is None
        assert info.image is not None
        assert info.image.output_resolutions == ["1024x1024", "2048x2048"]
        assert info.image.min_height is None  # Grok: not user-controllable
        assert "t2i" in info.capabilities

    def test_aisha_model_with_height_control(self) -> None:
        info = ModelInfo(
            model_key="aisha-image",
            name="Aisha Image",
            description="ComfyUI-based generation",
            capabilities=["t2i", "i2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
            image=ImageConstraints(
                min_height=256,
                max_height=2048,
                default_height=1024,
            ),
        )
        assert info.image is not None
        assert info.image.min_height == 256
        assert info.image.max_height == 2048
        assert info.image.output_resolutions is None

    def test_video_model_fields(self) -> None:
        info = ModelInfo(
            model_key="grok-imagine-video",
            name="Grok Imagine Video",
            description="Video gen",
            capabilities=["t2v", "i2v", "v2v", "flf2v"],
            is_enabled=True,
            max_images=1,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1", "16:9"],
            video=VideoConstraints(max_duration=15, resolutions=["480p", "720p"]),
        )
        assert info.video is not None
        assert info.video.max_duration == 15

    def test_roundtrip_with_video(self) -> None:
        info = ModelInfo(
            model_key="test",
            name="Test",
            description="",
            capabilities=["t2v"],
            is_enabled=True,
            max_images=1,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1"],
            video=VideoConstraints(max_duration=15, resolutions=["480p", "720p"]),
        )
        encoded = msgspec.json.encode(info)
        decoded = msgspec.json.decode(encoded, type=ModelInfo)
        assert decoded.video is not None
        assert decoded.video.resolutions == ["480p", "720p"]


class TestProviderInfoSchema:
    def test_provider_with_models(self) -> None:
        model = ModelInfo(
            model_key="aisha-image",
            name="Aisha Image",
            description="ComfyUI",
            capabilities=["t2i", "i2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
        )
        provider = ProviderInfo(
            provider="aisha",
            name="Aisha",
            available=True,
            models=[model],
        )
        assert len(provider.models) == 1
        assert provider.models[0].supports_negative_prompt is True

    def test_provider_with_empty_models(self) -> None:
        provider = ProviderInfo(
            provider="grok",
            name="xAI Grok",
            available=False,
            models=[],
        )
        assert provider.available is False
        assert provider.models == []


class TestCapabilitiesDerivation:
    """Verify capabilities list matches ModelType enum properties."""

    def test_grok_imagine_image_capabilities(self) -> None:
        mt = ModelType.GROK_IMAGINE_IMAGE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i", "i2i"]

    def test_grok_2_image_capabilities(self) -> None:
        mt = ModelType.GROK_2_IMAGE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i"]

    def test_grok_imagine_video_capabilities(self) -> None:
        mt = ModelType.GROK_IMAGINE_VIDEO
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2v", "i2v", "v2v"]

    def test_aisha_image_capabilities(self) -> None:
        mt = ModelType.AISHA_IMAGE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i", "i2i"]

    def test_aisha_video_capabilities(self) -> None:
        mt = ModelType.AISHA_VIDEO
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2v", "i2v", "flf2v"]

    def test_aspect_ratios_are_valid_enum_members(self) -> None:
        """Every model's aspect_ratios must be valid AspectRatio enum values."""
        from src.core.enums import AspectRatio

        all_values = {ar.value for ar in AspectRatio}
        for mt in ModelType:
            meta = get_model_meta(mt)
            model_values = {ar.value for ar in meta.aspect_ratios}
            assert model_values.issubset(all_values), (
                f"{mt.value} has invalid aspect ratios: {model_values - all_values}"
            )
            assert len(meta.aspect_ratios) > 0, f"{mt.value} has no aspect ratios"
