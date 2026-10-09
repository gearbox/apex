"""Registry-driven video request validation (arc 1, step 3): C5-C9, C11 helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import (
    GenerationService,
    UnsupportedGenerationParameterError,
)
from src.api.services.generation.source_media import (
    ResolvedSourceMedia,
    SourceMediaValidationError,
)
from src.api.services.generation.video_validation import (
    input_video_seconds,
    unsupported_video_parameters,
)
from src.core.enums import (
    AspectRatio,
    GenerationType,
    MediaKind,
    ModelType,
    Provider,
    VideoParameter,
    VideoResolution,
)
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.core.model_registry import get_model_meta
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit

_PROVIDER_FOR = {
    ModelType.GROK_IMAGINE_VIDEO: Provider.GROK,
    ModelType.AISHA_VIDEO: Provider.AISHA,
    ModelType.WAN_3_VIDEO: Provider.ALIBABA,
}


def _service() -> GenerationService:
    billing = AsyncMock()
    billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    billing.assert_sufficient_balance = AsyncMock()
    billing.check_and_reserve = AsyncMock()
    pricing = AsyncMock()
    pricing.quote = AsyncMock(return_value=50)
    return GenerationService(
        providers={provider: MagicMock() for provider in Provider},  # type: ignore[misc]
        billing_service=billing,
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
    )


def _request(model: ModelType, **overrides: Any) -> UnifiedGenerationRequest:
    fields: dict[str, Any] = {
        "prompt": "a fox",
        "generation_type": GenerationType.T2V,
        "model": model,
    } | overrides
    return UnifiedGenerationRequest(**fields)


def _video_source(duration_ms: int | None, *, position: int = 0) -> ResolvedSourceMedia:
    asset_id = uuid4()
    return ResolvedSourceMedia(
        position=position,
        ref=AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset_id),
        asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, asset_id),
        media_kind=MediaKind.VIDEO,
        content_type="video/mp4",
        storage_key=f"uploads/{asset_id}.mp4",
        size_bytes=1,
        duration_ms=duration_ms,
        job_id=None,
    )


async def _generate(service: GenerationService, request: UnifiedGenerationRequest) -> None:
    await service.generate(request, user_id=uuid4(), session=AsyncMock(), product_config=VEX_CONFIG)


def _assert_no_billing_work(service: GenerationService) -> None:
    service._pricing.quote.assert_not_awaited()  # type: ignore[attr-defined]
    service._billing.assert_sufficient_balance.assert_not_awaited()  # type: ignore[attr-defined]
    service._billing.check_and_reserve.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("model", "overrides", "message"),
    [
        (ModelType.GROK_IMAGINE_VIDEO, {"duration": 16}, "durations of 1-15"),
        (ModelType.AISHA_VIDEO, {"duration": 11}, "durations of 1-10"),
        (ModelType.WAN_3_VIDEO, {"duration": 1}, "durations of 2-30"),
        (
            ModelType.GROK_IMAGINE_VIDEO,
            {"resolution": VideoResolution.RES_1080P},
            "does not support resolution '1080p'",
        ),
        (
            ModelType.AISHA_VIDEO,
            {"aspect_ratio": AspectRatio.RATIO_4_3},
            "does not support aspect ratio '4:3'",
        ),
        (
            ModelType.WAN_3_VIDEO,
            {"aspect_ratio": AspectRatio.RATIO_2_3},
            "does not support aspect ratio '2:3'",
        ),
    ],
)
async def test_c5_c6_c8_out_of_range_values_fail_before_any_billing_work(
    model: ModelType, overrides: dict[str, Any], message: str
) -> None:
    service = _service()

    with pytest.raises(ValueError, match=message):
        await _generate(service, _request(model, **overrides))

    _assert_no_billing_work(service)


async def test_c6_wan_accepts_1080p_and_the_full_duration_range() -> None:
    service = _service()
    request = _request(ModelType.WAN_3_VIDEO, duration=30, resolution=VideoResolution.RES_1080P)

    # Validation passes; the call then stops at the next gate (model row lookup).
    with (
        patch(
            "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(Exception, match="disabled"),
    ):
        await _generate(service, request)


def test_c6_wan_31_seconds_is_rejected_by_the_schema() -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            b'{"prompt":"x","generation_type":"t2v","model":"wan3.0-video","duration":31}',
            type=UnifiedGenerationRequest,
        )


@pytest.mark.parametrize(
    "model",
    [ModelType.GROK_IMAGINE_VIDEO, ModelType.AISHA_VIDEO, ModelType.GROK_IMAGINE_IMAGE],
)
@pytest.mark.parametrize(
    ("field", "value"), [("audio", True), ("audio", False), ("prompt_extend", False)]
)
async def test_c7_undeclared_video_parameters_are_rejected_by_name(
    model: ModelType, field: str, value: bool
) -> None:
    service = _service()
    generation_type = (
        GenerationType.T2I if get_model_meta(model).video is None else GenerationType.T2V
    )
    request = _request(model, generation_type=generation_type, **{field: value})

    with pytest.raises(UnsupportedGenerationParameterError) as raised:
        await _generate(service, request)

    assert raised.value.parameters == (field,)
    _assert_no_billing_work(service)


def test_c7_none_is_always_accepted() -> None:
    request = _request(ModelType.GROK_IMAGINE_VIDEO)

    assert unsupported_video_parameters(request, frozenset()) == []


def test_c7_declared_parameters_are_accepted() -> None:
    request = _request(ModelType.WAN_3_VIDEO, audio=False, prompt_extend=True)
    supported = get_model_meta(ModelType.WAN_3_VIDEO).video.supported_parameters  # type: ignore[union-attr]

    assert unsupported_video_parameters(request, supported) == []
    assert supported == {VideoParameter.AUDIO, VideoParameter.PROMPT_EXTEND}


async def test_c8_wan_rejects_aspect_ratio_when_source_media_is_present() -> None:
    service = _service()
    request = _request(
        ModelType.WAN_3_VIDEO,
        generation_type=GenerationType.I2V,
        aspect_ratio=AspectRatio.RATIO_16_9,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
    )

    with pytest.raises(ValueError, match="omit aspect_ratio"):
        await _generate(service, request)

    _assert_no_billing_work(service)


@pytest.mark.parametrize("model", [ModelType.GROK_IMAGINE_VIDEO, ModelType.AISHA_VIDEO])
def test_c8_other_models_keep_accepting_aspect_ratio_with_media(model: ModelType) -> None:
    video = get_model_meta(model).video
    assert video is not None
    assert video.aspect_ratio_with_input_media is True


@pytest.mark.parametrize(
    ("duration_ms", "requested", "message"),
    [
        (None, 5, "no recorded duration"),
        (15_001, 5, "maximum is 15s"),
        (15_000, 16, "exceeds the 30s limit"),
        (14_200, 16, "exceeds the 30s limit"),
    ],
)
async def test_c9_wan_v2v_source_video_limits(
    duration_ms: int | None, requested: int, message: str
) -> None:
    service = _service()
    request = _request(
        ModelType.WAN_3_VIDEO,
        generation_type=GenerationType.V2V,
        duration=requested,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
    )
    resolved = [_video_source(duration_ms)]

    with (
        patch(
            "src.api.services.generation.service.SourceMediaResolver.resolve",
            new=AsyncMock(return_value=resolved),
        ),
        pytest.raises(SourceMediaValidationError, match=message),
    ):
        await _generate(service, request)

    _assert_no_billing_work(service)


def test_c11_input_video_seconds_counts_only_when_the_model_bills_them() -> None:
    wan = get_model_meta(ModelType.WAN_3_VIDEO).video
    grok = get_model_meta(ModelType.GROK_IMAGINE_VIDEO).video
    assert wan is not None
    assert grok is not None
    sources = [_video_source(7_200)]

    assert input_video_seconds(sources, wan) == 8  # 7.2s rounds up
    assert input_video_seconds(sources, grok) == 0
    assert input_video_seconds([], wan) == 0


def test_c11_unknown_input_duration_fails_closed_for_billing() -> None:
    wan = get_model_meta(ModelType.WAN_3_VIDEO).video
    assert wan is not None

    with pytest.raises(SourceMediaValidationError, match="no recorded duration"):
        input_video_seconds([_video_source(None)], wan)
