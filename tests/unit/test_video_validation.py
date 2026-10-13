"""Registry-driven video request validation (arc 1, step 3): C5-C9, C11 helpers."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.generation.base import ProviderSubmitResult
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
    validate_video_request,
    validate_video_sources,
)
from src.api.services.model_studio.wan_payload import build_wan_video_payload
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

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        providers={provider: MagicMock() for provider in Provider},
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


_WAN_ANCHORED = [GenerationType.I2V, GenerationType.FLF2V, GenerationType.V2V]


@pytest.mark.parametrize("generation_type", _WAN_ANCHORED)
async def test_c8_wan_rejects_aspect_ratio_in_input_anchored_modes(
    generation_type: GenerationType,
) -> None:
    service = _service()
    request = _request(
        ModelType.WAN_3_VIDEO,
        generation_type=generation_type,
        aspect_ratio=AspectRatio.RATIO_16_9,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
    )

    with pytest.raises(ValueError, match="omit aspect_ratio"):
        await _generate(service, request)

    _assert_no_billing_work(service)


def test_c8_wan_r2v_accepts_an_explicit_aspect_ratio_and_sends_it() -> None:
    """R2V references are subjects, not frames: the album's 16:9 master must be requestable."""
    meta = get_model_meta(ModelType.WAN_3_VIDEO)
    request = _request(
        ModelType.WAN_3_VIDEO,
        generation_type=GenerationType.R2V,
        aspect_ratio=AspectRatio.RATIO_16_9,
        source_media=[
            SourceMediaReference(asset_ref=f"upload:{uuid4()}"),
            SourceMediaReference(asset_ref=f"upload:{uuid4()}"),
        ],
    )

    assert validate_video_request(request, meta) is meta.video

    body = build_wan_video_payload(
        model=request.model,
        prompt=request.prompt,
        contract=meta.generation_modes[GenerationType.R2V].source_media,
        media=[(MediaKind.IMAGE, "https://x/1.png"), (MediaKind.IMAGE, "https://x/2.png")],
        duration=request.duration,
        resolution=request.resolution,
        aspect_ratio=request.aspect_ratio,
        audio=None,
        prompt_extend=None,
        seed=None,
    )
    assert body["parameters"]["ratio"] == "16:9"  # type: ignore[index]


def test_c8_wan_r2v_without_aspect_ratio_is_still_adaptive() -> None:
    meta = get_model_meta(ModelType.WAN_3_VIDEO)

    body = build_wan_video_payload(
        model=ModelType.WAN_3_VIDEO,
        prompt="Image 1 walks",
        contract=meta.generation_modes[GenerationType.R2V].source_media,
        media=[(MediaKind.IMAGE, "https://x/1.png")],
        duration=5,
        resolution=VideoResolution.RES_720P,
        aspect_ratio=None,
        audio=None,
        prompt_extend=None,
        seed=None,
    )

    assert body["parameters"]["ratio"] == "adaptive"  # type: ignore[index]


def test_c8_wan_t2v_accepts_aspect_ratio() -> None:
    meta = get_model_meta(ModelType.WAN_3_VIDEO)
    request = _request(ModelType.WAN_3_VIDEO, aspect_ratio=AspectRatio.RATIO_9_16)

    assert validate_video_request(request, meta) is meta.video


def test_c8_wan_declares_exactly_the_frame_anchored_modes() -> None:
    video = get_model_meta(ModelType.WAN_3_VIDEO).video
    assert video is not None
    assert video.input_anchored_modes == frozenset(_WAN_ANCHORED)


@pytest.mark.parametrize("model", [ModelType.GROK_IMAGINE_VIDEO, ModelType.AISHA_VIDEO])
def test_c8_other_models_keep_accepting_aspect_ratio_with_media(model: ModelType) -> None:
    video = get_model_meta(model).video
    assert video is not None
    assert video.input_anchored_modes == frozenset()


@pytest.mark.parametrize(
    "model",
    [ModelType.GROK_IMAGINE_VIDEO, ModelType.AISHA_VIDEO, ModelType.WAN_3_VIDEO],
)
def test_c8_anchored_modes_are_declared_modes_with_video_output(model: ModelType) -> None:
    meta = get_model_meta(model)
    assert meta.video is not None
    for mode in meta.video.input_anchored_modes:
        assert mode in meta.generation_modes
        assert mode.output_kind is MediaKind.VIDEO


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


def _v2v_request(model: ModelType, *, duration: int = 5) -> UnifiedGenerationRequest:
    return _request(
        model,
        generation_type=GenerationType.V2V,
        duration=duration,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
    )


@contextmanager
def _resolved_v2v_source(source: ResolvedSourceMedia) -> Iterator[None]:
    """Patch the owned-library lookups so ``generate`` sees exactly ``source``."""
    with (
        _ENABLED_MODEL,
        patch(
            "src.api.services.generation.service.SourceMediaResolver.resolve",
            new=AsyncMock(return_value=[source]),
        ),
        patch(
            "src.api.services.generation.service.UserImageRepository.touch_expiry_many",
            new=AsyncMock(return_value=1),
        ),
        patch(
            "src.api.services.generation.service.GenerationJobSourceRepository.create_many",
            new=AsyncMock(),
        ),
    ):
        yield


@pytest.mark.parametrize(
    ("content_type", "size_bytes", "duration_ms", "message"),
    [
        ("video/webm", 5 * 1024 * 1024, 6_000, "webm is not accepted; use one of mov, mp4"),
        ("video/mp4", 150 * 1024 * 1024, 6_000, "100000000"),
        ("video/mp4", 1_000, 500, "minimum is 1000 ms"),
    ],
)
async def test_r2_wan_v2v_source_outside_the_provider_limits_fails_before_any_billing_work(
    content_type: str, size_bytes: int, duration_ms: int, message: str
) -> None:
    service = _submitting_service()
    source = dataclasses.replace(
        _video_source(duration_ms), content_type=content_type, size_bytes=size_bytes
    )

    with (
        _resolved_v2v_source(source),
        pytest.raises(SourceMediaValidationError, match=message),
    ):
        await _generate(service, _v2v_request(ModelType.WAN_3_VIDEO))

    _assert_no_billing_work(service)
    service._providers[Provider.ALIBABA].submit.assert_not_awaited()  # type: ignore[attr-defined]


def test_r2_the_container_message_lists_the_accepted_formats_sorted() -> None:
    wan = get_model_meta(ModelType.WAN_3_VIDEO).video
    assert wan is not None
    source = dataclasses.replace(_video_source(6_000), content_type="video/webm")

    with pytest.raises(SourceMediaValidationError) as raised:
        validate_video_sources([source], _v2v_request(ModelType.WAN_3_VIDEO), wan)

    assert str(raised.value) == (
        "source_media position 0: webm is not accepted; use one of mov, mp4"
    )
    assert "uploads/" not in str(raised.value)  # never leaks the storage key


def test_r2_an_unknown_container_is_named_by_its_content_type() -> None:
    wan = get_model_meta(ModelType.WAN_3_VIDEO).video
    assert wan is not None
    source = dataclasses.replace(_video_source(6_000), content_type="video/x-matroska")

    with pytest.raises(SourceMediaValidationError, match="video/x-matroska is not accepted"):
        validate_video_sources([source], _v2v_request(ModelType.WAN_3_VIDEO), wan)


@pytest.mark.parametrize(
    "source",
    [
        dataclasses.replace(_video_source(1_000), size_bytes=100_000_000),  # both limits, exactly
        dataclasses.replace(_video_source(15_000), content_type="video/quicktime"),
    ],
)
def test_r2_wan_accepts_a_source_exactly_at_the_provider_limits(
    source: ResolvedSourceMedia,
) -> None:
    wan = get_model_meta(ModelType.WAN_3_VIDEO).video
    assert wan is not None

    validate_video_sources([source], _v2v_request(ModelType.WAN_3_VIDEO, duration=5), wan)


async def test_r2_grok_v2v_still_accepts_a_webm_source() -> None:
    service = _service()
    job = MagicMock(id=uuid4(), status="queued", name="x", created_at=datetime.now(UTC))
    provider = MagicMock()
    provider.validate = MagicMock()
    provider.submit = AsyncMock(
        return_value=ProviderSubmitResult(job=job, balance_after=1, balance_event=None)
    )
    service._providers[Provider.GROK] = provider
    source = dataclasses.replace(_video_source(60_000), content_type="video/webm")

    with _resolved_v2v_source(source):
        await _generate(service, _v2v_request(ModelType.GROK_IMAGINE_VIDEO))

    assert service._pricing.quote.await_count == 1  # type: ignore[attr-defined]
    provider.submit.assert_awaited_once()


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


def _submitting_service() -> GenerationService:
    """A service whose provider accepts the submit, so ``generate`` reaches the quote."""
    service = _service()
    job = MagicMock(id=uuid4(), status="queued", name="x", created_at=datetime.now(UTC))
    provider = MagicMock()
    provider.validate = MagicMock()
    provider.submit = AsyncMock(
        return_value=ProviderSubmitResult(job=job, balance_after=1, balance_event=None)
    )
    service._providers[Provider.ALIBABA] = provider
    return service


_ENABLED_MODEL = patch(
    "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
    new=AsyncMock(return_value=MagicMock(is_enabled=True)),
)


async def test_c11_quote_receives_duration_and_resolution_for_a_text_video() -> None:
    service = _submitting_service()
    request = _request(ModelType.WAN_3_VIDEO, duration=12, resolution=VideoResolution.RES_1080P)

    with _ENABLED_MODEL:
        await _generate(service, request)

    kwargs = service._pricing.quote.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["billed_seconds"] == 12
    assert kwargs["video_resolution"] is VideoResolution.RES_1080P


async def test_c11_v2v_quote_counts_the_input_video_seconds_for_wan() -> None:
    service = _submitting_service()
    request = _request(
        ModelType.WAN_3_VIDEO,
        generation_type=GenerationType.V2V,
        duration=10,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
    )

    with (
        _ENABLED_MODEL,
        patch(
            "src.api.services.generation.service.SourceMediaResolver.resolve",
            new=AsyncMock(return_value=[_video_source(7_200)]),
        ),
        patch(
            "src.api.services.generation.service.UserImageRepository.touch_expiry_many",
            new=AsyncMock(return_value=1),
        ),
        patch(
            "src.api.services.generation.service.GenerationJobSourceRepository.create_many",
            new=AsyncMock(),
        ),
    ):
        await _generate(service, request)

    kwargs = service._pricing.quote.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["billed_seconds"] == 10 + 8
    assert kwargs["source_media_count"] == 1


async def test_c11_image_quote_has_no_video_basis() -> None:
    service = _service()
    job = MagicMock(id=uuid4(), status="completed", name="x", created_at=datetime.now(UTC))
    provider = MagicMock()
    provider.supports_image_sizing = False
    provider.submit = AsyncMock(
        return_value=ProviderSubmitResult(job=job, balance_after=1, balance_event=None)
    )
    service._providers[Provider.GROK] = provider
    request = _request(ModelType.GROK_IMAGINE_IMAGE, generation_type=GenerationType.T2I, duration=9)

    with _ENABLED_MODEL:
        await _generate(service, request)

    kwargs = service._pricing.quote.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["billed_seconds"] == 0
    assert kwargs["video_resolution"] is None
