"""R2 / P2-2: a model's declared seed range is enforced before any billing work."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.request_validation import validate_seed
from src.core.enums import GenerationType, ModelType, Provider
from src.core.model_registry import MODEL_METADATA, SeedRange, get_model_meta
from tests.unit.test_video_validation import (
    _ENABLED_MODEL,
    _assert_no_billing_work,
    _generate,
    _request,
    _service,
    _submitting_service,
)

pytestmark = pytest.mark.unit

_WAN_MESSAGE = "Model 'wan3.0-video' accepts seeds 0-2147483647; omit seed for a random one"


def _wan_request(seed: int | None) -> UnifiedGenerationRequest:
    """Built through the wire schema, as a client would send it."""
    return msgspec.convert(
        {"prompt": "a fox", "generation_type": "t2v", "model": "wan3.0-video", "seed": seed},
        UnifiedGenerationRequest,
    )


@pytest.mark.parametrize("seed", [-1, -5, 2**31, 2**40])
async def test_wan_rejects_an_out_of_range_seed_before_any_billing_work(seed: int) -> None:
    service = _submitting_service()

    with _ENABLED_MODEL, pytest.raises(ValueError, match="accepts seeds 0-2147483647") as raised:
        await _generate(service, _wan_request(seed))

    assert str(raised.value) == _WAN_MESSAGE
    _assert_no_billing_work(service)
    service._providers[Provider.ALIBABA].submit.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.parametrize("seed", [0, 1, 2_147_483_647, None])
async def test_wan_accepts_a_seed_inside_the_range_and_an_omitted_one(seed: int | None) -> None:
    service = _submitting_service()

    with _ENABLED_MODEL:
        await _generate(service, _wan_request(seed))

    assert service._pricing.quote.await_count == 1  # type: ignore[attr-defined]
    submitted = service._providers[Provider.ALIBABA].submit.await_args.args[0]  # type: ignore[attr-defined]
    assert submitted.seed == seed


@pytest.mark.parametrize(
    "model", [m for m in ModelType if get_model_meta(m).seed_range is None], ids=lambda m: m.value
)
@pytest.mark.parametrize("seed", [-5, 2**31, 2**40, 2**63 - 1])
def test_a_model_without_a_declared_range_accepts_any_seed(model: ModelType, seed: int) -> None:
    mode = next(iter(get_model_meta(model).generation_modes))
    request = _request(model, generation_type=mode, seed=seed)

    validate_seed(request, get_model_meta(model))


async def test_grok_image_with_a_large_seed_still_reaches_the_provider() -> None:
    service = _service()
    job = MagicMock(id=uuid4(), status="completed", name="x", created_at=datetime.now(UTC))
    provider = MagicMock()
    provider.supports_image_sizing = False
    provider.submit = AsyncMock(
        return_value=ProviderSubmitResult(
            receipt_held=False, job=job, balance_after=1, balance_event=None
        )
    )
    service._providers[Provider.GROK] = provider
    request = _request(ModelType.GROK_IMAGINE_IMAGE, generation_type=GenerationType.T2I, seed=2**40)

    with _ENABLED_MODEL:
        await _generate(service, request)

    provider.submit.assert_awaited_once()


def test_only_wan_declares_a_range_and_it_matches_the_upstream_documentation() -> None:
    declared = {m: meta.seed_range for m, meta in MODEL_METADATA.items() if meta.seed_range}

    assert declared == {ModelType.WAN_3_VIDEO: SeedRange(0, 2_147_483_647)}


def test_an_empty_range_is_rejected_at_declaration() -> None:
    with pytest.raises(ValueError, match="min must be <= max"):
        SeedRange(5, 4)
