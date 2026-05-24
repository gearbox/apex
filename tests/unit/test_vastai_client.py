"""Tests for Vast.ai API client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import msgspec
import pytest

from src.api.services.vastai.client import VastAIClient
from src.api.services.vastai.exceptions import (
    InstanceNotFoundError,
    NoCapacityError,
    OfferTakenError,
    VastAIError,
    VastAIPaymentError,
    VastAIRateLimitError,
)
from src.api.services.vastai.schemas import VastAIInstance, VastAIOffer
from src.core.bundle_config import HardwareRequirements

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HARDWARE = HardwareRequirements(
    gpu_whitelist=("RTX_4090", "A100_SXM4"),
    min_disk_gb=100,
    min_network_upload_mbps=500,
    min_network_download_mbps=1000,
    cuda_min_version="12.1",
    num_gpus=1,
)


def _make_offer(id: int, dph_total: float = 0.5) -> dict[str, object]:
    return {
        "id": id,
        "gpu_name": "RTX_4090",
        "num_gpus": 1,
        "dph_total": dph_total,
    }


def _mock_response(
    status_code: int,
    body: object,
    *,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    content = msgspec.json.encode(body)
    resp.content = content
    resp.text = content.decode()
    resp.headers = headers or {}
    return resp


def _make_settings(**overrides: Any) -> MagicMock:
    settings = MagicMock()
    settings.vastai_max_429_retries = overrides.get("max_429_retries", 3)
    settings.vastai_max_retry_after_seconds = overrides.get("max_retry_after_seconds", 10.0)
    return settings


def _make_client(mock_http: AsyncMock, **settings_overrides: Any) -> VastAIClient:
    return VastAIClient(
        http_client=mock_http,
        api_key="test-api-key",
        settings=_make_settings(**settings_overrides),
    )


# ---------------------------------------------------------------------------
# search_offers
# ---------------------------------------------------------------------------


async def test_search_offers_success() -> None:
    offers_data = [
        _make_offer(1, dph_total=0.3),
        _make_offer(2, dph_total=0.5),
        _make_offer(3, dph_total=0.1),
    ]
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(200, {"offers": offers_data}))
    client = _make_client(mock_http)
    result = await client.search_offers(_HARDWARE)
    assert len(result) == 3
    assert all(isinstance(o, VastAIOffer) for o in result)
    assert {o.id for o in result} == {1, 2, 3}


async def test_search_offers_no_results() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(200, {"offers": []}))
    client = _make_client(mock_http)
    with pytest.raises(NoCapacityError):
        await client.search_offers(_HARDWARE)


async def test_search_offers_builds_correct_filters() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(200, {"offers": [_make_offer(1)]}))
    client = _make_client(mock_http)
    await client.search_offers(_HARDWARE, limit=5)

    call_kwargs = mock_http.post.call_args.kwargs
    payload = call_kwargs["json"]

    assert payload["gpu_name"] == {"in": list(_HARDWARE.gpu_whitelist)}
    assert payload["num_gpus"] == {"gte": _HARDWARE.num_gpus}
    assert payload["disk_space"] == {"gte": _HARDWARE.min_disk_gb}
    assert payload["inet_up"] == {"gte": _HARDWARE.min_network_upload_mbps}
    assert payload["inet_down"] == {"gte": _HARDWARE.min_network_download_mbps}
    assert payload["cuda_max_good"] == {"gte": float(_HARDWARE.cuda_min_version)}
    assert payload["verified"] == {"eq": True}
    assert payload["rentable"] == {"eq": True}
    assert payload["rented"] == {"eq": False}
    assert payload["type"] == "ondemand"
    assert payload["order"] == [["dph_total", "asc"]]
    assert "order_dir" not in payload, "order_dir was removed from the Vast.ai API; do not send it"
    assert payload["limit"] == 5


async def test_search_offers_order_is_nested_array() -> None:
    """Vast.ai API requires order as an array of [field, direction] pairs."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(200, {"offers": [_make_offer(1)]}))
    client = _make_client(mock_http)
    await client.search_offers(_HARDWARE)

    payload = mock_http.post.await_args.kwargs["json"]
    order = payload["order"]
    assert isinstance(order, list), f"order must be a list, got {type(order).__name__}"
    assert len(order) >= 1, "order must have at least one sort key"
    for entry in order:
        assert isinstance(entry, list), f"each order entry must be a list, got {entry!r}"
        assert len(entry) == 2, (
            f"each order entry must have 2 elements (field, direction), got {entry!r}"
        )
        field, direction = entry
        assert isinstance(field, str)
        assert direction in ("asc", "desc"), f"direction must be 'asc' or 'desc', got {direction!r}"


async def test_search_offers_api_error() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(500, {"error": "server error"}))
    client = _make_client(mock_http)
    with pytest.raises(VastAIError) as exc_info:
        await client.search_offers(_HARDWARE)
    assert exc_info.value.status_code == 500


async def test_search_offers_invalid_cuda_version_raises_clear_error() -> None:
    """Non-numeric cuda_min_version should raise VastAIError without hitting the network."""
    bad_hardware = HardwareRequirements(
        gpu_whitelist=("RTX_4090",),
        min_disk_gb=100,
        min_network_upload_mbps=500,
        min_network_download_mbps=1000,
        cuda_min_version="12.1+",  # not a plain float
        num_gpus=1,
    )
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(200, {"offers": [_make_offer(1)]}))
    client = _make_client(mock_http)

    with pytest.raises(VastAIError) as exc_info:
        await client.search_offers(bad_hardware)

    assert "cuda_min_version" in str(exc_info.value)
    assert "12.1+" in str(exc_info.value)
    # Must fail BEFORE the HTTP call
    mock_http.post.assert_not_called()


# ---------------------------------------------------------------------------
# create_instance
# ---------------------------------------------------------------------------


async def test_create_instance_success() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(
        return_value=_mock_response(200, {"new_contract": 42, "success": True})
    )
    client = _make_client(mock_http)
    instance_id = await client.create_instance(
        offer_id=7,
        image="vastai/comfy:latest",
        disk_gb=100,
        env={"-p 8188:8188": "1"},
        onstart_cmd="bash /start.sh",
    )
    assert instance_id == 42


async def test_create_instance_offer_taken() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(return_value=_mock_response(409, {"error": "conflict"}))
    client = _make_client(mock_http)
    with pytest.raises(OfferTakenError) as exc_info:
        await client.create_instance(
            offer_id=7,
            image="vastai/comfy:latest",
            disk_gb=100,
            env={},
            onstart_cmd="bash /start.sh",
        )
    assert exc_info.value.status_code == 409


async def test_create_instance_payment_error() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(return_value=_mock_response(402, {"error": "payment required"}))
    client = _make_client(mock_http)
    with pytest.raises(VastAIPaymentError) as exc_info:
        await client.create_instance(
            offer_id=7,
            image="vastai/comfy:latest",
            disk_gb=100,
            env={},
            onstart_cmd="bash /start.sh",
        )
    assert exc_info.value.status_code == 402


async def test_create_instance_generic_server_error() -> None:
    """Non-2xx responses other than 402/409 should raise VastAIError via _raise_for_status."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(return_value=_mock_response(500, {"error": "internal"}))
    client = _make_client(mock_http)
    with pytest.raises(VastAIError) as exc_info:
        await client.create_instance(
            offer_id=7,
            image="vastai/comfy:latest",
            disk_gb=100,
            env={},
            onstart_cmd="bash /start.sh",
        )
    # Must be the base class, not one of the specific subclasses
    assert type(exc_info.value) is VastAIError
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_instance
# ---------------------------------------------------------------------------


async def test_get_instance_success() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "instances": {
                    "id": 123,
                    "actual_status": "running",
                    "cur_state": "running",
                    "status_msg": "all good",
                    "dph_total": 0.234567,
                }
            },
        )
    )
    client = _make_client(mock_http)
    instance = await client.get_instance(123)
    assert isinstance(instance, VastAIInstance)
    assert instance.id == 123
    assert instance.actual_status == "running"
    assert instance.cur_state == "running"


async def test_get_instance_unwraps_envelope() -> None:
    """Regression: /instances/{id}/ returns the instance wrapped in an
    'instances' key. apex must unwrap before returning to callers."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "instances": {
                    "id": 36610640,
                    "actual_status": "running",
                    "cur_state": "running",
                    "machine_id": 87355,
                    "gpu_name": "RTX 4090",
                }
            },
        )
    )
    client = _make_client(mock_http)
    instance = await client.get_instance(36610640)
    assert instance is not None
    assert instance.id == 36610640
    assert instance.actual_status == "running"
    assert instance.cur_state == "running"


async def test_get_instance_returns_none_when_instances_is_null() -> None:
    """Regression for 2026-05-12: Vast.ai returns {"instances": null} during
    transient states (image pulling, scheduling). apex must return None, not raise."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get = AsyncMock(return_value=_mock_response(200, {"instances": None}))
    client = _make_client(mock_http)
    result = await client.get_instance(36610640)
    assert result is None


async def test_get_instance_returns_none_when_instances_key_absent() -> None:
    """If the 'instances' key is missing entirely (rare API drift), treat as None."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "id": 36610640,  # root-level, no envelope — GetInstanceResponse.instances defaults None
                "actual_status": "running",
            },
        )
    )
    client = _make_client(mock_http)
    result = await client.get_instance(36610640)
    assert result is None


async def test_get_instance_not_found() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get = AsyncMock(return_value=_mock_response(404, {"error": "not found"}))
    client = _make_client(mock_http)
    with pytest.raises(InstanceNotFoundError) as exc_info:
        await client.get_instance(999)
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# stop_instance / start_instance
# ---------------------------------------------------------------------------


async def test_stop_instance_success() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(return_value=_mock_response(200, {"success": True}))
    client = _make_client(mock_http)
    await client.stop_instance(123)
    call_kwargs = mock_http.put.call_args.kwargs
    assert call_kwargs["json"] == {"state": "stopped"}


async def test_start_instance_success() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(return_value=_mock_response(200, {"success": True}))
    client = _make_client(mock_http)
    await client.start_instance(123)
    call_kwargs = mock_http.put.call_args.kwargs
    assert call_kwargs["json"] == {"state": "running"}


# ---------------------------------------------------------------------------
# destroy_instance
# ---------------------------------------------------------------------------


async def test_destroy_instance_success() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(return_value=_mock_response(200, {"success": True}))
    client = _make_client(mock_http)
    await client.destroy_instance(123)
    mock_http.delete.assert_awaited_once()


async def test_destroy_instance_idempotent_on_404() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(return_value=_mock_response(404, {"error": "not found"}))
    client = _make_client(mock_http)
    # Should NOT raise
    await client.destroy_instance(999)


# ---------------------------------------------------------------------------
# Auth header is sent on all requests
# ---------------------------------------------------------------------------


async def test_auth_header_sent_on_search() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=_mock_response(200, {"offers": [_make_offer(1)]}))
    client = VastAIClient(http_client=mock_http, api_key="my-secret-key", settings=_make_settings())
    await client.search_offers(_HARDWARE)
    headers = mock_http.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# dph_total_micros property
# ---------------------------------------------------------------------------


def test_dph_total_micros_conversion() -> None:
    offer = VastAIOffer(id=1, gpu_name="RTX_4090", dph_total=0.234567)
    assert offer.dph_total_micros == 234567


def test_dph_total_micros_handles_float_precision_edge_cases() -> None:
    """Some 6-decimal values aren't exactly representable in float64.

    E.g. ``0.258607 * 1_000_000 == 258606.99999999997`` — truncation via
    ``int()`` would yield ``258606`` (wrong). ``round()`` yields ``258607``.
    """
    cases = [
        (0.258607, 258607),
        (0.517488, 517488),
        (0.509597, 509597),
        (0.259947, 259947),
        (0.261941, 261941),
    ]
    for dph, expected_micros in cases:
        offer = VastAIOffer(id=1, gpu_name="RTX_4090", dph_total=dph)
        assert offer.dph_total_micros == expected_micros, (
            f"dph={dph}: expected {expected_micros}, got {offer.dph_total_micros}"
        )


# ---------------------------------------------------------------------------
# Schema regression: live API response shape
# ---------------------------------------------------------------------------


async def test_search_offers_accepts_live_response_shape() -> None:
    """Regression: live API returns verification:str, not verified:bool.

    apex's old schema crashed with msgspec.ValidationError. After the trim,
    this response parses cleanly because the schema no longer declares
    'verified' at all.
    """
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "offers": [
                    {
                        "id": 35689141,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.2688888888888889,
                        "num_gpus": 1,
                        # Live response uses verification:str — apex doesn't read it
                        # but it's in the payload; forbid_unknown_fields=False ignores it.
                        "verification": "verified",
                        "gpu_ram": 24564,
                        "disk_space": 360.57,
                        "inet_up": 1270.6,
                        "inet_down": 1282.9,
                        "cuda_max_good": 13.0,
                        "geolocation": "California, US",
                        "machine_id": 87355,
                        "reliability": 0.9749691,
                    }
                ]
            },
        )
    )
    client = _make_client(mock_http)
    offers = await client.search_offers(_HARDWARE)
    assert len(offers) == 1
    assert offers[0].id == 35689141
    assert offers[0].gpu_name == "RTX 4090"
    assert offers[0].dph_total == pytest.approx(0.2688888888888889)
    assert offers[0].dph_total_micros == 268889
    assert offers[0].num_gpus == 1


async def test_search_offers_accepts_minimal_response_shape() -> None:
    """Regression: schema tolerates absence of all optional fields."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                "offers": [
                    {
                        "id": 99999,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.30,
                        # no num_gpus, no other fields
                    }
                ]
            },
        )
    )
    client = _make_client(mock_http)
    offers = await client.search_offers(_HARDWARE)
    assert len(offers) == 1
    assert offers[0].num_gpus is None  # optional, defaulted


async def test_search_offers_rejects_missing_dph_total() -> None:
    """dph_total is required — apex billing depends on it. Missing dph_total
    must crash loudly as VastAIError, not silently provision at unknown cost."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(
        return_value=_mock_response(
            200,
            {"offers": [{"id": 1, "gpu_name": "RTX 4090"}]},  # no dph_total
        )
    )
    client = _make_client(mock_http)
    with pytest.raises(VastAIError, match="SearchOffersResponse"):
        await client.search_offers(_HARDWARE)


async def test_search_offers_raises_vastaierror_on_malformed_response() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(
        return_value=_mock_response(
            200,
            {
                # missing "offers" key
                "something_else": [],
            },
        )
    )
    client = _make_client(mock_http)
    with pytest.raises(VastAIError, match="SearchOffersResponse"):
        await client.search_offers(_HARDWARE)


async def test_create_instance_raises_vastaierror_on_malformed_response() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(
        return_value=_mock_response(
            200,
            {
                # missing required new_contract / success
                "weird": "shape",
            },
        )
    )
    client = _make_client(mock_http)
    with pytest.raises(VastAIError, match="CreateInstanceResponse"):
        await client.create_instance(
            offer_id=1,
            image="vastai/comfy:test",
            disk_gb=80,
            env={},
            onstart_cmd="echo hi",
        )


# ---------------------------------------------------------------------------
# 429 retry — regression for 2026-05-13 incident
# ---------------------------------------------------------------------------


async def test_destroy_instance_retries_on_429() -> None:
    """Regression for 2026-05-13: 429 must be retried, not raised immediately."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(
        side_effect=[
            _mock_response(429, "", headers={"Retry-After": "1"}),
            _mock_response(200, ""),
        ]
    )
    with patch("src.api.services.vastai.client.asyncio.sleep", new_callable=AsyncMock):
        client = _make_client(mock_http, max_429_retries=3)
        await client.destroy_instance(12345)
    assert mock_http.delete.await_count == 2


async def test_request_with_retry_exhausts_budget_then_raises() -> None:
    """After max_429_retries, surface VastAIRateLimitError."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(return_value=_mock_response(429, "", headers={"Retry-After": "1"}))
    with patch("src.api.services.vastai.client.asyncio.sleep", new_callable=AsyncMock):
        client = _make_client(mock_http, max_429_retries=2)
        with pytest.raises(VastAIRateLimitError) as exc_info:
            await client.destroy_instance(12345)
    # 1 initial + 2 retries = 3 calls total
    assert mock_http.delete.await_count == 3
    assert exc_info.value.retry_after_seconds == pytest.approx(1.0)
    assert exc_info.value.status_code == 429


async def test_retry_after_capped_at_max() -> None:
    """A pathological Retry-After is clamped to vastai_max_retry_after_seconds."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(
        side_effect=[
            _mock_response(429, "", headers={"Retry-After": "999"}),
            _mock_response(200, ""),
        ]
    )
    with patch(
        "src.api.services.vastai.client.asyncio.sleep", new_callable=AsyncMock
    ) as sleep_mock:
        client = _make_client(mock_http, max_429_retries=3, max_retry_after_seconds=10.0)
        await client.destroy_instance(12345)
    sleep_mock.assert_awaited_once_with(10.0)


async def test_retry_after_missing_uses_default_one_second() -> None:
    """Missing Retry-After header defaults to 1.0s wait."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(
        side_effect=[
            _mock_response(429, ""),  # no Retry-After header
            _mock_response(200, ""),
        ]
    )
    with patch(
        "src.api.services.vastai.client.asyncio.sleep", new_callable=AsyncMock
    ) as sleep_mock:
        client = _make_client(mock_http, max_429_retries=3)
        await client.destroy_instance(12345)
    sleep_mock.assert_awaited_once_with(1.0)


async def test_vastaierror_is_base_of_vastaieratelimiterror() -> None:
    """VastAIRateLimitError must be catchable as VastAIError."""
    err = VastAIRateLimitError("exhausted", retry_after_seconds=5.0)
    assert isinstance(err, VastAIError)
    assert err.status_code == 429
    assert err.retry_after_seconds == pytest.approx(5.0)


async def test_retry_applies_to_get_instance() -> None:
    """Smoke-test: 429 retry is wired into get_instance, not just destroy."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get = AsyncMock(
        side_effect=[
            _mock_response(429, "", headers={"Retry-After": "1"}),
            _mock_response(
                200,
                {"instances": {"id": 1, "actual_status": "running", "cur_state": "running"}},
            ),
        ]
    )
    with patch("src.api.services.vastai.client.asyncio.sleep", new_callable=AsyncMock):
        client = _make_client(mock_http, max_429_retries=3)
        result = await client.get_instance(1)
    assert result is not None
    assert mock_http.get.await_count == 2


async def test_zero_retries_raises_immediately_on_429() -> None:
    """vastai_max_429_retries=0 disables retry; raises VastAIRateLimitError on first 429."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.delete = AsyncMock(return_value=_mock_response(429, "", headers={"Retry-After": "5"}))
    client = _make_client(mock_http, max_429_retries=0)
    with pytest.raises(VastAIRateLimitError):
        await client.destroy_instance(12345)
    assert mock_http.delete.await_count == 1


# ---------------------------------------------------------------------------
# create_instance — template_hash_id vs image path
# ---------------------------------------------------------------------------


async def test_create_instance_with_template_hash_id_omits_image_and_onstart() -> None:
    """When template_hash_id is set, payload must NOT contain 'image' or 'onstart'."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(
        return_value=_mock_response(200, {"new_contract": 99, "success": True})
    )
    client = _make_client(mock_http)
    instance_id = await client.create_instance(
        offer_id=7,
        disk_gb=100,
        env={"-p 8188:8188": "1"},
        template_hash_id="abc123hash",
    )

    assert instance_id == 99
    payload = mock_http.put.call_args[1]["json"]
    assert payload["template_hash_id"] == "abc123hash"
    assert "image" not in payload
    assert "onstart" not in payload


async def test_create_instance_with_image_and_onstart_works() -> None:
    """Legacy path: image + onstart_cmd → no template_hash_id in payload."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.put = AsyncMock(
        return_value=_mock_response(200, {"new_contract": 55, "success": True})
    )
    client = _make_client(mock_http)
    instance_id = await client.create_instance(
        offer_id=7,
        disk_gb=100,
        env={},
        image="vastai/comfy:v0.15.1-cuda-12.9-py312",
        onstart_cmd="bash /start.sh",
    )

    assert instance_id == 55
    payload = mock_http.put.call_args[1]["json"]
    assert payload["image"] == "vastai/comfy:v0.15.1-cuda-12.9-py312"
    assert payload["onstart"] == "bash /start.sh"
    assert "template_hash_id" not in payload


async def test_create_instance_raises_when_neither_template_nor_image_provided() -> None:
    """create_instance must raise ValueError when called with neither template_hash_id nor image."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    client = _make_client(mock_http)
    with pytest.raises(
        ValueError, match="create_instance requires either template_hash_id or image"
    ):
        await client.create_instance(
            offer_id=7,
            disk_gb=100,
            env={},
        )
