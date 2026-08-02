"""Tests for the Grok API client."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.provider_failures import ProviderFailureKind
from src.api.services.grok import (
    GrokAPIError,
    GrokClient,
    GrokDeferredTerminalError,
    GrokInvalidRequestError,
    GrokModerationError,
    GrokVideoResult,
)
from src.api.services.grok.enums import ResponseImageFormat
from src.core.enums import AspectRatio, ModelType


class _DeferredStatus:
    DONE = 1
    PENDING = 2
    FAILED = 3
    EXPIRED = 4


@dataclass(frozen=True, slots=True)
class _Video:
    url: str = ""
    respect_moderation: bool = True


@dataclass(frozen=True, slots=True)
class _Error:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class _ImageResponse:
    url: str = ""
    b64_json: str = ""
    revised_prompt: str = ""


class _ModeratedImageResponse:
    @property
    def url(self) -> str:
        raise ValueError("Image did not respect moderation rules; URL is not available.")

    @property
    def revised_prompt(self) -> None:
        return None


class _ModeratedBase64ImageResponse:
    @property
    def base64(self) -> str:
        raise ValueError("Image did not respect moderation rules; base64 is not available.")

    @property
    def revised_prompt(self) -> None:
        return None


class _Base64ImageResponse:
    base64 = "aGVsbG8="
    revised_prompt = None


class _WhitespaceWrappedBase64ImageResponse:
    base64 = "aGVs\n bG8=\r\n"
    revised_prompt = None


class _CorruptBase64ImageResponse:
    base64 = "not valid base64 !!!"
    revised_prompt = None


class _DeferredPayload:
    def __init__(self, *, video: _Video | None = None, error: _Error | None = None) -> None:
        self.video = video
        self.error = error

    def HasField(self, name: str) -> bool:
        return getattr(self, name) is not None


class _DeferredResponse:
    def __init__(self, status: int, payload: _DeferredPayload | None = None) -> None:
        self.status = status
        self.response = payload

    def HasField(self, name: str) -> bool:
        return name == "response" and self.response is not None


class TestParseDeferredVideoResult:
    def test_done_returns_video_result(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.DONE,
            _DeferredPayload(video=_Video(url="https://example.test/video.mp4")),
        )

        result = GrokClient._parse_deferred_video_result(
            response,
            _DeferredStatus,
            request_id="req-1",
        )

        assert result == GrokVideoResult(url="https://example.test/video.mp4")

    def test_pending_returns_none(self) -> None:
        result = GrokClient._parse_deferred_video_result(
            _DeferredResponse(_DeferredStatus.PENDING),
            _DeferredStatus,
            request_id="req-1",
        )

        assert result is None

    def test_done_without_response_raises_api_error(self) -> None:
        with pytest.raises(GrokAPIError, match="no response returned"):
            GrokClient._parse_deferred_video_result(
                _DeferredResponse(_DeferredStatus.DONE),
                _DeferredStatus,
                request_id="req-1",
            )

    def test_video_missing_response_is_not_billable(self) -> None:
        """A truly empty deferred response is unproven malfunction, not
        proven undelivered output — it stays MALFORMED_RESPONSE/non-billable
        even with every charge flag on (D1, invariant 14)."""
        with pytest.raises(GrokAPIError) as exc_info:
            GrokClient._parse_deferred_video_result(
                _DeferredResponse(_DeferredStatus.DONE),
                _DeferredStatus,
                request_id="req-1",
            )

        failure = exc_info.value.failure
        assert failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
        assert (
            ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")
            .apply(failure)
            .billable
            is False
        )

    def test_video_missing_url_is_not_billable(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.DONE,
            _DeferredPayload(video=_Video(respect_moderation=True)),
        )

        with pytest.raises(GrokAPIError, match="URL missing from completed response") as exc_info:
            GrokClient._parse_deferred_video_result(
                response,
                _DeferredStatus,
                request_id="req-1",
            )

        failure = exc_info.value.failure
        assert failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
        assert (
            ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")
            .apply(failure)
            .billable
            is False
        )

    def test_done_without_url_and_failed_moderation_raises_moderation_error(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.DONE,
            _DeferredPayload(video=_Video(respect_moderation=False)),
        )

        with pytest.raises(GrokModerationError, match="flagged by moderation"):
            GrokClient._parse_deferred_video_result(
                response,
                _DeferredStatus,
                request_id="req-1",
            )

    def test_failed_with_error_payload_raises_api_error_with_code_and_message(self) -> None:
        response = _DeferredResponse(
            _DeferredStatus.FAILED,
            _DeferredPayload(error=_Error(code="E42", message="upstream failed")),
        )

        with pytest.raises(GrokDeferredTerminalError, match=r"\[E42\] upstream failed"):
            GrokClient._parse_deferred_video_result(
                response,
                _DeferredStatus,
                request_id="req-1",
            )

    @pytest.mark.parametrize(
        ("code", "kind"),
        [
            ("RATE_LIMITED", ProviderFailureKind.RATE_LIMITED),
            ("RESOURCE_EXHAUSTED", ProviderFailureKind.RATE_LIMITED),
            ("TIMEOUT", ProviderFailureKind.TIMEOUT),
            ("DEADLINE_EXCEEDED", ProviderFailureKind.TIMEOUT),
        ],
    )
    def test_failed_terminal_status_is_not_transient_even_for_retryable_kind(
        self,
        code: str,
        kind: ProviderFailureKind,
    ) -> None:
        response = _DeferredResponse(
            _DeferredStatus.FAILED,
            _DeferredPayload(error=_Error(code=code, message="provider terminal failure")),
        )

        with pytest.raises(GrokDeferredTerminalError) as exc_info:
            GrokClient._parse_deferred_video_result(
                response,
                _DeferredStatus,
                request_id="req-terminal",
            )

        assert exc_info.value.failure.kind == kind

    def test_expired_raises_api_error(self) -> None:
        with pytest.raises(GrokAPIError, match="expired before completion"):
            GrokClient._parse_deferred_video_result(
                _DeferredResponse(_DeferredStatus.EXPIRED),
                _DeferredStatus,
                request_id="req-1",
            )

    def test_unknown_status_is_treated_as_pending(self) -> None:
        result = GrokClient._parse_deferred_video_result(
            _DeferredResponse(999),
            _DeferredStatus,
            request_id="req-1",
        )

        assert result is None


class TestImageEditing:
    @staticmethod
    def _client_with_image(image: object) -> GrokClient:
        client = GrokClient.__new__(GrokClient)
        client._client = cast("Any", SimpleNamespace(image=image))
        return client

    async def test_edit_image_single_output_uses_sample_batch_with_source_image(
        self,
    ) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(
                return_value=[
                    _ImageResponse(
                        url="https://example.test/edited.png",
                        revised_prompt="revised prompt",
                    )
                ]
            )
        )
        client = self._client_with_image(image)

        results = await client.edit_image(
            "make it cinematic",
            "https://example.test/input.png",
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=1,
            aspect_ratio=AspectRatio.RATIO_16_9,
            image_format=ResponseImageFormat.URL,
        )

        assert len(results) == 1
        assert results[0].url == "https://example.test/edited.png"
        image.sample_batch.assert_awaited_once_with(
            model="grok-imagine-image",
            prompt="make it cinematic",
            n=1,
            image_url="https://example.test/input.png",
            image_urls=None,
            aspect_ratio="16:9",
            image_format="url",
        )

    async def test_edit_image_multiple_outputs_sends_source_image_and_n(
        self,
    ) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(
                return_value=[
                    _ImageResponse(url="https://example.test/edited-1.png"),
                    _ImageResponse(url="https://example.test/edited-2.png"),
                ]
            )
        )
        client = self._client_with_image(image)

        results = await client.edit_image(
            "make two options",
            "https://example.test/input.png",
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=2,
            aspect_ratio=AspectRatio.RATIO_9_16,
            image_format=ResponseImageFormat.URL,
        )

        assert [result.url for result in results] == [
            "https://example.test/edited-1.png",
            "https://example.test/edited-2.png",
        ]
        image.sample_batch.assert_awaited_once_with(
            model="grok-imagine-image",
            prompt="make two options",
            n=2,
            image_url="https://example.test/input.png",
            image_urls=None,
            aspect_ratio="9:16",
            image_format="url",
        )

    async def test_edit_image_supports_image_urls_without_image_url(self) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(
                return_value=[_ImageResponse(url="https://example.test/edited.png")]
            )
        )
        client = self._client_with_image(image)

        await client.edit_image(
            "combine these references",
            image_urls=[
                "https://example.test/source-1.png",
                "https://example.test/source-2.png",
            ],
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=1,
            image_format=ResponseImageFormat.URL,
        )

        image.sample_batch.assert_awaited_once_with(
            model="grok-imagine-image",
            prompt="combine these references",
            n=1,
            image_url=None,
            image_urls=[
                "https://example.test/source-1.png",
                "https://example.test/source-2.png",
            ],
            aspect_ratio=None,
            image_format="url",
        )

    async def test_edit_image_rejects_image_url_with_image_urls(self) -> None:
        image = SimpleNamespace(sample_batch=AsyncMock())
        client = self._client_with_image(image)

        with pytest.raises(GrokInvalidRequestError, match="Only one of image_url or image_urls"):
            await client.edit_image(
                "conflicting inputs",
                "https://example.test/source.png",
                image_urls=["https://example.test/source-2.png"],
            )

        image.sample_batch.assert_not_awaited()

    async def test_edit_image_omits_aspect_ratio_when_none(self) -> None:
        """Regression test for 03c179e: None must omit the protobuf field, not stretch the source."""
        image = SimpleNamespace(
            sample_batch=AsyncMock(
                return_value=[_ImageResponse(url="https://example.test/edited.png")]
            )
        )
        client = self._client_with_image(image)

        await client.edit_image(
            "keep source aspect",
            "https://example.test/input.png",
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=1,
            aspect_ratio=None,
            image_format=ResponseImageFormat.URL,
        )

        image.sample_batch.assert_awaited_once_with(
            model="grok-imagine-image",
            prompt="keep source aspect",
            n=1,
            image_url="https://example.test/input.png",
            image_urls=None,
            aspect_ratio=None,
            image_format="url",
        )

    async def test_edit_image_passes_explicit_aspect_ratio(self) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(
                return_value=[_ImageResponse(url="https://example.test/edited.png")]
            )
        )
        client = self._client_with_image(image)

        await client.edit_image(
            "reshape it",
            "https://example.test/input.png",
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=1,
            aspect_ratio=AspectRatio.RATIO_16_9,
            image_format=ResponseImageFormat.URL,
        )

        image.sample_batch.assert_awaited_once_with(
            model="grok-imagine-image",
            prompt="reshape it",
            n=1,
            image_url="https://example.test/input.png",
            image_urls=None,
            aspect_ratio="16:9",
            image_format="url",
        )

    async def test_edit_image_classifies_raising_url_property(self) -> None:
        image = SimpleNamespace(sample_batch=AsyncMock(return_value=[_ModeratedImageResponse()]))
        client = self._client_with_image(image)

        with pytest.raises(GrokModerationError) as exc_info:
            await client.edit_image("unsafe edit", "https://example.test/input.png")

        assert exc_info.value.failure.kind is ProviderFailureKind.MODERATION_REJECTED
        assert exc_info.value.failure.provider_request_accepted is True


class TestImageGeneration:
    @staticmethod
    def _client_with_image(image: object) -> GrokClient:
        client = GrokClient.__new__(GrokClient)
        client._client = cast("Any", SimpleNamespace(image=image))
        return client

    async def test_generate_image_omits_aspect_ratio_when_none(self) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(
                return_value=[_ImageResponse(url="https://example.test/generated.png")]
            )
        )
        client = self._client_with_image(image)

        await client.generate_image(
            "a cat in a tree",
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=1,
            aspect_ratio=None,
            image_format=ResponseImageFormat.URL,
        )

        image.sample_batch.assert_awaited_once_with(
            model="grok-imagine-image",
            prompt="a cat in a tree",
            n=1,
            aspect_ratio=None,
            image_format="url",
        )

    async def test_missing_output_url_is_a_malformed_provider_response(self) -> None:
        image = SimpleNamespace(sample_batch=AsyncMock(return_value=[_ImageResponse(url="")]))
        client = self._client_with_image(image)

        with pytest.raises(GrokAPIError) as exc_info:
            await client.generate_image(
                "a cat",
                model=ModelType.GROK_IMAGINE_IMAGE,
                image_format=ResponseImageFormat.URL,
            )

        failure = exc_info.value.failure
        assert failure.kind == ProviderFailureKind.MALFORMED_RESPONSE
        assert (
            ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")
            .apply(failure)
            .billable
            is False
        )

    async def test_empty_image_results_is_a_malformed_provider_response(self) -> None:
        image = SimpleNamespace(sample_batch=AsyncMock(return_value=[]))
        client = self._client_with_image(image)

        with pytest.raises(GrokAPIError) as exc_info:
            await client.generate_image(
                "a cat",
                model=ModelType.GROK_IMAGINE_IMAGE,
                n=0,
                image_format=ResponseImageFormat.URL,
            )

        failure = exc_info.value.failure
        assert failure.kind == ProviderFailureKind.MALFORMED_RESPONSE
        assert (
            ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")
            .apply(failure)
            .billable
            is False
        )

    async def test_generate_image_classifies_raising_url_property(self) -> None:
        image = SimpleNamespace(sample_batch=AsyncMock(return_value=[_ModeratedImageResponse()]))
        client = self._client_with_image(image)

        with pytest.raises(GrokModerationError) as exc_info:
            await client.generate_image("unsafe image")

        assert exc_info.value.failure.kind is ProviderFailureKind.MODERATION_REJECTED
        assert exc_info.value.failure.provider_request_accepted is True

    async def test_generate_image_decodes_sdk_base64_response(self) -> None:
        image = SimpleNamespace(sample_batch=AsyncMock(return_value=[_Base64ImageResponse()]))
        client = self._client_with_image(image)

        results = await client.generate_image("a cat", image_format=ResponseImageFormat.BASE64)

        assert results[0].base64_data == b"hello"

    async def test_whitespace_wrapped_base64_decodes(self) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(return_value=[_WhitespaceWrappedBase64ImageResponse()])
        )
        client = self._client_with_image(image)

        results = await client.generate_image("a cat", image_format=ResponseImageFormat.BASE64)

        assert results[0].base64_data == b"hello"

    async def test_corrupt_base64_is_malformed_and_not_billable(self) -> None:
        """An undecodable payload is a corrupt response, not a proven delivery
        failure — we cannot show the user anything or prove xAI produced a
        valid image, so it stays non-billable regardless of policy (D1)."""
        image = SimpleNamespace(
            sample_batch=AsyncMock(return_value=[_CorruptBase64ImageResponse()])
        )
        client = self._client_with_image(image)

        with pytest.raises(GrokAPIError) as exc_info:
            await client.generate_image("a cat", image_format=ResponseImageFormat.BASE64)

        failure = exc_info.value.failure
        assert failure.kind is ProviderFailureKind.MALFORMED_RESPONSE
        assert failure.provider_request_accepted is True
        assert (
            ProviderBillingPolicyRegistry.with_grok_moderation_policy("charge", "charge")
            .apply(failure)
            .billable
            is False
        )

    async def test_generate_image_classifies_raising_base64_property(self) -> None:
        image = SimpleNamespace(
            sample_batch=AsyncMock(return_value=[_ModeratedBase64ImageResponse()])
        )
        client = self._client_with_image(image)

        with pytest.raises(GrokModerationError) as exc_info:
            await client.generate_image("unsafe image", image_format=ResponseImageFormat.BASE64)

        assert exc_info.value.failure.kind is ProviderFailureKind.MODERATION_REJECTED
        assert exc_info.value.failure.provider_request_accepted is True
