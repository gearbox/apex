"""Grok API client service wrapping xai-sdk.

Provides async interface for xAI's image and video generation APIs.
The xai-sdk uses gRPC internally and provides both sync (Client) and async (AsyncClient).

Supports:
- grok-imagine-image: T2I and I2I (image editing)
- grok-2-image-1212: T2I only
- grok-imagine-video: T2V, I2V, V2V
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.config import Settings
from src.core.enums import AspectRatio, ModelType, VideoResolution

from .enums import ResponseImageFormat

if TYPE_CHECKING:
    from xai_sdk import AsyncClient as XAIAsyncClient
    from xai_sdk.aio.image import ImageResponse
    from xai_sdk.aio.video import VideoResponse
    from xai_sdk.proto.v6.deferred_pb2 import StartDeferredResponse
    from xai_sdk.proto.v6.video_pb2 import GetDeferredVideoResponse

logger = structlog.get_logger(__name__)


class GrokClientError(Exception):
    """Base exception for Grok client errors."""

    pass


class GrokConnectionError(GrokClientError):
    """Raised when connection to xAI API fails."""

    pass


class GrokAPIError(GrokClientError):
    """Raised when xAI API returns an error."""

    def __init__(
        self,
        message: str,
        *,
        grpc_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.grpc_code = grpc_code


class GrokRateLimitError(GrokAPIError):
    """Raised when rate limited by xAI API."""

    pass


class GrokInvalidRequestError(GrokAPIError):
    """Raised when request is invalid."""

    pass


class GrokTimeoutError(GrokAPIError):
    """Raised when request times out."""

    pass


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrokImageResult:
    """Result of an image generation request."""

    url: str | None
    """URL of the generated image on xAI storage (if image_format=url)."""

    base64_data: bytes | None
    """Raw image bytes (if image_format=base64)."""

    revised_prompt: str | None
    """The prompt after revision by the model."""

    @property
    def has_url(self) -> bool:
        """Check if result has a URL."""
        return self.url is not None

    @property
    def has_data(self) -> bool:
        """Check if result has binary data."""
        return self.base64_data is not None


@dataclass(frozen=True, slots=True)
class GrokVideoResult:
    """Result of a completed video generation request."""

    url: str
    """URL of the generated video on xAI storage."""


@dataclass(frozen=True, slots=True)
class GrokVideoJobStarted:
    """Result of starting an async video generation job."""

    request_id: str
    """Request ID for polling the result."""


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------


class GrokClient:
    """Async client for xAI Grok image and video generation APIs.

    Uses the official xai-sdk which communicates via gRPC.
    The AsyncClient is used for async operations.

    Example:
        >>> client = GrokClient(settings)
        >>> await client.connect()
        >>> results = await client.generate_image("A cat in a tree", n=2)
        >>> for result in results:
        ...     print(result.url)
        >>> await client.close()
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize Grok client.

        Args:
            settings: Application settings containing xAI API configuration.

        Raises:
            GrokClientError: If Grok is not configured (XAI_API_KEY missing).
        """
        if not settings.grok_configured:
            raise GrokClientError("Grok API key not configured (XAI_API_KEY)")

        self._settings = settings
        self._client: XAIAsyncClient | None = None

    async def __aenter__(self) -> GrokClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize the xAI SDK AsyncClient.

        Creates a new gRPC connection to xAI's API.
        """
        if self._client is None:
            from xai_sdk import AsyncClient

            self._client = AsyncClient(
                api_key=self._settings.xai_api_key,
                timeout=self._settings.xai_timeout,
            )
            logger.info("grok.client_initialized", timeout=self._settings.xai_timeout)

    async def close(self) -> None:
        """Close the client connection.

        Releases gRPC resources.
        """
        if self._client is None:
            return
        # AsyncClient may have a close method for cleanup
        # The xai-sdk AsyncClient uses gRPC channels that should be cleaned up
        try:
            if hasattr(self._client, "close"):
                import asyncio
                import inspect

                close_method = self._client.close
                if inspect.iscoroutinefunction(close_method):
                    await close_method()
                elif callable(close_method):
                    result = close_method()
                    # Handle if close returns a coroutine object
                    if asyncio.iscoroutine(result):
                        await result
        except Exception as e:
            logger.warning("grok.close_failed", error=str(e))
        finally:
            self._client = None
            logger.info("grok.client_closed")

    @property
    def client(self) -> XAIAsyncClient:
        """Get the xAI SDK async client.

        Returns:
            The initialized AsyncClient.

        Raises:
            GrokConnectionError: If client not connected.
        """
        if self._client is None:
            raise GrokConnectionError("Client not connected. Call connect() first.")
        return self._client

    # -------------------------------------------------------------------------
    # Image Generation
    # -------------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        model: ModelType = ModelType.GROK_IMAGINE_IMAGE,
        n: int = 1,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1,
        image_format: ResponseImageFormat = ResponseImageFormat.URL,
    ) -> list[GrokImageResult]:
        """Generate images from text prompt.

        Args:
            prompt: Text description of the image to generate.
            model: Model to use (grok-imagine-image or grok-2-image-1212).
            n: Number of images to generate (1-10).
            aspect_ratio: Aspect ratio for the generated image.
            image_format: Output format - "url" or "base64".

        Returns:
            List of GrokImageResult with URLs or base64 data.

        Raises:
            GrokAPIError: If API request fails.
            GrokInvalidRequestError: If parameters are invalid.
        """
        if model not in (ModelType.GROK_IMAGINE_IMAGE, ModelType.GROK_2_IMAGE):
            raise GrokInvalidRequestError(
                f"Model {model.value} does not support image generation. "
                f"Use {ModelType.GROK_IMAGINE_IMAGE.value} or {ModelType.GROK_2_IMAGE.value}."
            )

        try:
            if n == 1:
                response: ImageResponse = await self.client.image.sample(
                    model=model.value,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio.value,
                    image_format=image_format.value,
                )
                return [self._parse_image_response(response, image_format)]
            else:
                responses: Sequence[ImageResponse] = await self.client.image.sample_batch(
                    model=model.value,
                    prompt=prompt,
                    n=n,
                    aspect_ratio=aspect_ratio.value,
                    image_format=image_format.value,
                )
                return [self._parse_image_response(r, image_format) for r in responses]

        except Exception as e:
            raise self._convert_exception(e) from e

    async def edit_image(
        self,
        prompt: str,
        image_url: str,
        *,
        model: ModelType = ModelType.GROK_IMAGINE_IMAGE,
        image_format: ResponseImageFormat = ResponseImageFormat.URL,
    ) -> GrokImageResult:
        """Edit an existing image with a text prompt.

        Args:
            prompt: Text description of the edit to perform.
            image_url: URL or base64 data URL of the source image.
                For base64: "data:image/jpeg;base64,..."
            model: Model to use (must be grok-imagine-image).
            image_format: Output format - "url" or "base64".

        Returns:
            GrokImageResult with the edited image.

        Raises:
            GrokAPIError: If API request fails.
            GrokInvalidRequestError: If model doesn't support editing.
        """
        if model != ModelType.GROK_IMAGINE_IMAGE:
            raise GrokInvalidRequestError(
                f"Model {model.value} does not support image editing. "
                f"Use {ModelType.GROK_IMAGINE_IMAGE.value}."
            )

        try:
            response: ImageResponse = await self.client.image.sample(
                model=model.value,
                prompt=prompt,
                image_url=image_url,
                image_format=image_format.value,
            )
            return self._parse_image_response(response, image_format)

        except Exception as e:
            raise self._convert_exception(e) from e

    # -------------------------------------------------------------------------
    # Video Generation
    # -------------------------------------------------------------------------

    async def generate_video(
        self,
        prompt: str,
        *,
        model: ModelType = ModelType.GROK_IMAGINE_VIDEO,
        duration: int = 5,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9,
        resolution: VideoResolution = VideoResolution.RES_720P,
        image_url: str | None = None,
        video_url: str | None = None,
    ) -> GrokVideoResult:
        """Generate video with automatic polling (blocking).

        This method blocks until the video is ready or fails.
        The SDK handles polling automatically.

        Args:
            prompt: Text description of the video to generate.
            model: Model to use (must be grok-imagine-video).
            duration: Video duration in seconds (1-15).
            aspect_ratio: Aspect ratio for the video.
            resolution: Video resolution (480p or 720p).
            image_url: Optional source image URL for I2V generation.
            video_url: Optional source video URL for V2V editing.
                Maximum input video length is 8.7 seconds.

        Returns:
            GrokVideoResult with the video URL.

        Raises:
            GrokAPIError: If API request fails.
            GrokInvalidRequestError: If model doesn't support video.
        """
        if model != ModelType.GROK_IMAGINE_VIDEO:
            raise GrokInvalidRequestError(
                f"Model {model.value} does not support video generation. "
                f"Use {ModelType.GROK_IMAGINE_VIDEO.value}."
            )

        try:
            kwargs: dict[str, Any] = {
                "model": model.value,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio.value,
                "resolution": resolution.value,
            }

            if image_url:
                kwargs["image_url"] = image_url
            if video_url:
                kwargs["video_url"] = video_url

            # SDK handles polling automatically
            response: VideoResponse = await self.client.video.generate(**kwargs)

            return GrokVideoResult(url=response.url)

        except Exception as e:
            raise self._convert_exception(e) from e

    async def start_video_generation(
        self,
        prompt: str,
        *,
        model: ModelType = ModelType.GROK_IMAGINE_VIDEO,
        duration: int = 5,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9,
        resolution: VideoResolution = VideoResolution.RES_720P,
        image_url: str | None = None,
        video_url: str | None = None,
    ) -> GrokVideoJobStarted:
        """Start async video generation (non-blocking).

        Returns immediately with a request_id for manual polling.
        Use get_video_result() to retrieve the completed video.

        Args:
            prompt: Text description of the video to generate.
            model: Model to use (must be grok-imagine-video).
            duration: Video duration in seconds (1-15).
            aspect_ratio: Aspect ratio for the video.
            resolution: Video resolution (480p or 720p).
            image_url: Optional source image URL for I2V generation.
            video_url: Optional source video URL for V2V editing.

        Returns:
            GrokVideoJobStarted with request_id for polling.

        Raises:
            GrokAPIError: If API request fails.
        """
        if model != ModelType.GROK_IMAGINE_VIDEO:
            raise GrokInvalidRequestError(
                f"Model {model.value} does not support video generation. "
                f"Use {ModelType.GROK_IMAGINE_VIDEO.value}."
            )

        try:
            kwargs: dict[str, Any] = {
                "model": model.value,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio.value,
                "resolution": resolution.value,
            }

            if image_url:
                kwargs["image_url"] = image_url
            if video_url:
                kwargs["video_url"] = video_url

            # start() returns immediately with request_id
            response: StartDeferredResponse = await self.client.video.start(**kwargs)

            return GrokVideoJobStarted(request_id=response.request_id)

        except Exception as e:
            raise self._convert_exception(e) from e

    async def get_video_result(self, request_id: str) -> GrokVideoResult | None:
        """Get the result of an async video generation.

        Args:
            request_id: Request ID from start_video_generation().

        Returns:
            GrokVideoResult if complete, None if still processing.

        Raises:
            GrokAPIError: If the generation failed.
        """
        try:
            response: GetDeferredVideoResponse = await self.client.video.get(request_id)

            # Check if the response has a URL (indicates completion)
            if hasattr(response, "url") and response.response.video.url:
                return GrokVideoResult(url=response.response.video.url)

            # Still processing
            return None

        except Exception as e:
            # Check if it's a "not ready yet" type error
            error_msg = str(e).lower()
            if any(
                phrase in error_msg
                for phrase in ("processing", "pending", "not ready", "in progress")
            ):
                return None
            raise self._convert_exception(e) from e

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def encode_image_to_data_url(
        image_path: str | Path,
        content_type: str = "image/jpeg",
    ) -> str:
        """Encode an image file to a base64 data URL.

        Args:
            image_path: Path to the image file.
            content_type: MIME type of the image.

        Returns:
            Data URL string (data:image/jpeg;base64,...)
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        b64_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{b64_string}"

    @staticmethod
    def encode_bytes_to_data_url(
        data: bytes,
        content_type: str = "image/jpeg",
    ) -> str:
        """Encode image bytes to a base64 data URL.

        Args:
            data: Raw image bytes.
            content_type: MIME type of the image.

        Returns:
            Data URL string (data:image/jpeg;base64,...)
        """
        b64_string = base64.b64encode(data).decode("utf-8")
        return f"data:{content_type};base64,{b64_string}"

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _parse_image_response(
        self, response: ImageResponse, image_format: ResponseImageFormat
    ) -> GrokImageResult:
        """Parse xai-sdk image response to our result type.

        Args:
            response: ImageResponse object from xai-sdk.
            image_format: The requested format ("url" or "base64").

        Returns:
            GrokImageResult with extracted data.
        """
        url: str | None = None
        base64_data: bytes | None = None
        revised_prompt: str | None = None

        if image_format == ResponseImageFormat.URL:
            url = getattr(response, "url", None)
        else:
            # For base64 format, SDK returns raw bytes in .image attribute
            base64_data = getattr(response, "image", None)

        revised_prompt = getattr(response, "revised_prompt", None)

        return GrokImageResult(
            url=url,
            base64_data=base64_data,
            revised_prompt=revised_prompt,
        )

    def _convert_exception(self, e: Exception) -> GrokAPIError:
        """Convert xai-sdk/gRPC exceptions to our error types.

        The xai-sdk raises grpc.aio.AioRpcError for async operations.

        Args:
            e: The original exception.

        Returns:
            Appropriate GrokAPIError subclass.
        """
        error_msg = str(e)
        grpc_code: str | None = None

        # Try to extract gRPC status code
        code_method = getattr(e, "code", None)
        if callable(code_method):
            with contextlib.suppress(Exception):
                code = code_method()
                grpc_code = getattr(code, "name", None)
        # Check for specific error types
        error_lower = error_msg.lower()

        if "resource_exhausted" in error_lower or "rate" in error_lower:
            return GrokRateLimitError(error_msg, grpc_code=grpc_code)

        if "invalid_argument" in error_lower or "invalid" in error_lower:
            return GrokInvalidRequestError(error_msg, grpc_code=grpc_code)

        if "deadline_exceeded" in error_lower or "timeout" in error_lower:
            return GrokTimeoutError(error_msg, grpc_code=grpc_code)

        # Generic API error
        return GrokAPIError(error_msg, grpc_code=grpc_code)


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    "GrokClient",
    "GrokClientError",
    "GrokConnectionError",
    "GrokAPIError",
    "GrokRateLimitError",
    "GrokInvalidRequestError",
    "GrokTimeoutError",
    "GrokImageResult",
    "GrokVideoResult",
    "GrokVideoJobStarted",
]
