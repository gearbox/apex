"""Async client for the Alibaba Model Studio (DashScope) video-synthesis API.

Two REST calls: create an asynchronous task, and query it. Nothing here
logs or raises with a response body, the API key, a result URL or a prompt;
only the request id, HTTP status and upstream ``code`` are ever recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn
from urllib.parse import quote

import httpx
import msgspec
import structlog

from src.api.services.generation.provider_failures import ProviderFailureKind
from src.api.services.model_studio.errors import ModelStudioAPIError, ModelStudioTransientError
from src.api.services.model_studio.failure_classifier import ModelStudioFailureClassifier
from src.api.services.model_studio.types import (
    ModelStudioTask,
    ModelStudioTaskRef,
    ModelStudioTaskStatus,
    WireError,
    WireTaskResponse,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import SecretStr

    from src.api.services.generation.provider_failures import ProviderFailure

logger = structlog.get_logger(__name__)

_CREATE_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
_TASK_PATH = "/api/v1/tasks/"
_TRANSIENT_KINDS = frozenset(
    {
        ProviderFailureKind.RATE_LIMITED,
        ProviderFailureKind.TIMEOUT,
        ProviderFailureKind.PROVIDER_UNAVAILABLE,
    }
)
_QUERY_ONLY_TRANSIENT_KINDS = frozenset({ProviderFailureKind.AUTHENTICATION_FAILED})
"""Transient for a task *query* only: on create, auth failures stay terminal and non-billable."""


class ModelStudioClient:
    """Create and poll Model Studio video tasks."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        timeout_seconds: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Configure the client; ``connect()`` opens the connection pool.

        Args:
            base_url: Workspace host, e.g. ``https://<id>.ap-southeast-1.maas.aliyuncs.com``.
            api_key: Model Studio key. Never logged.
            timeout_seconds: Per HTTP call.
            transport: Test seam (``httpx.MockTransport``).
        """
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._classifier = ModelStudioFailureClassifier()

    async def connect(self) -> None:
        """Open the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(float(self._timeout_seconds), connect=10.0),
                transport=self._transport,
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ModelStudioClient is not connected. Call connect() first.")
        return self._client

    def _headers(self, *, asynchronous: bool) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        if asynchronous:
            headers["Content-Type"] = "application/json"
            # Mandatory: upstream rejects synchronous video-synthesis calls.
            headers["X-DashScope-Async"] = "enable"
        return headers

    async def create_video_task(self, payload: Mapping[str, object]) -> ModelStudioTaskRef:
        """Submit an asynchronous video-synthesis task.

        Raises:
            ModelStudioAPIError: On any failure. ``provider_request_accepted`` is
                ``False`` for a 4xx, and ``None`` (ambiguous: upstream may have
                created a billable task) for timeouts, connection errors, 5xx and
                an unreadable success body. The create is never retried here.
        """
        try:
            response = await self._http().post(
                _CREATE_PATH,
                content=msgspec.json.encode(payload),
                headers=self._headers(asynchronous=True),
            )
        except httpx.TimeoutException as exc:
            logger.warning("alibaba.submit_ambiguous", reason="timeout")
            raise ModelStudioAPIError(
                self._classifier.timeout(provider_request_accepted=None)
            ) from exc
        except httpx.TransportError as exc:
            logger.warning("alibaba.submit_ambiguous", reason="connection_error")
            raise ModelStudioAPIError(
                self._classifier.unavailable(provider_request_accepted=None)
            ) from exc

        if response.status_code >= 400:
            error = self._decode_error(response)
            failure = self._classifier.classify(
                code=error.code,
                http_status=response.status_code,
                provider_request_accepted=False if response.status_code < 500 else None,
                provider_request_id=error.request_id,
            )
            if response.status_code >= 500:
                logger.warning("alibaba.submit_ambiguous", reason="http_5xx")
            self._log_failure("alibaba.create_failed", failure)
            raise ModelStudioAPIError(failure)

        try:
            body = msgspec.json.decode(response.content, type=WireTaskResponse)
        except msgspec.DecodeError as exc:
            self._raise_ambiguous_malformed(response.status_code, cause=exc)
        task_id = body.output.task_id if body.output is not None else None
        if not task_id:
            self._raise_ambiguous_malformed(response.status_code, body.request_id)
        logger.info("alibaba.task_created", request_id=body.request_id)
        return ModelStudioTaskRef(task_id=str(task_id), request_id=body.request_id)

    async def get_task(self, task_id: str) -> ModelStudioTask:
        """Fetch one task.

        Raises:
            ModelStudioTransientError: Rate limits, timeouts, upstream 5xx and
                authentication failures; the caller retries on the next tick without
                failing the job. An auth failure on a *query* is the operator's problem
                (rotated key, workspace permission), not the task's: the task keeps
                running and billing upstream, so failing and refunding it would lose
                the video. The job's own overdue check bounds the retries.
            ModelStudioAPIError: Any other query failure (malformed, rejected task).
        """
        try:
            response = await self._http().get(
                f"{_TASK_PATH}{quote(task_id, safe='')}",
                headers=self._headers(asynchronous=False),
            )
        except httpx.TimeoutException as exc:
            raise ModelStudioTransientError(
                self._classifier.timeout(
                    provider_request_accepted=True, provider_request_id=task_id
                )
            ) from exc
        except httpx.TransportError as exc:
            raise ModelStudioTransientError(
                self._classifier.unavailable(
                    provider_request_accepted=True, provider_request_id=task_id
                )
            ) from exc

        if response.status_code >= 400:
            error = self._decode_error(response)
            failure = self._classifier.classify(
                code=error.code,
                http_status=response.status_code,
                provider_request_accepted=True,
                provider_request_id=task_id,
            )
            if failure.kind is ProviderFailureKind.AUTHENTICATION_FAILED:
                self._log_poll_auth_failed(failure)
            else:
                self._log_failure("alibaba.poll_failed", failure)
            raise self._query_error(failure)

        try:
            body = msgspec.json.decode(response.content, type=WireTaskResponse)
        except msgspec.DecodeError as exc:
            failure = self._classifier.malformed(
                provider_request_accepted=True,
                provider_request_id=task_id,
                http_status=response.status_code,
            )
            self._log_failure("alibaba.poll_failed", failure)
            raise ModelStudioAPIError(failure) from exc
        output = body.output
        if output is None or not output.task_status:
            failure = self._classifier.malformed(
                provider_request_accepted=True,
                provider_request_id=task_id,
                http_status=response.status_code,
            )
            self._log_failure("alibaba.poll_failed", failure)
            raise ModelStudioAPIError(failure)
        try:
            status = ModelStudioTaskStatus(output.task_status)
        except ValueError:
            status = ModelStudioTaskStatus.UNKNOWN
        return ModelStudioTask(
            task_id=output.task_id or task_id,
            task_status=status,
            video_url=output.video_url,
            code=output.code,
            message=output.message,
            usage=body.usage,
            request_id=body.request_id,
        )

    @staticmethod
    def _decode_error(response: httpx.Response) -> WireError:
        try:
            return msgspec.json.decode(response.content, type=WireError)
        except msgspec.DecodeError:
            return WireError()

    @staticmethod
    def _query_error(failure: ProviderFailure) -> ModelStudioAPIError:
        """Map a *query* failure; ``create_video_task`` never goes through here."""
        if failure.kind in _TRANSIENT_KINDS or failure.kind in _QUERY_ONLY_TRANSIENT_KINDS:
            return ModelStudioTransientError(failure)
        return ModelStudioAPIError(failure)

    @staticmethod
    def _log_poll_auth_failed(failure: ProviderFailure) -> None:
        """Operator-actionable: every in-flight job is stuck until the key/permission is fixed."""
        logger.error(
            "alibaba.poll_auth_failed",
            provider=failure.provider.value,
            provider_status=failure.provider_status_code,
            provider_error_code=failure.provider_error_code,
            provider_job_id=failure.provider_request_id,
        )

    def _raise_ambiguous_malformed(
        self,
        status: int,
        request_id: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> NoReturn:
        # A 2xx we cannot read may still have created a billable task.
        logger.warning(
            "alibaba.submit_ambiguous", reason="malformed_success", request_id=request_id
        )
        raise ModelStudioAPIError(
            self._classifier.malformed(
                provider_request_accepted=None, provider_request_id=request_id, http_status=status
            )
        ) from cause

    @staticmethod
    def _log_failure(event: str, failure: ProviderFailure) -> None:
        logger.warning(
            event,
            provider=failure.provider.value,
            failure_kind=failure.kind.value,
            provider_status=failure.provider_status_code,
            provider_error_code=failure.provider_error_code,
            provider_job_id=failure.provider_request_id,
            retryable=failure.retryable,
        )
