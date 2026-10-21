"""Small collaborators of the shared video completion path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.api.services.billing import BillingService
from src.api.services.job_state_transition import JobStateTransitionService
from src.api.services.video_jobs.download import VideoDownloadClient
from src.api.services.video_jobs.transition import make_transition_service_factory

pytestmark = pytest.mark.unit


async def test_download_client_requires_connect_and_never_carries_credentials() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"video")

    download = VideoDownloadClient(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="connect"):
        _ = download.client

    await download.connect()
    await download.connect()  # idempotent
    response = await download.client.get("https://oss.example.com/v.mp4?Signature=x")
    await download.close()
    await download.close()  # idempotent

    assert response.content == b"video"
    assert "authorization" not in seen[0].headers


def test_transition_factory_builds_a_service_per_session_with_one_billing_service() -> None:
    billing = MagicMock(spec=BillingService)
    event_bus = MagicMock()
    factory = make_transition_service_factory(
        billing_service=billing, event_bus=event_bus, ops_event_bus=None
    )

    first = factory(MagicMock())
    second = factory(MagicMock())

    assert isinstance(first, JobStateTransitionService)
    assert first is not second


def test_transition_factory_falls_back_to_a_stateless_billing_service() -> None:
    with patch("src.api.services.billing.BillingService") as billing_cls:
        make_transition_service_factory(billing_service=None, event_bus=None, ops_event_bus=None)

    billing_cls.assert_called_once_with()
