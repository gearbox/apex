"""Tests for Grok job orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.generation.provider_failures import ProviderModerationRejectedError
from src.api.services.grok import GrokClient, GrokImageResult, GrokModerationError
from src.api.services.grok.enums import ResponseImageFormat
from src.api.services.grok.job_service import GrokJobService
from src.core.enums import AspectRatio, GenerationType, JobStatus, ModelType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService


class _FakeJobRepository:
    def __init__(self) -> None:
        self.job: SimpleNamespace | None = None

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.job = SimpleNamespace(**kwargs)
        return self.job

    async def update_status(
        self,
        job_id: object,
        status: JobStatus,
        **kwargs: object,
    ) -> SimpleNamespace:
        if self.job is None:
            self.job = SimpleNamespace(id=job_id)
        self.job.status = status
        for key, value in kwargs.items():
            setattr(self.job, key, value)
        return self.job


async def test_create_image_job_i2i_preserves_requested_output_count() -> None:
    job_repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock())
    input_image_url = "https://example.test/source.png"
    prompt = "make two edited options"
    grok = SimpleNamespace(
        edit_image=AsyncMock(
            return_value=[
                GrokImageResult(
                    url="https://example.test/edited-1.png",
                    base64_data=None,
                    revised_prompt="revised prompt",
                ),
                GrokImageResult(
                    url="https://example.test/edited-2.png",
                    base64_data=None,
                    revised_prompt=None,
                ),
            ]
        ),
        generate_image=AsyncMock(),
    )
    billing_service = SimpleNamespace(
        check_and_reserve=AsyncMock(
            return_value=SimpleNamespace(
                txn=SimpleNamespace(id=uuid4(), balance_after=75),
                event=None,
            )
        ),
        refund=AsyncMock(),
    )
    service = GrokJobService(cast("GrokClient", grok), MagicMock())
    store_image_result = AsyncMock()

    with (
        patch("src.api.services.grok.job_service.JobRepository", return_value=job_repo),
        patch(
            "src.api.services.grok.job_service.OutputRepository",
            return_value=SimpleNamespace(),
        ),
        patch.object(service, "_store_image_result", store_image_result),
    ):
        result = await service.create_image_job(
            cast("AsyncSession", session),
            user_id=uuid4(),
            prompt=prompt,
            model=ModelType.GROK_IMAGINE_IMAGE,
            generation_type=GenerationType.I2I,
            n=2,
            aspect_ratio=AspectRatio.RATIO_9_16,
            input_image_url=input_image_url,
            billing_service=cast("BillingService", billing_service),
            account_id=uuid4(),
            token_cost=25,
            product_id="grok-image",
        )

    grok.edit_image.assert_awaited_once_with(
        prompt=prompt,
        image_url=input_image_url,
        image_urls=None,
        model=ModelType.GROK_IMAGINE_IMAGE,
        n=2,
        aspect_ratio=AspectRatio.RATIO_9_16,
        image_format=ResponseImageFormat.URL,
    )
    grok.generate_image.assert_not_awaited()
    assert store_image_result.await_count == 2
    assert result.job.status == JobStatus.COMPLETED
    assert result.balance_after == 75


async def test_create_image_job_i2i_without_resolved_input_url_fails_and_refunds() -> None:
    job_repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock())
    grok = SimpleNamespace(
        edit_image=AsyncMock(),
        generate_image=AsyncMock(),
    )
    billing_service = SimpleNamespace(
        check_and_reserve=AsyncMock(
            return_value=SimpleNamespace(
                txn=SimpleNamespace(id=uuid4(), balance_after=75),
                event=None,
            )
        ),
        refund=AsyncMock(),
    )
    service = GrokJobService(cast("GrokClient", grok), MagicMock())
    store_image_result = AsyncMock()

    with (
        patch("src.api.services.grok.job_service.JobRepository", return_value=job_repo),
        patch(
            "src.api.services.grok.job_service.OutputRepository",
            return_value=SimpleNamespace(),
        ),
        patch.object(service, "_store_image_result", store_image_result),
        pytest.raises(ValueError, match="requires a resolved input image URL"),
    ):
        await service.create_image_job(
            cast("AsyncSession", session),
            user_id=uuid4(),
            prompt="missing input",
            model=ModelType.GROK_IMAGINE_IMAGE,
            generation_type=GenerationType.I2I,
            n=1,
            aspect_ratio=AspectRatio.RATIO_1_1,
            billing_service=cast("BillingService", billing_service),
            account_id=uuid4(),
            token_cost=25,
            product_id="grok-image",
        )

    grok.edit_image.assert_not_awaited()
    grok.generate_image.assert_not_awaited()
    store_image_result.assert_not_awaited()
    assert job_repo.job is not None
    assert job_repo.job.status == JobStatus.FAILED
    assert job_repo.job.error_message == "I2I generation requires a resolved input image URL"
    billing_service.refund.assert_awaited_once()
    assert billing_service.refund.await_args.args[0] == job_repo.job.id
    assert billing_service.refund.await_args.kwargs["product_id"] == "grok-image"


async def test_create_image_job_moderation_rejection_is_billable_and_safe() -> None:
    job_repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock())
    raw_provider_message = (
        "Image did not respect moderation rules; URL is not available. xai-key=secret"
    )
    grok = SimpleNamespace(
        edit_image=AsyncMock(),
        generate_image=AsyncMock(side_effect=GrokModerationError(raw_provider_message)),
    )
    billing_service = _billing_stub()
    service = GrokJobService(cast("GrokClient", grok), MagicMock())

    with (
        patch("src.api.services.grok.job_service.JobRepository", return_value=job_repo),
        patch(
            "src.api.services.grok.job_service.OutputRepository",
            return_value=SimpleNamespace(),
        ),
        pytest.raises(ProviderModerationRejectedError) as exc_info,
    ):
        await service.create_image_job(
            cast("AsyncSession", session),
            user_id=uuid4(),
            prompt="unsafe request",
            model=ModelType.GROK_IMAGINE_IMAGE,
            generation_type=GenerationType.T2I,
            billing_service=cast("BillingService", billing_service),
            account_id=uuid4(),
            token_cost=25,
            product_id="grok-image",
        )

    assert job_repo.job is not None
    assert job_repo.job.status == JobStatus.FAILED
    assert job_repo.job.failure_code == "provider_moderation_rejected"
    assert job_repo.job.error_message == exc_info.value.public_message
    assert raw_provider_message not in job_repo.job.error_message
    assert billing_service.refund.await_count == 0


def _billing_stub() -> SimpleNamespace:
    return SimpleNamespace(
        check_and_reserve=AsyncMock(
            return_value=SimpleNamespace(
                txn=SimpleNamespace(id=uuid4(), balance_after=75),
                event=None,
            )
        ),
        refund=AsyncMock(),
    )


async def test_create_image_job_t2i_defaults_aspect_to_1_1_when_unset() -> None:
    job_repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock())
    grok = SimpleNamespace(
        edit_image=AsyncMock(),
        generate_image=AsyncMock(
            return_value=[
                GrokImageResult(
                    url="https://example.test/out.png", base64_data=None, revised_prompt=None
                )
            ]
        ),
    )
    billing_service = _billing_stub()
    service = GrokJobService(cast("GrokClient", grok), MagicMock())

    with (
        patch("src.api.services.grok.job_service.JobRepository", return_value=job_repo),
        patch(
            "src.api.services.grok.job_service.OutputRepository",
            return_value=SimpleNamespace(),
        ),
        patch.object(service, "_store_image_result", AsyncMock()),
    ):
        await service.create_image_job(
            cast("AsyncSession", session),
            user_id=uuid4(),
            prompt="a cat",
            model=ModelType.GROK_IMAGINE_IMAGE,
            generation_type=GenerationType.T2I,
            n=1,
            aspect_ratio=None,
            billing_service=cast("BillingService", billing_service),
            account_id=uuid4(),
            token_cost=25,
            product_id="grok-image",
        )

    grok.generate_image.assert_awaited_once_with(
        prompt="a cat",
        model=ModelType.GROK_IMAGINE_IMAGE,
        n=1,
        aspect_ratio=AspectRatio.RATIO_1_1,
        image_format=ResponseImageFormat.URL,
    )
    assert job_repo.job is not None
    assert job_repo.job.aspect_ratio == "1:1"


async def test_create_image_job_i2i_stores_null_aspect_when_unset() -> None:
    job_repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock())
    input_image_url = "https://example.test/source.png"
    grok = SimpleNamespace(
        edit_image=AsyncMock(
            return_value=[
                GrokImageResult(
                    url="https://example.test/edited.png", base64_data=None, revised_prompt=None
                )
            ]
        ),
        generate_image=AsyncMock(),
    )
    billing_service = _billing_stub()
    service = GrokJobService(cast("GrokClient", grok), MagicMock())

    with (
        patch("src.api.services.grok.job_service.JobRepository", return_value=job_repo),
        patch(
            "src.api.services.grok.job_service.OutputRepository",
            return_value=SimpleNamespace(),
        ),
        patch.object(service, "_store_image_result", AsyncMock()),
    ):
        await service.create_image_job(
            cast("AsyncSession", session),
            user_id=uuid4(),
            prompt="edit without reshape",
            model=ModelType.GROK_IMAGINE_IMAGE,
            generation_type=GenerationType.I2I,
            n=1,
            aspect_ratio=None,
            input_image_url=input_image_url,
            billing_service=cast("BillingService", billing_service),
            account_id=uuid4(),
            token_cost=25,
            product_id="grok-image",
        )

    grok.edit_image.assert_awaited_once_with(
        prompt="edit without reshape",
        image_url=input_image_url,
        image_urls=None,
        model=ModelType.GROK_IMAGINE_IMAGE,
        n=1,
        aspect_ratio=None,
        image_format=ResponseImageFormat.URL,
    )
    assert job_repo.job is not None
    assert job_repo.job.aspect_ratio is None


async def test_start_video_job_defaults_aspect_to_16_9_when_unset() -> None:
    job_repo = _FakeJobRepository()
    session = SimpleNamespace(flush=AsyncMock())
    grok = SimpleNamespace(
        start_video_generation=AsyncMock(return_value=SimpleNamespace(request_id="req-123")),
    )
    billing_service = _billing_stub()
    service = GrokJobService(cast("GrokClient", grok), MagicMock())

    with patch("src.api.services.grok.job_service.JobRepository", return_value=job_repo):
        await service.start_video_job(
            cast("AsyncSession", session),
            user_id=uuid4(),
            prompt="a cat running",
            aspect_ratio=None,
            billing_service=cast("BillingService", billing_service),
            account_id=uuid4(),
            token_cost=25,
            product_id="grok-video",
        )

    grok.start_video_generation.assert_awaited_once()
    assert grok.start_video_generation.await_args.kwargs["aspect_ratio"] == AspectRatio.RATIO_16_9
    assert job_repo.job is not None
    assert job_repo.job.aspect_ratio == "16:9"
