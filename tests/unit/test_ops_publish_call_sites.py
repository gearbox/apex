"""Tests for the four ops-event publish call sites and OpsEventBus itself.

Covers: OpsEventBus.publish semantics (enabled/disabled/never-raises),
AuthService.register -> USER_REGISTERED, JobStateTransitionService
.transition_to_failed -> GENERATION_FAILED, and gpu_session
publish_status_event's D9 provisioning->active-only GPU_NODE_STARTED gate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.ops_events import GpuNodeStartedOpsPayload, OpsEventType
from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.auth import AuthService
from src.api.services.gpu_session._events import publish_status_event
from src.api.services.job_state_transition import JobStateTransitionService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import GpuSessionStatus, JobStatus
from src.db.models import RefreshToken, User


class TestOpsEventBus:
    async def test_disabled_bus_never_touches_redis(self) -> None:
        bus = OpsEventBus(enabled=False)
        with patch("src.api.services.ops_event_bus.get_redis_client") as mock_get_client:
            await bus.publish(
                event_type=OpsEventType.USER_REGISTERED, product_id="vex", payload=object()
            )
            mock_get_client.assert_not_called()

    async def test_enabled_bus_publishes_to_ops_events_channel(self) -> None:
        bus = OpsEventBus(enabled=True)
        mock_client = AsyncMock()

        with patch("src.api.services.ops_event_bus.get_redis_client", return_value=mock_client):
            await bus.publish(
                event_type=OpsEventType.USER_REGISTERED,
                product_id="vex",
                payload=GpuNodeStartedOpsPayload(
                    session_id=uuid4(), user_id=uuid4(), model_type="aisha-image"
                ),
            )

        mock_client.publish.assert_awaited_once()
        channel, _data = mock_client.publish.call_args[0]
        assert channel == "ops:events"

    async def test_publish_failure_never_propagates(self) -> None:
        """A Redis blip during ops publish must never raise to the caller (D contract)."""
        bus = OpsEventBus(enabled=True)
        mock_client = AsyncMock()
        mock_client.publish = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch("src.api.services.ops_event_bus.get_redis_client", return_value=mock_client):
            await bus.publish(
                event_type=OpsEventType.USER_REGISTERED, product_id="vex", payload=object()
            )
        # No exception raised — test passes by not raising.


class TestAuthRegisterPublishesUserRegistered:
    @pytest.fixture
    def jwt_service(self) -> JWTService:
        config = JWTConfig(
            secret_key="test_secret_key_for_testing_only_256bits",
            access_token_expire_minutes=15,
            refresh_token_expire_days=7,
        )
        return JWTService(config)

    async def test_register_publishes_user_registered_ops_event(
        self, jwt_service: JWTService
    ) -> None:
        mock_repository = AsyncMock()
        mock_repository.email_exists.return_value = False
        mock_user = MagicMock(spec=User)
        mock_user.email = "new@example.com"
        mock_user.is_active = True
        mock_repository.create_user.return_value = mock_user
        mock_repository.create_refresh_token.return_value = MagicMock(spec=RefreshToken)

        ops_bus = AsyncMock()
        service = AuthService(
            repository=mock_repository,
            jwt_service=jwt_service,
            password_service=PasswordService(),
            token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
            ops_event_bus=ops_bus,
        )

        await service.register(email="new@example.com", password="pw", product_id="vex")

        ops_bus.publish.assert_awaited_once()
        _, kwargs = ops_bus.publish.call_args
        assert kwargs["event_type"] == OpsEventType.USER_REGISTERED
        assert kwargs["product_id"] == "vex"
        # register() generates its own user_id and passes it to both
        # create_user() and the ops payload — verify they match.
        created_user_id = mock_repository.create_user.call_args.kwargs["id"]
        assert kwargs["payload"].user_id == created_user_id


class TestJobStateTransitionPublishesGenerationFailed:
    async def test_transition_to_failed_publishes_ops_event(self) -> None:
        job_id = uuid4()
        user_id = uuid4()
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.user_id = user_id
        mock_job.status = JobStatus.RUNNING.value
        mock_job.provider = "grok"
        mock_job.generation_type = "t2i"

        session = AsyncMock()
        session.get = AsyncMock(return_value=mock_job)
        execute_result = MagicMock()
        execute_result.rowcount = 1
        session.execute = AsyncMock(return_value=execute_result)
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        ops_bus = AsyncMock()
        service = JobStateTransitionService(
            session=session,
            event_bus=None,
            billing_service=MagicMock(),
            ops_event_bus=ops_bus,
        )

        _job, did_transition = await service.transition_to_failed(
            job_id, error_message="boom", refund=False, product_id="vex"
        )

        assert did_transition is True
        ops_bus.publish.assert_awaited_once()
        _, kwargs = ops_bus.publish.call_args
        assert kwargs["event_type"] == OpsEventType.GENERATION_FAILED
        assert kwargs["product_id"] == "vex"
        assert kwargs["payload"].job_id == job_id
        assert kwargs["payload"].user_id == user_id

    async def test_no_transition_does_not_publish(self) -> None:
        """Idempotent no-op path (already terminal) must not publish."""
        job_id = uuid4()
        mock_job = MagicMock()
        mock_job.status = JobStatus.COMPLETED.value  # already terminal

        session = AsyncMock()
        session.get = AsyncMock(return_value=mock_job)

        ops_bus = AsyncMock()
        service = JobStateTransitionService(
            session=session,
            event_bus=None,
            billing_service=MagicMock(),
            ops_event_bus=ops_bus,
        )

        _job, did_transition = await service.transition_to_failed(
            job_id, error_message="boom", refund=False, product_id="vex"
        )

        assert did_transition is False
        ops_bus.publish.assert_not_awaited()


class TestGpuNodeStartedFiresOnlyOnProvisioningToActive:
    def _make_session(self) -> MagicMock:
        session = MagicMock()
        session.id = uuid4()
        session.user_id = uuid4()
        session.product_id = "vex"
        session.status = GpuSessionStatus.active
        session.model_type = "aisha-image"
        session.tunnel_hostname = None
        return session

    async def test_provisioning_to_active_fires_gpu_node_started(self) -> None:
        session_row = self._make_session()
        ops_bus = AsyncMock()

        await publish_status_event(
            None,
            session_row,
            previous_status=GpuSessionStatus.provisioning,
            ops_event_bus=ops_bus,
        )

        ops_bus.publish.assert_awaited_once()
        _, kwargs = ops_bus.publish.call_args
        assert kwargs["event_type"] == OpsEventType.GPU_NODE_STARTED
        assert kwargs["product_id"] == "vex"
        assert kwargs["payload"].session_id == session_row.id

    async def test_resuming_to_active_does_not_fire(self) -> None:
        """D9: resume is not a new node — must not fire."""
        session_row = self._make_session()
        ops_bus = AsyncMock()

        await publish_status_event(
            None,
            session_row,
            previous_status=GpuSessionStatus.resuming,
            ops_event_bus=ops_bus,
        )

        ops_bus.publish.assert_not_awaited()

    async def test_non_active_status_does_not_fire(self) -> None:
        session_row = self._make_session()
        session_row.status = GpuSessionStatus.stopping
        ops_bus = AsyncMock()

        await publish_status_event(
            None,
            session_row,
            previous_status=GpuSessionStatus.provisioning,
            ops_event_bus=ops_bus,
        )

        ops_bus.publish.assert_not_awaited()

    async def test_none_ops_event_bus_is_a_no_op(self) -> None:
        session_row = self._make_session()

        # Must not raise even though ops_event_bus is None.
        await publish_status_event(
            None,
            session_row,
            previous_status=GpuSessionStatus.provisioning,
            ops_event_bus=None,
        )
