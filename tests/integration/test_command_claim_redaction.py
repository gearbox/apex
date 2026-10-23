"""Integration coverage for N1 (review r2): the command claim's node-supplied ``agent_id``
is redacted at the service boundary before it is stored in ``gpu_session_commands`` or
logged — and redaction does not break an agent re-claiming its own command, because the
claim ``UPDATE`` matches on ``agent_id`` equality and redaction is deterministic.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

import structlog
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.services.gpu_session.command_payload import ProvisionCommand
from src.api.services.gpu_session.command_service import GpuSessionCommandService
from src.core.enums import CommandStatus, GpuSessionStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository

_CALLBACK_TOKEN = "command-claim-redaction-token"
_ROTATING_APP_TOKEN = "ghs_RotatingAppInstallationToken0123456789ab"


class _StubSettings:
    """Just the attributes GpuSessionCommandService reads."""

    github_content_token = ""
    hf_token = ""
    civitai_api_token = ""
    gpu_command_provision_timeout_seconds = 3600
    gpu_command_removal_timeout_seconds = 600
    gpu_command_restart_timeout_seconds = 300


async def test_a_github_token_in_agent_id_is_redacted_in_the_row_and_logs(
    db_engine,  # type: ignore[no-untyped-def]
) -> None:
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"command-claim-redaction-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.active,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
        callback_token_hash=hashlib.sha256(_CALLBACK_TOKEN.encode()).hexdigest(),
    )
    service = GpuSessionCommandService(
        session_factory=session_factory,
        settings=_StubSettings(),  # type: ignore[arg-type]
    )
    agent_id = f"{gpu_session.id}:{_ROTATING_APP_TOKEN}"

    try:
        async with session_factory() as db, db.begin():
            db.add(user)
            await db.flush()
            db.add(gpu_session)
        command = await service.enqueue(
            session_id=gpu_session.id,
            product_id="vex",
            command=ProvisionCommand(bundle="wan_2.2_i2v:260105-01", mode="additive"),
        )

        with structlog.testing.capture_logs() as logs:
            status, envelope = await service.claim(
                session_id=gpu_session.id, bearer_token=_CALLBACK_TOKEN, agent_id=agent_id
            )
            # A lost-response retry from the same agent: same raw agent_id.
            retry_status, retry_envelope = await service.claim(
                session_id=gpu_session.id, bearer_token=_CALLBACK_TOKEN, agent_id=agent_id
            )

        assert status == 200
        assert envelope is not None
        assert envelope["command_id"] == str(command.id)

        # Re-claim idempotency survives redaction: the same held command comes back.
        assert retry_status == 200
        assert retry_envelope is not None
        assert retry_envelope["command_id"] == envelope["command_id"]

        async with session_factory() as db:
            stored = await GpuSessionCommandRepository(db).get_by_operation(command.operation_id)
        assert stored is not None
        assert stored.status == CommandStatus.claimed
        assert stored.agent_id == f"{gpu_session.id}:[REDACTED]"
        assert _ROTATING_APP_TOKEN not in (stored.agent_id or "")

        claimed_logs = [entry for entry in logs if entry["event"] == "gpu_session.command.claimed"]
        assert len(claimed_logs) == 2
        assert all(entry["agent_id"] == stored.agent_id for entry in claimed_logs)
        assert all(_ROTATING_APP_TOKEN not in str(entry) for entry in logs)
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
