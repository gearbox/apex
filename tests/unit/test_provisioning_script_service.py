"""Unit tests for ProvisioningScriptService (Change 1 / D1-D6)."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from src.api.services.provisioning_script import (
    ProvisioningScriptRefNotFoundError,
    ProvisioningScriptService,
    ProvisioningScriptUnavailableError,
    _cache_key,
)
from src.core.enums import ScriptVariant


class _FakePipeline:
    """Minimal stand-in for a redis-py pipeline: sync command methods, async execute()."""

    def __init__(self, store: dict[str, dict[str, str]]) -> None:
        self._store = store
        self._pending_key: str | None = None
        self._pending_mapping: dict[str, str] | None = None

    def hset(self, key: str, mapping: dict[str, str]) -> _FakePipeline:
        self._pending_key = key
        self._pending_mapping = mapping
        return self

    def expire(self, key: str, ttl: int) -> _FakePipeline:  # noqa: ARG002
        return self

    async def execute(self) -> list[Any]:
        if self._pending_key is not None and self._pending_mapping is not None:
            self._store[self._pending_key] = dict(self._pending_mapping)
        return [1, True]


class _FakeRedis:
    """In-memory stand-in for the subset of redis.asyncio.Redis this service uses."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, str]] = {}
        self.raise_on_read = False
        self.raise_on_write = False

    async def hgetall(self, key: str) -> dict[str, str]:
        if self.raise_on_read:
            raise ConnectionError("redis down")
        return self.store.get(key, {})

    def pipeline(self) -> _FakePipeline:
        if self.raise_on_write:

            class _Boom(_FakePipeline):
                async def execute(self) -> list[Any]:
                    raise ConnectionError("redis down")

            return _Boom(self.store)
        return _FakePipeline(self.store)


def _make_settings(**overrides: Any) -> MagicMock:
    settings = MagicMock()
    settings.ai_bundles_github_token = "ghp_test_token"
    settings.provisioning_script_ref = "v1.0.0"
    settings.provisioning_script_dev_ref = None
    settings.provisioning_script_cache_ttl_seconds = 86400
    settings.provisioning_script_dev_cache_ttl_seconds = 60
    settings.environment = "staging"
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_response(
    status_code: int, *, content: bytes = b"", headers: dict[str, str] | None = None
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode()
    resp.headers = headers or {}
    return resp


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


class TestResolve:
    async def test_cache_miss_fetches_and_caches(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"#!/bin/sh\necho hi\n")
        redis = _FakeRedis()
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        result = await service.resolve(ScriptVariant.comfyui, "v1.2.3")

        assert result.cache_hit is False
        assert result.content == "#!/bin/sh\necho hi\n"
        assert result.sha256 == hashlib.sha256(b"#!/bin/sh\necho hi\n").hexdigest()
        http.get.assert_awaited_once()
        # Never leaks repo/path from the request — assert the outbound URL is the
        # hard-coded (repo, path) pair, not derived from anything caller-supplied.
        called_url = http.get.await_args.args[0]
        assert (
            called_url
            == "https://api.github.com/repos/gearbox/aisha/contents/scripts/aisha-provision-comfyui.sh"
        )
        assert http.get.await_args.kwargs["params"] == {"ref": "v1.2.3"}

    async def test_second_call_is_a_cache_hit_with_zero_outbound_calls(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"script body")
        redis = _FakeRedis()
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        first = await service.resolve(ScriptVariant.comfyui, "v1.2.3")
        http.get.reset_mock()
        second = await service.resolve(ScriptVariant.comfyui, "v1.2.3")

        assert second.cache_hit is True
        assert second.sha256 == first.sha256
        assert second.content == first.content
        http.get.assert_not_awaited()

    async def test_github_404_raises_ref_not_found_and_does_not_cache(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(404)
        redis = _FakeRedis()
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        with pytest.raises(ProvisioningScriptRefNotFoundError):
            await service.resolve(ScriptVariant.comfyui, "v9.9.9")

        assert redis.store == {}

    async def test_github_500_raises_unavailable_and_does_not_cache(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(500)
        redis = _FakeRedis()
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        with pytest.raises(ProvisioningScriptUnavailableError):
            await service.resolve(ScriptVariant.comfyui, "v1.2.3")

        assert redis.store == {}

    async def test_github_rate_limited_403_raises_unavailable(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(403, headers={"X-RateLimit-Remaining": "0"})
        redis = _FakeRedis()
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        with pytest.raises(ProvisioningScriptUnavailableError):
            await service.resolve(ScriptVariant.comfyui, "v1.2.3")

    async def test_httpx_error_raises_unavailable(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.side_effect = httpx.ConnectError("refused")
        redis = _FakeRedis()
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        with pytest.raises(ProvisioningScriptUnavailableError):
            await service.resolve(ScriptVariant.comfyui, "v1.2.3")

    async def test_redis_read_failure_falls_through_to_live_fetch(self) -> None:
        """Fail-open: a cache read error must not fail the request."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"body")
        redis = _FakeRedis()
        redis.raise_on_read = True
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        result = await service.resolve(ScriptVariant.comfyui, "v1.2.3")

        assert result.content == "body"
        http.get.assert_awaited_once()

    async def test_redis_write_failure_does_not_fail_a_successful_fetch(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"body")
        redis = _FakeRedis()
        redis.raise_on_write = True
        service = ProvisioningScriptService(
            http=http,
            redis=redis,  # type: ignore[arg-type]
            settings=_make_settings(),
        )

        result = await service.resolve(ScriptVariant.comfyui, "v1.2.3")

        assert result.content == "body"

    async def test_none_redis_never_caches(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"body")
        service = ProvisioningScriptService(http=http, redis=None, settings=_make_settings())

        await service.resolve(ScriptVariant.comfyui, "v1.2.3")
        await service.resolve(ScriptVariant.comfyui, "v1.2.3")

        assert http.get.await_count == 2

    def test_cache_key_is_scoped_by_repo_ref_and_path_hash(self) -> None:
        key = _cache_key("gearbox/aisha", "v1.2.3", "scripts/aisha-provision-comfyui.sh")
        assert key.startswith("provisioning:script:gearbox/aisha:v1.2.3:")


# ---------------------------------------------------------------------------
# serve_for_session()
# ---------------------------------------------------------------------------


class TestServeForSession:
    def _make_session_row(self, *, token: str = "correct-token") -> MagicMock:
        row = MagicMock()
        row.callback_token_hash = hashlib.sha256(token.encode()).hexdigest()
        return row

    async def test_unknown_variant_is_bad_request_with_no_outbound_call(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        service = ProvisioningScriptService(http=http, redis=None, settings=_make_settings())

        result = await service.serve_for_session(
            db=AsyncMock(),
            session_id=uuid4(),
            token="tok",
            variant="not-a-real-variant",
            ref="v1.2.3",
        )

        assert result.outcome == "bad_request"
        http.get.assert_not_awaited()

    @pytest.mark.parametrize("bad_ref", ["../../etc/passwd", "master", "x", ""])
    async def test_bad_ref_is_bad_request_with_no_outbound_call(self, bad_ref: str) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        service = ProvisioningScriptService(http=http, redis=None, settings=_make_settings())

        result = await service.serve_for_session(
            db=AsyncMock(), session_id=uuid4(), token="tok", variant="comfyui", ref=bad_ref
        )

        assert result.outcome == "bad_request"
        http.get.assert_not_awaited()

    async def test_dev_ref_allowed_outside_production(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"body")
        settings = _make_settings(
            provisioning_script_dev_ref="my-test-branch", environment="staging"
        )
        service = ProvisioningScriptService(http=http, redis=None, settings=settings)
        session_id = uuid4()
        db = AsyncMock()
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"

        from unittest.mock import patch

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(
                return_value=self._make_session_row(token="tok")
            )
            result = await service.serve_for_session(
                db=db, session_id=session_id, token="tok", variant="comfyui", ref="my-test-branch"
            )

        assert result.outcome == "ok"

    async def test_dev_ref_rejected_in_production(self) -> None:
        http = AsyncMock(spec=httpx.AsyncClient)
        settings = _make_settings(
            provisioning_script_dev_ref="my-test-branch", environment="production"
        )
        service = ProvisioningScriptService(http=http, redis=None, settings=settings)

        result = await service.serve_for_session(
            db=AsyncMock(), session_id=uuid4(), token="tok", variant="comfyui", ref="my-test-branch"
        )

        assert result.outcome == "bad_request"
        http.get.assert_not_awaited()

    async def test_missing_session_or_token_is_unauthorized(self) -> None:
        service = ProvisioningScriptService(
            http=AsyncMock(spec=httpx.AsyncClient), redis=None, settings=_make_settings()
        )

        missing_session = await service.serve_for_session(
            db=AsyncMock(), session_id=None, token="tok", variant="comfyui", ref="v1.0.0"
        )
        missing_token = await service.serve_for_session(
            db=AsyncMock(), session_id=uuid4(), token=None, variant="comfyui", ref="v1.0.0"
        )

        assert missing_session.outcome == "unauthorized"
        assert missing_token.outcome == "unauthorized"

    async def test_wrong_token_is_unauthorized(self) -> None:
        from unittest.mock import patch

        service = ProvisioningScriptService(
            http=AsyncMock(spec=httpx.AsyncClient), redis=None, settings=_make_settings()
        )
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(
                return_value=self._make_session_row(token="correct-token")
            )
            result = await service.serve_for_session(
                db=AsyncMock(),
                session_id=uuid4(),
                token="wrong-token",
                variant="comfyui",
                ref="v1.0.0",
            )

        assert result.outcome == "unauthorized"

    async def test_token_from_a_different_session_is_unauthorized(self) -> None:
        """A token that hashes correctly for a DIFFERENT session's hash must not validate."""
        from unittest.mock import patch

        service = ProvisioningScriptService(
            http=AsyncMock(spec=httpx.AsyncClient), redis=None, settings=_make_settings()
        )
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"
        other_sessions_row = self._make_session_row(token="other-session-token")

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=other_sessions_row)
            result = await service.serve_for_session(
                db=AsyncMock(),
                session_id=uuid4(),
                token="my-session-token",
                variant="comfyui",
                ref="v1.0.0",
            )

        assert result.outcome == "unauthorized"

    async def test_unknown_session_is_unauthorized(self) -> None:
        from unittest.mock import patch

        service = ProvisioningScriptService(
            http=AsyncMock(spec=httpx.AsyncClient), redis=None, settings=_make_settings()
        )
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
            result = await service.serve_for_session(
                db=AsyncMock(), session_id=uuid4(), token="tok", variant="comfyui", ref="v1.0.0"
            )

        assert result.outcome == "unauthorized"

    async def test_valid_request_resolves_and_returns_ok(self) -> None:
        from unittest.mock import patch

        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(200, content=b"script body")
        service = ProvisioningScriptService(http=http, redis=None, settings=_make_settings())
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(
                return_value=self._make_session_row(token="tok")
            )
            result = await service.serve_for_session(
                db=AsyncMock(), session_id=uuid4(), token="tok", variant="comfyui", ref="v1.0.0"
            )

        assert result.outcome == "ok"
        assert result.script is not None
        assert result.script.content == "script body"

    async def test_github_404_maps_to_not_found_outcome(self) -> None:
        from unittest.mock import patch

        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(404)
        service = ProvisioningScriptService(http=http, redis=None, settings=_make_settings())
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(
                return_value=self._make_session_row(token="tok")
            )
            result = await service.serve_for_session(
                db=AsyncMock(), session_id=uuid4(), token="tok", variant="comfyui", ref="v1.0.0"
            )

        assert result.outcome == "not_found"

    async def test_github_5xx_maps_to_unavailable_outcome(self) -> None:
        from unittest.mock import patch

        http = AsyncMock(spec=httpx.AsyncClient)
        http.get.return_value = _make_response(503)
        service = ProvisioningScriptService(http=http, redis=None, settings=_make_settings())
        repo_patch_target = "src.api.services.provisioning_script.GpuSessionRepository"

        with patch(repo_patch_target) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(
                return_value=self._make_session_row(token="tok")
            )
            result = await service.serve_for_session(
                db=AsyncMock(), session_id=uuid4(), token="tok", variant="comfyui", ref="v1.0.0"
            )

        assert result.outcome == "unavailable"
