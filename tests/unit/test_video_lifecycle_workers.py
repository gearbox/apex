"""Wiring of the in-process async-video lifecycle workers (poller + receipt reconciler)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.dependencies.common import _start_video_lifecycle_workers
from tests.unit.helpers import hermetic_settings


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider_label": "alibaba",
        "workers_enabled": True,
        "job_service": MagicMock(),
        "compensator": MagicMock(),
        "poller_in_process": True,
        "poll_interval_seconds": 15,
        "max_concurrent_polls": 8,
        "reconciler_name": "alibaba_video_receipts",
        "standalone_module": "src.workers.wan_video",
        "settings": MagicMock(
            video_task_receipt_grace_seconds=60,
            video_task_receipt_reconcile_interval_seconds=30,
        ),
        "db_manager": MagicMock(),
        "receipt_db": MagicMock(),
        "event_bus": MagicMock(),
        "ops_event_bus": MagicMock(),
        "redis_enabled": False,
    }
    base.update(overrides)
    return base


@pytest.fixture
def started() -> Any:
    """Replace both worker classes with start-recording fakes."""
    with (
        patch("src.api.dependencies.common.AsyncVideoJobWorker") as poller_cls,
        patch("src.api.dependencies.common.build_receipt_reconciler_worker") as build,
    ):
        poller_cls.return_value.start = AsyncMock()
        build.return_value.start = AsyncMock()
        yield poller_cls, build


@pytest.mark.parametrize(
    ("label", "name", "module"),
    [
        ("grok", "grok_video_receipts", "src.workers.grok_video"),
        ("alibaba", "alibaba_video_receipts", "src.workers.wan_video"),
    ],
)
class TestLifecycleWiring:
    async def test_reconciler_runs_without_the_poller(
        self, started: Any, label: str, name: str, module: str
    ) -> None:
        poller_cls, build = started

        result = await _start_video_lifecycle_workers(
            **_kwargs(
                provider_label=label,
                reconciler_name=name,
                standalone_module=module,
                poller_in_process=False,
            )
        )

        assert result.poller is None
        assert result.reconciler is build.return_value
        build.return_value.start.assert_awaited_once()
        poller_cls.assert_not_called()
        assert build.call_args.kwargs["name"] == name

    async def test_both_start_when_the_poller_is_in_process(
        self, started: Any, label: str, name: str, module: str
    ) -> None:
        poller_cls, build = started

        result = await _start_video_lifecycle_workers(
            **_kwargs(provider_label=label, reconciler_name=name, standalone_module=module)
        )

        assert result.poller is poller_cls.return_value
        assert result.reconciler is build.return_value
        poller_cls.return_value.start.assert_awaited_once()
        build.return_value.start.assert_awaited_once()

    async def test_nothing_starts_when_workers_are_disabled(
        self, started: Any, label: str, name: str, module: str
    ) -> None:
        poller_cls, build = started

        result = await _start_video_lifecycle_workers(
            **_kwargs(
                provider_label=label,
                reconciler_name=name,
                standalone_module=module,
                workers_enabled=False,
            )
        )

        assert (result.poller, result.reconciler) == (None, None)
        poller_cls.assert_not_called()
        build.assert_not_called()

    @pytest.mark.parametrize("missing", ["job_service", "compensator"])
    async def test_nothing_starts_for_an_unconfigured_provider(
        self, started: Any, label: str, name: str, module: str, missing: str
    ) -> None:
        poller_cls, build = started

        result = await _start_video_lifecycle_workers(
            **_kwargs(
                provider_label=label,
                reconciler_name=name,
                standalone_module=module,
                **{missing: None},
            )
        )

        assert (result.poller, result.reconciler) == (None, None)
        poller_cls.assert_not_called()
        build.assert_not_called()


async def test_a_disabled_poller_logs_a_warning(started: Any) -> None:
    del started
    with patch("src.api.dependencies.common.logger") as log:
        await _start_video_lifecycle_workers(**_kwargs(poller_in_process=False))

    event = log.warning.call_args.args[0]
    assert event == "alibaba.video_poller_disabled"
    assert "settle only on read" in log.warning.call_args.kwargs["hint"]


def test_the_poller_flags_default_to_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "test_secret_key_for_testing_only_256bits_long")
    settings = hermetic_settings()

    assert settings.grok_video_worker_in_process is True
    assert settings.wan_video_worker_in_process is True


def test_no_reconciler_is_built_under_a_poller_flag_guard() -> None:
    """Structural guard: reconcilers must never be gated by ``*_video_worker_in_process``."""
    tree = ast.parse(Path("src/api/dependencies/common.py").read_text())
    calls: list[tuple[ast.Call, list[str]]] = []

    def visit(node: ast.AST, guards: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = guards
            if isinstance(node, ast.If) and child in node.body:
                inner = [*guards, ast.unparse(node.test)]
            if isinstance(child, ast.Call):
                func = ast.unparse(child.func)
                if func.endswith(
                    ("AcceptedTaskReconcilerWorker", "build_receipt_reconciler_worker")
                ):
                    calls.append((child, inner))
            visit(child, inner)

    visit(tree, [])

    assert calls
    assert not [g for _, guards in calls for g in guards if "video_worker_in_process" in g]
    assert not [c for c, _ in calls if ast.unparse(c.func).endswith("AcceptedTaskReconcilerWorker")]
