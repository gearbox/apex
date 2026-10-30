"""Unit tests for the handshake helpers in ``tests/integration/pg_locks.py``.

The helpers exist so a concurrency test whose producer fails reports that
failure instead of hanging until the CI job limit. Each failure mode is pinned
here with short deadlines; none of these tests touches a database.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.integration.pg_locks import wait_for_signal, wait_for_start_signal


class _ProducerError(Exception):
    """Distinct failure, so a test can prove the producer's own error surfaced."""


async def _cancel(task: asyncio.Task[object]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestWaitForSignal:
    async def test_returns_once_producer_signals(self) -> None:
        signal = asyncio.Event()
        release = asyncio.Event()

        async def producer() -> None:
            signal.set()
            await release.wait()

        task = asyncio.create_task(producer())
        try:
            await wait_for_signal(signal, task, timeout=1.0)
            assert not task.done()
        finally:
            release.set()
            await task

    async def test_surfaces_the_producers_own_exception(self) -> None:
        signal = asyncio.Event()

        async def producer() -> None:
            raise _ProducerError

        task = asyncio.create_task(producer())
        with pytest.raises(_ProducerError):
            await wait_for_signal(signal, task, timeout=1.0)

    async def test_producer_finishing_without_signal_fails_fast(self) -> None:
        signal = asyncio.Event()

        async def producer() -> None:
            return

        task = asyncio.create_task(producer())
        with pytest.raises(AssertionError, match="without signalling"):
            await wait_for_signal(signal, task, timeout=1.0)

    async def test_times_out_when_nobody_signals(self) -> None:
        signal = asyncio.Event()

        async def producer() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(producer())
        try:
            with pytest.raises(TimeoutError, match="not set within"):
                await wait_for_signal(signal, task, timeout=0.05)
        finally:
            await _cancel(task)


class TestWaitForStartSignal:
    async def test_returns_once_signal_is_set(self) -> None:
        signal = asyncio.Event()
        signal.set()
        await wait_for_start_signal(signal, timeout=1.0)

    async def test_times_out_with_an_explanatory_message(self) -> None:
        with pytest.raises(TimeoutError, match="first racer finished"):
            await wait_for_start_signal(asyncio.Event(), timeout=0.05)
