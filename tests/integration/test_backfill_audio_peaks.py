"""The detail-peaks backfill CLI (S3a W6 / W9.8): real PostgreSQL, real ffmpeg, real
``R2StorageService`` over a fake S3 client.

The S3 client is the only fake: a missing key raises the ``NoSuchKey`` ``ClientError`` the real
service maps to ``StorageNotFoundError``. The backfill scans every owner's profiles, so each test
first fills whatever other tests left behind.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError
from sqlalchemy import select, text
from structlog.testing import capture_logs
from typer.testing import CliRunner

from src.api.services.media_ingest.audio import compute_detail_peaks
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.cli.backfill_audio_peaks import BackfillStats, app, run_backfill
from src.db.models.library import LibraryAudioProfile
from tests.audio_support import require_ffmpeg, run
from tests.media_ingest_support import make_media_ingestor

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from uuid import UUID

    from src.db.models.storage import UserImage
    from src.db.models.user import User
    from tests.library_holds_support import HoldWorld


@pytest.fixture(autouse=True)
async def _no_leftover_work(hold_world: HoldWorld) -> AsyncGenerator[None]:
    """Profiles other tests left without detail are none of this module's business."""
    async with hold_world.factory() as session:
        await session.execute(
            text(
                "UPDATE library_audio_profiles SET detail_peaks = '\\x0000', "
                "detail_peaks_version = 1 WHERE detail_peaks IS NULL"
            )
        )
        await session.commit()
    yield


@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """An 8 s sine: exactly 400 detail buckets."""
    path = tmp_path_factory.mktemp("backfill") / "sine8.wav"
    run(
        [
            require_ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8:sample_rate=44100",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path.read_bytes()


class _Bucket:
    """The fake S3 client behind a real ``R2StorageService``."""

    def __init__(
        self,
        objects: dict[str, bytes],
        *,
        before_get: Callable[[str], Awaitable[None]] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        self.objects = objects
        self.before_get = before_get
        self.errors = errors or {}
        self.requested: list[str] = []

    async def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: ARG002
        self.requested.append(Key)
        if self.before_get is not None:
            await self.before_get(Key)
        code = self.errors.get(Key) or (None if Key in self.objects else "NoSuchKey")
        if code is not None:
            raise ClientError({"Error": {"Code": code, "Message": "nope"}}, "GetObject")
        body = AsyncMock()
        body.read.return_value = self.objects[Key]
        return {"Body": body}


@asynccontextmanager
async def _r2(bucket: _Bucket) -> AsyncIterator[R2StorageService]:
    storage = R2StorageService(
        R2StorageSettings(
            account_id="test",
            access_key_id="test",
            secret_access_key="test",
            bucket_name="apex-user-content",
        )
    )

    @asynccontextmanager
    async def client() -> AsyncIterator[_Bucket]:
        yield bucket

    with patch.object(storage, "_get_client", client):
        yield storage


async def _run(
    world: HoldWorld, bucket: _Bucket, *, batch_size: int = 50, dry_run: bool = False
) -> BackfillStats:
    async with _r2(bucket) as r2, world.factory() as session:
        return await run_backfill(
            session, r2, make_media_ingestor(), batch_size=batch_size, dry_run=dry_run
        )


async def _audios(world: HoldWorld, user: User, count: int) -> list[UserImage]:
    return [await world.audio(user, duration_ms=8_000) for _ in range(count)]


async def _profile(world: HoldWorld, upload_id: UUID) -> LibraryAudioProfile:
    async with world.factory() as session:
        return (
            await session.execute(
                select(LibraryAudioProfile).where(LibraryAudioProfile.upload_id == upload_id)
            )
        ).scalar_one()


async def test_it_fills_every_missing_row_across_keyset_pages(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    audios = await _audios(hold_world, user, 5)
    overview = (await _profile(hold_world, audios[0].id)).peaks
    bucket = _Bucket({audio.storage_key: sine_wav for audio in audios})

    stats = await _run(hold_world, bucket, batch_size=2)

    assert (stats.scanned, stats.done, stats.skipped, dict(stats.failed)) == (5, 5, 0, {})
    for audio in audios:
        profile = await _profile(hold_world, audio.id)
        assert profile.detail_peaks is not None
        assert len(profile.detail_peaks) == 400 * 2
        assert profile.detail_peaks_version == 1
        assert profile.peaks == overview, "the overview is never touched"
        assert profile.peaks_version == 1
    assert sorted(bucket.requested) == sorted(audio.storage_key for audio in audios)


async def test_the_values_are_those_of_ingest_s_decode(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    (audio,) = await _audios(hold_world, user, 1)
    prepared = await make_media_ingestor().prepare_audio(sine_wav)

    await _run(hold_world, _Bucket({audio.storage_key: prepared.data}))

    assert (await _profile(hold_world, audio.id)).detail_peaks == prepared.detail_peaks


async def test_a_second_run_finds_nothing_to_do(hold_world: HoldWorld, sine_wav: bytes) -> None:
    user = await hold_world.user()
    audios = await _audios(hold_world, user, 2)
    bucket = _Bucket({audio.storage_key: sine_wav for audio in audios})
    await _run(hold_world, bucket)
    before = [(await _profile(hold_world, audio.id)).detail_peaks for audio in audios]
    again = _Bucket(dict(bucket.objects))

    stats = await _run(hold_world, again)

    assert (stats.scanned, stats.done, stats.skipped, dict(stats.failed)) == (0, 0, 0, {})
    assert again.requested == []
    assert [(await _profile(hold_world, audio.id)).detail_peaks for audio in audios] == before


async def test_a_row_that_already_has_detail_is_neither_read_nor_rewritten(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    done = await hold_world.audio(user, detail_peaks=b"\x01\x02")
    (missing,) = await _audios(hold_world, user, 1)
    bucket = _Bucket({missing.storage_key: sine_wav, done.storage_key: sine_wav})

    stats = await _run(hold_world, bucket)

    assert (stats.scanned, stats.done) == (1, 1)
    assert bucket.requested == [missing.storage_key]
    assert (await _profile(hold_world, done.id)).detail_peaks == b"\x01\x02"


async def test_two_runs_at_once_write_each_row_once(hold_world: HoldWorld, sine_wav: bytes) -> None:
    """Both runs read the same three rows before either writes; the conditional UPDATE lets
    exactly one of them write each."""
    user = await hold_world.user()
    audios = await _audios(hold_world, user, 3)
    both_scanned = asyncio.Barrier(2)

    def bucket() -> _Bucket:
        waited: list[bool] = []

        async def before_get(_key: str) -> None:
            if not waited:  # each run's first download: its keyset page is already read
                waited.append(True)
                await asyncio.wait_for(both_scanned.wait(), 20)

        return _Bucket({a.storage_key: sine_wav for a in audios}, before_get=before_get)

    first, second = await asyncio.gather(_run(hold_world, bucket()), _run(hold_world, bucket()))

    assert (first.scanned, second.scanned) == (3, 3)
    assert first.done + second.done == 3, (first, second)
    assert first.skipped + second.skipped == 3, (first, second)
    assert not first.failed
    assert not second.failed
    for audio in audios:
        profile = await _profile(hold_world, audio.id)
        assert profile.detail_peaks is not None
        assert len(profile.detail_peaks) == 800


async def test_a_missing_object_counts_as_missing_and_the_scan_goes_on(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    gone, kept = await _audios(hold_world, user, 2)

    with capture_logs() as logs:
        stats = await _run(hold_world, _Bucket({kept.storage_key: sine_wav}), batch_size=1)

    assert (stats.scanned, stats.done, dict(stats.failed)) == (2, 1, {"missing": 1})
    assert (await _profile(hold_world, gone.id)).detail_peaks is None
    assert (await _profile(hold_world, kept.id)).detail_peaks is not None
    (failure,) = [log for log in logs if log["event"] == "backfill.audio_peaks.row_failed"]
    assert failure["error_type"] == "missing"
    assert set(failure) == {"event", "profile_id", "error_type", "log_level"}


async def test_other_failures_are_counted_by_type_and_do_not_stop_the_scan(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    corrupt, forbidden, good = await _audios(hold_world, user, 3)
    bucket = _Bucket(
        {corrupt.storage_key: b"not audio" * 200, good.storage_key: sine_wav},
        errors={forbidden.storage_key: "AccessDenied"},
    )

    stats = await _run(hold_world, bucket)

    assert dict(stats.failed) == {"InvalidAudioError": 1, "StorageDownloadError": 1}
    assert (stats.scanned, stats.done) == (3, 1)
    assert (await _profile(hold_world, good.id)).detail_peaks is not None
    for failed in (corrupt, forbidden):
        assert (await _profile(hold_world, failed.id)).detail_peaks is None


async def test_a_database_failure_on_one_row_is_rolled_back_and_the_scan_goes_on(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    audios = await _audios(hold_world, user, 2)
    bucket = _Bucket({audio.storage_key: sine_wav for audio in audios})
    calls: list[int] = []

    def too_long_once(pcm: bytes) -> bytes:
        calls.append(1)
        # the first row gets a level the CHECK refuses; the next one a good level
        return bytes(90_102) if len(calls) == 1 else compute_detail_peaks(pcm)

    with patch("src.api.services.media_ingest.service.compute_detail_peaks", too_long_once):
        stats = await _run(hold_world, bucket)

    assert (stats.scanned, stats.done, dict(stats.failed)) == (2, 1, {"IntegrityError": 1})


async def test_a_dry_run_reports_and_writes_nothing(hold_world: HoldWorld, sine_wav: bytes) -> None:
    user = await hold_world.user()
    good, gone = await _audios(hold_world, user, 2)

    stats = await _run(hold_world, _Bucket({good.storage_key: sine_wav}), dry_run=True)

    assert (stats.scanned, stats.done, dict(stats.failed)) == (2, 1, {"missing": 1})
    assert (await _profile(hold_world, good.id)).detail_peaks is None
    assert (await _profile(hold_world, gone.id)).detail_peaks is None


async def test_an_audio_output_is_filled_from_its_own_key(
    hold_world: HoldWorld, sine_wav: bytes
) -> None:
    user = await hold_world.user()
    job = await hold_world.job(user)
    output = await hold_world.output(user, job, content_type="audio/mpeg")
    profile_id = uuid4()
    async with hold_world.factory() as session:
        session.add(
            LibraryAudioProfile(
                id=profile_id,
                output_id=output.id,
                codec="mp3",
                sample_rate_hz=44100,
                channels=2,
                peaks=bytes(2048),
                peaks_version=1,
            )
        )
        await session.commit()

    stats = await _run(hold_world, _Bucket({output.storage_key: sine_wav}))

    assert (stats.scanned, stats.done) == (1, 1)
    async with hold_world.factory() as session:
        profile = await session.get(LibraryAudioProfile, profile_id)
        assert profile is not None
        assert profile.detail_peaks is not None


def test_the_command_refuses_to_run_without_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "")

    result = CliRunner().invoke(app, ["--dry-run"])

    assert result.exit_code == 1, result.output
    assert "R2 is not configured" in result.output
