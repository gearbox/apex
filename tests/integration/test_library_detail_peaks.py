"""``GET /v1/library/assets/{ref}/peaks?level=detail`` against a real PostgreSQL (S3a W6 / W9.8).

The stored detail level is a recognisable byte pattern, so a window is right only if it is the
exact slice.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.library import LibraryController
from src.api.services.library import LibraryService
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.db.models.library import LibraryAudioProfile
from tests.library_holds_support import HoldWorld, bearer, build_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from httpx import Response
    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.storage import UserImage
    from src.db.models.user import User

BUCKETS = 1_000
"""20 s of audio."""
STORED = bytes((index * 7) % 256 for index in range(2 * BUCKETS))


def _signed(values: bytes) -> list[int]:
    return [b - 256 if b > 127 else b for b in values]


def _slice(first: int, end: int) -> list[int]:
    return _signed(STORED[2 * first : 2 * end])


@pytest.fixture
def factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    app = build_app(factory, [LibraryController])
    async with AsyncTestClient(app=app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def user(hold_world: HoldWorld) -> User:
    return await hold_world.user()


@pytest_asyncio.fixture
async def audio(hold_world: HoldWorld, user: User) -> UserImage:
    return await hold_world.audio(user, duration_ms=BUCKETS * 20, detail_peaks=STORED)


async def _peaks(
    client: AsyncTestClient[Litestar], user: User, asset_id: object, query: str = "level=detail"
) -> Response:
    return await client.get(
        f"/v1/library/assets/upload:{asset_id}/peaks?{query}", headers=bearer(user.id)
    )


class TestWindows:
    async def test_without_a_window_it_is_the_whole_level(
        self, client: AsyncTestClient[Litestar], user: User, audio: UserImage
    ) -> None:
        response = await _peaks(client, user, audio.id)

        assert response.status_code == 200
        assert response.json() == {
            "level": "detail",
            "version": 1,
            "bucket_ms": 20,
            "total_buckets": BUCKETS,
            "from_bucket": 0,
            "peaks": _signed(STORED),
        }

    @pytest.mark.parametrize(
        ("from_ms", "to_ms", "first", "end"),
        [
            (0, 20, 0, 1),
            (2_000, 3_000, 100, 150),
            (2_010, 3_001, 100, 151),  # [from // 20, ceil(to / 20))
            (2_019, 2_020, 100, 101),
            (19_980, 20_000, 999, 1_000),
        ],
    )
    async def test_a_window_is_the_exact_slice(
        self,
        client: AsyncTestClient[Litestar],
        user: User,
        audio: UserImage,
        from_ms: int,
        to_ms: int,
        first: int,
        end: int,
    ) -> None:
        response = await _peaks(
            client, user, audio.id, f"level=detail&from_ms={from_ms}&to_ms={to_ms}"
        )

        body = response.json()
        assert response.status_code == 200, body
        assert (body["from_bucket"], body["total_buckets"]) == (first, BUCKETS)
        assert body["peaks"] == _slice(first, end)

    @pytest.mark.parametrize(
        ("from_ms", "to_ms", "first", "end"),
        [
            pytest.param(19_000, 100_000, 950, 1_000, id="runs-past-the-end"),
            pytest.param(20_000, 21_000, 1_000, 1_000, id="starts-at-the-end"),
            pytest.param(500_000, 600_000, 1_000, 1_000, id="far-past-the-end"),
            pytest.param(10**30, 10**30 + 1, 1_000, 1_000, id="absurd"),
        ],
    )
    async def test_a_window_past_the_end_is_clamped(
        self,
        client: AsyncTestClient[Litestar],
        user: User,
        audio: UserImage,
        from_ms: int,
        to_ms: int,
        first: int,
        end: int,
    ) -> None:
        response = await _peaks(
            client, user, audio.id, f"level=detail&from_ms={from_ms}&to_ms={to_ms}"
        )

        body = response.json()
        assert response.status_code == 200, body
        assert body["from_bucket"] == first
        assert body["peaks"] == _slice(first, end)

    async def test_the_widest_window_is_accepted(
        self, client: AsyncTestClient[Litestar], user: User, audio: UserImage
    ) -> None:
        response = await _peaks(client, user, audio.id, "level=detail&from_ms=0&to_ms=600000")

        assert response.status_code == 200
        assert response.json()["peaks"] == _signed(STORED)

    @pytest.mark.parametrize(
        "query",
        [
            "level=detail&from_ms=0&to_ms=600001",
            "level=detail&from_ms=100&to_ms=100",
            "level=detail&from_ms=200&to_ms=100",
            "level=detail&from_ms=-1&to_ms=100",
            "level=detail&from_ms=0",
            "level=detail&to_ms=100",
            "from_ms=0&to_ms=100",  # a window on the overview
            "level=overview&to_ms=100",
        ],
    )
    async def test_a_bad_window_is_422(
        self, client: AsyncTestClient[Litestar], user: User, audio: UserImage, query: str
    ) -> None:
        response = await _peaks(client, user, audio.id, query)

        assert (response.status_code, response.json()["error"]) == (422, "invalid_peaks_window")

    @pytest.mark.parametrize("query", ["level=words", "level=detail&from_ms=a&to_ms=2"])
    async def test_a_malformed_parameter_is_400(
        self, client: AsyncTestClient[Litestar], user: User, audio: UserImage, query: str
    ) -> None:
        response = await _peaks(client, user, audio.id, query)

        assert response.status_code == 400

    async def test_the_overview_is_unchanged_but_for_its_level(
        self, client: AsyncTestClient[Litestar], user: User, audio: UserImage
    ) -> None:
        implicit = await _peaks(client, user, audio.id, "")
        explicit = await _peaks(client, user, audio.id, "level=overview")

        assert implicit.json() == explicit.json()
        assert implicit.json()["level"] == "overview"
        assert implicit.json()["buckets"] == 1024


class TestWhatCannotBeRead:
    async def test_detail_not_computed_yet_is_409_while_the_overview_works(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld, user: User
    ) -> None:
        old = await hold_world.audio(user)  # stored before S3a: detail is NULL

        detail = await _peaks(client, user, old.id)
        windowed = await _peaks(client, user, old.id, "level=detail&from_ms=0&to_ms=100")
        overview = await _peaks(client, user, old.id, "")

        assert (detail.status_code, detail.json()["error"]) == (409, "peaks_not_ready")
        assert (windowed.status_code, windowed.json()["error"]) == (409, "peaks_not_ready")
        assert overview.status_code == 200

    async def test_an_image_is_422_media_kind_not_supported(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld, user: User
    ) -> None:
        image = await hold_world.upload(user)

        response = await _peaks(client, user, image.id)

        assert (response.status_code, response.json()["error"]) == (
            422,
            "media_kind_not_supported",
        )

    @pytest.mark.parametrize("case", ["foreign", "unknown", "malformed", "other_product"])
    async def test_it_is_one_404_whatever_the_reason(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld, user: User, case: str
    ) -> None:
        if case == "foreign":
            asset = await hold_world.audio(await hold_world.user(), detail_peaks=STORED)
            ref = f"upload:{asset.id}"
        elif case == "other_product":
            other = await hold_world.user(product_id="synthara")
            ref = f"upload:{(await hold_world.audio(other, detail_peaks=STORED)).id}"
        else:
            ref = f"upload:{uuid4()}" if case == "unknown" else "nope"

        response = await client.get(
            f"/v1/library/assets/{ref}/peaks?level=detail", headers=bearer(user.id)
        )

        assert (response.status_code, response.json()["error"]) == (404, "asset_not_found")

    async def test_an_audio_output_serves_its_detail_too(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld, user: User
    ) -> None:
        job = await hold_world.job(user)
        output = await hold_world.output(user, job, content_type="audio/mpeg")
        async with hold_world.factory() as session:
            session.add(
                LibraryAudioProfile(
                    id=uuid4(),
                    output_id=output.id,
                    codec="mp3",
                    sample_rate_hz=44100,
                    channels=2,
                    peaks=bytes(2048),
                    peaks_version=1,
                    detail_peaks=STORED,
                    detail_peaks_version=1,
                )
            )
            await session.commit()

        response = await client.get(
            f"/v1/library/assets/output:{output.id}/peaks?level=detail&from_ms=40&to_ms=80",
            headers=bearer(user.id),
        )

        assert response.json()["peaks"] == _slice(2, 4)

    async def test_an_audio_without_a_profile_is_a_broken_invariant(
        self, hold_world: HoldWorld, user: User
    ) -> None:
        bare = await hold_world.audio(user, with_profile=False)

        async with hold_world.factory() as session:
            with pytest.raises(RuntimeError, match="no audio profile"):
                await LibraryService(session=session).get_audio_detail_peaks(
                    format_asset_ref(LibraryAssetSource.UPLOAD, bare.id),
                    user.id,
                    "vex",
                    session=session,
                )


class TestOneStatement:
    async def test_a_window_is_one_select_that_slices_in_sql(
        self, db_engine: AsyncEngine, hold_world: HoldWorld, user: User, audio: UserImage
    ) -> None:
        """Invariant 6: one statement, and only the window ever leaves the database."""
        statements: list[str] = []

        def on_execute(_conn: object, _cursor: object, statement: str, *_: object) -> None:
            statements.append(" ".join(statement.split()).lower())

        event.listen(db_engine.sync_engine, "before_cursor_execute", on_execute)
        try:
            async with hold_world.factory() as session:
                peaks = await LibraryService(session=session).get_audio_detail_peaks(
                    format_asset_ref(LibraryAssetSource.UPLOAD, audio.id),
                    user.id,
                    "vex",
                    session=session,
                    from_ms=2_000,
                    to_ms=3_000,
                )
        finally:
            event.remove(db_engine.sync_engine, "before_cursor_execute", on_execute)

        assert peaks is not None
        assert len(peaks.peaks) == 2 * 50
        assert len(statements) == 1, statements
        (statement,) = statements
        assert statement.startswith("select")
        # the whole column is named only inside substring() and octet_length()
        uses = re.findall(r"(\w+)\(library_audio_profiles\.detail_peaks\b(?!_)", statement)
        assert sorted(uses) == ["octet_length", "substring"]
        assert len(re.findall(r"library_audio_profiles\.detail_peaks\b(?!_)", statement)) == 2


class TestTheDatabaseChecks:
    @pytest.mark.parametrize(
        ("values", "constraint"),
        [
            pytest.param(
                {"detail_peaks": bytes(4), "detail_peaks_version": None},
                "ck_library_audio_profiles_detail_peaks_pair",
                id="peaks-without-version",
            ),
            pytest.param(
                {"detail_peaks": None, "detail_peaks_version": 1},
                "ck_library_audio_profiles_detail_peaks_pair",
                id="version-without-peaks",
            ),
            pytest.param(
                {"detail_peaks": bytes(3), "detail_peaks_version": 1},
                "ck_library_audio_profiles_detail_peaks_size",
                id="odd",
            ),
            pytest.param(
                {"detail_peaks": bytes(90_102), "detail_peaks_version": 1},
                "ck_library_audio_profiles_detail_peaks_size",
                id="too-long",
            ),
        ],
    )
    async def test_bad_detail_columns_are_refused(
        self,
        hold_world: HoldWorld,
        audio: UserImage,
        values: dict[str, object],
        constraint: str,
    ) -> None:
        async with hold_world.factory() as session:
            with pytest.raises(IntegrityError, match=constraint):
                await session.execute(
                    text(
                        "UPDATE library_audio_profiles SET detail_peaks = :detail_peaks, "
                        "detail_peaks_version = :detail_peaks_version WHERE upload_id = :upload"
                    ),
                    {**values, "upload": audio.id},
                )

    async def test_the_largest_level_fits(self, hold_world: HoldWorld, user: User) -> None:
        await hold_world.audio(user, detail_peaks=bytes(90_100))
