"""The effective expiry has ONE definition, and the ``expiring_soon`` cursor reads exactly it (R1).

The SQL ordering of ``expiring_soon`` is computed from ``LibraryAssetRow.expiry_key`` (a held asset
is ``PINNED_EXPIRY``); the next cursor must carry that same column - never the stored
``expires_at`` of a held row (in the past) and never a value recomputed in Python - or the keyset
resumes in the wrong place. The behaviour is pinned in
``tests/integration/test_library_expiry_holds.py``; these tests pin the wiring.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.api.schemas.pagination import decode_library_cursor, encode_library_cursor
from src.api.services import library as library_service_module
from src.api.services.library import LibraryService, cursor_timestamp
from src.core.enums import LibrarySort
from src.core.library_limits import PINNED_EXPIRY
from src.core.library_ref import LibraryAssetSource
from src.db.repositories.library import LibraryAssetRow


def _row(*, expires_at: datetime, expiry_key: datetime, created_at: datetime) -> LibraryAssetRow:
    return LibraryAssetRow(
        source=LibraryAssetSource.UPLOAD,
        id=uuid4(),
        created_at=created_at,
        expires_at=expires_at,
        expiry_key=expiry_key,
        content_type="image/png",
        size_bytes=1,
        width=1,
        height=1,
        duration_ms=None,
        original_filename=None,
        display_filename=None,
        job_id=None,
        model=None,
        generation_type=None,
        display_title=None,
        is_favorite=False,
        project_id=None,
        held=expiry_key == PINNED_EXPIRY,
    )


NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
HELD_ROW = _row(
    expires_at=NOW - timedelta(hours=48),
    expiry_key=PINNED_EXPIRY,
    created_at=NOW - timedelta(days=9),
)


class TestPinnedExpiry:
    def test_is_the_last_day_of_9999_in_utc(self) -> None:
        assert datetime(9999, 12, 31, tzinfo=UTC) == PINNED_EXPIRY
        assert PINNED_EXPIRY.utcoffset() == timedelta(0)

    def test_survives_a_cursor_round_trip(self) -> None:
        cursor = encode_library_cursor(
            PINNED_EXPIRY, "output", uuid4(), sort=LibrarySort.EXPIRING_SOON.value
        )

        decoded, source, _ = decode_library_cursor(cursor, expected_sort="expiring_soon")

        assert decoded == PINNED_EXPIRY
        assert source == "output"


class TestCursorTimestamp:
    def test_expiring_soon_uses_the_effective_expiry_of_a_held_row(self) -> None:
        assert cursor_timestamp(HELD_ROW, LibrarySort.EXPIRING_SOON) == PINNED_EXPIRY
        assert cursor_timestamp(HELD_ROW, LibrarySort.EXPIRING_SOON) != HELD_ROW.expires_at

    def test_expiring_soon_uses_the_stored_expiry_of_an_unheld_row(self) -> None:
        unheld = _row(
            expires_at=NOW + timedelta(days=3),
            expiry_key=NOW + timedelta(days=3),
            created_at=NOW - timedelta(days=4),
        )

        assert cursor_timestamp(unheld, LibrarySort.EXPIRING_SOON) == unheld.expires_at

    def test_the_other_sorts_keep_their_created_at_key(self) -> None:
        for sort in (LibrarySort.NEWEST, LibrarySort.OLDEST):
            assert cursor_timestamp(HELD_ROW, sort) == HELD_ROW.created_at


def _attribute_names(function: ast.AsyncFunctionDef | ast.FunctionDef) -> set[str]:
    return {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}


def _function(source: str, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


class TestNoExpiresAtInTheCursorPath:
    """A static hold: the ``expiring_soon`` cursor never reads the raw ``expires_at`` column."""

    def test_cursor_timestamp_reads_expiry_key_and_no_expires_at(self) -> None:
        function = _function(inspect.getsource(cursor_timestamp), "cursor_timestamp")

        attributes = _attribute_names(function)

        assert "expiry_key" in attributes
        assert "expires_at" not in attributes

    def test_list_assets_builds_the_cursor_through_cursor_timestamp_only(self) -> None:
        module = ast.parse(inspect.getsource(library_service_module))
        [service] = [
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == LibraryService.__name__
        ]
        [list_assets] = [
            node
            for node in service.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "list_assets"
        ]

        called = {
            node.func.id
            for node in ast.walk(list_assets)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        assert {"cursor_timestamp", "encode_library_cursor"} <= called
        assert "expires_at" not in _attribute_names(list_assets)
