"""Studio plan schema v2: track-level ``instrumental``; a plan may have no sections.

Revision ID: 056
Revises: 055
Create Date: 2026-11-13 00:00:00.000000

**Data only** — no table or column changes, so ``alembic check`` stays clean.

``PlanDocument`` (``src/studio/documents.py``) gained ``instrumental: bool = False`` between
``tempo_bpm`` and ``global_style``, and ``sections`` may now be empty. A breaking change to a
stored document is a schema bump plus a data migration (D5: no read-time shims), so every plan
version row at ``schema_version = 1`` is rewritten to ``2``:

* ``content`` is re-encoded with the new field order (``instrumental`` = ``false``);
* ``content_hash`` is recomputed, because it is ``sha256("plan\\n\\n" + canonical-json)`` and the
  canonical JSON just changed — without this, an identical re-create of a pre-existing plan would
  no longer be recognised as a replay and ``version_unchanged`` could not fire;
* ``schema_version`` becomes ``2``.

Lyrics versions are untouched (their hash covers the plan version's *id*, not its hash).

This migration is **self-contained**: it never imports application code, so a later change to
``PlanDocument`` cannot change what an old migration does. The frozen ``msgspec`` copies below
must keep the application Structs' field order and defaults (msgspec encodes every field in
declaration order); ``tests/integration/test_studio_plan_schema_v2_migration.py`` proves the
resulting hash equals the application's ``content_hash``.

Rows are processed ``_BATCH`` at a time (keyset on ``id``), so memory stays bounded.

Downgrade refuses (raises, changing nothing) while any v2 plan holds something v1 cannot say: an
``instrumental`` plan, or a plan without sections.
"""

import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

import msgspec
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "056"
down_revision: str | Sequence[str] | None = "055"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BATCH = 500
_PLAN_HEAD = b"plan\n\n"  # f"{kind}\n{based_on_plan_version_id or ''}\n" for a plan


# -- frozen copies of the stored document shapes (ids and kinds are plain strings) -------------


class _StyleV(msgspec.Struct, kw_only=True, frozen=True, forbid_unknown_fields=True):
    tags: list[str] = []
    avoid: list[str] = []


class _SectionV(msgspec.Struct, kw_only=True, frozen=True, forbid_unknown_fields=True):
    id: str
    kind: str
    label: str | None = None
    duration_seconds: int | None = None
    style: _StyleV = _StyleV()
    delivery: list[str] = []
    instrumental: bool = False
    notes: str | None = None


class _PlanV1(msgspec.Struct, kw_only=True, frozen=True, forbid_unknown_fields=True):
    title: str
    idea: str | None = None
    tempo_bpm: int | None = None
    global_style: _StyleV = _StyleV()
    references: list[str] = []
    sections: list[_SectionV]


class _PlanV2(msgspec.Struct, kw_only=True, frozen=True, forbid_unknown_fields=True):
    title: str
    idea: str | None = None
    tempo_bpm: int | None = None
    instrumental: bool = False
    global_style: _StyleV = _StyleV()
    references: list[str] = []
    sections: list[_SectionV] = []


_VERSIONS = sa.table(
    "studio_document_versions",
    sa.column("id", postgresql.UUID(as_uuid=True)),
    sa.column("kind", sa.String),
    sa.column("schema_version", sa.SmallInteger),
    sa.column("content", postgresql.JSONB),
    sa.column("content_hash", sa.String),
)


def _plan_batches(schema_version: int) -> Iterator[list[Any]]:
    """``(id, content)`` rows of the plan versions at ``schema_version``, ``_BATCH`` at a time.

    Keyset pagination on ``id`` (never ``OFFSET``), so memory stays bounded however many versions
    exist and the batches stay valid while earlier ones are being rewritten.
    """
    last: Any = None
    while True:
        query = (
            sa.select(_VERSIONS.c.id, _VERSIONS.c.content)
            .where(_VERSIONS.c.kind == "plan", _VERSIONS.c.schema_version == schema_version)
            .order_by(_VERSIONS.c.id)
            .limit(_BATCH)
        )
        if last is not None:
            query = query.where(_VERSIONS.c.id > last)
        batch = list(op.get_bind().execute(query))
        if not batch:
            return
        yield batch
        last = batch[-1][0]


def _rewrite(rows: list[tuple[Any, bytes]], schema_version: int) -> None:
    """Store ``(id, canonical json)`` pairs with the schema version and the matching hash."""
    statement = (
        sa.update(_VERSIONS)
        .where(_VERSIONS.c.id == sa.bindparam("b_id"))
        .values(
            content=sa.bindparam("b_content", type_=postgresql.JSONB),
            content_hash=sa.bindparam("b_hash"),
            schema_version=schema_version,
        )
    )
    op.get_bind().execute(
        statement,
        [
            {
                "b_id": row_id,
                "b_content": msgspec.json.decode(encoded),
                "b_hash": hashlib.sha256(_PLAN_HEAD + encoded).hexdigest(),
            }
            for row_id, encoded in rows
        ],
    )


def _is_representable_in_v1(plan: _PlanV2) -> bool:
    return not plan.instrumental and bool(plan.sections)


def upgrade() -> None:
    """Rewrite every v1 plan version as v2."""
    for batch in _plan_batches(1):
        _rewrite(
            [
                (row_id, msgspec.json.encode(msgspec.convert(content, _PlanV2)))
                for row_id, content in batch
            ],
            2,
        )


def downgrade() -> None:
    """Rewrite every v2 plan version as v1 — unless one cannot be represented there.

    The check runs over **all** rows before the first one is rewritten, so a refusal changes
    nothing.

    Raises:
        RuntimeError: Some v2 plans are ``instrumental`` or have no sections. Nothing was changed;
            delete or edit those versions first.
    """
    unrepresentable = sum(
        not _is_representable_in_v1(msgspec.convert(content, _PlanV2))
        for batch in _plan_batches(2)
        for _, content in batch
    )
    if unrepresentable:
        raise RuntimeError(
            f"cannot downgrade to 055: {unrepresentable} studio plan version(s) are instrumental "
            "or have no sections, which schema_version 1 cannot represent; "
            "delete or edit those versions first"
        )
    for batch in _plan_batches(2):
        rewritten: list[tuple[Any, bytes]] = []
        for row_id, content in batch:
            plan = msgspec.convert(content, _PlanV2)
            v1 = _PlanV1(
                title=plan.title,
                idea=plan.idea,
                tempo_bpm=plan.tempo_bpm,
                global_style=plan.global_style,
                references=plan.references,
                sections=plan.sections,
            )
            rewritten.append((row_id, msgspec.json.encode(v1)))
        _rewrite(rewritten, 1)
