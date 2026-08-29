"""CLI command: backfill sm/md WEBP thumbnail variants for pre-existing content.

Usage:
    python -m src.cli.backfill_thumbnails run [--product vex] [--only all|outputs|uploads]
           [--dry-run] [--batch-size 100] [--limit N] [--include-video]
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import structlog
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import literal, select, tuple_

from src.api.services.image_thumbnail import make_image_thumbnails
from src.api.services.storage import (
    R2StorageService,
    R2StorageSettings,
    StorageDownloadError,
    StorageNotFoundError,
    StorageType,
)
from src.api.services.thumbnail import extract_video_thumbnail
from src.core.config import Settings
from src.core.enums import MediaKind, media_kind_from_content_type
from src.core.thumbnails import THUMBNAIL_SPECS, label_for_max_edge
from src.db.models.storage import GenerationOutput, UserImage
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository
from src.db.session import DatabaseManager

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)
console = Console()

app = typer.Typer(
    name="backfill-thumbnails",
    help="Backfill sm/md WEBP thumbnail variants for pre-existing content",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _Only(enum.StrEnum):
    all = "all"
    outputs = "outputs"
    uploads = "uploads"


class _RowStatus(enum.Enum):
    SKIPPED_COMPLETE = "skipped_complete"
    SKIPPED_NO_POSTER = "skipped_no_poster"
    UPDATED = "updated"
    FAILED = "failed"


@dataclass
class _RowResult:
    status: _RowStatus
    variants_created: int = 0


@dataclass
class _Stats:
    scanned: int = 0
    skipped_complete: int = 0
    skipped_no_poster: int = 0
    updated: int = 0
    variants_created: int = 0
    failed: int = 0


def _tally(stats: _Stats, result: _RowResult) -> None:
    match result.status:
        case _RowStatus.SKIPPED_COMPLETE:
            stats.skipped_complete += 1
        case _RowStatus.SKIPPED_NO_POSTER:
            stats.skipped_no_poster += 1
        case _RowStatus.UPDATED:
            stats.updated += 1
            stats.variants_created += result.variants_created
        case _RowStatus.FAILED:
            stats.failed += 1


# ---------------------------------------------------------------------------
# Per-row backfill logic
# ---------------------------------------------------------------------------


async def _backfill_output_row(
    session: AsyncSession,
    r2: R2StorageService,
    full: GenerationOutput,
    derivatives: list[GenerationOutput],
    *,
    dry_run: bool,
    include_video: bool,
) -> _RowResult:
    # Count any derivative occupying a known size slot — WEBP or not.
    # A JPEG legacy poster with thumbnail_max_edge=512 occupies "md" and must not
    # be joined by a new WEBP md (build_media would emit two md variants).
    valid: set[str] = set()
    for d in derivatives:
        lbl = label_for_max_edge(d.thumbnail_max_edge)
        if lbl is not None:
            valid.add(lbl)

    missing = [s for s in THUMBNAIL_SPECS if s.label not in valid]
    if not missing:
        return _RowResult(_RowStatus.SKIPPED_COMPLETE)

    source_bytes: bytes
    if _is_video := media_kind_from_content_type(full.content_type) is MediaKind.VIDEO:
        poster = next(
            (d for d in derivatives if d.content_type.startswith("image/")),
            None,
        )
        if poster is not None:
            try:
                source_bytes = await r2.download(poster.storage_key)
            except (StorageNotFoundError, StorageDownloadError):
                logger.warning(
                    "backfill.row.failed",
                    full_id=str(full.id),
                    reason="poster_download_failed",
                )
                return _RowResult(_RowStatus.FAILED)
        elif include_video:
            try:
                video_bytes = await r2.download(full.storage_key)
            except (StorageNotFoundError, StorageDownloadError):
                logger.warning(
                    "backfill.row.failed",
                    full_id=str(full.id),
                    reason="video_download_failed",
                )
                return _RowResult(_RowStatus.FAILED)
            frame = await extract_video_thumbnail(video_bytes)
            if not frame:
                logger.warning(
                    "backfill.row.failed",
                    full_id=str(full.id),
                    reason="frame_extraction_failed",
                )
                return _RowResult(_RowStatus.FAILED)
            source_bytes = frame
        else:
            logger.info("backfill.row.skipped_no_poster", full_id=str(full.id))
            return _RowResult(_RowStatus.SKIPPED_NO_POSTER)
    else:
        try:
            source_bytes = await r2.download(full.storage_key)
        except (StorageNotFoundError, StorageDownloadError):
            logger.warning(
                "backfill.row.failed",
                full_id=str(full.id),
                reason="source_download_failed",
            )
            return _RowResult(_RowStatus.FAILED)

    generated = await make_image_thumbnails(source_bytes, specs=missing)
    if not generated:
        logger.warning(
            "backfill.row.failed",
            full_id=str(full.id),
            reason="thumbnail_generation_failed",
        )
        return _RowResult(_RowStatus.FAILED)

    variants_created = 0
    output_repo = OutputRepository(session)

    for g in generated:
        if dry_run:
            variants_created += 1
            continue

        try:
            up = await r2.upload(
                user_id=full.user_id,
                data=g.result.data,
                content_type=g.result.content_type,
                storage_type=StorageType.OUTPUT,
                job_id=full.job_id,
            )
        except Exception:
            logger.warning(
                "backfill.row.failed",
                full_id=str(full.id),
                reason="r2_upload_failed",
                label=g.spec.label,
            )
            return _RowResult(_RowStatus.FAILED, variants_created)

        try:
            async with session.begin_nested():
                await output_repo.create(
                    id=up.id,
                    user_id=full.user_id,
                    job_id=full.job_id,
                    storage_key=up.storage_key,
                    content_type=g.result.content_type,
                    size_bytes=len(g.result.data),
                    format=g.result.format,
                    output_index=full.output_index,
                    expires_at=full.expires_at,
                    product_id=full.product_id,
                    is_thumbnail=True,
                    parent_output_id=full.id,
                    thumbnail_max_edge=g.spec.max_edge,
                    width=g.result.width,
                    height=g.result.height,
                )
            variants_created += 1
            logger.info(
                "backfill.row.updated",
                full_id=str(full.id),
                label=g.spec.label,
            )
        except Exception:
            with contextlib.suppress(Exception):
                await r2.delete(up.storage_key)
            logger.warning(
                "backfill.row.failed",
                full_id=str(full.id),
                reason="db_create_failed",
                label=g.spec.label,
            )
            return _RowResult(_RowStatus.FAILED, variants_created)

    if variants_created > 0:
        return _RowResult(_RowStatus.UPDATED, variants_created)
    return _RowResult(_RowStatus.SKIPPED_COMPLETE)


async def _backfill_upload_row(
    session: AsyncSession,
    r2: R2StorageService,
    full: UserImage,
    derivatives: list[UserImage],
    *,
    dry_run: bool,
) -> _RowResult:
    valid: set[str] = set()
    for d in derivatives:
        lbl = label_for_max_edge(d.thumbnail_max_edge)
        if lbl is not None:
            valid.add(lbl)

    missing = [s for s in THUMBNAIL_SPECS if s.label not in valid]
    if not missing:
        return _RowResult(_RowStatus.SKIPPED_COMPLETE)

    try:
        source_bytes = await r2.download(full.storage_key)
    except (StorageNotFoundError, StorageDownloadError):
        logger.warning(
            "backfill.row.failed",
            full_id=str(full.id),
            reason="source_download_failed",
        )
        return _RowResult(_RowStatus.FAILED)

    generated = await make_image_thumbnails(source_bytes, specs=missing)
    if not generated:
        logger.warning(
            "backfill.row.failed",
            full_id=str(full.id),
            reason="thumbnail_generation_failed",
        )
        return _RowResult(_RowStatus.FAILED)

    variants_created = 0
    image_repo = UserImageRepository(session)

    for g in generated:
        if dry_run:
            variants_created += 1
            continue

        try:
            up = await r2.upload(
                user_id=full.user_id,
                data=g.result.data,
                content_type=g.result.content_type,
                storage_type=StorageType.UPLOAD,
            )
        except Exception:
            logger.warning(
                "backfill.row.failed",
                full_id=str(full.id),
                reason="r2_upload_failed",
                label=g.spec.label,
            )
            return _RowResult(_RowStatus.FAILED, variants_created)

        try:
            async with session.begin_nested():
                await image_repo.create(
                    id=up.id,
                    user_id=full.user_id,
                    storage_key=up.storage_key,
                    original_filename=f"{up.id}.{g.result.format}",
                    content_type=g.result.content_type,
                    size_bytes=len(g.result.data),
                    format=g.result.format,
                    expires_at=full.expires_at,
                    product_id=full.product_id,
                    is_thumbnail=True,
                    parent_image_id=full.id,
                    thumbnail_max_edge=g.spec.max_edge,
                    width=g.result.width,
                    height=g.result.height,
                )
            variants_created += 1
            logger.info(
                "backfill.row.updated",
                full_id=str(full.id),
                label=g.spec.label,
            )
        except Exception:
            with contextlib.suppress(Exception):
                await r2.delete(up.storage_key)
            logger.warning(
                "backfill.row.failed",
                full_id=str(full.id),
                reason="db_create_failed",
                label=g.spec.label,
            )
            return _RowResult(_RowStatus.FAILED, variants_created)

    if variants_created > 0:
        return _RowResult(_RowStatus.UPDATED, variants_created)
    return _RowResult(_RowStatus.SKIPPED_COMPLETE)


# ---------------------------------------------------------------------------
# Batch scan loops
# ---------------------------------------------------------------------------


async def _scan_outputs(
    session: AsyncSession,
    r2: R2StorageService,
    stats: _Stats,
    *,
    product: str | None,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
    include_video: bool,
) -> None:
    cursor_ts: datetime | None = None
    cursor_id: UUID | None = None
    output_repo = OutputRepository(session)

    while limit is None or stats.scanned < limit:
        effective_batch = batch_size
        if limit is not None:
            effective_batch = min(batch_size, limit - stats.scanned)

        q = select(GenerationOutput).where(GenerationOutput.is_thumbnail.is_(False))
        if product is not None:
            q = q.where(GenerationOutput.product_id == product)
        if cursor_ts is not None and cursor_id is not None:
            q = q.where(
                tuple_(GenerationOutput.created_at, GenerationOutput.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )
        q = q.order_by(
            GenerationOutput.created_at.desc(),
            GenerationOutput.id.desc(),
        ).limit(effective_batch + 1)

        result = await session.execute(q)
        rows = list(result.scalars().all())

        has_more = len(rows) > effective_batch
        batch = rows[:effective_batch]

        if not batch:
            break

        parent_ids: list[UUID] = [row.id for row in batch]
        derivatives_map = await output_repo.batch_derivatives(parent_ids)

        for full in batch:
            row_derivatives = derivatives_map.get(full.id, [])
            row_result = await _backfill_output_row(
                session,
                r2,
                full,
                row_derivatives,
                dry_run=dry_run,
                include_video=include_video,
            )
            stats.scanned += 1
            _tally(stats, row_result)

        if not dry_run:
            await session.commit()

        if not has_more:
            break

        last = batch[-1]
        cursor_ts = last.created_at
        cursor_id = last.id


async def _scan_uploads(
    session: AsyncSession,
    r2: R2StorageService,
    stats: _Stats,
    *,
    product: str | None,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    cursor_ts: datetime | None = None
    cursor_id: UUID | None = None
    image_repo = UserImageRepository(session)

    while limit is None or stats.scanned < limit:
        effective_batch = batch_size
        if limit is not None:
            effective_batch = min(batch_size, limit - stats.scanned)

        q = select(UserImage).where(UserImage.is_thumbnail.is_(False))
        if product is not None:
            q = q.where(UserImage.product_id == product)
        if cursor_ts is not None and cursor_id is not None:
            q = q.where(
                tuple_(UserImage.created_at, UserImage.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )
        q = q.order_by(
            UserImage.created_at.desc(),
            UserImage.id.desc(),
        ).limit(effective_batch + 1)

        result = await session.execute(q)
        rows = list(result.scalars().all())

        has_more = len(rows) > effective_batch
        batch = rows[:effective_batch]

        if not batch:
            break

        parent_ids = [row.id for row in batch]
        derivatives_map = await image_repo.batch_derivatives(parent_ids)

        for full in batch:
            row_derivatives = derivatives_map.get(full.id, [])
            row_result = await _backfill_upload_row(
                session,
                r2,
                full,
                row_derivatives,
                dry_run=dry_run,
            )
            stats.scanned += 1
            _tally(stats, row_result)

        if not dry_run:
            await session.commit()

        if not has_more:
            break

        last = batch[-1]
        cursor_ts = last.created_at
        cursor_id = last.id


# ---------------------------------------------------------------------------
# Public async entry point (used by CLI and tests)
# ---------------------------------------------------------------------------


async def run_backfill(
    session: AsyncSession,
    r2: R2StorageService,
    *,
    product: str | None,
    only: _Only,
    dry_run: bool,
    batch_size: int,
    limit: int | None,
    include_video: bool,
) -> tuple[_Stats, _Stats]:
    """Run the backfill and return (output_stats, upload_stats)."""
    output_stats = _Stats()
    upload_stats = _Stats()

    if only != _Only.uploads:
        await _scan_outputs(
            session,
            r2,
            output_stats,
            product=product,
            batch_size=batch_size,
            limit=limit,
            dry_run=dry_run,
            include_video=include_video,
        )

    if only != _Only.outputs:
        await _scan_uploads(
            session,
            r2,
            upload_stats,
            product=product,
            batch_size=batch_size,
            limit=limit,
            dry_run=dry_run,
        )

    return output_stats, upload_stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(
    output_stats: _Stats,
    upload_stats: _Stats,
    *,
    dry_run: bool,
) -> None:
    title = "Backfill Summary" + (" (dry-run)" if dry_run else "")
    table = Table(title=title)
    table.add_column("Target", style="bold")
    table.add_column("Scanned", justify="right")
    table.add_column("Complete", justify="right", style="green")
    table.add_column("No poster", justify="right", style="yellow")
    table.add_column("Updated", justify="right", style="cyan")
    table.add_column("Variants", justify="right", style="cyan")
    table.add_column("Failed", justify="right", style="red")

    for label, s in (("Outputs", output_stats), ("Uploads", upload_stats)):
        table.add_row(
            label,
            str(s.scanned),
            str(s.skipped_complete),
            str(s.skipped_no_poster),
            str(s.updated),
            str(s.variants_created),
            str(s.failed),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


async def _run_impl(
    *,
    product: str | None,
    only: _Only,
    dry_run: bool,
    batch_size: int,
    limit: int | None,
    include_video: bool,
) -> None:
    settings = Settings()

    if not settings.r2_configured:
        console.print(
            "[red]R2 is not configured — set R2_ACCOUNT_ID, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.[/red]"
        )
        raise typer.Exit(1)

    r2_settings = R2StorageSettings(
        account_id=settings.r2_account_id,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket_name=settings.r2_bucket_name,
        public_url_base=settings.r2_public_url_base,
        retention_days=settings.retention_days,
    )
    r2 = R2StorageService(r2_settings)
    db = DatabaseManager(settings.database_url)
    session = db.session_factory()

    try:
        output_stats, upload_stats = await run_backfill(
            session,
            r2,
            product=product,
            only=only,
            dry_run=dry_run,
            batch_size=batch_size,
            limit=limit,
            include_video=include_video,
        )
    finally:
        await session.close()
        await db.close()

    _print_summary(output_stats, upload_stats, dry_run=dry_run)


@app.command("run")
def run(
    product: Annotated[
        str | None,
        typer.Option("--product", help="Product slug filter (e.g. vex)"),
    ] = None,
    only: Annotated[
        _Only,
        typer.Option("--only", help="Target: all, outputs, or uploads"),
    ] = _Only.all,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Scan and report without writing anything"),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Rows per keyset batch"),
    ] = 100,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Max rows per target (for testing / incremental runs)"),
    ] = None,
    include_video: Annotated[
        bool,
        typer.Option(
            "--include-video",
            help="Download MP4 and extract frame for poster-less video outputs",
        ),
    ] = False,
) -> None:
    """Backfill sm/md WEBP thumbnail variants for pre-existing content."""
    asyncio.run(
        _run_impl(
            product=product,
            only=only,
            dry_run=dry_run,
            batch_size=batch_size,
            limit=limit,
            include_video=include_video,
        )
    )


if __name__ == "__main__":
    app()
