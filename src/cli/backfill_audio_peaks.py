"""CLI command: compute the 20 ms detail peaks of audio stored before S3a.

Usage:
    python -m src.cli.backfill_audio_peaks [--dry-run] [--batch-size 50]

Run it once after migration 063, on staging and production. It reads only the profiles whose
``detail_peaks`` is NULL, downloads each stored object from R2, decodes it exactly as ingest does
(``MediaIngestService.compute_audio_detail_peaks``, under the video slot) and writes the result
with ``UPDATE … WHERE detail_peaks IS NULL``. It is therefore idempotent, and two runs at once
write each row once. Each row has its own short transaction, and none is open while a file
downloads or ffmpeg runs. A row that fails is logged and counted by error type (a missing R2
object as ``missing``) and the scan goes on. The overview peaks are never touched.

``--dry-run`` does everything but the write: it reports what would be filled and what would fail.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final

import structlog
import typer
from rich.console import Console
from rich.table import Table

from src.api.services.media_ingest.audio import DETAIL_PEAKS_VERSION
from src.api.services.media_ingest.factory import build_media_ingest_service
from src.api.services.storage import R2StorageService, R2StorageSettings, StorageNotFoundError
from src.core.config import Settings
from src.db.repositories.library_audio import LibraryAudioProfileRepository
from src.db.session import DatabaseManager

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.media_ingest import MediaIngestService
    from src.db.repositories.library_audio import MissingDetailPeaksRow

logger = structlog.get_logger(__name__)
console = Console()

MISSING: Final = "missing"
"""The failure type of a profile whose stored object is not in R2."""

app = typer.Typer(
    name="backfill-audio-peaks",
    help="Compute the detail waveform peaks of audio stored before S3a",
    add_completion=False,
)


@dataclass
class BackfillStats:
    scanned: int = 0
    done: int = 0
    """Rows filled (on a dry run: rows that would be)."""
    skipped: int = 0
    """Rows another run filled while this one was working on them."""
    failed: Counter[str] = field(default_factory=Counter)
    """Failures by error type."""


async def run_backfill(
    session: AsyncSession,
    r2: R2StorageService,
    ingestor: MediaIngestService,
    *,
    batch_size: int,
    dry_run: bool,
) -> BackfillStats:
    """Fill every profile without detail peaks, one row at a time.

    Args:
        session: A session this function commits and rolls back (one transaction per row).
        r2: Where the stored objects are.
        ingestor: Decodes them as ingest does.
        batch_size: Profiles per keyset page.
        dry_run: Compute but write nothing.
    """
    repository = LibraryAudioProfileRepository(session)
    stats = BackfillStats()
    after: tuple[datetime, UUID] | None = None
    while True:
        batch = await repository.list_missing_detail_peaks(limit=batch_size, after=after)
        await session.commit()  # nothing stays open while the files download and decode
        for row in batch:
            stats.scanned += 1
            await _backfill_row(session, r2, ingestor, row, stats, dry_run=dry_run)
        if len(batch) < batch_size:
            return stats
        # The keyset moves past failed rows too, so one bad object never stalls the scan.
        after = (batch[-1].created_at, batch[-1].profile_id)


async def _backfill_row(
    session: AsyncSession,
    r2: R2StorageService,
    ingestor: MediaIngestService,
    row: MissingDetailPeaksRow,
    stats: BackfillStats,
    *,
    dry_run: bool,
) -> None:
    try:
        data = await r2.download(row.storage_key)
        detail_peaks = await ingestor.compute_audio_detail_peaks(data)
        if dry_run:
            stats.done += 1
            return
        filled = await LibraryAudioProfileRepository(session).fill_detail_peaks(
            row.profile_id,
            detail_peaks=detail_peaks,
            detail_peaks_version=DETAIL_PEAKS_VERSION,
        )
        await session.commit()
    except StorageNotFoundError:
        _record_failure(stats, row, MISSING)
    except Exception as exc:
        # One bad row must not end the run. Only the type is logged: a message could quote a
        # tag value, a file name or a statement's parameters.
        await session.rollback()
        _record_failure(stats, row, type(exc).__name__)
    else:
        if filled:
            stats.done += 1
        else:
            stats.skipped += 1


def _record_failure(stats: BackfillStats, row: MissingDetailPeaksRow, error_type: str) -> None:
    stats.failed[error_type] += 1
    logger.warning(
        "backfill.audio_peaks.row_failed", profile_id=str(row.profile_id), error_type=error_type
    )


def _print_summary(stats: BackfillStats, *, dry_run: bool) -> None:
    table = Table(title="Audio detail peaks backfill" + (" (dry-run)" if dry_run else ""))
    table.add_column("Outcome", style="bold")
    table.add_column("Rows", justify="right")
    table.add_row("scanned", str(stats.scanned))
    table.add_row("would fill" if dry_run else "done", str(stats.done), style="cyan")
    table.add_row("skipped", str(stats.skipped), style="yellow")
    table.add_row("failed", str(stats.failed.total()), style="red")
    for error_type, count in sorted(stats.failed.items()):
        table.add_row(f"  failed: {error_type}", str(count), style="red")
    console.print(table)


async def _run_impl(*, dry_run: bool, batch_size: int) -> None:
    settings = Settings()
    if not settings.r2_configured:
        console.print(
            "[red]R2 is not configured — set R2_ACCOUNT_ID, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.[/red]"
        )
        raise typer.Exit(1)

    r2 = R2StorageService(
        R2StorageSettings(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket_name=settings.r2_bucket_name,
            retention_days=settings.retention_days,
            max_upload_size_mb=settings.max_upload_size_mb,
            audio_upload_max_mb=settings.audio_upload_max_mb,
        )
    )
    db = DatabaseManager(settings.database_url)
    session = db.session_factory()
    try:
        stats = await run_backfill(
            session,
            r2,
            build_media_ingest_service(settings),
            batch_size=batch_size,
            dry_run=dry_run,
        )
    finally:
        await session.close()
        await db.close()
    _print_summary(stats, dry_run=dry_run)


@app.command()
def run(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Download and decode, but write nothing"),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, max=1000, help="Profiles per keyset batch"),
    ] = 50,
) -> None:
    """Compute the detail waveform peaks of every audio profile that has none."""
    asyncio.run(_run_impl(dry_run=dry_run, batch_size=batch_size))


if __name__ == "__main__":
    app()
