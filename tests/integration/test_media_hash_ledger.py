"""PostgreSQL contracts for durable PDQ ledger rows and asyncpg BIT values."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import struct
import subprocess
import zlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy import bindparam, func, select
from sqlalchemy.exc import IntegrityError

from src.api.services.billing import BillingService
from src.api.services.frames.worker import FrameExtractionWorker
from src.api.services.grok import GrokImageResult
from src.api.services.grok.job_service import GrokJobService
from src.api.services.media_hash_ledger import MediaHashLedger
from src.api.services.user_content import UserContentService
from src.core.media_hash import HashSample, HashSet, PdqHash
from src.core.uid import new_id
from src.db.models.media_hash import MediaHash
from src.db.models.storage import GenerationOutput, UserImage
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository
from src.db.types import PdqBit256
from src.workers.aisha_job_poller import AishaJobPoller

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.media_ingest import MediaIngestService
    from tests.integration.conftest import (
        FrameExtractionJobFactory,
        GpuSessionFactory,
        JobFactory,
        UserFactory,
        UserImageFactory,
    )

pytestmark = pytest.mark.asyncio

_CANARY = b"private-metadata-canary"


def _image_with_descriptive_carriers(image_format: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(output, format=image_format)
    source = output.getvalue()
    if image_format == "PNG":

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        carriers = b"".join(
            chunk(kind, payload)
            for kind, payload in [
                (b"tEXt", b"Comment\x00" + _CANARY),
                (b"iTXt", b"Comment\x00\x00\x00\x00\x00" + _CANARY),
                (b"zTXt", b"Comment\x00\x00" + zlib.compress(_CANARY)),
                (b"eXIf", b"Exif\x00\x00" + _CANARY),
                (b"caBX", _CANARY),
            ]
        )
        return source[:33] + carriers + source[33:]
    if image_format == "JPEG":
        carriers = b"".join(
            b"\xff" + bytes([marker]) + struct.pack(">H", len(_CANARY) + 2) + _CANARY
            for marker in [0xE1, 0xEB, 0xED, 0xFE, 0xE2]
        )
        return source[:2] + carriers + source[2:] + _CANARY
    assert image_format == "WEBP"
    for kind in [b"EXIF", b"XMP ", b"JUNK"]:
        payload = kind + struct.pack("<I", len(_CANARY)) + _CANARY + b"\x00"
        body = source[12:] + payload
        source = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body
    return source


def _row(*, user_id: UUID, source_id: UUID, job_id: UUID | None = None) -> MediaHash:
    return MediaHash(
        id=new_id(),
        product_id="vex",
        user_id=user_id,
        job_id=job_id,
        source_kind="output",
        source_id=source_id,
        source_media_type="image",
        hash_profile="pdq-image-rgb-white-v1",
        sampling_profile="still-v1",
        sample_index=0,
        frame_timestamp_ms=None,
        pdq=bytes(range(32)),
        pdq_quality=73,
    )


async def test_asyncpg_bit_round_trip_typed_query_and_fk_lifecycle(
    db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
) -> None:
    """Raw 32-byte PDQ values round-trip and survive only the intended deletes."""
    user = await make_user(email=f"hash-{uuid4().hex}@example.com")
    job = await make_job(user=user)
    row = _row(user_id=user.id, job_id=job.id, source_id=uuid4())
    db_session.add(row)
    await db_session.flush()

    stored = (
        await db_session.execute(select(MediaHash.pdq).where(MediaHash.id == row.id))
    ).scalar_one()
    assert stored == bytes(range(32))

    query_hash = bindparam("query_hash", bytes(range(32)), type_=PdqBit256())
    distance = (
        await db_session.execute(
            select(func.bit_count(MediaHash.pdq.bitwise_xor(query_hash))).where(
                MediaHash.id == row.id
            )
        )
    ).scalar_one()
    assert distance == 0

    await db_session.delete(job)
    await db_session.flush()
    assert (
        await db_session.execute(select(MediaHash.job_id).where(MediaHash.id == row.id))
    ).scalar_one() is None

    user_id = user.id
    await db_session.delete(user)
    await db_session.flush()
    remaining = select(func.count()).select_from(MediaHash).where(MediaHash.user_id == user_id)
    assert (await db_session.execute(remaining)).scalar_one() == 0


async def test_ledger_sample_identity_is_unique(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email=f"hash-unique-{uuid4().hex}@example.com")
    source_id = uuid4()
    db_session.add(_row(user_id=user.id, source_id=source_id))
    await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(_row(user_id=user.id, source_id=source_id))
            await db_session.flush()

    stored = select(func.count()).select_from(MediaHash).where(MediaHash.source_id == source_id)
    assert (await db_session.execute(stored)).scalar_one() == 1


@pytest.mark.parametrize("product_id", ["vex", "synthara"])
async def test_upload_and_output_ledger_inherit_product_and_user(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_job: JobFactory,
    make_user_image: UserImageFactory,
    product_id: str,
) -> None:
    user = await make_user(email=f"ledger-{uuid4().hex}@example.com", product_id=product_id)
    job = await make_job(user=user, product_id=product_id)
    upload = await make_user_image(user=user, product_id=product_id)
    output = GenerationOutput(
        id=new_id(),
        user_id=user.id,
        job_id=job.id,
        product_id=product_id,
        storage_key=f"test/{uuid4()}.png",
        content_type="image/png",
        size_bytes=12,
        format="png",
        output_index=0,
        expires_at=job.created_at,
    )
    db_session.add(output)
    await db_session.flush()
    hash_set = HashSet(
        profile_id="pdq-image-rgb-white-v1",
        sampling_profile="still-v1",
        samples=(HashSample(pdq=PdqHash(bits=b"\x00" * 32, quality=80), sample_index=0),),
    )
    ledger = MediaHashLedger(db_session)
    await ledger.register_upload(upload, hash_set)
    await ledger.register_output(output, hash_set)
    await db_session.flush()
    rows = (
        (
            await db_session.execute(
                select(MediaHash).where(MediaHash.source_id.in_([upload.id, output.id]))
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert {(row.product_id, row.user_id) for row in rows} == {(product_id, user.id)}
    assert {row.source_kind for row in rows} == {"upload", "output"}


async def test_ledger_integrity_failure_rolls_back_original_upload(
    db_session: AsyncSession, make_user: UserFactory, make_user_image: UserImageFactory
) -> None:
    user = await make_user(email=f"ledger-rollback-{uuid4().hex}@example.com")
    upload_id = uuid4()
    conflicting = _row(user_id=user.id, source_id=upload_id)
    conflicting.source_kind = "upload"
    db_session.add(conflicting)
    await db_session.flush()
    hash_set = HashSet(
        profile_id="pdq-image-rgb-white-v1",
        sampling_profile="still-v1",
        samples=(HashSample(pdq=PdqHash(bits=b"\x01" * 32, quality=80), sample_index=0),),
    )
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            upload = await make_user_image(user=user, image_id=upload_id)
            await MediaHashLedger(db_session).register_upload(upload, hash_set)
            await db_session.flush()
    assert await db_session.get(UserImage, upload_id) is None
    assert (
        await db_session.execute(select(MediaHash).where(MediaHash.source_id == upload_id))
    ).scalars().one().id == conflicting.id


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
async def test_real_upload_stores_sanitized_bytes_and_stages_ledger_with_row(
    db_session: AsyncSession,
    make_user: UserFactory,
    media_ingestor: MediaIngestService,
    image_format: str,
) -> None:
    user = await make_user(email=f"ledger-upload-{uuid4().hex}@example.com")
    source = _image_with_descriptive_carriers(image_format)
    object_id = uuid4()
    storage = MagicMock()
    storage.upload = AsyncMock(
        return_value=MagicMock(id=object_id, storage_key=f"test/{object_id}.{image_format.lower()}")
    )
    service = UserContentService(
        storage,
        db_session,
        product_id=user.product_id,
        media_ingestor=media_ingestor,
    )
    with patch(
        "src.api.services.user_content.make_image_thumbnails", new=AsyncMock(return_value=[])
    ):
        uploaded = await service.upload_image(
            user_id=user.id,
            data=source,
            filename=f"photo.{image_format.lower()}",
            content_type=f"image/{image_format.lower()}",
        )
    assert _CANARY not in storage.upload.await_args.kwargs["data"]
    assert uploaded.id == object_id
    upload = await db_session.get(UserImage, object_id)
    ledger = (
        (await db_session.execute(select(MediaHash).where(MediaHash.source_id == object_id)))
        .scalars()
        .all()
    )
    assert upload is not None
    assert len(ledger) == 1
    assert ledger[0].user_id == user.id
    assert ledger[0].product_id == user.product_id
    assert ledger[0].source_kind == "upload"


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
async def test_grok_image_stores_sanitized_bytes_and_stages_ledger_with_output(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_job: JobFactory,
    media_ingestor: MediaIngestService,
    image_format: str,
) -> None:
    user = await make_user(email=f"ledger-grok-{uuid4().hex}@example.com")
    job = await make_job(user=user)
    response = MagicMock(content=_image_with_descriptive_carriers(image_format))
    response.raise_for_status = MagicMock()
    storage = MagicMock()
    storage.build_storage_key.return_value = f"test/grok/output.{image_format.lower()}"
    storage.put_raw = AsyncMock()
    service = GrokJobService(MagicMock(), storage, media_ingestor=media_ingestor)
    service._http_client = MagicMock(get=AsyncMock(return_value=response))
    with patch.object(service, "_store_output_thumbnails", new=AsyncMock()):
        await service._store_image_result(
            session=db_session,
            output_repo=OutputRepository(db_session),
            user_id=user.id,
            job_id=job.id,
            result=GrokImageResult(
                url=f"https://provider.invalid/image.{image_format.lower()}",
                base64_data=None,
                revised_prompt=None,
            ),
            output_index=0,
            input_image_id=None,
            product_id=user.product_id,
        )
    assert _CANARY not in storage.put_raw.await_args.args[1]
    output = (
        (
            await db_session.execute(
                select(GenerationOutput).where(GenerationOutput.job_id == job.id)
            )
        )
        .scalars()
        .one()
    )
    ledger = (
        (await db_session.execute(select(MediaHash).where(MediaHash.source_id == output.id)))
        .scalars()
        .all()
    )
    assert len(ledger) == 1
    assert ledger[0].source_kind == "output"
    assert ledger[0].user_id == user.id
    assert ledger[0].product_id == user.product_id


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
async def test_frame_writer_stores_sanitized_bytes_and_stages_ledger_with_upload(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_frame_extraction_job: FrameExtractionJobFactory,
    tmp_path: Path,
    media_ingestor: MediaIngestService,
    image_format: str,
) -> None:
    user = await make_user(email=f"ledger-frame-{uuid4().hex}@example.com")
    job = await make_frame_extraction_job(user=user)
    source = _image_with_descriptive_carriers(image_format)
    object_id = uuid4()
    storage = MagicMock()
    storage.upload = AsyncMock(
        return_value=MagicMock(id=object_id, storage_key=f"test/{object_id}.{image_format.lower()}")
    )
    settings = MagicMock(
        frame_extract_poll_interval_seconds=30,
        frame_extract_ffmpeg_timeout_seconds=30,
        frame_preview_max_edge=512,
        retention_days=7,
        frame_extract_stale_running_seconds=300,
    )
    worker = FrameExtractionWorker(
        MagicMock(),
        storage,
        settings,
        media_ingestor=media_ingestor,
        redis_client_factory=MagicMock(),
    )
    with (
        patch(
            "src.api.services.frames.worker.frame_ffmpeg.extract_frame",
            new=AsyncMock(return_value=source),
        ),
        patch(
            "src.api.services.frames.worker.make_image_thumbnails",
            new=AsyncMock(return_value=[]),
        ),
    ):
        saved_id = await worker._extract_and_save_frame(
            job,
            tmp_path / "unused.mp4",
            1000,
            image_repo=UserImageRepository(db_session),
            ledger=MediaHashLedger(db_session),
            session=db_session,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            uploaded_keys=[],
        )
    assert saved_id == object_id
    assert _CANARY not in storage.upload.await_args.kwargs["data"]
    upload = await db_session.get(UserImage, object_id)
    ledger = (
        (await db_session.execute(select(MediaHash).where(MediaHash.source_id == object_id)))
        .scalars()
        .all()
    )
    assert upload is not None
    assert len(ledger) == 1
    assert ledger[0].source_kind == "upload"
    assert ledger[0].user_id == user.id


async def test_video_upload_stores_remuxed_bytes_and_stages_frame_hashes(
    db_session: AsyncSession,
    make_user: UserFactory,
    tmp_path: Path,
    media_ingestor: MediaIngestService,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required for video ingest tests in CI")
        pytest.skip("ffmpeg is unavailable locally")
    source = tmp_path / "private.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-metadata",
            "title=private-metadata-canary",
            "-metadata",
            "comment=private-metadata-canary",
            "-metadata",
            "location=+12.34+56.78/",
            "-metadata:s:v:0",
            "handler_name=private-metadata-canary",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    user = await make_user(email=f"ledger-video-{uuid4().hex}@example.com")
    object_id = uuid4()
    storage = MagicMock()
    storage.upload = AsyncMock(
        return_value=MagicMock(id=object_id, storage_key=f"test/{object_id}.mp4")
    )
    service = UserContentService(
        storage,
        db_session,
        product_id=user.product_id,
        media_ingestor=media_ingestor,
    )
    with patch(
        "src.api.services.user_content.extract_video_thumbnail", new=AsyncMock(return_value=None)
    ):
        uploaded = await service.upload_image(
            user_id=user.id,
            data=source.read_bytes(),
            filename="private.mp4",
            content_type="video/mp4",
        )
    assert uploaded.id == object_id
    assert b"private-metadata-canary" not in storage.upload.await_args.kwargs["data"]
    upload = await db_session.get(UserImage, object_id)
    ledger = (
        (await db_session.execute(select(MediaHash).where(MediaHash.source_id == object_id)))
        .scalars()
        .all()
    )
    assert upload is not None
    assert upload.duration_ms == 2000
    assert len(ledger) == 2
    assert all(row.source_media_type == "video" for row in ledger)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
async def test_aisha_writer_stores_sanitized_bytes_and_stages_ledger_with_output(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_job: JobFactory,
    make_gpu_session: GpuSessionFactory,
    media_ingestor: MediaIngestService,
    image_format: str,
) -> None:
    user = await make_user(email=f"ledger-aisha-{uuid4().hex}@example.com")
    gpu_session = await make_gpu_session(user=user)
    job = await make_job(
        user=user, provider="aisha", status="running", gpu_session_id=gpu_session.id
    )
    source = _image_with_descriptive_carriers(image_format)
    object_id = uuid4()
    storage = MagicMock()
    storage.upload = AsyncMock(
        return_value=MagicMock(id=object_id, storage_key=f"test/{object_id}.{image_format.lower()}")
    )
    client = MagicMock(get_image=AsyncMock(return_value=source))
    config = MagicMock(
        tick_interval_seconds=30,
        tunnel_allowed_suffix="gpu.example.test",
        tunnel_allowed_prefix=None,
    )
    poller = AishaJobPoller(
        session_factory=MagicMock(),
        event_bus=None,
        billing_service=BillingService(),
        r2_storage=storage,
        config=config,
        media_ingestor=media_ingestor,
        redis_client_factory=MagicMock(),
    )
    with patch(
        "src.workers.aisha_job_poller.make_image_thumbnails", new=AsyncMock(return_value=[])
    ):
        outcome = await poller._download_and_upload(
            client=client,
            job=job,
            img_info={
                "filename": f"output.{image_format.lower()}",
                "subfolder": "",
                "type": "output",
            },
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )
    assert _CANARY not in storage.upload.await_args.kwargs["data"]
    assert len(outcome.outputs) == 1
    await poller._make_transition_service(db_session).transition_to_completed(
        job.id, outputs=list(outcome.outputs), product_id=user.product_id
    )
    output = (
        (
            await db_session.execute(
                select(GenerationOutput).where(GenerationOutput.job_id == job.id)
            )
        )
        .scalars()
        .one()
    )
    ledger = (
        (await db_session.execute(select(MediaHash).where(MediaHash.source_id == output.id)))
        .scalars()
        .all()
    )
    assert len(ledger) == 1
    assert ledger[0].source_kind == "output"
    assert ledger[0].user_id == user.id
