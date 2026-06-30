"""Tests for gallery endpoint schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec

from src.api.schemas.gallery import (
    GalleryGridItem,
    GalleryGroupDetail,
    GalleryLineage,
    GalleryOutputItem,
)
from src.api.schemas.media import MediaObject, MediaOriginal
from src.core.enums import (
    GalleryBadge,
    GallerySourceType,
    GenerationType,
    OutputMediaType,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _make_media(
    content_type: str = "image/jpeg",
    url: str = "/v1/content/outputs/abc",
) -> MediaObject:
    return MediaObject(
        media_type=OutputMediaType.IMAGE
        if content_type.startswith("image/")
        else OutputMediaType.VIDEO,
        original=MediaOriginal(
            url=url,
            width=512,
            height=512,
            content_type=content_type,
            size_bytes=12345,
        ),
        variants=[],
    )


class TestOutputMediaTypeEnum:
    def test_image_value(self) -> None:
        assert OutputMediaType.IMAGE.value == "image"

    def test_video_value(self) -> None:
        assert OutputMediaType.VIDEO.value == "video"


class TestGalleryBadgeEnum:
    def test_image_value(self) -> None:
        assert GalleryBadge.IMAGE.value == "image"

    def test_prompt_value(self) -> None:
        assert GalleryBadge.PROMPT.value == "prompt"


class TestGallerySourceTypeEnum:
    def test_upload_value(self) -> None:
        assert GallerySourceType.UPLOAD.value == "upload"

    def test_generation_value(self) -> None:
        assert GallerySourceType.GENERATION.value == "generation"


class TestGalleryGridItem:
    def _make(self, **overrides: object) -> GalleryGridItem:
        defaults: dict[str, object] = {
            "job_id": uuid4(),
            "cover": _make_media(),
            "badge": GalleryBadge.PROMPT,
            "output_count": 1,
            "generation_type": GenerationType.T2I,
            "prompt_snippet": "a cat",
            "created_at": _now(),
        } | overrides
        return GalleryGridItem(**defaults)  # type: ignore[arg-type]

    def test_roundtrip(self) -> None:
        item = self._make()
        encoded = msgspec.json.encode(item)
        decoded = msgspec.json.decode(encoded, type=GalleryGridItem)
        assert decoded.job_id == item.job_id
        assert decoded.cover.original.url == "/v1/content/outputs/abc"
        assert decoded.badge == GalleryBadge.PROMPT

    def test_aspect_ratio_optional(self) -> None:
        item = self._make(aspect_ratio=None)
        assert item.aspect_ratio is None

    def test_aspect_ratio_string(self) -> None:
        item = self._make(aspect_ratio="16:9")
        assert item.aspect_ratio == "16:9"

    def test_model_optional(self) -> None:
        item = self._make(model=None)
        assert item.model is None

    def test_model_string(self) -> None:
        item = self._make(model="grok-imagine-image")
        assert item.model == "grok-imagine-image"

    def test_enum_values_serialize(self) -> None:
        item = self._make(badge=GalleryBadge.IMAGE)
        data = msgspec.json.decode(msgspec.json.encode(item), type=dict)
        assert data["badge"] == "image"

    def test_cover_media_object_serializes(self) -> None:
        cover = _make_media(content_type="video/mp4", url="/v1/content/outputs/vid")
        item = self._make(cover=cover)
        data = msgspec.json.decode(msgspec.json.encode(item), type=dict)
        assert data["cover"]["media_type"] == "video"
        assert data["cover"]["original"]["url"] == "/v1/content/outputs/vid"

    def test_cover_variants_round_trip(self) -> None:
        from src.api.schemas.media import ImageVariant

        cover = MediaObject(
            media_type=OutputMediaType.IMAGE,
            original=MediaOriginal(
                url="/v1/content/outputs/full",
                width=1024,
                height=1024,
                content_type="image/png",
                size_bytes=50000,
            ),
            variants=[
                ImageVariant(label="sm", width=100, height=100, url="/v1/content/outputs/sm"),
                ImageVariant(label="md", width=400, height=400, url="/v1/content/outputs/md"),
            ],
        )
        item = self._make(cover=cover)
        decoded = msgspec.json.decode(msgspec.json.encode(item), type=GalleryGridItem)
        assert len(decoded.cover.variants) == 2
        assert decoded.cover.variants[0].label == "sm"


class TestGalleryOutputItem:
    def _make(self, **overrides: object) -> GalleryOutputItem:
        defaults: dict[str, object] = {
            "id": uuid4(),
            "output_index": 0,
            "created_at": _now(),
            "media": _make_media(),
        } | overrides
        return GalleryOutputItem(**defaults)  # type: ignore[arg-type]

    def test_roundtrip(self) -> None:
        item_id = uuid4()
        item = self._make(id=item_id)
        encoded = msgspec.json.encode(item)
        decoded = msgspec.json.decode(encoded, type=GalleryOutputItem)
        assert decoded.id == item_id
        assert decoded.media.original.content_type == "image/jpeg"

    def test_media_object_url_round_trip(self) -> None:
        item = self._make(media=_make_media(url="/v1/content/outputs/specific"))
        decoded = msgspec.json.decode(msgspec.json.encode(item), type=GalleryOutputItem)
        assert decoded.media.original.url == "/v1/content/outputs/specific"

    def test_output_index_preserved(self) -> None:
        item = self._make(output_index=5)
        assert item.output_index == 5


class TestGalleryLineage:
    def test_roundtrip_generation_source(self) -> None:
        lineage = GalleryLineage(
            source_type=GallerySourceType.GENERATION,
            source_job_id=uuid4(),
            source_job_name="My Job",
            source_output_id=uuid4(),
        )
        encoded = msgspec.json.encode(lineage)
        decoded = msgspec.json.decode(encoded, type=GalleryLineage)
        assert decoded.source_type == GallerySourceType.GENERATION
        assert decoded.source_job_name == "My Job"

    def test_roundtrip_upload_source(self) -> None:
        lineage = GalleryLineage(
            source_type=GallerySourceType.UPLOAD,
            source_upload_id=uuid4(),
        )
        encoded = msgspec.json.encode(lineage)
        decoded = msgspec.json.decode(encoded, type=GalleryLineage)
        assert decoded.source_type == GallerySourceType.UPLOAD
        assert decoded.source_upload_id is not None
        assert decoded.source_job_id is None


class TestGalleryGroupDetail:
    def _output_item(self) -> GalleryOutputItem:
        return GalleryOutputItem(
            id=uuid4(),
            output_index=0,
            created_at=_now(),
            media=_make_media(),
        )

    def test_roundtrip(self) -> None:
        detail = GalleryGroupDetail(
            job_id=uuid4(),
            badge=GalleryBadge.PROMPT,
            prompt="a cat",
            outputs=[self._output_item()],
            media_type=OutputMediaType.IMAGE,
            provider="grok",
            generation_type=GenerationType.T2I,
            created_at=_now(),
        )
        encoded = msgspec.json.encode(detail)
        decoded = msgspec.json.decode(encoded, type=GalleryGroupDetail)
        assert decoded.job_id == detail.job_id
        assert decoded.badge == GalleryBadge.PROMPT
        assert len(decoded.outputs) == 1
        assert decoded.input_media is None

    def test_with_lineage(self) -> None:
        lineage = GalleryLineage(
            source_type=GallerySourceType.GENERATION,
            source_job_id=uuid4(),
        )
        detail = GalleryGroupDetail(
            job_id=uuid4(),
            badge=GalleryBadge.IMAGE,
            prompt="edit this",
            outputs=[],
            media_type=OutputMediaType.IMAGE,
            provider="grok",
            generation_type=GenerationType.I2I,
            created_at=_now(),
            lineage=lineage,
        )
        data = msgspec.json.decode(msgspec.json.encode(detail), type=dict)
        assert data["lineage"]["source_type"] == "generation"

    def test_aspect_ratio_included(self) -> None:
        detail = GalleryGroupDetail(
            job_id=uuid4(),
            badge=GalleryBadge.PROMPT,
            prompt="wide image",
            outputs=[],
            media_type=OutputMediaType.IMAGE,
            provider="grok",
            generation_type=GenerationType.T2I,
            aspect_ratio="16:9",
            created_at=_now(),
        )
        data = msgspec.json.decode(msgspec.json.encode(detail), type=dict)
        assert data["aspect_ratio"] == "16:9"

    def test_input_media_round_trip(self) -> None:
        input_media = _make_media(url="/v1/content/uploads/src-img")
        detail = GalleryGroupDetail(
            job_id=uuid4(),
            badge=GalleryBadge.IMAGE,
            prompt="riff on this",
            outputs=[],
            media_type=OutputMediaType.IMAGE,
            provider="grok",
            generation_type=GenerationType.I2I,
            created_at=_now(),
            input_media=input_media,
        )
        decoded = msgspec.json.decode(msgspec.json.encode(detail), type=GalleryGroupDetail)
        assert decoded.input_media is not None
        assert decoded.input_media.original.url == "/v1/content/uploads/src-img"

    def test_input_media_none_when_text_only(self) -> None:
        detail = GalleryGroupDetail(
            job_id=uuid4(),
            badge=GalleryBadge.PROMPT,
            prompt="from scratch",
            outputs=[],
            media_type=OutputMediaType.IMAGE,
            provider="grok",
            generation_type=GenerationType.T2I,
            created_at=_now(),
        )
        assert detail.input_media is None
