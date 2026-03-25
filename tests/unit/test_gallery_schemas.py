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
from src.core.enums import (
    GalleryBadge,
    GallerySourceType,
    GenerationType,
    OutputMediaType,
)


def _now() -> datetime:
    return datetime.now(UTC)


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
            "cover_url": "/v1/content/outputs/abc",
            "badge": GalleryBadge.PROMPT,
            "media_type": OutputMediaType.IMAGE,
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
        assert decoded.cover_url == item.cover_url
        assert decoded.badge == GalleryBadge.PROMPT
        assert decoded.media_type == OutputMediaType.IMAGE

    def test_aspect_ratio_optional(self) -> None:
        item = self._make(aspect_ratio=None)
        assert item.aspect_ratio is None

    def test_aspect_ratio_string(self) -> None:
        item = self._make(aspect_ratio="16:9")
        assert item.aspect_ratio == "16:9"

    def test_video_url_optional(self) -> None:
        item = self._make(video_url=None)
        assert item.video_url is None

    def test_video_url_set(self) -> None:
        item = self._make(video_url="/v1/content/outputs/vid")
        assert item.video_url == "/v1/content/outputs/vid"

    def test_model_optional(self) -> None:
        item = self._make(model=None)
        assert item.model is None

    def test_enum_values_serialize(self) -> None:
        item = self._make(badge=GalleryBadge.IMAGE, media_type=OutputMediaType.VIDEO)
        data = msgspec.json.decode(msgspec.json.encode(item), type=dict)
        assert data["badge"] == "image"
        assert data["media_type"] == "video"


class TestGalleryOutputItem:
    def _make(self, **overrides: object) -> GalleryOutputItem:
        defaults: dict[str, object] = {
            "id": uuid4(),
            "url": "/v1/content/outputs/abc",
            "content_type": "image/jpeg",
            "media_type": OutputMediaType.IMAGE,
            "format": "jpeg",
            "size_bytes": 12345,
            "output_index": 0,
            "created_at": _now(),
        } | overrides
        return GalleryOutputItem(**defaults)  # type: ignore[arg-type]

    def test_roundtrip(self) -> None:
        item = self._make()
        encoded = msgspec.json.encode(item)
        decoded = msgspec.json.decode(encoded, type=GalleryOutputItem)
        assert decoded.id == item.id
        assert decoded.content_type == "image/jpeg"

    def test_thumbnail_url_optional(self) -> None:
        item = self._make()
        assert item.thumbnail_url is None

    def test_thumbnail_url_set(self) -> None:
        item = self._make(thumbnail_url="/v1/content/outputs/thumb")
        assert item.thumbnail_url == "/v1/content/outputs/thumb"


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
            url="/v1/content/outputs/x",
            content_type="image/jpeg",
            media_type=OutputMediaType.IMAGE,
            format="jpeg",
            size_bytes=100,
            output_index=0,
            created_at=_now(),
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
