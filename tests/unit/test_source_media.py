"""Focused contracts for normalized owned-library generation inputs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from src.api.schemas.unified_generation import (
    SourceImageReference,
    SourceMediaReference,
    UnifiedGenerationRequest,
)
from src.api.services.generation.source_media import (
    SourceMediaResolver,
    SourceMediaValidationError,
    normalize_source_media,
)
from src.core.enums import GenerationType, ModelType


def _i2i_request(
    *,
    input_image_id: UUID | None = None,
    source_media: list[SourceMediaReference] | None = None,
    source_images: list[SourceImageReference] | None = None,
) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="Edit this image",
        generation_type=GenerationType.I2I,
        model=ModelType.GROK_IMAGINE_IMAGE,
        input_image_id=input_image_id,
        source_media=source_media,
        source_images=source_images,
    )


def test_legacy_source_images_normalize_in_order() -> None:
    upload_id = uuid4()
    output_id = uuid4()

    normalized = normalize_source_media(
        _i2i_request(
            source_images=[
                SourceImageReference(input_image_id=upload_id),
                SourceImageReference(source_output_id=output_id),
            ]
        )
    )

    assert normalized.source_media is not None
    assert [source.asset_ref for source in normalized.source_media] == [
        f"upload:{upload_id}",
        f"output:{output_id}",
    ]
    assert normalized.input_image_id is None
    assert normalized.source_output_id is None
    assert normalized.source_images is None


def test_source_media_and_legacy_alias_are_rejected() -> None:
    with pytest.raises(SourceMediaValidationError, match="cannot be combined"):
        normalize_source_media(
            _i2i_request(
                input_image_id=uuid4(),
                source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
            )
        )


async def test_resolver_returns_interleaved_sources_in_request_order(monkeypatch) -> None:
    upload_id = uuid4()
    output_id = uuid4()
    user_id = uuid4()
    upload_repo = SimpleNamespace(
        get_many=AsyncMock(
            return_value={
                upload_id: SimpleNamespace(
                    is_thumbnail=False,
                    product_id="vex",
                    content_type="image/png",
                    storage_key="uploads/input.png",
                    size_bytes=10,
                )
            }
        )
    )
    output_repo = SimpleNamespace(
        get_many=AsyncMock(
            return_value={
                output_id: SimpleNamespace(
                    is_thumbnail=False,
                    product_id="vex",
                    content_type="video/mp4",
                    storage_key="outputs/input.mp4",
                    size_bytes=20,
                    job_id=uuid4(),
                )
            }
        )
    )
    monkeypatch.setattr(
        "src.api.services.generation.source_media.UserImageRepository",
        lambda _session: upload_repo,
    )
    monkeypatch.setattr(
        "src.api.services.generation.source_media.OutputRepository",
        lambda _session: output_repo,
    )

    resolved = await SourceMediaResolver().resolve(
        [
            SourceMediaReference(asset_ref=f"output:{output_id}"),
            SourceMediaReference(asset_ref=f"upload:{upload_id}"),
        ],
        user_id=user_id,
        session=AsyncMock(),
        product_id="vex",
    )

    assert [item.asset_ref for item in resolved] == [f"output:{output_id}", f"upload:{upload_id}"]
    assert [item.position for item in resolved] == [0, 1]
    upload_repo.get_many.assert_awaited_once_with([upload_id], user_id=user_id)
    output_repo.get_many.assert_awaited_once_with([output_id], user_id=user_id)


async def test_resolver_malformed_reference_does_not_echo_raw_value() -> None:
    raw = "upload:not-a-uuid-secret"
    with pytest.raises(SourceMediaValidationError) as exc_info:
        await SourceMediaResolver().resolve(
            [SourceMediaReference(asset_ref=raw)],
            user_id=uuid4(),
            session=AsyncMock(),
        )

    assert "position 0" in str(exc_info.value)
    assert raw not in str(exc_info.value)


async def test_resolver_rejects_duplicate_reference_at_its_position() -> None:
    asset_id = uuid4()

    with pytest.raises(SourceMediaValidationError, match="position 1 duplicates"):
        await SourceMediaResolver().resolve(
            [
                SourceMediaReference(asset_ref=f"upload:{asset_id}"),
                SourceMediaReference(asset_ref=f"upload:{asset_id}"),
            ],
            user_id=uuid4(),
            session=AsyncMock(),
        )


@pytest.mark.parametrize(
    ("is_thumbnail", "product_id"),
    [(True, "vex"), (False, "other-product")],
)
async def test_resolver_hides_thumbnail_and_wrong_product_as_unavailable(
    monkeypatch,
    is_thumbnail: bool,
    product_id: str,
) -> None:
    asset_id = uuid4()
    upload_repo = SimpleNamespace(
        get_many=AsyncMock(
            return_value={
                asset_id: SimpleNamespace(
                    is_thumbnail=is_thumbnail,
                    product_id=product_id,
                    content_type="image/png",
                    storage_key="uploads/input.png",
                    size_bytes=10,
                )
            }
        )
    )
    output_repo = SimpleNamespace(get_many=AsyncMock(return_value={}))
    monkeypatch.setattr(
        "src.api.services.generation.source_media.UserImageRepository",
        lambda _session: upload_repo,
    )
    monkeypatch.setattr(
        "src.api.services.generation.source_media.OutputRepository",
        lambda _session: output_repo,
    )

    with pytest.raises(
        SourceMediaValidationError,
        match="position 0 does not name an available asset",
    ):
        await SourceMediaResolver().resolve(
            [SourceMediaReference(asset_ref=f"upload:{asset_id}")],
            user_id=uuid4(),
            session=AsyncMock(),
            product_id="vex",
        )
