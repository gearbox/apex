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
from src.api.services.generation.generation_modes import resolve_generation_modes
from src.api.services.generation.service import GenerationService
from src.api.services.generation.source_media import (
    ResolvedSourceMedia,
    SourceMediaResolver,
    SourceMediaValidationError,
    normalize_source_media,
)
from src.api.services.workflow.capabilities import derive_capabilities
from src.api.services.workflow.contract import (
    BoundWorkflow,
    WorkflowMap,
    WorkflowMediaInput,
    WorkflowRole,
)
from src.core.enums import (
    GenerationType,
    MediaKind,
    MediaSlot,
    ModelType,
    Resolution,
    Sampler,
    Scheduler,
)
from src.core.generation_config import (
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref


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


def _mode_request(
    model: ModelType,
    generation_type: GenerationType,
    media_kinds: tuple[MediaKind, ...],
) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="Validate per-mode input",
        model=model,
        generation_type=generation_type,
        source_media=(
            [SourceMediaReference(asset_ref=f"upload:{uuid4()}") for _ in media_kinds]
            if media_kinds
            else None
        ),
    )


def _resolved_sources(media_kinds: tuple[MediaKind, ...]) -> list[ResolvedSourceMedia]:
    sources: list[ResolvedSourceMedia] = []
    for position, media_kind in enumerate(media_kinds):
        asset_id = uuid4()
        ref = AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset_id)
        sources.append(
            ResolvedSourceMedia(
                position=position,
                ref=ref,
                asset_ref=format_asset_ref(ref.source, ref.asset_id),
                media_kind=media_kind,
                content_type="image/png" if media_kind is MediaKind.IMAGE else "video/mp4",
                storage_key=f"uploads/source-{position}",
                size_bytes=1,
                job_id=None,
            )
        )
    return sources


def _aisha_video_capabilities():
    """Derive the bundle contract used by the bundle-backed matrix rows."""
    media_inputs = tuple(
        WorkflowMediaInput(
            id=slot.value,
            class_name="LoadImage",
            input="image",
            kind=MediaKind.IMAGE,
            slot=slot,
            target_role=WorkflowRole.POSITIVE_PROMPT,
            target_input=slot.value,
        )
        for slot in (MediaSlot.FIRST_FRAME, MediaSlot.LAST_FRAME)
    )
    bound = BoundWorkflow(
        map=WorkflowMap(
            contract_version=2,
            media=MediaKind.VIDEO,
            nodes={},
            media_inputs=media_inputs,
            model_inputs=(),
        ),
        api_graph={},
    )
    return derive_capabilities(
        bound,
        BundleGenerationConfig(
            defaults=GenerationDefaults(
                resolution=Resolution.STANDARD,
                steps=12,
                cfg=1.1,
                sampler=Sampler.EULER,
                scheduler=Scheduler.BETA,
                denoise=1.0,
            ),
            constraints=GenerationConstraints(
                max_megapixels=1.0,
                latent_multiple=16,
                max_edge=1536,
                min_steps=1,
                max_steps=20,
                min_cfg=0.0,
                max_cfg=30.0,
                allowed_samplers=frozenset(),
                allowed_schedulers=frozenset(),
                max_batch_size=1,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("declaration", "model", "generation_type", "media_kinds", "valid"),
    [
        ("registry", ModelType.GROK_IMAGINE_VIDEO, GenerationType.T2V, (), True),
        ("registry", ModelType.GROK_IMAGINE_VIDEO, GenerationType.T2V, (MediaKind.IMAGE,), False),
        ("registry", ModelType.GROK_IMAGINE_VIDEO, GenerationType.I2V, (MediaKind.IMAGE,), True),
        (
            "registry",
            ModelType.GROK_IMAGINE_VIDEO,
            GenerationType.I2V,
            (MediaKind.IMAGE, MediaKind.IMAGE),
            False,
        ),
        ("registry", ModelType.GROK_IMAGINE_IMAGE, GenerationType.I2I, (MediaKind.IMAGE,), True),
        (
            "registry",
            ModelType.GROK_IMAGINE_IMAGE,
            GenerationType.I2I,
            (MediaKind.IMAGE,) * 4,
            True,
        ),
        (
            "registry",
            ModelType.GROK_IMAGINE_IMAGE,
            GenerationType.I2I,
            (MediaKind.IMAGE,) * 5,
            False,
        ),
        ("registry", ModelType.GROK_IMAGINE_VIDEO, GenerationType.V2V, (MediaKind.VIDEO,), True),
        ("registry", ModelType.GROK_IMAGINE_VIDEO, GenerationType.V2V, (MediaKind.IMAGE,), False),
        ("registry", ModelType.GROK_IMAGINE_VIDEO, GenerationType.V2V, (), False),
        ("bundle", ModelType.AISHA_VIDEO, GenerationType.T2V, (), True),
        ("bundle", ModelType.AISHA_VIDEO, GenerationType.T2V, (MediaKind.IMAGE,), False),
        ("bundle", ModelType.AISHA_VIDEO, GenerationType.I2V, (MediaKind.IMAGE,), True),
        (
            "bundle",
            ModelType.AISHA_VIDEO,
            GenerationType.I2V,
            (MediaKind.IMAGE, MediaKind.IMAGE),
            False,
        ),
        (
            "bundle",
            ModelType.AISHA_VIDEO,
            GenerationType.FLF2V,
            (MediaKind.IMAGE, MediaKind.IMAGE),
            True,
        ),
        ("bundle", ModelType.AISHA_VIDEO, GenerationType.FLF2V, (MediaKind.IMAGE,), False),
        (
            "bundle",
            ModelType.AISHA_VIDEO,
            GenerationType.FLF2V,
            (MediaKind.VIDEO, MediaKind.IMAGE),
            False,
        ),
    ],
)
def test_per_mode_source_media_validation_matrix(
    declaration: str,
    model: ModelType,
    generation_type: GenerationType,
    media_kinds: tuple[MediaKind, ...],
    valid: bool,
) -> None:
    """Registry and bundle contracts drive the identical validator path."""
    capabilities = _aisha_video_capabilities() if declaration == "bundle" else None
    contract = resolve_generation_modes(model, capabilities=capabilities)[
        generation_type
    ].source_media
    request = _mode_request(model, generation_type, media_kinds)

    if valid:
        GenerationService._validate_source_cardinality(request, contract)
        GenerationService._validate_resolved_sources(_resolved_sources(media_kinds), contract)
    else:
        with pytest.raises(SourceMediaValidationError):
            GenerationService._validate_source_cardinality(request, contract)
            GenerationService._validate_resolved_sources(_resolved_sources(media_kinds), contract)


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
