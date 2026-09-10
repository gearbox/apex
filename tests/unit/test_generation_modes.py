"""Contracts for the shared per-generation-mode resolver."""

from __future__ import annotations

from src.api.services.generation.generation_modes import resolve_generation_modes
from src.api.services.workflow.contract import BundleCapabilities
from src.core.enums import GenerationType, MediaKind, MediaSlot, ModelType
from src.core.generation_mode import GenerationModeMeta, SourceMediaConstraints


def test_bundle_modes_narrow_static_registration_and_win_contract_details() -> None:
    bundle_contract = SourceMediaConstraints(
        min=1,
        max=2,
        media_types=frozenset({MediaKind.IMAGE}),
        roles=(),
    )
    capabilities = BundleCapabilities(
        media=MediaKind.IMAGE,
        generation_modes={
            GenerationType.I2I: GenerationModeMeta(bundle_contract),
            GenerationType.V2V: GenerationModeMeta(
                SourceMediaConstraints(
                    min=1,
                    max=1,
                    media_types=frozenset({MediaKind.VIDEO}),
                    roles=(MediaSlot.SOURCE,),
                )
            ),
        },
        supports_negative_prompt=False,
        writable=frozenset(),
        max_batch_size=1,
    )

    modes = resolve_generation_modes(ModelType.AISHA_IMAGE, capabilities=capabilities)

    assert modes == {GenerationType.I2I: GenerationModeMeta(bundle_contract)}


def test_registry_modes_are_used_without_bundle_capabilities() -> None:
    modes = resolve_generation_modes(ModelType.GROK_IMAGINE_VIDEO, capabilities=None)

    assert set(modes) == {GenerationType.T2V, GenerationType.I2V, GenerationType.V2V}
    source_media = modes[GenerationType.V2V].source_media
    assert source_media is not None
    assert source_media.media_types == frozenset({MediaKind.VIDEO})
