"""Contracts for the shared per-generation-mode resolver."""

from __future__ import annotations

import pytest

from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.generation.generation_modes import resolve_generation_modes
from src.api.services.generation.service import GenerationService
from src.api.services.generation.source_media import SourceMediaValidationError
from src.api.services.workflow.contract import BundleCapabilities
from src.core.enums import GenerationType, MediaKind, MediaSlot, ModelType
from src.core.generation_mode import GenerationModeMeta, SourceMediaConstraints


def _capabilities(
    generation_modes: dict[GenerationType, GenerationModeMeta],
    *,
    media: MediaKind = MediaKind.IMAGE,
) -> BundleCapabilities:
    return BundleCapabilities(
        media=media,
        generation_modes=generation_modes,
        supports_negative_prompt=False,
        writable=frozenset(),
        max_batch_size=1,
    )


def test_bundle_contract_wider_than_provider_is_clamped_to_provider() -> None:
    """A two-reference bundle cannot advertise more than Aisha can execute."""
    capabilities = _capabilities(
        {
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(1, 2, frozenset({MediaKind.IMAGE}))
            )
        }
    )

    modes = resolve_generation_modes(ModelType.AISHA_IMAGE, capabilities=capabilities)

    contract = modes[GenerationType.I2I].source_media
    assert contract is not None
    assert (contract.min, contract.max) == (1, 1)


def test_bundle_contract_narrows_provider_contract() -> None:
    capabilities = _capabilities(
        {
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(2, 2, frozenset({MediaKind.IMAGE}))
            )
        }
    )

    modes = resolve_generation_modes(ModelType.GROK_IMAGINE_IMAGE, capabilities=capabilities)

    contract = modes[GenerationType.I2I].source_media
    assert contract is not None
    assert (contract.min, contract.max) == (2, 2)


def test_disjoint_media_types_drop_a_mode() -> None:
    capabilities = _capabilities(
        {
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(1, 1, frozenset({MediaKind.VIDEO}))
            )
        }
    )

    modes = resolve_generation_modes(ModelType.GROK_IMAGINE_IMAGE, capabilities=capabilities)

    assert GenerationType.I2I not in modes


def test_conflicting_roles_drop_a_mode_and_compatible_roles_survive() -> None:
    conflicting = _capabilities(
        {
            GenerationType.I2V: GenerationModeMeta(
                SourceMediaConstraints(
                    1,
                    1,
                    frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.LAST_FRAME,),
                )
            )
        },
        media=MediaKind.VIDEO,
    )
    compatible = _capabilities(
        {
            GenerationType.I2V: GenerationModeMeta(
                SourceMediaConstraints(
                    1,
                    1,
                    frozenset({MediaKind.IMAGE}),
                    roles=(MediaSlot.FIRST_FRAME,),
                )
            )
        },
        media=MediaKind.VIDEO,
    )

    assert resolve_generation_modes(ModelType.AISHA_VIDEO, capabilities=conflicting) == {}
    contract = resolve_generation_modes(ModelType.AISHA_VIDEO, capabilities=compatible)[
        GenerationType.I2V
    ].source_media
    assert contract is not None
    assert contract.roles == (MediaSlot.FIRST_FRAME,)


def test_mismatched_source_media_nullness_drops_a_mode() -> None:
    capabilities = _capabilities(
        {
            GenerationType.T2I: GenerationModeMeta(
                SourceMediaConstraints(1, 1, frozenset({MediaKind.IMAGE}))
            )
        }
    )

    assert resolve_generation_modes(ModelType.GROK_IMAGINE_IMAGE, capabilities=capabilities) == {}


def test_advertised_aisha_contract_rejects_two_assets_before_submission() -> None:
    capabilities = _capabilities(
        {
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(1, 2, frozenset({MediaKind.IMAGE}))
            )
        }
    )
    contract = resolve_generation_modes(ModelType.AISHA_IMAGE, capabilities=capabilities)[
        GenerationType.I2I
    ].source_media
    request = UnifiedGenerationRequest(
        prompt="Edit",
        generation_type=GenerationType.I2I,
        model=ModelType.AISHA_IMAGE,
        source_media=[
            SourceMediaReference(asset_ref="upload:00000000-0000-0000-0000-000000000001"),
            SourceMediaReference(asset_ref="upload:00000000-0000-0000-0000-000000000002"),
        ],
    )

    assert contract is not None
    assert contract.max == 1
    with pytest.raises(SourceMediaValidationError, match="requires between"):
        GenerationService._validate_source_cardinality(request, contract)


def test_registry_modes_are_used_without_bundle_capabilities() -> None:
    modes = resolve_generation_modes(ModelType.GROK_IMAGINE_VIDEO, capabilities=None)

    assert set(modes) == {GenerationType.T2V, GenerationType.I2V, GenerationType.V2V}
    source_media = modes[GenerationType.V2V].source_media
    assert source_media is not None
    assert source_media.media_types == frozenset({MediaKind.VIDEO})
