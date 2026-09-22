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
from tests.unit.helpers import qwen_rapid_aio_capabilities


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
    """A bundle declaring more references than the provider can execute is clamped."""
    capabilities = _capabilities(
        {
            GenerationType.I2I: GenerationModeMeta(
                SourceMediaConstraints(1, 4, frozenset({MediaKind.IMAGE}))
            )
        }
    )

    modes = resolve_generation_modes(ModelType.AISHA_IMAGE, capabilities=capabilities)

    contract = modes[GenerationType.I2I].source_media
    assert contract is not None
    assert (contract.min, contract.max) == (1, 3)


def test_two_slot_bundle_narrows_the_aisha_image_provider_limit_to_two() -> None:
    """Registry 3 ∩ qwen.rapid.aio's two reference slots = 2."""
    modes = resolve_generation_modes(
        ModelType.AISHA_IMAGE, capabilities=qwen_rapid_aio_capabilities(2)
    )

    contract = modes[GenerationType.I2I].source_media
    assert contract is not None
    assert (contract.min, contract.max) == (1, 2)


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


def _aisha_i2i_request(source_count: int) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="Edit",
        generation_type=GenerationType.I2I,
        model=ModelType.AISHA_IMAGE,
        source_media=[
            SourceMediaReference(asset_ref=f"upload:00000000-0000-0000-0000-00000000000{i + 1}")
            for i in range(source_count)
        ],
    )


def test_advertised_aisha_contract_accepts_two_assets_and_rejects_three() -> None:
    contract = resolve_generation_modes(
        ModelType.AISHA_IMAGE, capabilities=qwen_rapid_aio_capabilities(2)
    )[GenerationType.I2I].source_media
    assert contract is not None
    assert contract.max == 2

    GenerationService._validate_source_cardinality(_aisha_i2i_request(1), contract)
    GenerationService._validate_source_cardinality(_aisha_i2i_request(2), contract)
    with pytest.raises(SourceMediaValidationError, match="requires between 1 and 2"):
        GenerationService._validate_source_cardinality(_aisha_i2i_request(3), contract)


def test_registry_modes_are_used_without_bundle_capabilities() -> None:
    modes = resolve_generation_modes(ModelType.GROK_IMAGINE_VIDEO, capabilities=None)

    assert set(modes) == {GenerationType.T2V, GenerationType.I2V, GenerationType.V2V}
    source_media = modes[GenerationType.V2V].source_media
    assert source_media is not None
    assert source_media.media_types == frozenset({MediaKind.VIDEO})
