"""Tests for src.core.resolution — pure dimension resolution logic."""

from __future__ import annotations

import pytest

from src.core.enums import AspectRatio, Resolution
from src.core.resolution import TIER_MEGAPIXELS, ResolvedDimensions, resolve_dimensions


class TestPixelBudgetStability:
    """STANDARD tier should produce ~constant pixel budget across all aspect ratios."""

    @pytest.mark.parametrize("ar", list(AspectRatio))
    def test_standard_budget_within_8_percent(self, ar: AspectRatio) -> None:
        dims = resolve_dimensions(
            aspect_ratio=ar,
            max_megapixels=2.0,
            latent_multiple=16,
            max_edge=2048,
            tier=Resolution.STANDARD,
        )
        mp = dims.megapixels
        target = TIER_MEGAPIXELS[Resolution.STANDARD]
        assert abs(mp - target) / target <= 0.08, (
            f"{ar.value}: {mp:.3f} MP is more than 8% from {target} MP"
        )


class TestSnapping:
    """Every edge must be a multiple of latent_multiple."""

    @pytest.mark.parametrize("ar", list(AspectRatio))
    @pytest.mark.parametrize("tier", list(Resolution))
    def test_edges_are_multiples_of_latent_multiple(
        self, ar: AspectRatio, tier: Resolution
    ) -> None:
        latent_multiple = 16
        dims = resolve_dimensions(
            aspect_ratio=ar,
            max_megapixels=2.0,
            latent_multiple=latent_multiple,
            max_edge=2048,
            tier=tier,
        )
        assert dims.width % latent_multiple == 0, (
            f"{ar.value}/{tier.value}: width {dims.width} not multiple of {latent_multiple}"
        )
        assert dims.height % latent_multiple == 0, (
            f"{ar.value}/{tier.value}: height {dims.height} not multiple of {latent_multiple}"
        )


class TestMegapixelClamp:
    """ULTRA tier (4 MP) with max_megapixels=1.05 must clamp to ≤1.05 MP."""

    def test_ultra_clamps_to_model_max(self) -> None:
        max_mp = 1.05
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_1_1,
            max_megapixels=max_mp,
            latent_multiple=16,
            max_edge=2048,
            tier=Resolution.ULTRA,
        )
        assert dims.megapixels <= max_mp + 0.01  # allow tiny rounding from snapping

    def test_high_clamps_to_model_max(self) -> None:
        max_mp = 0.5
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_16_9,
            max_megapixels=max_mp,
            latent_multiple=16,
            max_edge=2048,
            tier=Resolution.HIGH,
        )
        assert dims.megapixels <= max_mp + 0.02


class TestMaxEdgeClamp:
    """No edge should exceed max_edge."""

    @pytest.mark.parametrize("ar", list(AspectRatio))
    def test_no_edge_exceeds_max_edge(self, ar: AspectRatio) -> None:
        max_edge = 1024
        dims = resolve_dimensions(
            aspect_ratio=ar,
            max_megapixels=4.0,
            latent_multiple=16,
            max_edge=max_edge,
            tier=Resolution.ULTRA,
        )
        assert dims.width <= max_edge, f"{ar.value}: width {dims.width} > max_edge {max_edge}"
        assert dims.height <= max_edge, f"{ar.value}: height {dims.height} > max_edge {max_edge}"


class TestExplicitOverride:
    """Explicit width+height are snapped and clamped."""

    def test_exact_explicit_dims_snapped_to_multiple(self) -> None:
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_1_1,
            max_megapixels=4.0,
            latent_multiple=16,
            max_edge=2048,
            explicit_width=512,
            explicit_height=512,
        )
        assert dims.width == 512
        assert dims.height == 512

    def test_explicit_dims_snapped_to_latent_multiple(self) -> None:
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_1_1,
            max_megapixels=4.0,
            latent_multiple=16,
            max_edge=2048,
            explicit_width=510,
            explicit_height=510,
        )
        assert dims.width % 16 == 0
        assert dims.height % 16 == 0

    def test_oversize_explicit_dims_clamped_under_mp_cap(self) -> None:
        max_mp = 1.0
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_1_1,
            max_megapixels=max_mp,
            latent_multiple=16,
            max_edge=4096,
            explicit_width=2048,
            explicit_height=2048,
        )
        # 2048*2048 = 4.19 MP; must clamp to ≤1.0 MP
        assert dims.megapixels <= max_mp + 0.05

    def test_explicit_dims_respect_max_edge(self) -> None:
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_1_1,
            max_megapixels=4.0,
            latent_multiple=16,
            max_edge=768,
            explicit_width=1024,
            explicit_height=1024,
        )
        assert dims.width <= 768
        assert dims.height <= 768


class TestRawFractionAspectRatio:
    """resolve_dimensions also accepts a raw (w, h) fraction, e.g. an i2i
    source image's exact aspect, in addition to an AspectRatio enum member."""

    def test_resolve_dimensions_accepts_raw_fraction(self) -> None:
        dims = resolve_dimensions(
            aspect_ratio=(1080, 1920),
            max_megapixels=2.0,
            latent_multiple=16,
            max_edge=2048,
            tier=Resolution.STANDARD,
        )
        # Portrait fraction (9:16-equivalent) -> taller than wide.
        assert dims.height > dims.width
        assert dims.width % 16 == 0
        assert dims.height % 16 == 0
        assert dims.megapixels <= 2.0 + 0.01

    def test_raw_fraction_matches_equivalent_enum_ratio(self) -> None:
        """(9, 16) should resolve to the same dims as the RATIO_9_16 enum."""
        from_enum = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_9_16,
            max_megapixels=2.0,
            latent_multiple=16,
            max_edge=2048,
            tier=Resolution.STANDARD,
        )
        from_fraction = resolve_dimensions(
            aspect_ratio=(9, 16),
            max_megapixels=2.0,
            latent_multiple=16,
            max_edge=2048,
            tier=Resolution.STANDARD,
        )
        assert from_enum == from_fraction


class TestFractionPositivityGuard:
    """Non-positive (w, h) fraction components must raise ValueError, not

    propagate into deep math errors (ZeroDivisionError / nonsense dims).
    """

    def test_resolve_dimensions_rejects_zero_fraction_component(self) -> None:
        with pytest.raises(ValueError, match="positive components"):
            resolve_dimensions(
                aspect_ratio=(0, 9),
                max_megapixels=2.0,
                latent_multiple=16,
                max_edge=2048,
                tier=Resolution.STANDARD,
            )

    def test_resolve_dimensions_rejects_negative_fraction_component(self) -> None:
        with pytest.raises(ValueError, match="positive components"):
            resolve_dimensions(
                aspect_ratio=(-16, 9),
                max_megapixels=2.0,
                latent_multiple=16,
                max_edge=2048,
                tier=Resolution.STANDARD,
            )


class TestResolvedDimensionsProperty:
    def test_megapixels_property(self) -> None:
        d = ResolvedDimensions(width=1000, height=1000)
        assert d.megapixels == pytest.approx(1.0)

    def test_megapixels_wide(self) -> None:
        d = ResolvedDimensions(width=1920, height=1080)
        assert d.megapixels == pytest.approx(2.0736)


class TestMaxEdgeSubTile:
    """When max_edge < latent_multiple the degenerate cap is honoured literally."""

    def test_both_edges_at_or_below_max_edge(self) -> None:
        dims = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_1_1,
            max_megapixels=4.0,
            latent_multiple=16,
            max_edge=8,
            tier=Resolution.STANDARD,
        )
        assert dims.width <= 8
        assert dims.height <= 8

    @pytest.mark.parametrize("ar", list(AspectRatio))
    @pytest.mark.parametrize(
        "tier,max_edge",
        [
            (Resolution.STANDARD, 512),
            (Resolution.HIGH, 1024),
            (Resolution.ULTRA, 2048),
        ],
    )
    def test_edges_never_exceed_max_edge(
        self, ar: AspectRatio, tier: Resolution, max_edge: int
    ) -> None:
        dims = resolve_dimensions(
            aspect_ratio=ar,
            max_megapixels=4.0,
            latent_multiple=16,
            max_edge=max_edge,
            tier=tier,
        )
        assert dims.width <= max_edge, (
            f"{ar.value}/{tier.value}: width {dims.width} > max_edge {max_edge}"
        )
        assert dims.height <= max_edge, (
            f"{ar.value}/{tier.value}: height {dims.height} > max_edge {max_edge}"
        )

    @pytest.mark.parametrize("ar", list(AspectRatio))
    @pytest.mark.parametrize(
        "tier,max_edge",
        [
            (Resolution.STANDARD, 512),
            (Resolution.HIGH, 1024),
        ],
    )
    def test_edges_are_multiples_of_latent_multiple_when_not_degenerate(
        self, ar: AspectRatio, tier: Resolution, max_edge: int
    ) -> None:
        latent_multiple = 16
        dims = resolve_dimensions(
            aspect_ratio=ar,
            max_megapixels=4.0,
            latent_multiple=latent_multiple,
            max_edge=max_edge,
            tier=tier,
        )
        assert dims.width % latent_multiple == 0, (
            f"{ar.value}/{tier.value}: width {dims.width} not multiple of {latent_multiple}"
        )
        assert dims.height % latent_multiple == 0, (
            f"{ar.value}/{tier.value}: height {dims.height} not multiple of {latent_multiple}"
        )
