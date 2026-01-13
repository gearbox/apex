"""Tests for API schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas.generation import (
    AspectRatio,
    GenerationRequest,
    ModelType,
)


class TestAspectRatio:
    """Tests for AspectRatio enum."""

    @pytest.mark.parametrize(
        "ratio,height,expected_width",
        [
            (AspectRatio.RATIO_1_1, 1024, 1024),
            (AspectRatio.RATIO_16_9, 1080, 1920),
            (AspectRatio.RATIO_9_16, 1920, 1080),
            (AspectRatio.RATIO_4_3, 768, 1024),
            (AspectRatio.RATIO_3_4, 1024, 768),
            (AspectRatio.RATIO_2_3, 1536, 1024),
            (AspectRatio.RATIO_3_2, 1024, 1536),
        ],
    )
    def test_calculate_width(
        self,
        ratio: AspectRatio,
        height: int,
        expected_width: int,
    ) -> None:
        """Test width calculation from aspect ratio."""
        width = ratio.calculate_width(height)
        # Allow small rounding differences due to multiple of 8 rounding
        assert abs(width - expected_width) <= 8

    def test_width_multiple_of_8(self) -> None:
        """Test that calculated width is multiple of 8."""
        for ratio in AspectRatio:
            for height in [512, 768, 1024, 1080, 1536]:
                width = ratio.calculate_width(height)
                assert width % 8 == 0, f"{ratio} with height {height} gave width {width}"


class TestGenerationRequest:
    """Tests for GenerationRequest schema."""

    def test_minimal_request(self) -> None:
        """Test creation with only required field."""
        request = GenerationRequest(prompt="A cat")

        assert request.prompt == "A cat"
        assert request.height == 1024
        assert request.aspect_ratio == AspectRatio.RATIO_1_1
        assert request.model_type == ModelType.AISHA
        assert request.max_images == 1
        assert request.steps == 12
        assert request.seed is not None  # Auto-generated

    def test_name_auto_generation(self) -> None:
        """Test name is generated from prompt if not provided."""
        request = GenerationRequest(prompt="A beautiful sunset over mountains")
        assert request.name == "A beautiful sunset over mountains"

        # Long prompt gets truncated
        long_prompt = "A" * 100
        request = GenerationRequest(prompt=long_prompt)
        assert len(request.name) <= 53  # 50 chars + "..."
        assert request.name.endswith("...")

    def test_seed_auto_generation(self) -> None:
        """Test seed is auto-generated if not provided."""
        request1 = GenerationRequest(prompt="test")
        request2 = GenerationRequest(prompt="test")

        # Seeds should be different (with very high probability)
        assert request1.seed != request2.seed

    def test_explicit_seed(self) -> None:
        """Test explicit seed is preserved."""
        request = GenerationRequest(prompt="test", seed=42)
        assert request.seed == 42

    def test_calculated_width(self) -> None:
        """Test width calculation property."""
        request = GenerationRequest(
            prompt="test",
            height=1080,
            aspect_ratio=AspectRatio.RATIO_16_9,
        )
        assert request.calculated_width == 1920

    def test_validation_prompt_required(self) -> None:
        """Test prompt is required."""
        with pytest.raises(ValidationError):
            GenerationRequest()

    def test_validation_prompt_min_length(self) -> None:
        """Test prompt minimum length."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="")

    def test_validation_height_range(self) -> None:
        """Test height validation."""
        # Too small
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", height=100)

        # Too large
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", height=3000)

        # Valid range
        GenerationRequest(prompt="test", height=256)
        GenerationRequest(prompt="test", height=2048)

    def test_validation_max_images(self) -> None:
        """Test max_images validation."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", max_images=0)

        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", max_images=5)

        # Valid range
        GenerationRequest(prompt="test", max_images=1)
        GenerationRequest(prompt="test", max_images=4)

    def test_validation_steps(self) -> None:
        """Test steps validation."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", steps=0)

        with pytest.raises(ValidationError):
            GenerationRequest(prompt="test", steps=25)

        # Valid range
        GenerationRequest(prompt="test", steps=1)
        GenerationRequest(prompt="test", steps=20)
