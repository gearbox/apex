"""Tests for WorkflowService."""

import json
from pathlib import Path

import pytest

from src.api.schemas.generation import AspectRatio, GenerationRequest, GenerationType, ModelType
from src.api.services.workflow_service import (
    NodeIDs,
    WorkflowNotFoundError,
    WorkflowService,
    WorkflowValidationError,
)


class TestWorkflowService:
    """Tests for WorkflowService."""

    def test_load_workflow(self, workflow_service: WorkflowService) -> None:
        """Test loading a workflow."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        assert isinstance(workflow, dict)
        assert NodeIDs.EMPTY_LATENT in workflow
        assert NodeIDs.KSAMPLER in workflow

    def test_workflow_caching(self, workflow_service: WorkflowService) -> None:
        """Test that workflows are cached."""
        workflow1 = workflow_service.load_workflow(ModelType.AISHA)
        workflow2 = workflow_service.load_workflow(ModelType.AISHA)

        # Should be different objects (deep copy)
        assert workflow1 is not workflow2
        # But with same content
        assert workflow1 == workflow2

    def test_workflow_not_found(self, tmp_path: Path) -> None:
        """Test error when workflow file missing."""
        service = WorkflowService(base_path=tmp_path)

        with pytest.raises(WorkflowNotFoundError):
            service.load_workflow(ModelType.AISHA)

    def test_apply_parameters(self, workflow_service: WorkflowService) -> None:
        """Test applying generation parameters to workflow."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        request = GenerationRequest(
            prompt="A beautiful cat",
            negative_prompt="ugly, blurry",
            height=1080,
            aspect_ratio=AspectRatio.RATIO_16_9,
            max_images=2,
            seed=42,
            steps=8,
        )

        modified = workflow_service.apply_parameters(
            workflow=workflow,
            request=request,
            filename_prefix="test_output",
        )

        # Check dimensions
        assert modified[NodeIDs.EMPTY_LATENT]["inputs"]["width"] == 1920
        assert modified[NodeIDs.EMPTY_LATENT]["inputs"]["height"] == 1080
        assert modified[NodeIDs.EMPTY_LATENT]["inputs"]["batch_size"] == 2

        # Check prompts
        assert modified[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"] == "A beautiful cat"
        assert modified[NodeIDs.NEGATIVE_PROMPT]["inputs"]["prompt"] == "ugly, blurry"

        # Check sampler
        assert modified[NodeIDs.KSAMPLER]["inputs"]["seed"] == 42
        assert modified[NodeIDs.KSAMPLER]["inputs"]["steps"] == 8

    def test_apply_parameters_with_images(self, workflow_service: WorkflowService) -> None:
        """Test applying input images to workflow."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        request = GenerationRequest(prompt="test", generation_type=GenerationType.I2I)

        modified = workflow_service.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1="uploaded_image1.png",
            input_image_2="uploaded_image2.png",
        )

        assert modified[NodeIDs.LOAD_IMAGE_1]["inputs"]["image"] == "uploaded_image1.png"
        assert modified[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] == "uploaded_image2.png"

    def test_apply_parameters_immutable(self, workflow_service: WorkflowService) -> None:
        """Test that apply_parameters doesn't modify original workflow."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)
        original_prompt = workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"]

        request = GenerationRequest(prompt="New prompt")
        workflow_service.apply_parameters(workflow=workflow, request=request)

        # Original should be unchanged
        assert workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"] == original_prompt

    def test_apply_parameters_t2i_disconnects_images(
        self, workflow_service: WorkflowService
    ) -> None:
        """Test that t2i mode disconnects image inputs."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        request = GenerationRequest(
            prompt="A beautiful cat",
            generation_type=GenerationType.T2I,
        )

        modified = workflow_service.apply_parameters(
            workflow=workflow,
            request=request,
        )

        # Image inputs should be disconnected from positive prompt encoder
        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert "image1" not in positive_inputs
        assert "image2" not in positive_inputs
        assert "image3" not in positive_inputs

    def test_apply_parameters_i2i_keeps_images(self, workflow_service: WorkflowService) -> None:
        """Test that i2i mode preserves image connections for uploaded images."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        request = GenerationRequest(
            prompt="Transform this image",
            generation_type=GenerationType.I2I,
        )

        modified = workflow_service.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1="my_image.png",
        )

        # Image should be set in LoadImage node
        assert modified[NodeIDs.LOAD_IMAGE_1]["inputs"]["image"] == "my_image.png"

        # Connection should be restored to positive prompt encoder
        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert "image1" in positive_inputs
        assert positive_inputs["image1"] == [NodeIDs.LOAD_IMAGE_1, 0]

        # image2 should NOT be connected (not uploaded)
        assert "image2" not in positive_inputs

    def test_apply_parameters_i2i_with_two_images(self, workflow_service: WorkflowService) -> None:
        """Test that i2i mode connects both images when both are provided."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        request = GenerationRequest(
            prompt="Blend these images",
            generation_type=GenerationType.I2I,
        )

        modified = workflow_service.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1="image_a.png",
            input_image_2="image_b.png",
        )

        # Both images should be set in LoadImage nodes
        assert modified[NodeIDs.LOAD_IMAGE_1]["inputs"]["image"] == "image_a.png"
        assert modified[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] == "image_b.png"

        # Both connections should be restored to positive prompt encoder
        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert positive_inputs["image1"] == [NodeIDs.LOAD_IMAGE_1, 0]
        assert positive_inputs["image2"] == [NodeIDs.LOAD_IMAGE_2, 0]

    def test_apply_parameters_i2i_second_image_only(
        self, workflow_service: WorkflowService
    ) -> None:
        """Test that i2i mode works when only second image is provided."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)

        request = GenerationRequest(
            prompt="Use this reference",
            generation_type=GenerationType.I2I,
        )

        modified = workflow_service.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1=None,
            input_image_2="only_second.png",
        )

        # Only image2 should be connected
        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert "image1" not in positive_inputs
        assert positive_inputs["image2"] == [NodeIDs.LOAD_IMAGE_2, 0]
        assert modified[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] == "only_second.png"

    def test_validate_workflow_valid(self, workflow_service: WorkflowService) -> None:
        """Test validation passes for valid workflow."""
        workflow = workflow_service.load_workflow(ModelType.AISHA)
        assert workflow_service.validate_workflow(workflow) is True

    def test_validate_workflow_missing_nodes(self, workflow_service: WorkflowService) -> None:
        """Test validation fails for incomplete workflow."""
        workflow = {"1": {"class_type": "SomeNode", "inputs": {}}}

        with pytest.raises(WorkflowValidationError) as exc_info:
            workflow_service.validate_workflow(workflow)

        assert "missing required nodes" in str(exc_info.value).lower()


class TestGuiToApiConversion:
    """Tests for GUI to API workflow format conversion."""

    def test_convert_simple_workflow(self, tmp_path: Path) -> None:
        """Test converting a simple GUI workflow to API format."""
        gui_workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "CheckpointLoaderSimple",
                    "inputs": [],
                    "widgets_values": ["model.safetensors"],
                },
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [
                        {"name": "model", "link": 1},
                    ],
                    "widgets_values": [12345, "fixed", 20, 7.0, "euler", "normal", 1.0],
                },
            ],
            "links": [
                [1, 1, 0, 2, 0, "MODEL"],  # link_id, src_node, src_slot, dst_node, dst_slot, type
            ],
        }

        # Create test workflow file
        bundle_dir = tmp_path / "config" / "bundles" / "qwen_rapid_aio" / "260103-18"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "workflow.json").write_text(json.dumps(gui_workflow))

        service = WorkflowService(base_path=tmp_path)
        api_workflow = service.load_workflow(ModelType.AISHA)

        # Check structure
        assert "1" in api_workflow
        assert "2" in api_workflow

        # Check class types
        assert api_workflow["1"]["class_type"] == "CheckpointLoaderSimple"
        assert api_workflow["2"]["class_type"] == "KSampler"

        # Check widget values converted to inputs
        assert api_workflow["1"]["inputs"]["ckpt_name"] == "model.safetensors"
        assert api_workflow["2"]["inputs"]["seed"] == 12345
        assert api_workflow["2"]["inputs"]["steps"] == 20

        # Check link converted to connection reference
        assert api_workflow["2"]["inputs"]["model"] == ["1", 0]
