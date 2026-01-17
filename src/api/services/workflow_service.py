"""Workflow manipulation service for ComfyUI workflows."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from src.api.schemas.generation import GenerationRequest, GenerationType, ModelType

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Base exception for workflow errors."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when workflow file is not found."""


class WorkflowValidationError(WorkflowError):
    """Raised when workflow validation fails."""


class NodeIDs:
    """Node IDs for qwen_rapid_aio workflow."""

    EMPTY_LATENT = "9"  # EmptyLatentImage - width, height, batch_size
    CHECKPOINT_LOADER = "1"  # CheckpointLoaderSimple
    POSITIVE_PROMPT = "3"  # TextEncodeQwenImageEditPlus - positive prompt
    NEGATIVE_PROMPT = "4"  # TextEncodeQwenImageEditPlus - negative prompt
    KSAMPLER = "2"  # KSampler - seed, steps, cfg, etc.
    LOAD_IMAGE_1 = "7"  # LoadImage - first input image
    LOAD_IMAGE_2 = "8"  # LoadImage - second input image
    SAVE_IMAGE = "11"  # SaveImage - output


class WorkflowService:
    """Service for loading and modifying ComfyUI workflows.

    Handles:
    - Loading workflow templates from disk
    - Applying user parameters to workflow nodes
    - Converting GUI workflow format to API format
    """

    # Mapping of model types to workflow paths
    MODEL_WORKFLOW_MAP: ClassVar[dict[ModelType, str]] = {
        ModelType.AISHA: "config/bundles/qwen_rapid_aio/260103-18/workflow.json",
    }

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize workflow service.

        Args:
            base_path: Base path for resolving workflow files.
                      Defaults to current working directory.
        """
        self._base_path = base_path or Path.cwd()
        self._workflow_cache: dict[str, dict[str, Any]] = {}

    def get_workflow_path(self, model_type: ModelType) -> Path:
        """Get workflow file path for a model type.

        Args:
            model_type: The model type to get workflow for.

        Returns:
            Path to the workflow JSON file.

        Raises:
            WorkflowNotFoundError: If model type has no mapped workflow.
        """
        if model_type not in self.MODEL_WORKFLOW_MAP:
            raise WorkflowNotFoundError(f"No workflow mapped for model type: {model_type}")

        return self._base_path / self.MODEL_WORKFLOW_MAP[model_type]

    def load_workflow(self, model_type: ModelType) -> dict[str, Any]:
        """Load and cache workflow template.

        Args:
            model_type: The model type to load workflow for.

        Returns:
            Parsed workflow dictionary (API format).

        Raises:
            WorkflowNotFoundError: If workflow file doesn't exist.
            WorkflowValidationError: If workflow JSON is invalid.
        """
        workflow_path = self.get_workflow_path(model_type)
        cache_key = str(workflow_path)

        if cache_key not in self._workflow_cache:
            if not workflow_path.exists():
                raise WorkflowNotFoundError(f"Workflow not found: {workflow_path}")

            try:
                with workflow_path.open() as f:
                    gui_workflow = json.load(f)

                # Convert from GUI format to API format
                api_workflow = self._convert_gui_to_api_format(gui_workflow)
                self._workflow_cache[cache_key] = api_workflow

            except json.JSONDecodeError as e:
                raise WorkflowValidationError(f"Invalid workflow JSON: {e}") from e

        # Return a deep copy to prevent mutation of cached workflow
        return copy.deepcopy(self._workflow_cache[cache_key])

    def validate_workflow(self, workflow: dict[str, Any]) -> bool:
        """Validate workflow has required nodes.

        Args:
            workflow: Workflow dictionary to validate.

        Returns:
            True if workflow is valid.

        Raises:
            WorkflowValidationError: If required nodes are missing.
        """
        required_nodes = [
            NodeIDs.EMPTY_LATENT,
            NodeIDs.POSITIVE_PROMPT,
            NodeIDs.KSAMPLER,
        ]

        if missing := [node for node in required_nodes if node not in workflow]:
            raise WorkflowValidationError(f"Workflow missing required nodes: {missing}")

        return True

    def _convert_gui_to_api_format(self, gui_workflow: dict[str, Any]) -> dict[str, Any]:
        """Convert GUI workflow format to API format.

        GUI format: {"nodes": [...], "links": [...]}
        API format: {"node_id": {"class_type": ..., "inputs": {...}}}

        Args:
            gui_workflow: Workflow in GUI/export format.

        Returns:
            Workflow in API format ready for /prompt endpoint.
        """
        api_workflow: dict[str, Any] = {}
        nodes = gui_workflow.get("nodes", [])
        links = gui_workflow.get("links", [])

        # Build link lookup: link_id -> (source_node_id, source_slot)
        link_map: dict[int, tuple[int, int]] = {}
        for link in links:
            # Link format: [link_id, source_node, source_slot, target_node, target_slot, type]
            link_id, source_node, source_slot = link[0], link[1], link[2]
            link_map[link_id] = (source_node, source_slot)

        for node in nodes:
            node_id = str(node["id"])
            class_type = node["type"]

            # Start with widget values as inputs
            inputs: dict[str, Any] = {}

            # Process widgets_values - these are the actual parameter values
            widgets_values = node.get("widgets_values", [])

            # Map widgets to input names based on node type
            inputs = self._map_widget_values(class_type, widgets_values, node)

            # Process input connections from links
            for input_def in node.get("inputs", []):
                input_name = input_def["name"]
                link_id = input_def.get("link")

                if link_id is not None and link_id in link_map:
                    source_node, source_slot = link_map[link_id]
                    # API format for connections: [source_node_id_string, source_slot_index]
                    inputs[input_name] = [str(source_node), source_slot]

            api_workflow[node_id] = {
                "class_type": class_type,
                "inputs": inputs,
            }

        return api_workflow

    def _map_widget_values(
        self,
        class_type: str,
        widgets_values: list[Any],
        node: dict[str, Any],
    ) -> dict[str, Any]:
        """Map widget values to input names based on node type.

        Args:
            class_type: The node class type.
            widgets_values: List of widget values from GUI format.
            node: Full node definition for additional context.

        Returns:
            Dictionary mapping input names to values.
        """
        inputs: dict[str, Any] = {}

        # Define widget mappings for known node types
        widget_mappings: dict[str, list[str]] = {
            "EmptyLatentImage": ["width", "height", "batch_size"],
            "CheckpointLoaderSimple": ["ckpt_name"],
            "TextEncodeQwenImageEditPlus": ["prompt"],
            "KSampler": [
                "seed",
                "control_after_generate",  # "fixed", "increment", etc.
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
            ],
            "LoadImage": ["image", "upload"],
            "SaveImage": ["filename_prefix"],
            "PreviewImage": [],
            "VAEDecode": [],
        }

        if class_type in widget_mappings:
            mapping = widget_mappings[class_type]
            for i, name in enumerate(mapping):
                if i < len(widgets_values):
                    inputs[name] = widgets_values[i]
        else:
            # Fallback: try to infer from node inputs with widgets
            widget_idx = 0
            for input_def in node.get("inputs", []):
                if "widget" in input_def:
                    widget_name = input_def["widget"].get("name", input_def["name"])
                    if widget_idx < len(widgets_values):
                        inputs[widget_name] = widgets_values[widget_idx]
                        widget_idx += 1

        return inputs

    def apply_parameters(
        self,
        workflow: dict[str, Any],
        request: GenerationRequest,
        input_image_1: str | None = None,
        input_image_2: str | None = None,
        filename_prefix: str = "generated",
    ) -> dict[str, Any]:
        """Apply generation parameters to workflow.

        Args:
            workflow: Base workflow in API format.
            request: Generation request with parameters.
            input_image_1: Filename of first uploaded image (optional).
            input_image_2: Filename of second uploaded image (optional).
            filename_prefix: Prefix for output filenames.

        Returns:
            Modified workflow with applied parameters.
        """
        workflow = copy.deepcopy(workflow)

        # Apply image dimensions and batch size
        if NodeIDs.EMPTY_LATENT in workflow:
            workflow[NodeIDs.EMPTY_LATENT]["inputs"]["width"] = request.get_calculated_width()
            workflow[NodeIDs.EMPTY_LATENT]["inputs"]["height"] = request.height
            workflow[NodeIDs.EMPTY_LATENT]["inputs"]["batch_size"] = request.max_images

        # Apply positive prompt
        if NodeIDs.POSITIVE_PROMPT in workflow:
            workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"] = request.prompt

        # Apply negative prompt
        if NodeIDs.NEGATIVE_PROMPT in workflow:
            workflow[NodeIDs.NEGATIVE_PROMPT]["inputs"]["prompt"] = request.negative_prompt

        # Apply KSampler parameters
        if NodeIDs.KSAMPLER in workflow:
            workflow[NodeIDs.KSAMPLER]["inputs"]["seed"] = request.seed
            workflow[NodeIDs.KSAMPLER]["inputs"]["steps"] = request.steps

        # Apply output filename prefix
        if NodeIDs.SAVE_IMAGE in workflow:
            workflow[NodeIDs.SAVE_IMAGE]["inputs"]["filename_prefix"] = filename_prefix

        # Handle image inputs based on generation type
        if request.generation_type == GenerationType.T2I:
            # For t2i: disconnect all image inputs from the prompt encoder
            self._disconnect_image_inputs(workflow)
            logger.debug("T2I mode: disconnected all image input nodes")
        else:
            # For i2i: connect only the images that were actually uploaded
            # First, disconnect all image inputs
            self._disconnect_image_inputs(workflow)

            # Then, reconnect and set only the provided images
            if input_image_1:
                # Set the image filename in LoadImage node
                if NodeIDs.LOAD_IMAGE_1 in workflow:
                    workflow[NodeIDs.LOAD_IMAGE_1]["inputs"]["image"] = input_image_1

                # Reconnect image1 to positive prompt encoder
                if NodeIDs.POSITIVE_PROMPT in workflow:
                    # Connection format: [source_node_id, output_slot]
                    workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["image1"] = [
                        NodeIDs.LOAD_IMAGE_1,
                        0,
                    ]

            if input_image_2:
                # Set the image filename in LoadImage node
                if NodeIDs.LOAD_IMAGE_2 in workflow:
                    workflow[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] = input_image_2

                # Reconnect image2 to positive prompt encoder
                if NodeIDs.POSITIVE_PROMPT in workflow:
                    workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["image2"] = [
                        NodeIDs.LOAD_IMAGE_2,
                        0,
                    ]

            logger.debug(
                f"I2I mode: connected images - image1={input_image_1}, image2={input_image_2}"
            )

        return workflow

    def _disconnect_image_inputs(self, workflow: dict[str, Any]) -> None:
        """Disconnect image inputs from the positive prompt encoder.

        Args:
            workflow: Workflow to modify in place.
        """
        if NodeIDs.POSITIVE_PROMPT not in workflow:
            return

        inputs = workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]

        # Remove image connection keys if present
        for key in ["image1", "image2", "image3"]:
            inputs.pop(key, None)
