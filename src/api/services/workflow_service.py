"""Workflow manipulation service for ComfyUI workflows."""

import copy
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from src.api.schemas.generation import GenerationRequest, ModelType

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Base exception for workflow errors."""

    pass


class WorkflowNotFoundError(WorkflowError):
    """Raised when workflow file is not found."""

    pass


class WorkflowValidationError(WorkflowError):
    """Raised when workflow validation fails."""

    pass


# Node IDs from the qwen_rapid_aio workflow
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
        cache_key = model_type.value

        if cache_key in self._workflow_cache:
            logger.debug(f"Using cached workflow for {model_type}")
            return copy.deepcopy(self._workflow_cache[cache_key])

        workflow_path = self.get_workflow_path(model_type)

        if not workflow_path.exists():
            raise WorkflowNotFoundError(f"Workflow file not found: {workflow_path}")

        try:
            with open(workflow_path) as f:
                gui_workflow = json.load(f)
        except json.JSONDecodeError as e:
            raise WorkflowValidationError(f"Invalid workflow JSON: {e}") from e

        # Convert GUI format to API format
        api_workflow = self._convert_gui_to_api(gui_workflow)
        self._workflow_cache[cache_key] = api_workflow

        logger.info(f"Loaded and cached workflow for {model_type} from {workflow_path}")
        return copy.deepcopy(api_workflow)

    def _convert_gui_to_api(self, gui_workflow: dict[str, Any]) -> dict[str, Any]:
        """Convert ComfyUI GUI workflow format to API format.

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
            workflow[NodeIDs.EMPTY_LATENT]["inputs"]["width"] = request.calculated_width
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

        # Apply input images if provided
        if input_image_1 and NodeIDs.LOAD_IMAGE_1 in workflow:
            workflow[NodeIDs.LOAD_IMAGE_1]["inputs"]["image"] = input_image_1

        if input_image_2 and NodeIDs.LOAD_IMAGE_2 in workflow:
            workflow[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] = input_image_2

        # Apply output filename prefix
        if NodeIDs.SAVE_IMAGE in workflow:
            workflow[NodeIDs.SAVE_IMAGE]["inputs"]["filename_prefix"] = filename_prefix

        logger.debug(
            f"Applied parameters: size={request.calculated_width}x{request.height}, "
            f"batch={request.max_images}, seed={request.seed}, steps={request.steps}"
        )

        return workflow

    def validate_workflow(self, workflow: dict[str, Any]) -> bool:
        """Validate that workflow has required nodes.

        Args:
            workflow: Workflow in API format.

        Returns:
            True if valid, raises exception otherwise.

        Raises:
            WorkflowValidationError: If required nodes are missing.
        """
        required_nodes = [
            NodeIDs.EMPTY_LATENT,
            NodeIDs.POSITIVE_PROMPT,
            NodeIDs.KSAMPLER,
        ]

        missing = [node_id for node_id in required_nodes if node_id not in workflow]

        if missing:
            raise WorkflowValidationError(f"Workflow missing required nodes: {missing}")

        return True
