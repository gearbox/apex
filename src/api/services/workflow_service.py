"""Workflow manipulation service for ComfyUI workflows."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.api.schemas.generation import GenerationRequest, GenerationType

logger = structlog.get_logger(__name__)


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
    - Loading workflow templates from the synced bundle cache
    - Injecting checkpoint names from bundle.yaml at submit time
    - Applying user parameters to workflow nodes
    - Converting GUI workflow format to API format
    """

    def __init__(self) -> None:
        self._workflow_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _bundle_version_dir(bundle_dir: Path, bundle_version: str | None) -> Path:
        """Return the versioned subdirectory for a bundle."""
        return bundle_dir / (bundle_version or "current")

    def load_workflow_from_bundle(
        self,
        bundle_dir: Path,
        bundle_version: str | None,
    ) -> dict[str, Any]:
        """Load and convert the workflow from a provisioned bundle directory.

        Args:
            bundle_dir: Path to the bundle root (e.g. cache_dir / "bundles/qwen_rapid_aio").
            bundle_version: Specific version (e.g. "260103-19"). None uses the "current" symlink.

        Returns:
            Parsed workflow in API format (deep copy from cache).

        Raises:
            WorkflowNotFoundError: If the workflow file does not exist.
            WorkflowValidationError: If the workflow JSON is invalid.
        """
        workflow_path = self._bundle_version_dir(bundle_dir, bundle_version) / "workflow.json"
        # When bundle_version is None we follow the "current" symlink; resolve it
        # so the cache key changes when the symlink is updated to a new version.
        if bundle_version is None:
            try:
                cache_key = str(workflow_path.resolve())
            except OSError as e:
                raise WorkflowNotFoundError(f"Workflow not found: {workflow_path}") from e
        else:
            cache_key = str(workflow_path)

        if cache_key not in self._workflow_cache:
            if not workflow_path.exists():
                raise WorkflowNotFoundError(f"Workflow not found: {workflow_path}")

            try:
                with workflow_path.open() as f:
                    gui_workflow = json.load(f)

                api_workflow = self._convert_gui_to_api_format(gui_workflow)
                self._workflow_cache[cache_key] = api_workflow

            except json.JSONDecodeError as e:
                raise WorkflowValidationError(f"Invalid workflow JSON: {e}") from e

        return copy.deepcopy(self._workflow_cache[cache_key])

    def inject_checkpoint(
        self,
        workflow: dict[str, Any],
        bundle_dir: Path,
        bundle_version: str | None,
        *,
        session_id: str,
    ) -> None:
        """Inject ckpt_name from bundle.yaml into the workflow (in-place).

        Guard: only injects when there is exactly one checkpoint file in bundle.yaml
        AND exactly one CheckpointLoaderSimple node in the workflow. Multi-loader
        (video) workflows are left untouched.

        Args:
            workflow: API-format workflow dict to modify in-place.
            bundle_dir: Path to the bundle root directory.
            bundle_version: Specific version. None uses the "current" symlink.
            session_id: Session ID for log context.
        """
        bundle_yaml_path = self._bundle_version_dir(bundle_dir, bundle_version) / "bundle.yaml"

        ckpt_files: list[str] = []
        try:
            with bundle_yaml_path.open() as f:
                bundle_data = yaml.safe_load(f)
            if isinstance(bundle_data, dict):
                for model in bundle_data.get("models", []) or []:
                    if isinstance(model, dict) and model.get("model_type") == "checkpoints":
                        ckpt_files.extend(
                            str(file_entry["filename"])
                            for file_entry in model.get("files", []) or []
                            if isinstance(file_entry, dict) and "filename" in file_entry
                        )
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "workflow.checkpoint_injection_skipped",
                session_id=session_id,
                reason="bundle_yaml_read_error",
                error=str(exc),
            )
            return

        ckpt_nodes = [
            (node_id, node)
            for node_id, node in workflow.items()
            if isinstance(node, dict) and node.get("class_type") == "CheckpointLoaderSimple"
        ]

        if len(ckpt_files) != 1 or len(ckpt_nodes) != 1:
            logger.info(
                "workflow.checkpoint_injection_skipped",
                session_id=session_id,
                checkpoint_file_count=len(ckpt_files),
                loader_node_count=len(ckpt_nodes),
            )
            return

        _node_id, node = ckpt_nodes[0]
        node.setdefault("inputs", {})["ckpt_name"] = ckpt_files[0]
        logger.info(
            "workflow.checkpoint_injected",
            session_id=session_id,
            ckpt_name=ckpt_files[0],
        )

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
            logger.debug("workflow.disconnected_t2i_mode")
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
                "workflow.configured",
                image1=input_image_1,
                image2=input_image_2,
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
