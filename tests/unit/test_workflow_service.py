"""Tests for WorkflowService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import structlog.testing

from src.api.schemas.generation import (
    AspectRatio,
    GenerationRequest,
    GenerationType,
)
from src.api.services.workflow_service import (
    NodeIDs,
    WorkflowConfigError,
    WorkflowNotFoundError,
    WorkflowService,
    WorkflowValidationError,
)
from tests.unit.conftest import _MINIMAL_GUI_WORKFLOW, make_bundle_dir


class TestLoadWorkflowFromBundle:
    """WorkflowService.load_workflow_from_bundle() — bundle cache path."""

    def test_load_uses_bundle_cache_not_vendored(self, tmp_path: Path) -> None:
        """Workflow is loaded from the bundle cache, not a vendored path."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()

        workflow = svc.load_workflow_from_bundle(bundle_root, bundle_version="260103-19")

        assert isinstance(workflow, dict)
        # Required nodes present
        assert NodeIDs.EMPTY_LATENT in workflow
        assert NodeIDs.KSAMPLER in workflow

    def test_load_uses_current_symlink_when_no_version(self, tmp_path: Path) -> None:
        """When bundle_version is None, 'current' symlink is used."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()

        workflow = svc.load_workflow_from_bundle(bundle_root, bundle_version=None)

        assert NodeIDs.EMPTY_LATENT in workflow

    def test_workflow_caching_returns_deep_copy(self, tmp_path: Path) -> None:
        """Successive calls return equal but distinct dicts (cache + deep copy)."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()

        w1 = svc.load_workflow_from_bundle(bundle_root, "260103-19")
        w2 = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        assert w1 == w2
        assert w1 is not w2

    def test_raises_workflow_not_found_when_missing(self, tmp_path: Path) -> None:
        bundle_root = tmp_path / "bundles" / "nonexistent"
        bundle_root.mkdir(parents=True)
        svc = WorkflowService()

        with pytest.raises(WorkflowNotFoundError):
            svc.load_workflow_from_bundle(bundle_root, "260103-19")

    def test_raises_validation_error_on_bad_json(self, tmp_path: Path) -> None:
        bundle_root = tmp_path / "bundles" / "bad"
        version_dir = bundle_root / "v1"
        version_dir.mkdir(parents=True)
        (version_dir / "workflow.json").write_text("{ invalid json")
        svc = WorkflowService()

        with pytest.raises(WorkflowValidationError):
            svc.load_workflow_from_bundle(bundle_root, "v1")


class TestInjectCheckpoint:
    """WorkflowService.inject_checkpoint() — single source of truth for ckpt_name."""

    def test_ckpt_name_from_bundle_yaml_overrides_stale_widget(self, tmp_path: Path) -> None:
        """Checkpoint filename from BundleIndexService replaces whatever is baked in the workflow."""
        bundle_root, _ = make_bundle_dir(tmp_path, version="260103-19")
        mock_bundles = MagicMock()
        mock_bundles.get_checkpoint_filenames.return_value = ["Qwen-Rapid-AIO-NSFW-v19.safetensors"]
        svc = WorkflowService(bundle_index=mock_bundles)
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        # The workflow bakes "STALE.safetensors" as the checkpoint name
        assert workflow[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"] == "STALE.safetensors"

        with structlog.testing.capture_logs() as cap:
            svc.inject_checkpoint(
                workflow, "qwen_rapid_aio", "260103-19", session_id="test-session"
            )

        assert workflow[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"] == (
            "Qwen-Rapid-AIO-NSFW-v19.safetensors"
        )
        mock_bundles.get_checkpoint_filenames.assert_called_once_with("qwen_rapid_aio", "260103-19")
        assert any(e["event"] == "workflow.checkpoint_injected" for e in cap)
        injected = next(e for e in cap if e["event"] == "workflow.checkpoint_injected")
        assert injected["ckpt_name"] == "Qwen-Rapid-AIO-NSFW-v19.safetensors"

    def test_injection_skipped_when_multiple_checkpoint_files(self, tmp_path: Path) -> None:
        """Guard: two checkpoint files reported by BundleIndexService → no injection."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        mock_bundles = MagicMock()
        mock_bundles.get_checkpoint_filenames.return_value = ["high.safetensors", "low.safetensors"]
        svc = WorkflowService(bundle_index=mock_bundles)
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")
        original_ckpt = workflow[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"]

        with structlog.testing.capture_logs() as cap:
            svc.inject_checkpoint(workflow, "qwen_rapid_aio", "260103-19", session_id="s1")

        # Widget value unchanged
        assert workflow[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"] == original_ckpt
        assert any(e["event"] == "workflow.checkpoint_injection_skipped" for e in cap)

    def test_injection_skipped_when_multiple_loader_nodes(self, tmp_path: Path) -> None:
        """Guard: two CheckpointLoaderSimple nodes → no injection."""
        nodes_raw = _MINIMAL_GUI_WORKFLOW["nodes"]
        assert isinstance(nodes_raw, list)
        extra_nodes = [
            *nodes_raw,
            {
                "id": 99,
                "type": "CheckpointLoaderSimple",
                "inputs": [],
                "widgets_values": ["extra.safetensors"],
            },
        ]
        multi_loader_workflow: dict[str, object] = {
            "nodes": extra_nodes,
            "links": _MINIMAL_GUI_WORKFLOW["links"],
        }
        bundle_root = tmp_path / "bundles" / "multi_loader"
        version_dir = bundle_root / "v1"
        version_dir.mkdir(parents=True)
        (version_dir / "workflow.json").write_text(json.dumps(multi_loader_workflow))

        mock_bundles = MagicMock()
        mock_bundles.get_checkpoint_filenames.return_value = ["single.safetensors"]
        svc = WorkflowService(bundle_index=mock_bundles)
        workflow = svc.load_workflow_from_bundle(bundle_root, "v1")
        original_ckpt = workflow[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"]

        with structlog.testing.capture_logs() as cap:
            svc.inject_checkpoint(workflow, "multi_loader", "v1", session_id="s2")

        assert workflow[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"] == original_ckpt
        assert any(e["event"] == "workflow.checkpoint_injection_skipped" for e in cap)

    def test_injection_raises_when_no_bundle_index_and_single_loader(self, tmp_path: Path) -> None:
        """When WorkflowService has no bundle_index and workflow has 1 loader, raise WorkflowConfigError."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()  # no bundle_index
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        with pytest.raises(WorkflowConfigError, match="bundle index is not configured"):
            svc.inject_checkpoint(workflow, "qwen_rapid_aio", "260103-19", session_id="no-idx")

    def test_injection_skipped_when_no_bundle_index_and_no_loader_nodes(self) -> None:
        """When WorkflowService has no bundle_index and workflow has 0 loader nodes, info-skip."""
        svc = WorkflowService()  # no bundle_index
        workflow: dict[str, object] = {}  # no CheckpointLoaderSimple nodes

        with structlog.testing.capture_logs() as cap:
            svc.inject_checkpoint(workflow, "some_bundle", None, session_id="no-idx")

        assert any(e["event"] == "workflow.checkpoint_injection_skipped" for e in cap)


class TestWiredInputsPreserved:
    """The converter correctly wires output nodes from the links array."""

    def test_wired_save_image_has_images_input(self, tmp_path: Path) -> None:
        """Node 11 (SaveImage, link 19 from VAEDecode node 5) is correctly wired."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()

        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        save_node = workflow[NodeIDs.SAVE_IMAGE]  # node "11"
        assert save_node["class_type"] == "SaveImage"
        # Link 19: VAEDecode (node 5) slot 0 → SaveImage (node 11)
        assert save_node["inputs"]["images"] == ["5", 0]


class TestEndToEndAishaImage:
    """End-to-end workflow shape for aisha-image with qwen_rapid_aio v19."""

    def test_final_workflow_has_correct_ckpt_and_wired_save_image(self, tmp_path: Path) -> None:
        """After load + inject + apply_parameters, ckpt_name is v19 and SaveImage is wired."""
        bundle_root, _ = make_bundle_dir(tmp_path, version="260103-19")
        mock_bundles = MagicMock()
        mock_bundles.get_checkpoint_filenames.return_value = ["Qwen-Rapid-AIO-NSFW-v19.safetensors"]
        svc = WorkflowService(bundle_index=mock_bundles)
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")
        svc.inject_checkpoint(workflow, "qwen_rapid_aio", "260103-19", session_id="e2e")

        request = GenerationRequest(
            prompt="A ginger cat on grass",
            generation_type=GenerationType.T2I,
        )
        configured = svc.apply_parameters(workflow=workflow, request=request)

        assert (
            configured[NodeIDs.CHECKPOINT_LOADER]["inputs"]["ckpt_name"]
            == "Qwen-Rapid-AIO-NSFW-v19.safetensors"
        )
        # All SaveImage nodes in the final workflow must have a non-null images input
        save_image_nodes = [v for v in configured.values() if v.get("class_type") == "SaveImage"]
        assert save_image_nodes, "Expected at least one SaveImage node"
        for node in save_image_nodes:
            assert "images" in node["inputs"], f"SaveImage node missing 'images' input: {node}"


class TestApplyParameters:
    """WorkflowService.apply_parameters() — parameter injection."""

    def test_apply_parameters(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(
            prompt="A beautiful cat",
            negative_prompt="ugly, blurry",
            height=1080,
            aspect_ratio=AspectRatio.RATIO_16_9,
            max_images=2,
            seed=42,
            steps=8,
        )
        modified = svc.apply_parameters(workflow=workflow, request=request, filename_prefix="test")

        assert modified[NodeIDs.EMPTY_LATENT]["inputs"]["width"] == 1920
        assert modified[NodeIDs.EMPTY_LATENT]["inputs"]["height"] == 1080
        assert modified[NodeIDs.EMPTY_LATENT]["inputs"]["batch_size"] == 2
        assert modified[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"] == "A beautiful cat"
        assert modified[NodeIDs.NEGATIVE_PROMPT]["inputs"]["prompt"] == "ugly, blurry"
        assert modified[NodeIDs.KSAMPLER]["inputs"]["seed"] == 42
        assert modified[NodeIDs.KSAMPLER]["inputs"]["steps"] == 8
        save_image_nodes = [v for v in modified.values() if v.get("class_type") == "SaveImage"]
        assert save_image_nodes, "Expected at least one SaveImage node"
        for node in save_image_nodes:
            assert node["inputs"].get("filename_prefix") == "test"

    def test_apply_parameters_immutable(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")
        original_prompt = workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"]

        svc.apply_parameters(workflow=workflow, request=GenerationRequest(prompt="New prompt"))

        assert workflow[NodeIDs.POSITIVE_PROMPT]["inputs"]["prompt"] == original_prompt

    def test_apply_parameters_with_images(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(prompt="test", generation_type=GenerationType.I2I)
        modified = svc.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1="uploaded_image1.png",
            input_image_2="uploaded_image2.png",
        )

        assert modified[NodeIDs.LOAD_IMAGE_1]["inputs"]["image"] == "uploaded_image1.png"
        assert modified[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] == "uploaded_image2.png"

    def test_t2i_disconnects_images(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(prompt="A cat", generation_type=GenerationType.T2I)
        modified = svc.apply_parameters(workflow=workflow, request=request)

        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert "image1" not in positive_inputs
        assert "image2" not in positive_inputs
        assert "image3" not in positive_inputs

    def test_i2i_keeps_images(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(prompt="Use this", generation_type=GenerationType.I2I)
        modified = svc.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1="ref1.png",
            input_image_2="ref2.png",
        )

        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert positive_inputs["image1"] == [NodeIDs.LOAD_IMAGE_1, 0]
        assert positive_inputs["image2"] == [NodeIDs.LOAD_IMAGE_2, 0]

    def test_i2i_single_image(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(prompt="Use this", generation_type=GenerationType.I2I)
        modified = svc.apply_parameters(
            workflow=workflow,
            request=request,
            input_image_1=None,
            input_image_2="only_second.png",
        )

        positive_inputs = modified[NodeIDs.POSITIVE_PROMPT]["inputs"]
        assert "image1" not in positive_inputs
        assert positive_inputs["image2"] == [NodeIDs.LOAD_IMAGE_2, 0]
        assert modified[NodeIDs.LOAD_IMAGE_2]["inputs"]["image"] == "only_second.png"


class TestInvalidateCache:
    def test_invalidate_cache_clears_workflow_cache(self, tmp_path: Path) -> None:
        """invalidate_cache() empties _workflow_cache."""
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()

        svc.load_workflow_from_bundle(bundle_root, "260103-19")
        assert len(svc._workflow_cache) == 1

        svc.invalidate_cache()

        assert svc._workflow_cache == {}

    def test_invalidate_cache_forces_reload_from_disk(self, tmp_path: Path) -> None:
        """After invalidate_cache(), load_workflow_from_bundle re-reads from disk."""
        bundle_root, version_dir = make_bundle_dir(tmp_path)
        svc = WorkflowService()

        svc.load_workflow_from_bundle(bundle_root, "260103-19")
        svc.invalidate_cache()

        # Replace workflow.json on disk with a minimal valid workflow
        new_workflow = {
            "nodes": [
                {
                    "id": 9,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "widgets_values": [512, 512, 1],
                },
                {
                    "id": 3,
                    "type": "TextEncodeQwenImageEditPlus",
                    "inputs": [],
                    "widgets_values": ["new prompt"],
                },
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [],
                    "widgets_values": [0, "fixed", 10, 1.0, "euler", "beta", 1.0],
                },
            ],
            "links": [],
        }
        (version_dir / "workflow.json").write_text(json.dumps(new_workflow))

        reloaded = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        assert reloaded[NodeIDs.EMPTY_LATENT]["inputs"]["width"] == 512

    def test_invalidate_cache_noop_when_cache_empty(self) -> None:
        """invalidate_cache() is safe when the cache is already empty."""
        svc = WorkflowService()
        svc.invalidate_cache()
        assert svc._workflow_cache == {}


class TestValidateWorkflow:
    def test_validate_valid(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")
        assert svc.validate_workflow(workflow) is True

    def test_validate_missing_nodes(self) -> None:
        svc = WorkflowService()
        with pytest.raises(WorkflowValidationError, match="missing required nodes"):
            svc.validate_workflow({"1": {"class_type": "SomeNode", "inputs": {}}})


class TestApplyParametersFullSamplerInjection:
    """apply_parameters injects cfg, sampler_name, scheduler, denoise into KSampler."""

    def test_injects_all_sampler_params(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(
            prompt="test",
            seed=99,
            steps=20,
            cfg=3.5,
            sampler="dpmpp_2m",
            scheduler="karras",
            denoise=0.8,
        )
        result = svc.apply_parameters(workflow=workflow, request=request)
        ks = result[NodeIDs.KSAMPLER]["inputs"]

        assert ks["seed"] == 99
        assert ks["steps"] == 20
        assert ks["cfg"] == pytest.approx(3.5)
        assert ks["sampler_name"] == "dpmpp_2m"
        assert ks["scheduler"] == "karras"
        assert ks["denoise"] == pytest.approx(0.8)

    def test_injects_explicit_width(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(
            prompt="test",
            height=768,
            width=1024,  # explicit; should override calculated
        )
        result = svc.apply_parameters(workflow=workflow, request=request)
        latent = result[NodeIDs.EMPTY_LATENT]["inputs"]
        assert latent["width"] == 1024
        assert latent["height"] == 768

    def test_defaults_injected_when_not_overridden(self, tmp_path: Path) -> None:
        bundle_root, _ = make_bundle_dir(tmp_path)
        svc = WorkflowService()
        workflow = svc.load_workflow_from_bundle(bundle_root, "260103-19")

        request = GenerationRequest(prompt="test")
        result = svc.apply_parameters(workflow=workflow, request=request)
        ks = result[NodeIDs.KSAMPLER]["inputs"]

        assert ks["cfg"] == pytest.approx(1.1)  # GenerationRequest default
        assert ks["sampler_name"] == "euler"
        assert ks["scheduler"] == "beta"
        assert ks["denoise"] == pytest.approx(1.0)


class TestGuiToApiConversion:
    """GUI → API workflow format conversion."""

    def test_convert_simple_workflow(self, tmp_path: Path) -> None:
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
                    "inputs": [{"name": "model", "link": 1}],
                    "widgets_values": [12345, "fixed", 20, 7.0, "euler", "normal", 1.0],
                },
            ],
            "links": [
                [1, 1, 0, 2, 0, "MODEL"],
            ],
        }
        version_dir = tmp_path / "bundles" / "test" / "v1"
        version_dir.mkdir(parents=True)
        (version_dir / "workflow.json").write_text(json.dumps(gui_workflow))

        svc = WorkflowService()
        api_workflow = svc.load_workflow_from_bundle(tmp_path / "bundles" / "test", "v1")

        assert api_workflow["1"]["class_type"] == "CheckpointLoaderSimple"
        assert api_workflow["2"]["class_type"] == "KSampler"
        assert api_workflow["1"]["inputs"]["ckpt_name"] == "model.safetensors"
        assert api_workflow["2"]["inputs"]["seed"] == 12345
        assert api_workflow["2"]["inputs"]["steps"] == 20
        # Link resolved: KSampler.model ← CheckpointLoaderSimple slot 0
        assert api_workflow["2"]["inputs"]["model"] == ["1", 0]
