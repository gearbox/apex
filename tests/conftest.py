"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest

from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide test settings."""
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=18188,
        debug=True,
    )


@pytest.fixture
def workflow_service(tmp_path: Path) -> WorkflowService:
    """Provide workflow service with test bundle."""
    # Create test bundle structure
    bundle_dir = tmp_path / "config" / "bundles" / "qwen_rapid_aio" / "260103-18"
    bundle_dir.mkdir(parents=True)

    # Copy actual workflow for testing
    source_workflow = (
        Path(__file__).parent.parent
        / "config"
        / "bundles"
        / "qwen_rapid_aio"
        / "260103-18"
        / "workflow.json"
    )

    if source_workflow.exists():
        import shutil

        shutil.copy(source_workflow, bundle_dir / "workflow.json")
    else:
        # Create minimal test workflow
        import json

        test_workflow = {
            "nodes": [
                {
                    "id": 9,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "widgets_values": [1024, 1024, 1],
                },
                {
                    "id": 3,
                    "type": "TextEncodeQwenImageEditPlus",
                    "inputs": [],
                    "widgets_values": ["test prompt"],
                },
                {
                    "id": 4,
                    "type": "TextEncodeQwenImageEditPlus",
                    "inputs": [],
                    "widgets_values": ["negative"],
                },
                {
                    "id": 2,
                    "type": "KSampler",
                    "inputs": [],
                    "widgets_values": [12345, "fixed", 12, 1.1, "euler", "beta", 1],
                },
            ],
            "links": [],
        }
        (bundle_dir / "workflow.json").write_text(json.dumps(test_workflow))

    return WorkflowService(base_path=tmp_path)
