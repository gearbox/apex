"""Unit contracts for v2 bundle-declared API workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas.generation import GenerationRequest
from src.api.services.workflow.applier import ModelInputResolutionError, WorkflowApplyError, apply
from src.api.services.workflow.binder import bind_workflow
from src.api.services.workflow.capabilities import (
    _GENERATION_TYPES_BY_MEDIA,
    derive_capabilities,
)
from src.api.services.workflow.contract import MediaSlot
from src.api.services.workflow.parser import WorkflowContractError, parse_workflow_map
from src.core.enums import GenerationType, MediaKind, Resolution, Sampler, Scheduler
from src.core.generation_config import (
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)


def _map() -> dict[str, object]:
    return {
        "contract_version": 2,
        "media": "image",
        "nodes": {
            "latent": {
                "id": "1",
                "class": "EmptyLatentImage",
                "inputs": {"width": "canvas_width", "height": "canvas_height"},
            },
            "positive_prompt": {
                "id": "2",
                "class": "CLIPTextEncode",
                "inputs": {"text": "prompt"},
            },
            "sampler": {
                "id": "3",
                "class": "KSampler",
                "inputs": {"sampler": "sampler_name", "steps": "steps"},
            },
            "save": {
                "id": "4",
                "class": "SaveImage",
                "inputs": {"filename_prefix": "prefix"},
            },
        },
        "media_inputs": [
            {
                "id": "5",
                "class": "LoadImage",
                "kind": "image",
                "slot": "reference",
                "target_role": "positive_prompt",
                "target_input": "reference_image",
            }
        ],
        "model_inputs": [
            {
                "id": "6",
                "class": "CheckpointLoaderSimple",
                "input": "ckpt_name",
                "model_type": "checkpoints",
                "filename": "model.safetensors",
            }
        ],
    }


def _graph() -> dict[str, object]:
    return {
        "1": {"class_type": "EmptyLatentImage", "inputs": {"canvas_width": 0, "canvas_height": 0}},
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"prompt": "", "reference_image": ["5", 0]},
        },
        "3": {"class_type": "KSampler", "inputs": {"sampler_name": "", "steps": 0}},
        "4": {"class_type": "SaveImage", "inputs": {"prefix": ""}},
        "5": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "6": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ""}},
    }


def _bound() -> object:
    return bind_workflow(parse_workflow_map(_map(), Path("bundle.yaml")), _graph())


def _generation() -> BundleGenerationConfig:
    return BundleGenerationConfig(
        defaults=GenerationDefaults(
            resolution=Resolution.STANDARD,
            steps=12,
            cfg=1.1,
            sampler=Sampler.EULER,
            scheduler=Scheduler.BETA,
            denoise=1.0,
        ),
        constraints=GenerationConstraints(
            max_megapixels=1.0,
            latent_multiple=16,
            max_edge=1536,
            min_steps=1,
            max_steps=20,
            min_cfg=0.0,
            max_cfg=30.0,
            allowed_samplers=frozenset(),
            allowed_schedulers=frozenset(),
            max_batch_size=3,
        ),
    )


def test_parser_accumulates_v2_contract_errors_with_source_path() -> None:
    raw = _map()
    raw["contract_version"] = 1
    raw["media"] = "image"
    raw["nodes"] = {"latent": raw["nodes"]["latent"]}  # type: ignore[index]

    with pytest.raises(WorkflowContractError) as error:
        parse_workflow_map(raw, Path("bundles/example/bundle.yaml"))

    text = str(error.value)
    assert "bundles/example/bundle.yaml" in text
    assert "contract_version" in text
    assert "positive_prompt" in text
    assert "sampler" in text


def test_binder_requires_save_and_linked_media_target() -> None:
    raw = _map()
    raw["nodes"].pop("save")  # type: ignore[index]
    graph = _graph()
    graph["2"]["inputs"]["reference_image"] = "not-a-link"  # type: ignore[index]

    with pytest.raises(WorkflowContractError) as error:
        bind_workflow(parse_workflow_map(raw, Path("bundle.yaml")), graph)

    assert "save role" in str(error.value)
    assert "must hold a graph link" in str(error.value)


def test_apply_uses_declared_input_names_and_slot_wiring() -> None:
    configured = apply(
        _bound(),  # type: ignore[arg-type]
        GenerationRequest(prompt="declared input", height=768, width=1024, sampler="heun"),
        media_filenames={MediaSlot.REFERENCE: ["source.png"]},
        filename_prefix="gen_123",
        model_filenames=lambda _model_type: ["model.safetensors"],
    )

    assert configured["2"]["inputs"]["prompt"] == "declared input"
    assert configured["3"]["inputs"]["sampler_name"] == "heun"
    assert "cfg" not in configured["3"]["inputs"]
    assert configured["5"]["inputs"]["image"] == "source.png"
    assert configured["2"]["inputs"]["reference_image"] == ["5", 0]
    assert configured["6"]["inputs"]["ckpt_name"] == "model.safetensors"


def test_apply_disconnects_media_and_fails_loudly_for_missing_model() -> None:
    with pytest.raises(ModelInputResolutionError, match="no filename"):
        apply(
            _bound(),  # type: ignore[arg-type]
            GenerationRequest(prompt="t2i", height=768, width=1024),
            media_filenames={},
            filename_prefix="gen_123",
            model_filenames=lambda _model_type: None,
        )

    raw = _map()
    raw["model_inputs"] = []
    configured = apply(
        bind_workflow(parse_workflow_map(raw, Path("bundle.yaml")), _graph()),
        GenerationRequest(prompt="t2i", height=768, width=1024),
        media_filenames={},
        filename_prefix="gen_123",
        model_filenames=lambda _model_type: None,
    )
    assert "reference_image" not in configured["2"]["inputs"]


def test_apply_rejects_oversupplied_media_filenames() -> None:
    with pytest.raises(WorkflowApplyError, match=r"received 2 filenames.*declares 1"):
        apply(
            _bound(),  # type: ignore[arg-type]
            GenerationRequest(prompt="i2i", height=768, width=1024),
            media_filenames={MediaSlot.REFERENCE: ["one.png", "two.png"]},
            filename_prefix="gen_123",
            model_filenames=lambda _model_type: ["model.safetensors"],
        )


def test_apply_tolerates_undersupplied_media_filenames() -> None:
    raw = _map()
    raw["media_inputs"].append(  # type: ignore[union-attr]
        {
            "id": "7",
            "class": "LoadImage",
            "kind": "image",
            "slot": "reference",
            "target_role": "positive_prompt",
            "target_input": "reference_image_2",
        }
    )
    graph = _graph()
    graph["2"]["inputs"]["reference_image_2"] = ["7", 0]  # type: ignore[index]
    graph["7"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
    bound = bind_workflow(parse_workflow_map(raw, Path("bundle.yaml")), graph)

    configured = apply(
        bound,
        GenerationRequest(prompt="i2i", height=768, width=1024),
        media_filenames={MediaSlot.REFERENCE: ["one.png"]},
        filename_prefix="gen_123",
        model_filenames=lambda _model_type: ["model.safetensors"],
    )

    assert configured["5"]["inputs"]["image"] == "one.png"  # type: ignore[index]
    assert "reference_image_2" not in configured["2"]["inputs"]  # type: ignore[index]


def test_capabilities_are_mechanical_from_the_bound_map() -> None:
    capabilities = derive_capabilities(_bound(), _generation())  # type: ignore[arg-type]

    assert capabilities.generation_types == frozenset({GenerationType.T2I, GenerationType.I2I})
    assert capabilities.max_batch_size == 1
    assert capabilities.max_reference_images == 1
    assert capabilities.supports_negative_prompt is False


@pytest.mark.parametrize("media_kind", MediaKind)
def test_every_media_kind_has_capability_derivation(media_kind: MediaKind) -> None:
    assert media_kind in _GENERATION_TYPES_BY_MEDIA


def test_missing_capability_derivation_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(_GENERATION_TYPES_BY_MEDIA, MediaKind.IMAGE)

    with pytest.raises(WorkflowContractError, match="image"):
        derive_capabilities(_bound(), _generation())  # type: ignore[arg-type]


def test_capabilities_do_not_advertise_inputs_without_request_sources() -> None:
    raw = _map()
    raw["nodes"]["model_sampling"] = {  # type: ignore[index]
        "id": "7",
        "class": "ModelSamplingAuraFlow",
        "inputs": {"shift": "shift"},
    }
    graph = _graph()
    graph["7"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 0.0}}

    capabilities = derive_capabilities(
        bind_workflow(parse_workflow_map(raw, Path("bundle.yaml")), graph),
        _generation(),
    )
    assert "model_sampling.shift" not in capabilities.writable
