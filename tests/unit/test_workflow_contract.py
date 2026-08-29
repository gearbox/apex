"""Unit contracts for v2 bundle-declared API workflows."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.api.schemas.generation import GenerationRequest
from src.api.services.workflow.applier import (
    ModelInputResolutionError,
    WorkflowApplyError,
    _select_model_filename,
    apply,
)
from src.api.services.workflow.binder import bind_workflow
from src.api.services.workflow.capabilities import (
    _GENERATION_TYPES_BY_MEDIA,
    derive_capabilities,
)
from src.api.services.workflow.contract import BoundWorkflow, MediaSlot, WorkflowRole
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


def _bound() -> BoundWorkflow:
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


def _wan_bound(wan_workflow_bundle: Path) -> BoundWorkflow:
    bundle_data = json.loads((wan_workflow_bundle / "bundle.yaml").read_text())
    graph = json.loads((wan_workflow_bundle / "workflow.api.json").read_text())
    return bind_workflow(
        parse_workflow_map(bundle_data["workflow"], wan_workflow_bundle / "bundle.yaml"), graph
    )


def test_wan_shaped_fixture_parses_binds_and_advertises_frame_to_video(
    wan_workflow_bundle: Path,
) -> None:
    """The video workflow contract is executable before a real WAN bundle lands."""
    capabilities = derive_capabilities(_wan_bound(wan_workflow_bundle), _generation())

    assert capabilities.generation_types == frozenset(
        {GenerationType.T2V, GenerationType.I2V, GenerationType.FLF2V}
    )


@pytest.mark.parametrize(
    ("media_inputs", "expected"),
    [
        pytest.param(
            lambda bound: bound.map.media_inputs[:1],
            frozenset({GenerationType.T2V, GenerationType.I2V}),
            id="first-frame",
        ),
        pytest.param(
            lambda bound: bound.map.media_inputs,
            frozenset({GenerationType.T2V, GenerationType.I2V, GenerationType.FLF2V}),
            id="first-and-last-frame",
        ),
        pytest.param(
            lambda bound: (
                replace(bound.map.media_inputs[0], kind=MediaKind.VIDEO, slot=MediaSlot.SOURCE),
            ),
            frozenset({GenerationType.T2V, GenerationType.V2V}),
            id="video-source",
        ),
        pytest.param(lambda _bound: (), frozenset({GenerationType.T2V}), id="no-media-inputs"),
    ],
)
def test_video_capabilities_follow_declared_kind_and_slot(
    wan_workflow_bundle: Path,
    media_inputs: object,
    expected: frozenset[GenerationType],
) -> None:
    bound = _wan_bound(wan_workflow_bundle)
    inputs = media_inputs(bound)  # type: ignore[operator]
    video_bound = replace(bound, map=replace(bound.map, media_inputs=inputs))

    assert derive_capabilities(video_bound, _generation()).generation_types == expected


def test_image_bundle_ignores_video_reference_slot_for_i2i_capability() -> None:
    bound = _bound()
    invalid_reference_kind = replace(bound.map.media_inputs[0], kind=MediaKind.VIDEO)
    malformed_bound = replace(bound, map=replace(bound.map, media_inputs=(invalid_reference_kind,)))

    capabilities = derive_capabilities(malformed_bound, _generation())

    assert capabilities.generation_types == frozenset({GenerationType.T2I})
    assert capabilities.max_reference_images == 0


def test_video_bundle_ignores_image_source_slot_for_v2v_capability(
    wan_workflow_bundle: Path,
) -> None:
    bound = _wan_bound(wan_workflow_bundle)
    invalid_source_kind = replace(
        bound.map.media_inputs[0], kind=MediaKind.IMAGE, slot=MediaSlot.SOURCE
    )
    malformed_bound = replace(bound, map=replace(bound.map, media_inputs=(invalid_source_kind,)))

    assert derive_capabilities(malformed_bound, _generation()).generation_types == frozenset(
        {GenerationType.T2V}
    )


def test_capabilities_use_the_bundle_batch_constraint_only_when_batch_is_mapped() -> None:
    bound = _bound()
    assert derive_capabilities(bound, _generation()).max_batch_size == 1

    raw = _map()
    raw["nodes"]["latent"]["inputs"]["batch_size"] = "batch_size"  # type: ignore[index]
    graph = _graph()
    graph["1"]["inputs"]["batch_size"] = 1  # type: ignore[index]
    batch_bound = bind_workflow(parse_workflow_map(raw, Path("bundle.yaml")), graph)

    assert derive_capabilities(batch_bound, _generation()).max_batch_size == 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda raw: raw.pop("contract_version"),
            "workflow.contract_version is required",
            id="missing-contract-version",
        ),
        pytest.param(
            lambda raw: raw.__setitem__("contract_version", 1),
            "workflow.contract_version must be 2, got 1",
            id="wrong-contract-version",
        ),
        pytest.param(
            lambda raw: raw.pop("media"), "workflow.media is required", id="missing-media"
        ),
        pytest.param(
            lambda raw: raw.__setitem__("media", "audio"),
            "workflow.media has unsupported value 'audio'",
            id="unknown-media",
        ),
        pytest.param(
            lambda raw: raw["nodes"]["latent"].__setitem__("id", []),  # type: ignore[index]
            "workflow.nodes.latent.id must be an integer or non-empty string",
            id="invalid-node-id-type",
        ),
        pytest.param(
            lambda raw: raw["nodes"]["latent"].__setitem__("id", ""),  # type: ignore[index]
            "workflow.nodes.latent.id must not be empty",
            id="empty-node-id",
        ),
        pytest.param(
            lambda raw: raw["nodes"].__setitem__("latent", "not-a-map"),  # type: ignore[index]
            "workflow.nodes.latent must be a mapping",
            id="role-not-a-mapping",
        ),
        pytest.param(
            lambda raw: raw["nodes"]["latent"]["inputs"].__setitem__("unknown", "x"),  # type: ignore[index]
            "workflow.nodes.latent.inputs has unsupported parameter 'unknown'",
            id="parameter-outside-role-vocabulary",
        ),
        pytest.param(
            lambda raw: raw["nodes"]["latent"]["inputs"].__setitem__("length", "length"),  # type: ignore[index]
            "parameter 'length' is video-only for media image",
            id="video-only-parameter-on-image",
        ),
        pytest.param(
            lambda raw: raw.__setitem__("media_inputs", "not-a-list"),
            "workflow.media_inputs must be a list",
            id="media-inputs-not-a-list",
        ),
        pytest.param(
            lambda raw: raw.__setitem__("model_inputs", "not-a-list"),
            "workflow.model_inputs must be a list",
            id="model-inputs-not-a-list",
        ),
        pytest.param(
            lambda raw: raw.__setitem__("model_inputs", ["not-a-map"]),
            "workflow.model_inputs[0] must be a mapping",
            id="model-input-not-a-mapping",
        ),
    ],
)
def test_parser_rejections_name_the_invalid_workflow_field(mutate: object, message: str) -> None:
    raw = copy.deepcopy(_map())
    mutate(raw)  # type: ignore[operator]

    with pytest.raises(WorkflowContractError) as error:
        parse_workflow_map(raw, Path("bundle.yaml"))

    assert message in str(error.value)


@pytest.mark.parametrize(
    ("slot", "message"),
    [
        pytest.param(MediaSlot.FIRST_FRAME, "slot 'first_frame' requires media video"),
        pytest.param(MediaSlot.LAST_FRAME, "slot 'last_frame' requires media video"),
        pytest.param(MediaSlot.SOURCE, "slot 'source' requires media video"),
    ],
)
def test_parser_rejects_video_slots_on_image_workflows(slot: MediaSlot, message: str) -> None:
    raw = _map()
    raw["media_inputs"][0]["slot"] = slot.value  # type: ignore[index]
    raw["media_inputs"][0]["kind"] = (  # type: ignore[index]
        MediaKind.VIDEO.value if slot is MediaSlot.SOURCE else MediaKind.IMAGE.value
    )

    with pytest.raises(WorkflowContractError) as error:
        parse_workflow_map(raw, Path("bundle.yaml"))

    assert message in str(error.value)


def test_parser_rejects_duplicate_media_slots_missing_first_frame_and_reused_ids() -> None:
    raw = _map()
    raw["media"] = "video"
    raw["media_inputs"] = [
        {
            "id": "5",
            "class": "LoadVideo",
            "kind": "video",
            "slot": "source",
            "target_role": "positive_prompt",
            "target_input": "source_video",
        },
        {
            "id": "6",
            "class": "LoadVideo",
            "kind": "video",
            "slot": "source",
            "target_role": "positive_prompt",
            "target_input": "source_video_2",
        },
        {
            "id": "7",
            "class": "LoadImage",
            "kind": "image",
            "slot": "last_frame",
            "target_role": "positive_prompt",
            "target_input": "last_frame",
        },
    ]

    with pytest.raises(WorkflowContractError) as error:
        parse_workflow_map(raw, Path("bundle.yaml"))

    text = str(error.value)
    assert "duplicate 'source' slot" in text
    assert "slot 'last_frame' requires a first_frame slot" in text
    assert "workflow node id '6' is reused by media_inputs[1], model_inputs[0]" in text


def test_parser_accumulates_independent_failures_in_one_response() -> None:
    raw = _map()
    raw["contract_version"] = 1
    raw.pop("media")
    raw["nodes"]["latent"]["id"] = ""  # type: ignore[index]

    with pytest.raises(WorkflowContractError) as error:
        parse_workflow_map(raw, Path("bundle.yaml"))

    text = str(error.value)
    assert "workflow.contract_version must be 2" in text
    assert "workflow.media is required" in text
    assert "workflow.nodes.latent.id must not be empty" in text


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda raw: raw.__setitem__("nodes", "not-a-map"),
            "workflow.nodes is required and must be a mapping",
            id="nodes-not-a-mapping",
        ),
        pytest.param(
            lambda raw: raw["nodes"].__setitem__("unsupported", {}),  # type: ignore[index]
            "workflow.nodes.unsupported has unsupported value 'unsupported'",
            id="unsupported-role",
        ),
        pytest.param(
            lambda raw: raw["nodes"]["latent"].__setitem__("class", ""),  # type: ignore[index]
            "workflow.nodes.latent.class must be a non-empty string",
            id="empty-class-name",
        ),
        pytest.param(
            lambda raw: raw.__setitem__("media_inputs", ["not-a-map"]),
            "workflow.media_inputs[0] must be a mapping",
            id="media-input-not-a-mapping",
        ),
        pytest.param(
            lambda raw: raw["media_inputs"][0].__setitem__("target_input", "prompt"),  # type: ignore[index]
            "target_input 'prompt' collides with a mapped parameter",
            id="media-target-input-collision",
        ),
    ],
)
def test_parser_rejections_cover_each_structural_level(mutate: object, message: str) -> None:
    raw = copy.deepcopy(_map())
    mutate(raw)  # type: ignore[operator]

    with pytest.raises(WorkflowContractError) as error:
        parse_workflow_map(raw, Path("bundle.yaml"))

    assert message in str(error.value)


@pytest.mark.parametrize(
    ("mutate_graph", "message"),
    [
        pytest.param(
            lambda graph: graph.pop("1"), "workflow.nodes.latent id '1' is absent", id="node-absent"
        ),
        pytest.param(
            lambda graph: graph["1"].__setitem__("class_type", "WrongClass"),  # type: ignore[index]
            "does not match API class_type",
            id="class-mismatch",
        ),
        pytest.param(
            lambda graph: graph["1"].__setitem__("inputs", []),  # type: ignore[index]
            "workflow API node '1'.inputs must be a mapping",
            id="inputs-not-a-mapping",
        ),
        pytest.param(
            lambda graph: graph["1"]["inputs"].pop("canvas_width"),  # type: ignore[index]
            "workflow.nodes.latent.inputs.width names absent API input 'canvas_width'",
            id="declared-input-absent",
        ),
        pytest.param(
            lambda graph: graph["5"]["inputs"].pop("image"),  # type: ignore[index]
            "workflow.media_inputs[0].input names absent API input 'image'",
            id="media-loader-input-absent",
        ),
        pytest.param(
            lambda graph: graph["2"]["inputs"].pop("reference_image"),  # type: ignore[index]
            "workflow.media_inputs[0].target_input names absent API input 'reference_image'",
            id="target-input-absent",
        ),
        pytest.param(
            lambda graph: graph["2"]["inputs"].__setitem__("reference_image", "scalar"),  # type: ignore[index]
            "target_input 'reference_image' must hold a graph link",
            id="target-input-scalar",
        ),
        pytest.param(
            lambda graph: graph["6"]["inputs"].pop("ckpt_name"),  # type: ignore[index]
            "workflow.model_inputs[0].input names absent API input 'ckpt_name'",
            id="model-loader-input-absent",
        ),
    ],
)
def test_binder_rejections_name_the_invalid_graph_address(
    mutate_graph: object, message: str
) -> None:
    graph = copy.deepcopy(_graph())
    mutate_graph(graph)  # type: ignore[operator]

    with pytest.raises(WorkflowContractError) as error:
        bind_workflow(parse_workflow_map(_map(), Path("bundle.yaml")), graph)

    assert message in str(error.value)


def test_binder_requires_save_role_and_declared_target_role() -> None:
    raw = _map()
    raw["nodes"].pop("save")  # type: ignore[index]
    parsed = parse_workflow_map(raw, Path("bundle.yaml"))

    with pytest.raises(WorkflowContractError, match="save role"):
        bind_workflow(parsed, _graph())

    bound = _bound()
    missing_target = replace(bound.map.media_inputs[0], target_role=WorkflowRole.NEGATIVE_PROMPT)
    malformed_map = replace(bound.map, media_inputs=(missing_target,))
    with pytest.raises(WorkflowContractError, match=r"target_role .* is absent"):
        bind_workflow(replace(bound, map=malformed_map).map, _graph())


def test_binder_accumulates_independent_faults() -> None:
    graph = _graph()
    graph.pop("1")
    graph["2"]["inputs"]["reference_image"] = "scalar"  # type: ignore[index]
    graph["5"]["inputs"].pop("image")  # type: ignore[index]

    with pytest.raises(WorkflowContractError) as error:
        bind_workflow(parse_workflow_map(_map(), Path("bundle.yaml")), graph)

    text = str(error.value)
    assert "workflow.nodes.latent id '1' is absent" in text
    assert "target_input 'reference_image' must hold a graph link" in text
    assert "media_inputs[0].input names absent API input 'image'" in text


def test_binder_reports_non_mapping_graph_nodes_and_missing_media_and_model_loaders() -> None:
    graph = _graph()
    graph["unstructured"] = "not-a-node"
    graph.pop("5")
    graph.pop("6")

    with pytest.raises(WorkflowContractError) as error:
        bind_workflow(parse_workflow_map(_map(), Path("bundle.yaml")), graph)

    text = str(error.value)
    assert "workflow API node 'unstructured' must be a mapping" in text
    assert "workflow.media_inputs[0] id '5' is absent" in text
    assert "workflow.model_inputs[0] id '6' is absent" in text


def test_binder_reports_missing_target_node_and_loader_class_mismatches() -> None:
    graph = _graph()
    graph.pop("2")
    graph["5"]["class_type"] = "WrongMediaLoader"  # type: ignore[index]
    graph["6"]["class_type"] = "WrongModelLoader"  # type: ignore[index]

    with pytest.raises(WorkflowContractError) as error:
        bind_workflow(parse_workflow_map(_map(), Path("bundle.yaml")), graph)

    text = str(error.value)
    assert "workflow.nodes.positive_prompt id '2' is absent" in text
    assert "workflow.nodes.positive_prompt declares node '2', absent" in text
    assert "workflow.media_inputs[0] id '5' class 'LoadImage' does not match" in text
    assert "workflow.model_inputs[0] id '6' class 'CheckpointLoaderSimple' does not match" in text


@pytest.mark.parametrize(
    ("model_type", "declared_filename", "available", "message"),
    [
        pytest.param(None, None, None, "neither model_type nor filename", id="neither-declared"),
        pytest.param(
            "unet", None, [], "model input type 'unet', but it has no filename", id="none-available"
        ),
        pytest.param(
            "unet",
            "declared.safetensors",
            ["other.safetensors"],
            "not present in model type 'unet'",
            id="declared-file-not-available",
        ),
        pytest.param(
            "unet",
            None,
            ["first.safetensors", "second.safetensors"],
            "Bundle model type 'unet' has 2 filenames",
            id="ambiguous-model-type",
        ),
    ],
)
def test_model_filename_resolution_errors_name_the_failed_contract(
    model_type: str | None,
    declared_filename: str | None,
    available: list[str] | None,
    message: str,
) -> None:
    with pytest.raises(ModelInputResolutionError, match=message):
        _select_model_filename(
            model_type=model_type,
            declared_filename=declared_filename,
            filenames=lambda _model_type: available,
        )


def test_model_filename_resolution_uses_static_or_unambiguous_bundle_filenames() -> None:
    assert (
        _select_model_filename(
            model_type=None,
            declared_filename="static.safetensors",
            filenames=lambda _model_type: None,
        )
        == "static.safetensors"
    )
    assert (
        _select_model_filename(
            model_type="unet",
            declared_filename=None,
            filenames=lambda _model_type: ["only.safetensors"],
        )
        == "only.safetensors"
    )


def test_apply_ignores_media_filenames_for_slots_not_declared_by_the_bundle() -> None:
    configured = apply(
        _bound(),
        GenerationRequest(prompt="t2i", height=768, width=1024),
        media_filenames={MediaSlot.SOURCE: ["ignored.mp4"]},
        filename_prefix="gen_123",
        model_filenames=lambda _model_type: ["model.safetensors"],
    )

    assert configured["5"]["inputs"]["image"] == ""
