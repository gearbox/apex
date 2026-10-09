"""C13: the Wan request body, one test per generation mode."""

from __future__ import annotations

from typing import Any

import pytest

from src.api.services.model_studio.wan_payload import build_wan_video_payload
from src.core.enums import AspectRatio, GenerationType, MediaKind, ModelType, VideoResolution
from src.core.model_registry import get_model_meta

pytestmark = pytest.mark.unit

_IMG = MediaKind.IMAGE
_VID = MediaKind.VIDEO


def _build(generation_type: GenerationType, media: list[tuple[MediaKind, str]], **kw: Any) -> Any:
    contract = get_model_meta(ModelType.WAN_3_VIDEO).generation_modes[generation_type].source_media
    args: dict[str, Any] = {
        "model": ModelType.WAN_3_VIDEO,
        "prompt": "a fox runs",
        "contract": contract,
        "media": media,
        "duration": 5,
        "resolution": VideoResolution.RES_720P,
        "aspect_ratio": None,
        "audio": None,
        "prompt_extend": None,
        "seed": None,
    } | kw
    return build_wan_video_payload(**args)


def test_t2v_has_no_media_and_defaults_to_16_9() -> None:
    body = _build(GenerationType.T2V, [])

    assert body == {
        "model": "wan3.0-video",
        "input": {"prompt": "a fox runs"},
        "parameters": {
            "resolution": "720P",
            "ratio": "16:9",
            "duration": 5,
            "watermark": False,
        },
    }


def test_i2v_types_the_image_as_first_frame_and_uses_adaptive_ratio() -> None:
    body = _build(GenerationType.I2V, [(_IMG, "https://r2/a")])

    assert body["input"]["media"] == [{"type": "first_frame", "url": "https://r2/a"}]
    assert body["parameters"]["ratio"] == "adaptive"


def test_flf2v_orders_first_then_last_frame() -> None:
    body = _build(GenerationType.FLF2V, [(_IMG, "https://r2/a"), (_IMG, "https://r2/b")])

    assert body["input"]["media"] == [
        {"type": "first_frame", "url": "https://r2/a"},
        {"type": "last_frame", "url": "https://r2/b"},
    ]


def test_v2v_types_the_video_as_reference_video() -> None:
    body = _build(GenerationType.V2V, [(_VID, "https://r2/v")])

    assert body["input"]["media"] == [{"type": "reference_video", "url": "https://r2/v"}]
    assert body["parameters"]["ratio"] == "adaptive"


def test_r2v_keeps_request_order_as_image_1_to_n() -> None:
    urls = [f"https://r2/{i}" for i in range(4)]
    body = _build(GenerationType.R2V, [(_IMG, u) for u in urls])

    assert body["input"]["media"] == [{"type": "reference_image", "url": u} for u in urls]


@pytest.mark.parametrize(
    ("resolution", "wire"),
    [
        (VideoResolution.RES_480P, "480P"),
        (VideoResolution.RES_720P, "720P"),
        (VideoResolution.RES_1080P, "1080P"),
    ],
)
def test_resolution_uses_the_explicit_wire_table(resolution: VideoResolution, wire: str) -> None:
    assert _build(GenerationType.T2V, [], resolution=resolution)["parameters"]["resolution"] == wire


def test_an_explicit_ratio_is_sent_as_is() -> None:
    body = _build(GenerationType.T2V, [], aspect_ratio=AspectRatio.RATIO_9_16)

    assert body["parameters"]["ratio"] == "9:16"


def test_optional_parameters_are_omitted_when_none_so_upstream_defaults_apply() -> None:
    parameters = _build(GenerationType.T2V, [])["parameters"]

    assert not {"audio", "prompt_extend", "seed"} & parameters.keys()


def test_optional_parameters_are_forwarded_when_set_including_false_and_zero() -> None:
    parameters = _build(GenerationType.T2V, [], audio=False, prompt_extend=False, seed=0)[
        "parameters"
    ]

    assert parameters["audio"] is False
    assert parameters["prompt_extend"] is False
    assert parameters["seed"] == 0


def test_watermark_is_always_false_and_unsupported_inputs_are_never_emitted() -> None:
    body = _build(GenerationType.R2V, [(_IMG, "https://r2/a")], audio=True, seed=7)
    wire = repr(body)

    assert body["parameters"]["watermark"] is False
    assert body["parameters"]["duration"] == 5
    for forbidden in ("negative_prompt", "'file'", "'link'", "reference_audio", "'duration': -1"):
        assert forbidden not in wire


def test_media_not_matching_the_role_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="roles"):
        _build(GenerationType.FLF2V, [(_IMG, "https://r2/a")])
