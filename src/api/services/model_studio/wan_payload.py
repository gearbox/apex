"""Pure builder for the Wan 3.0 ``video-synthesis`` request body.

No I/O and no settings: everything the body depends on is an argument, so each
generation mode is unit-testable against the exact wire format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.core.enums import AspectRatio, MediaKind, MediaSlot, ModelType, VideoResolution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.core.generation_mode import SourceMediaConstraints

_SLOT_TO_WAN_TYPE: Final[Mapping[MediaSlot, str]] = {
    MediaSlot.FIRST_FRAME: "first_frame",
    MediaSlot.LAST_FRAME: "last_frame",
}
_KIND_TO_WAN_REFERENCE: Final[Mapping[MediaKind, str]] = {
    MediaKind.IMAGE: "reference_image",
    MediaKind.VIDEO: "reference_video",
}
_RESOLUTION_TO_WAN: Final[Mapping[VideoResolution, str]] = {
    VideoResolution.RES_480P: "480P",
    VideoResolution.RES_720P: "720P",
    VideoResolution.RES_1080P: "1080P",
}
_ADAPTIVE_RATIO: Final = "adaptive"


def build_wan_video_payload(
    *,
    model: ModelType,
    prompt: str,
    contract: SourceMediaConstraints | None,
    media: Sequence[tuple[MediaKind, str]],
    duration: int,
    resolution: VideoResolution,
    aspect_ratio: AspectRatio | None,
    audio: bool | None,
    prompt_extend: bool | None,
    seed: int | None,
) -> dict[str, object]:
    """Build the JSON body for ``POST .../video-synthesis``.

    Args:
        model: Wan model key (sent as ``model``).
        prompt: Text prompt; reference modes cite "Image 1..N" in ``media`` order.
        contract: The mode's source-media contract; positional roles decide
            first/last-frame typing, otherwise the item's kind decides.
        media: ``(kind, presigned_url)`` in request order.
        duration: Output seconds. Upstream's smart duration is never requested.
        resolution: Output resolution.
        aspect_ratio: Requested ratio, or ``None`` for ``adaptive`` (inputs present)
            / ``16:9`` (text-to-video).
        audio: ``None`` leaves the upstream default.
        prompt_extend: ``None`` leaves the upstream default.
        seed: ``None`` leaves the upstream (random) default.

    Returns:
        The request body.

    Raises:
        ValueError: ``media`` does not line up with the contract's roles.
    """
    roles = contract.roles if contract is not None else ()
    if roles and len(roles) != len(media):
        raise ValueError(f"expected {len(roles)} media items for roles, got {len(media)}")

    wire_media: list[dict[str, str]] = []
    for index, (kind, url) in enumerate(media):
        wire_type = _SLOT_TO_WAN_TYPE[roles[index]] if roles else _KIND_TO_WAN_REFERENCE[kind]
        wire_media.append({"type": wire_type, "url": url})

    if aspect_ratio is not None:
        ratio = aspect_ratio.value
    elif media:
        ratio = _ADAPTIVE_RATIO
    else:
        ratio = AspectRatio.RATIO_16_9.value

    parameters: dict[str, object] = {
        "resolution": _RESOLUTION_TO_WAN[resolution],
        "ratio": ratio,
        "duration": duration,
        "watermark": False,
    }
    if audio is not None:
        parameters["audio"] = audio
    if prompt_extend is not None:
        parameters["prompt_extend"] = prompt_extend
    if seed is not None:
        parameters["seed"] = seed

    input_body: dict[str, object] = {"prompt": prompt}
    if wire_media:
        input_body["media"] = wire_media
    return {"model": model.value, "input": input_body, "parameters": parameters}
