"""Registry-driven validation and billing basis for video generation requests.

Pure functions over ``VideoMeta`` so every rule is unit-testable without the
orchestrator. ``GenerationService.generate`` calls these before any billing
work, in two phases: ``validate_video_request`` (request only) and
``validate_video_sources`` (needs the ownership-checked, resolved assets).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.api.services.generation.source_media import SourceMediaValidationError
from src.core.enums import MediaFormat, MediaKind, VideoParameter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.core.model_registry import InputVideoConstraints, ModelMeta, VideoMeta


def unsupported_video_parameters(
    request: UnifiedGenerationRequest, supported: frozenset[VideoParameter]
) -> list[str]:
    """Names of optional video parameters sent as non-``None`` but not declared by the model."""
    return [
        parameter.value
        for parameter in VideoParameter
        if getattr(request, parameter.value) is not None and parameter not in supported
    ]


def validate_video_request(request: UnifiedGenerationRequest, meta: ModelMeta) -> VideoMeta:
    """Check duration, resolution and aspect ratio against the model's video constraints.

    Returns:
        The model's ``VideoMeta`` for the caller's later phases.

    Raises:
        ValueError: The model has no video constraints (a registry bug) or a value is
            outside what the model supports.
    """
    video = meta.video
    if video is None:
        raise ValueError(f"Model '{request.model.value}' has no video constraints")

    if not video.min_duration <= request.duration <= video.max_duration:
        raise ValueError(
            f"Model '{request.model.value}' supports video durations of "
            f"{video.min_duration}-{video.max_duration} seconds; got {request.duration}"
        )
    if request.resolution not in video.resolutions:
        raise ValueError(
            f"Model '{request.model.value}' does not support resolution "
            f"'{request.resolution.value}'; supported: {[r.value for r in video.resolutions]}"
        )
    if request.aspect_ratio is not None:
        if request.generation_type in video.input_anchored_modes:
            raise ValueError(
                f"Model '{request.model.value}' derives the output aspect ratio from its input "
                f"media in '{request.generation_type.value}' mode; omit aspect_ratio"
            )
        if request.aspect_ratio not in meta.aspect_ratios:
            raise ValueError(
                f"Model '{request.model.value}' does not support aspect ratio "
                f"'{request.aspect_ratio.value}'; supported: "
                f"{[a.value for a in meta.aspect_ratios]}"
            )
    return video


def _needs_input_durations(video: VideoMeta) -> bool:
    """Whether a video source must carry a known duration (limits or billing depend on it)."""
    clip = video.input_video
    return (
        (clip is not None and (clip.min_duration_ms > 0 or clip.max_duration_ms is not None))
        or video.max_combined_duration is not None
        or video.bills_input_video_seconds
    )


def _video_source_seconds(sources: Sequence[ResolvedSourceMedia], video: VideoMeta) -> list[int]:
    """Whole seconds (rounded up) of each video source; fails closed on unknown length."""
    if not _needs_input_durations(video):
        return []
    seconds: list[int] = []
    for source in sources:
        if source.media_kind is not MediaKind.VIDEO:
            continue
        if source.duration_ms is None:
            raise SourceMediaValidationError(
                f"source_media position {source.position}: source video has no recorded duration"
            )
        seconds.append(math.ceil(source.duration_ms / 1000))
    return seconds


def _validate_clip(source: ResolvedSourceMedia, clip: InputVideoConstraints) -> None:
    """Check one video source's container, size and duration, in that order.

    Every message names the limit and the position only; none carries a storage key or URL.
    """
    position = source.position
    try:
        container: MediaFormat | None = MediaFormat.from_content_type(source.content_type)
    except ValueError:
        container = None
    if container is None or container not in clip.formats:
        shown = container.value if container is not None else source.content_type
        allowed = ", ".join(sorted(fmt.value for fmt in clip.formats))
        raise SourceMediaValidationError(
            f"source_media position {position}: {shown} is not accepted; use one of {allowed}"
        )
    if clip.max_bytes is not None and source.size_bytes > clip.max_bytes:
        raise SourceMediaValidationError(
            f"source_media position {position}: source video is {source.size_bytes} bytes; "
            f"the maximum is {clip.max_bytes} bytes"
        )
    if clip.min_duration_ms == 0 and clip.max_duration_ms is None:
        return
    if source.duration_ms is None:
        raise SourceMediaValidationError(
            f"source_media position {position}: source video has no recorded duration"
        )
    if source.duration_ms < clip.min_duration_ms:
        raise SourceMediaValidationError(
            f"source_media position {position}: source video is {source.duration_ms} ms; "
            f"the minimum is {clip.min_duration_ms} ms"
        )
    if clip.max_duration_ms is not None and source.duration_ms > clip.max_duration_ms:
        raise SourceMediaValidationError(
            f"source_media position {position}: source video is "
            f"{math.ceil(source.duration_ms / 1000)}s; "
            f"the maximum is {clip.max_duration_ms / 1000:g}s"
        )


def validate_video_sources(
    sources: Sequence[ResolvedSourceMedia],
    request: UnifiedGenerationRequest,
    video: VideoMeta,
) -> None:
    """Enforce container, size, per-clip and combined-duration limits on video inputs."""
    clips = [source for source in sources if source.media_kind is MediaKind.VIDEO]
    if not clips:
        return
    if video.input_video is None:
        # Source cardinality/kind validation rejects this earlier; fail closed on a
        # registry that lets a video through to a model that takes none.
        raise SourceMediaValidationError(
            f"Model '{request.model.value}' does not accept a video source"
        )
    for source in clips:
        _validate_clip(source, video.input_video)
    input_seconds = _video_source_seconds(sources, video)
    if (
        video.max_combined_duration is not None
        and input_seconds
        and sum(input_seconds) + request.duration > video.max_combined_duration
    ):
        raise SourceMediaValidationError(
            f"source video ({sum(input_seconds)}s) plus requested duration "
            f"({request.duration}s) exceeds the {video.max_combined_duration}s limit"
        )


def input_video_seconds(sources: Sequence[ResolvedSourceMedia], video: VideoMeta) -> int:
    """Input-video seconds counted toward ``billed_seconds`` (0 unless the model bills them)."""
    if not video.bills_input_video_seconds:
        return 0
    return sum(_video_source_seconds(sources, video))
