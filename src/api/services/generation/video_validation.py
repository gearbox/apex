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
from src.core.enums import MediaKind, VideoParameter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.core.model_registry import ModelMeta, VideoMeta


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
        if request.source_media and not video.aspect_ratio_with_input_media:
            raise ValueError(
                f"Model '{request.model.value}' derives the output aspect ratio from its input "
                f"media; omit aspect_ratio when source_media is supplied"
            )
        if request.aspect_ratio not in meta.aspect_ratios:
            raise ValueError(
                f"Model '{request.model.value}' does not support aspect ratio "
                f"'{request.aspect_ratio.value}'; supported: "
                f"{[a.value for a in meta.aspect_ratios]}"
            )
    return video


def _needs_input_durations(video: VideoMeta) -> bool:
    return (
        video.max_input_video_seconds is not None
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


def validate_video_sources(
    sources: Sequence[ResolvedSourceMedia],
    request: UnifiedGenerationRequest,
    video: VideoMeta,
) -> None:
    """Enforce per-clip and combined-duration limits on resolved video inputs."""
    input_seconds = _video_source_seconds(sources, video)
    for seconds in input_seconds:
        if video.max_input_video_seconds is not None and seconds > video.max_input_video_seconds:
            raise SourceMediaValidationError(
                f"source video is {seconds}s; the maximum is {video.max_input_video_seconds}s"
            )
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
