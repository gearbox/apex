"""Prepare newly stored originals without taking ownership of storage or DB work."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from PIL import Image, ImageOps, JpegImagePlugin

from src.api.services.image_normalization import (
    ImageNormalizationError,
    ImageTooLargeError,
    NormalizedImage,
    check_image_pixel_limit,
    normalize_image,
    sniff_format,
)
from src.api.services.media_ingest.audio import (
    AUDIO_MAX_SECONDS,
    MAX_PEAKS_SAMPLES,
    PEAKS_SAMPLE_RATE_HZ,
    PEAKS_VERSION,
    PROBE_SHOW_ENTRIES,
    AudioProbe,
    compute_peaks,
    parse_audio_probe,
    remux_arguments,
)
from src.api.services.media_ingest.errors import (
    AudioCode,
    InvalidAudioError,
    InvalidMediaError,
    MediaProcessingError,
    UnsupportedMediaError,
)
from src.api.services.media_ingest.image_strip import strip_image_metadata
from src.api.services.media_ingest.mp3_frames import FrameScan, audible_frames, frames_only
from src.api.services.media_ingest.pdq import pdq_from_image_bytes
from src.api.services.media_ingest.types import (
    ImageIngestPolicy,
    PreparedAudio,
    PreparedImage,
    PreparedVideo,
    VideoStreamProfile,
)
from src.api.services.media_tools import (
    MediaToolError,
    MediaToolExitError,
    ProcessResult,
    run_media_command,
)
from src.core.enums import MediaFormat
from src.core.media_hash import HashSample, HashSet

# Docker installs these at /usr/bin.  The explicit lookup also supports local
# development installations without ever accepting an executable from input.
_FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"
_ORIENTATION_TAG = 0x0112
_PTS_RE = re.compile(r"pts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_DURATION_TAG_RE = re.compile(r"^(\d{2,}):(\d{2}):(\d{2})\.(\d{9})$")
_MP4_BRANDS = {b"isom", b"mp41", b"mp42", b"avc1", b"dash", b"M4V ", b"MSNV"} | {
    f"iso{n}".encode() for n in range(2, 10)
}
logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Coroutine


class DurationSource(StrEnum):
    """Where a probed visual duration came from, in resolution order."""

    STREAM = "stream"
    TIMELINE = "timeline"
    TAG = "tag"
    CONTAINER = "container"
    UNKNOWN = "unknown"


# Only these describe the selected visual stream itself. A container duration
# can span a longer audio track, so it is not comparable across a remux.
_STREAM_LEVEL_SOURCES = frozenset(
    {DurationSource.STREAM, DurationSource.TIMELINE, DurationSource.TAG}
)
_REMUX_DURATION_TOLERANCE_MS = 250


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    format: MediaFormat
    width: int
    height: int
    duration_ms: int | None
    duration_source: DurationSource
    video_stream_index: int
    audio_stream_index: int | None
    codec: str
    codec_profile: str | None
    pix_fmt: str | None
    color_transfer: str | None
    color_primaries: str | None
    rotation_degrees: int
    sample_aspect_ratio: str | None

    def stream_profile(self) -> VideoStreamProfile:
        return VideoStreamProfile(
            container=self.format,
            codec=self.codec,
            codec_profile=self.codec_profile,
            pix_fmt=self.pix_fmt,
            color_transfer=self.color_transfer,
            color_primaries=self.color_primaries,
            rotation_degrees=self.rotation_degrees,
            sample_aspect_ratio=self.sample_aspect_ratio,
            has_audio=self.audio_stream_index is not None,
        )


class MediaIngestService:
    """One process-local image/video preparation service.

    Worker replicas each own their own limits, so deployment capacity is the
    configured concurrency multiplied by the number of API/worker processes.
    """

    def __init__(
        self,
        *,
        max_image_megapixels: float,
        max_input_bytes: int,
        video_max_frames: int = 60,
        video_max_edge: int = 512,
        video_concurrency: int = 2,
        image_concurrency: int = 4,
        admission_wait_seconds: float = 10.0,
        video_deadline_seconds: float = 90.0,
        stage_timeout_seconds: float = 30.0,
        max_animation_frames: int = 100,
        video_slots: asyncio.Semaphore | None = None,
        image_slots: asyncio.Semaphore | None = None,
        max_audio_bytes: int | None = None,
    ) -> None:
        if max_image_megapixels <= 0 or max_input_bytes <= 0:
            raise ValueError("media byte and pixel limits must be positive")
        if max_audio_bytes is not None and max_audio_bytes <= 0:
            raise ValueError("media byte and pixel limits must be positive")
        if not 1 <= video_max_frames <= 60 or not 1 <= video_max_edge <= 512:
            raise ValueError("video sample settings are outside their supported bounds")
        if video_concurrency < 1 or image_concurrency < 1 or max_animation_frames < 1:
            raise ValueError("media concurrency and animation limits must be positive")
        if min(admission_wait_seconds, video_deadline_seconds, stage_timeout_seconds) <= 0:
            raise ValueError("media timeouts must be positive")
        self._max_image_megapixels = max_image_megapixels
        self._max_input_bytes = max_input_bytes
        # Audio has its own byte budget (``AUDIO_UPLOAD_MAX_MB``); unset, it follows the general one.
        self._max_audio_bytes = max_audio_bytes if max_audio_bytes is not None else max_input_bytes
        self._video_max_frames = video_max_frames
        self._video_max_edge = video_max_edge
        self._admission_wait_seconds = admission_wait_seconds
        self._video_deadline_seconds = video_deadline_seconds
        self._stage_timeout_seconds = stage_timeout_seconds
        self._max_animation_frames = max_animation_frames
        # Supplying a semaphore lets several instances (different byte budgets)
        # share one process-wide admission budget; ``None`` keeps the own-slot default.
        self._image_slots = (
            image_slots if image_slots is not None else asyncio.Semaphore(image_concurrency)
        )
        self._video_slots = (
            video_slots if video_slots is not None else asyncio.Semaphore(video_concurrency)
        )

    async def prepare_image(self, data: bytes, *, policy: ImageIngestPolicy) -> PreparedImage:
        """Normalize/strip an image, then compute PDQ from exactly stored bytes."""
        if len(data) > self._max_input_bytes:
            raise InvalidMediaError("image input exceeds the configured byte limit")
        try:
            await asyncio.wait_for(self._image_slots.acquire(), self._admission_wait_seconds)
        except TimeoutError as exc:
            raise MediaProcessingError("image preparation capacity is exhausted") from exc

        task = asyncio.create_task(self._prepare_image(data, policy))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            # ``to_thread`` work cannot be cancelled safely. Keep its slot
            # occupied until it has finished so cancelled callers cannot
            # over-admit expensive image decode/native hashing work.
            task.add_done_callback(self._cancelled_image_done)
            raise
        except Exception:
            self._image_slots.release()
            raise
        else:
            self._image_slots.release()
            return result

    def _cancelled_image_done(self, task: asyncio.Task[PreparedImage]) -> None:
        self._image_slots.release()
        self._log_cancelled_task_failure(task)

    @staticmethod
    def _log_cancelled_task_failure(task: asyncio.Task[object]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("media_ingest.cancelled_task_failed", error=str(exc))

    async def _prepare_image(self, data: bytes, policy: ImageIngestPolicy) -> PreparedImage:
        normalized = await self._normalize_for_policy(data, policy)
        return await asyncio.to_thread(self._prepare_image_sync, normalized)

    async def prepare_video(
        self, data: bytes, *, max_duration_seconds: float | None = None
    ) -> PreparedVideo:
        """Probe, remux, and sample a video under one bounded end-to-end budget."""
        if len(data) > self._max_input_bytes:
            raise InvalidMediaError("video input exceeds the configured byte limit")
        if max_duration_seconds is not None and (
            not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
        ):
            raise ValueError("max_duration_seconds must be finite and positive")
        return await self._in_video_slot(
            self._prepare_video(data, max_duration_seconds),
            busy_message="video preparation capacity is exhausted",
        )

    async def prepare_audio(self, data: bytes) -> PreparedAudio:
        """Probe, clean the container using the format policy, then compute waveform peaks.

        Shares the **video** admission slots and deadline: it is ffmpeg work of the same weight.
        The container is sniffed by ffprobe, never taken from the client's MIME; the byte limit
        is re-checked here against the audio budget, because the route only knows the declared
        type.

        Raises:
            InvalidMediaError: Over the audio byte budget (``UnsupportedAudioError`` /
                ``InvalidAudioError`` carry the typed ``422`` code for everything else).
            MediaProcessingError: Out of capacity, or a local failure (never the client's fault).
        """
        if len(data) > self._max_audio_bytes:
            raise InvalidMediaError("audio input exceeds the configured byte limit")
        return await self._in_video_slot(
            self._prepare_audio(data), busy_message="audio preparation capacity is exhausted"
        )

    async def _in_video_slot[T](self, work: Coroutine[Any, Any, T], *, busy_message: str) -> T:
        """Run ``work`` holding one video slot; a cancelled caller keeps it until ffmpeg is reaped."""
        try:
            await asyncio.wait_for(self._video_slots.acquire(), self._admission_wait_seconds)
        except TimeoutError as exc:
            work.close()
            raise MediaProcessingError(busy_message) from exc
        task = asyncio.create_task(work)
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(self._cancelled_slot_done)
            raise
        except Exception:
            self._video_slots.release()
            raise
        else:
            self._video_slots.release()
            return result

    def _cancelled_slot_done(self, task: asyncio.Task[object]) -> None:
        self._video_slots.release()
        self._log_cancelled_task_failure(task)

    async def _normalize_for_policy(
        self, data: bytes, policy: ImageIngestPolicy
    ) -> NormalizedImage:
        if policy is ImageIngestPolicy.UPLOAD:
            try:
                return await normalize_image(data, max_megapixels=self._max_image_megapixels)
            except ImageTooLargeError:
                raise
            except ImageNormalizationError as exc:
                raise InvalidMediaError("file is not a decodable image") from exc
        sniffed = sniff_format(data)
        mapping = {
            "png": MediaFormat.PNG,
            "jpeg": MediaFormat.JPEG,
            "webp": MediaFormat.WEBP,
            "webp_animated": MediaFormat.WEBP,
        }
        image_format = mapping.get(sniffed.value)
        if image_format is None:
            raise UnsupportedMediaError("provider image must be PNG, JPEG, or WebP")
        try:
            with Image.open(io.BytesIO(data)) as image:
                check_image_pixel_limit(image, max_megapixels=self._max_image_megapixels)
        except ImageTooLargeError as exc:
            raise InvalidMediaError(str(exc)) from exc
        except Exception as exc:
            raise InvalidMediaError("provider image is not decodable") from exc
        return NormalizedImage(
            data=data,
            format=image_format,
            content_type=image_format.content_type,
            converted=False,
            sniffed=sniffed,
        )

    def _prepare_image_sync(self, normalized: NormalizedImage) -> PreparedImage:
        orientation = self._orientation(normalized.data)
        animated = self._animation_frame_count(normalized.data)
        if animated > self._max_animation_frames:
            raise InvalidMediaError("image animation exceeds the configured frame limit")
        if animated > 1 and orientation != 1:
            raise UnsupportedMediaError("orientation-bearing animations are not supported")
        orientation_baked = orientation != 1
        data = normalized.data
        if orientation_baked:
            data = self._bake_orientation(data, normalized.format)
        try:
            data = strip_image_metadata(data, normalized.format)
            with Image.open(io.BytesIO(data)) as image:
                check_image_pixel_limit(image, max_megapixels=self._max_image_megapixels)
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.seek(0)
                image.load()
                width, height = image.size
        except (InvalidMediaError, ImageTooLargeError):
            raise
        except Exception as exc:
            raise InvalidMediaError("sanitized image is not decodable") from exc
        profile = (
            "pdq-animation-first-frame-rgb-white-v1" if animated > 1 else "pdq-image-rgb-white-v1"
        )
        sampling = "first-animation-frame-v1" if profile.startswith("pdq-animation") else "still-v1"
        return PreparedImage(
            data=data,
            format=normalized.format,
            width=width,
            height=height,
            hash_set=HashSet(
                profile_id=profile,
                sampling_profile=sampling,
                samples=(HashSample(pdq=pdq_from_image_bytes(data), sample_index=0),),
            ),
            orientation_baked=orientation_baked,
            converted=normalized.converted or orientation_baked,
        )

    @staticmethod
    def _orientation(data: bytes) -> int:
        try:
            with Image.open(io.BytesIO(data)) as image:
                orientation = image.getexif().get(_ORIENTATION_TAG, 1)
        except Exception as exc:
            logger.warning("media_ingest.orientation_ignored", raw_type=type(exc).__name__)
            return 1
        if type(orientation) is not int or orientation not in range(2, 9):
            if type(orientation) is not int or orientation != 1:
                logger.warning(
                    "media_ingest.orientation_ignored",
                    raw_type=type(orientation).__name__,
                    raw_value=orientation if type(orientation) is int else None,
                )
            return 1
        return orientation

    @staticmethod
    def _animation_frame_count(data: bytes) -> int:
        try:
            with Image.open(io.BytesIO(data)) as image:
                return int(getattr(image, "n_frames", 1))
        except Exception as exc:
            raise InvalidMediaError("image animation metadata is malformed") from exc

    def _bake_orientation(self, data: bytes, image_format: MediaFormat) -> bytes:
        try:
            with Image.open(io.BytesIO(data)) as source:
                check_image_pixel_limit(source, max_megapixels=self._max_image_megapixels)
                image = ImageOps.exif_transpose(source)
                image.load()
                output = io.BytesIO()
                icc_profile = source.info.get("icc_profile")
                save_options: dict[str, object] = {"icc_profile": icc_profile}
                if dpi := self._rotated_dpi(source, orientation=self._orientation(data)):
                    save_options["dpi"] = dpi
                if image_format is MediaFormat.PNG:
                    image.save(output, format="PNG", **save_options)
                elif image_format is MediaFormat.JPEG:
                    if image.mode not in {"RGB", "L"}:
                        image = image.convert("RGB")
                    sampling = JpegImagePlugin.get_sampling(source)
                    if sampling >= 0:
                        save_options["subsampling"] = sampling
                    image.save(output, format="JPEG", quality=95, **save_options)
                elif image_format is MediaFormat.WEBP:
                    image.save(
                        output,
                        format="WEBP",
                        quality=95,
                        lossless=bool(source.info.get("lossless", False)),
                        **save_options,
                    )
                else:  # pragma: no cover - guarded by caller
                    self._raise_unsupported_orientation_format()
                return output.getvalue()
        except (ImageTooLargeError, UnsupportedMediaError):
            raise
        except Exception as exc:
            raise InvalidMediaError("image orientation could not be baked") from exc

    @staticmethod
    def _raise_unsupported_orientation_format() -> None:
        raise UnsupportedMediaError("unsupported orientation format")

    @staticmethod
    def _rotated_dpi(source: Image.Image, *, orientation: int) -> tuple[float, float] | None:
        dpi = source.info.get("dpi")
        if (
            not isinstance(dpi, tuple)
            or len(dpi) != 2
            or not all(isinstance(value, int | float) and value > 0 for value in dpi)
        ):
            return None
        horizontal, vertical = float(dpi[0]), float(dpi[1])
        return (vertical, horizontal) if orientation in {5, 6, 7, 8} else (horizontal, vertical)

    async def _prepare_video(
        self, data: bytes, max_duration_seconds: float | None
    ) -> PreparedVideo:
        deadline = asyncio.get_running_loop().time() + self._video_deadline_seconds
        temp_dir: Path | None = None
        try:
            # Inside the try so ENOSPC/EACCES/missing TMPDIR classify as an
            # operational failure; off the loop because it is a blocking syscall.
            temp_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="apex-media-ingest-"))
            input_path = temp_dir / "input.bin"
            await asyncio.to_thread(input_path.write_bytes, data)
            source_probe = await self._probe_video(input_path, deadline)
            # Browser MediaRecorder WebM carries no duration at all; the remux
            # writes one, so the source cap is only an early exit when known.
            if source_probe.duration_ms is not None:
                self._validate_video_duration(source_probe.duration_ms, max_duration_seconds)
            output_path = temp_dir / f"prepared.{source_probe.format.value}"
            await self._remux(input_path, output_path, source_probe, deadline)
            prepared_probe = await self._probe_video(output_path, deadline)
            prepared_duration_ms = self._prepared_duration_ms(prepared_probe, max_duration_seconds)
            self._validate_remux_duration(source_probe, prepared_probe)
            samples, sampling_profile = await self._sample_video(
                output_path, prepared_duration_ms, prepared_probe, temp_dir, deadline
            )
            prepared_bytes = await asyncio.to_thread(output_path.read_bytes)
            self._validate_prepared_video_size(prepared_bytes)
            return PreparedVideo(
                data=prepared_bytes,
                format=prepared_probe.format,
                width=prepared_probe.width,
                height=prepared_probe.height,
                duration_ms=prepared_duration_ms,
                hash_set=HashSet(
                    profile_id=f"pdq-video-rgb-white-v2-edge-{self._video_max_edge}",
                    sampling_profile=sampling_profile,
                    samples=tuple(samples),
                ),
                stream_profile=prepared_probe.stream_profile(),
            )
        except (InvalidMediaError, UnsupportedMediaError, MediaProcessingError):
            raise
        except (MediaToolError, TimeoutError, OSError) as exc:
            raise MediaProcessingError("video preparation failed operationally") from exc
        finally:
            if temp_dir is not None:
                await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    async def _prepare_audio(self, data: bytes) -> PreparedAudio:
        deadline = asyncio.get_running_loop().time() + self._video_deadline_seconds
        temp_dir: Path | None = None
        try:
            temp_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="apex-media-ingest-"))
            input_path = temp_dir / "input.bin"
            await asyncio.to_thread(input_path.write_bytes, data)
            source = await self._probe_audio(input_path, deadline)
            output_path = temp_dir / f"prepared.{source.format.value}"
            if source.format is MediaFormat.MP3:
                first_remux_path = temp_dir / "first-remux.mp3"
                await self._remux_audio(input_path, first_remux_path, source, deadline)
                first_remux = await asyncio.to_thread(first_remux_path.read_bytes)
                first_scan = await asyncio.to_thread(frames_only, first_remux)
                if first_scan.dropped_bytes:
                    logger.info(
                        "media_ingest.audio_junk_dropped", dropped_bytes=first_scan.dropped_bytes
                    )
                    frames_path = temp_dir / "frames.mp3"
                    await asyncio.to_thread(frames_path.write_bytes, first_scan.frames)
                    await self._remux_audio(frames_path, output_path, source, deadline)
                else:
                    output_path = first_remux_path

                final_mp3 = await asyncio.to_thread(output_path.read_bytes)
                final_scan = await asyncio.to_thread(frames_only, final_mp3)
                self._require_clean_mp3(final_scan)
                source_scan = await asyncio.to_thread(frames_only, data)
                input_frames = source_scan.frames
                output_frames = final_scan.frames
                # The remux writes a new Xing/Info frame; each side must omit its own
                # metadata frame because its gapless trim values can differ.
                input_frames = audible_frames(input_frames)
                output_frames = audible_frames(output_frames)
                input_frames_path = temp_dir / "input-frames.mp3"
                output_frames_path = temp_dir / "output-frames.mp3"
                await asyncio.to_thread(input_frames_path.write_bytes, input_frames)
                await asyncio.to_thread(output_frames_path.write_bytes, output_frames)
                await self._assert_decoded_audio_equal(
                    input_frames_path, output_frames_path, deadline
                )
            else:
                await self._remux_audio(input_path, output_path, source, deadline)
            # The stored bytes are what gets described and gated: probe the remuxed file again.
            prepared = await self._probe_audio(output_path, deadline)
            self._validate_audio_remux(source, prepared)
            if source.format is MediaFormat.FLAC:
                self._require_same_flac_sample_format(source, prepared)
                await self._assert_decoded_audio_equal(input_path, output_path, deadline)
            peaks = await self._audio_peaks(output_path, deadline)
            prepared_bytes = await asyncio.to_thread(output_path.read_bytes)
            self._validate_prepared_audio_size(prepared_bytes)
            return PreparedAudio(
                data=prepared_bytes,
                format=prepared.format,
                codec=prepared.codec,
                sample_rate_hz=prepared.sample_rate_hz,
                channels=prepared.channels,
                bit_rate=prepared.bit_rate,
                duration_ms=prepared.duration_ms,
                peaks=peaks,
                peaks_version=PEAKS_VERSION,
            )
        except (InvalidMediaError, UnsupportedMediaError, MediaProcessingError):
            raise
        except (MediaToolError, TimeoutError, OSError) as exc:
            raise MediaProcessingError("audio preparation failed operationally") from exc
        finally:
            if temp_dir is not None:
                await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    @staticmethod
    def _validate_audio_remux(source: AudioProbe, prepared: AudioProbe) -> None:
        """A stream copy keeps the container: anything else is our fault, not the client's."""
        if prepared.format is not source.format:
            raise MediaProcessingError("audio remux changed the container")

    @staticmethod
    def _require_clean_mp3(scan: FrameScan) -> None:
        if scan.dropped_bytes:
            raise MediaProcessingError("MP3 remux left bytes outside audio frames")

    @staticmethod
    def _require_same_flac_sample_format(source: AudioProbe, prepared: AudioProbe) -> None:
        if prepared.sample_fmt != source.sample_fmt:
            raise MediaProcessingError("FLAC remux changed the decoded sample format")

    def _validate_prepared_audio_size(self, data: bytes) -> None:
        if len(data) > self._max_audio_bytes:
            raise InvalidMediaError("prepared audio exceeds the configured byte limit")

    async def _run_audio_stage(self, stage: str, args: list[str], deadline: float) -> ProcessResult:
        """Run an audio ffmpeg/ffprobe stage; a non-zero exit is the file's fault, not ours.

        Unlike the video stages this logs **no stderr**: decoder warnings can quote tag values.
        """
        try:
            return await run_media_command(args, timeout_seconds=self._remaining(deadline))
        except MediaToolExitError as exc:
            if exc.returncode > 0:
                logger.warning("media_ingest.audio_not_decodable", stage=stage)
                raise InvalidAudioError(AudioCode.INVALID, "audio is not decodable") from exc
            raise

    async def _probe_audio(self, path: Path, deadline: float) -> AudioProbe:
        result = await self._run_audio_stage(
            "probe",
            [
                _FFPROBE,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_entries",
                PROBE_SHOW_ENTRIES,
                "-of",
                "json",
                str(path),
            ],
            deadline,
        )
        try:
            decoded = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise InvalidAudioError(AudioCode.INVALID, "audio is not decodable") from exc
        return parse_audio_probe(decoded)

    async def _remux_audio(
        self, source: Path, destination: Path, probe: AudioProbe, deadline: float
    ) -> None:
        await self._run_audio_stage(
            "remux",
            [
                _FFMPEG,
                "-y",
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-protocol_whitelist",
                "file",
                "-i",
                str(source),
                *remux_arguments(probe.format, sample_fmt=probe.sample_fmt),
                str(destination),
            ],
            deadline,
        )

    async def _assert_decoded_audio_equal(
        self, source: Path, prepared: Path, deadline: float
    ) -> None:
        """Require the same decoded PCM samples; an encoder mismatch is a service failure."""

        async def decoded_md5(path: Path) -> str:
            result = await self._run_audio_stage(
                "sample_verification",
                [
                    _FFMPEG,
                    "-hide_banner",
                    "-nostats",
                    "-loglevel",
                    "error",
                    "-protocol_whitelist",
                    "file",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "pcm_s32le",
                    "-f",
                    "md5",
                    "-",
                ],
                deadline,
            )
            return result.stdout.decode("ascii", errors="strict").strip()

        source_md5, prepared_md5 = await asyncio.gather(decoded_md5(source), decoded_md5(prepared))
        if not source_md5 or source_md5 != prepared_md5:
            raise MediaProcessingError("audio cleanup changed decoded samples")

    async def _audio_peaks(self, path: Path, deadline: float) -> bytes:
        """Decode the stored file to mono 11,025 Hz PCM and reduce it to the peaks."""
        result = await self._run_audio_stage(
            "peaks",
            [
                _FFMPEG,
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-protocol_whitelist",
                "file",
                "-i",
                str(path),
                "-t",
                str(AUDIO_MAX_SECONDS + 1),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                str(PEAKS_SAMPLE_RATE_HZ),
                "-f",
                "s16le",
                "-",
            ],
            deadline,
        )
        if len(result.stdout) // 2 > MAX_PEAKS_SAMPLES:
            raise InvalidAudioError(AudioCode.INVALID, "the audio decodes longer than allowed")
        return await asyncio.to_thread(compute_peaks, result.stdout)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise MediaProcessingError("video preparation exceeded its total deadline")
        return min(remaining, self._stage_timeout_seconds)

    @staticmethod
    def _validate_video_duration(duration_ms: int, maximum_seconds: float | None) -> None:
        if maximum_seconds is not None and duration_ms > round(maximum_seconds * 1000):
            raise InvalidMediaError("video duration exceeds the configured limit")

    @classmethod
    def _prepared_duration_ms(cls, probe: _VideoProbe, maximum_seconds: float | None) -> int:
        """Require a duration on the prepared output and apply the authoritative cap.

        The remux writes a duration even when the source had none, and the
        prepared file is what is stored and sampled.
        """
        if probe.duration_ms is None:
            raise InvalidMediaError("video duration is unavailable")
        cls._validate_video_duration(probe.duration_ms, maximum_seconds)
        return probe.duration_ms

    @staticmethod
    def _validate_remux_duration(source: _VideoProbe, prepared: _VideoProbe) -> None:
        """Reject a remux that silently lost part of the visual timeline.

        Some demuxers only warn for a file cut mid-cluster and ffmpeg still
        exits zero; the prepared timeline exposes that loss. The comparison is
        only meaningful when both sides describe the visual stream itself, so
        container fallbacks (which may span a longer audio track) and unknown
        durations skip it.
        """
        if (
            source.duration_ms is None
            or prepared.duration_ms is None
            or source.duration_source not in _STREAM_LEVEL_SOURCES
            or prepared.duration_source not in _STREAM_LEVEL_SOURCES
        ):
            return
        if source.duration_ms - prepared.duration_ms > _REMUX_DURATION_TOLERANCE_MS:
            logger.warning(
                "media_ingest.video_duration_lost",
                stage="prepared_probe",
                source_duration_ms=source.duration_ms,
                source_duration_source=source.duration_source.value,
                prepared_duration_ms=prepared.duration_ms,
                prepared_duration_source=prepared.duration_source.value,
            )
            raise InvalidMediaError("video is not decodable")

    def _validate_prepared_video_size(self, data: bytes) -> None:
        if len(data) > self._max_input_bytes:
            raise InvalidMediaError("prepared video exceeds the configured byte limit")

    async def _run_stage(self, stage: str, args: list[str], deadline: float) -> ProcessResult:
        try:
            return await run_media_command(args, timeout_seconds=self._remaining(deadline))
        except MediaToolExitError as exc:
            if exc.returncode > 0:
                # A malformed upload is the user's input, not a service fault:
                # warn without a traceback. Provider callers log their own
                # error-level events for pipeline anomalies.
                logger.warning(
                    "media_ingest.video_not_decodable",
                    stage=stage,
                    stderr_excerpt=exc.stderr_excerpt.strip(),
                )
                raise InvalidMediaError("video is not decodable") from exc
            raise

    async def _probe_video(self, path: Path, deadline: float) -> _VideoProbe:
        result = await self._run_stage(
            "probe",
            [
                _FFPROBE,
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,width,height,duration,duration_ts,time_base,disposition,pix_fmt,profile,color_transfer,color_primaries,sample_aspect_ratio:stream_tags=DURATION:stream_side_data=rotation:format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            deadline,
        )
        # Outside the parse ``try``: an ``OSError`` here is operational and must
        # reach ``_prepare_video``'s handler, not become ``InvalidMediaError``.
        header = await asyncio.to_thread(_read_header, path)
        try:
            decoded = json.loads(result.stdout)
            container = self._detect_container(header, decoded["format"]["format_name"])
            streams = decoded["streams"]
            visual = next(
                stream
                for stream in streams
                if stream.get("codec_type") == "video"
                and not bool((stream.get("disposition") or {}).get("attached_pic"))
            )
            audio = next(
                (stream for stream in streams if stream.get("codec_type") == "audio"), None
            )
            duration_seconds, duration_source = self._stream_duration_seconds(
                visual, decoded["format"]
            )
            width, height = int(visual["width"]), int(visual["height"])
            self._validate_probe_values(width, height, visual.get("codec_name"))
            self._validate_video_pixel_limit(width, height)
            return _VideoProbe(
                format=container,
                width=width,
                height=height,
                duration_ms=(
                    round(duration_seconds * 1000) if duration_seconds is not None else None
                ),
                duration_source=duration_source,
                video_stream_index=int(visual["index"]),
                audio_stream_index=int(audio["index"]) if audio is not None else None,
                codec=str(visual["codec_name"]),
                codec_profile=self._optional_text(visual.get("profile")),
                pix_fmt=self._optional_text(visual.get("pix_fmt")),
                color_transfer=self._optional_text(visual.get("color_transfer")),
                color_primaries=self._optional_text(visual.get("color_primaries")),
                rotation_degrees=self._rotation_degrees(visual.get("side_data_list")),
                sample_aspect_ratio=self._optional_text(visual.get("sample_aspect_ratio")),
            )
        except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
            raise InvalidMediaError("video has no usable visual stream") from exc

    @staticmethod
    def _optional_text(value: object) -> str | None:
        """Informational probe fact: anything but a non-empty string becomes ``None``."""
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _rotation_degrees(side_data: object) -> int:
        """Display-matrix rotation in degrees normalized into ``[0, 360)`` (ffprobe reports e.g. ``-90``).

        Informational: a missing or odd value is ``0``, never a rejection.
        """
        if not isinstance(side_data, list):
            return 0
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                try:
                    return int(float(entry["rotation"])) % 360
                except (TypeError, ValueError, OverflowError):
                    return 0
        return 0

    @staticmethod
    def _stream_duration_seconds(
        stream: dict[str, object], container: dict[str, object]
    ) -> tuple[float | None, DurationSource]:
        """Resolve the selected visual stream's duration and its provenance.

        Never uses an audio stream's duration. Falls back to the container
        duration, and reports ``UNKNOWN`` (``None``) when nothing is available;
        callers decide whether an unknown duration is acceptable.
        """

        def valid(value: object) -> float | None:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) and number > 0 else None

        format_duration = valid(container.get("duration"))
        candidates: list[tuple[float | None, DurationSource]] = [
            (valid(stream.get("duration")), DurationSource.STREAM)
        ]
        duration_ts = stream.get("duration_ts")
        time_base = stream.get("time_base")
        timeline: float | None = None
        if isinstance(time_base, str) and isinstance(duration_ts, (str, int)):
            with contextlib.suppress(ValueError, ZeroDivisionError):
                numerator, denominator = time_base.split("/", 1)
                timeline = valid(int(duration_ts) * int(numerator) / int(denominator))
        candidates.append((timeline, DurationSource.TIMELINE))
        tags = stream.get("tags")
        tag = tags.get("DURATION") if isinstance(tags, dict) else None
        tagged: float | None = None
        if isinstance(tag, str) and (match := _DURATION_TAG_RE.fullmatch(tag)):
            hours, minutes, seconds, fraction = match.groups()
            if int(minutes) < 60 and int(seconds) < 60:
                tagged = valid(
                    int(hours) * 3600
                    + int(minutes) * 60
                    + int(seconds)
                    + int(fraction) / 1_000_000_000
                )
        candidates.append((tagged, DurationSource.TAG))
        for candidate, source in candidates:
            if candidate is not None:
                if format_duration is not None and candidate > format_duration + 1:
                    raise InvalidMediaError("visual stream duration exceeds container duration")
                return candidate, source
        if format_duration is not None:
            return format_duration, DurationSource.CONTAINER
        return None, DurationSource.UNKNOWN

    @staticmethod
    def _validate_probe_values(width: int, height: int, codec: object) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("non-positive video metadata")
        if not codec:
            raise ValueError("missing video codec")

    def _validate_video_pixel_limit(self, width: int, height: int) -> None:
        if width * height > self._max_image_megapixels * 1_000_000:
            raise ValueError("video frame exceeds configured pixel limit")

    @staticmethod
    def _detect_container(data: bytes, format_name: str) -> MediaFormat:
        if "mov,mp4" in format_name:
            if len(data) < 12 or data[4:8] != b"ftyp":
                return MediaFormat.MOV
            box_size = int.from_bytes(data[:4], "big")
            if box_size < 16 or box_size > len(data):
                raise UnsupportedMediaError("invalid ISO-BMFF brand box")
            major = data[8:12]
            if major == b"qt  ":
                return MediaFormat.MOV
            brands = {major} | {data[i : i + 4] for i in range(16, box_size, 4)}
            if brands & _MP4_BRANDS:
                return MediaFormat.MP4
            raise UnsupportedMediaError("unsupported ISO-BMFF video brand")
        if "matroska,webm" in format_name and data.startswith(b"\x1aE\xdf\xa3"):
            if MediaIngestService._ebml_doctype(data[:4096]) == b"webm":
                return MediaFormat.WEBM
            raise UnsupportedMediaError("Matroska is not accepted as WebM")
        raise UnsupportedMediaError("unsupported video container")

    @staticmethod
    def _ebml_doctype(data: bytes) -> bytes | None:
        """Read the actual EBML header DocType within a bounded sniff."""

        def vint(offset: int, *, identifier: bool = False) -> tuple[int, int]:
            if offset >= len(data) or data[offset] == 0:
                raise ValueError("invalid EBML variable integer")
            first = data[offset]
            width = 9 - first.bit_length()
            if width > 8 or offset + width > len(data):
                raise ValueError("truncated EBML variable integer")
            value = first if identifier else first & ((1 << (8 - width)) - 1)
            for octet in data[offset + 1 : offset + width]:
                value = (value << 8) | octet
            return value, offset + width

        with contextlib.suppress(ValueError):
            size, position = vint(4)
            end = min(position + size, len(data))
            while position < end:
                element_id, position = vint(position, identifier=True)
                element_size, position = vint(position)
                if position + element_size > end:
                    return None
                if element_id == 0x4282:
                    return data[position : position + element_size]
                position += element_size
        return None

    async def _remux(
        self, source: Path, destination: Path, probe: _VideoProbe, deadline: float
    ) -> None:
        args = [
            _FFMPEG,
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            f"0:{probe.video_stream_index}",
        ]
        if probe.audio_stream_index is not None:
            args.extend(["-map", f"0:{probe.audio_stream_index}"])
        args.extend(
            [
                "-map_metadata",
                "-1",
                "-map_metadata:s",
                "-1",
                "-map_chapters",
                "-1",
                "-c",
                "copy",
                "-fflags",
                "+bitexact",
            ]
        )
        if probe.format in {MediaFormat.MP4, MediaFormat.MOV}:
            args.extend(["-movflags", "+faststart"])
        args.extend(["-f", probe.format.value, str(destination)])
        await self._run_stage("remux", args, deadline)

    async def _sample_video(
        self, path: Path, duration_ms: int, probe: _VideoProbe, temp_dir: Path, deadline: float
    ) -> tuple[list[HashSample], str]:
        interval = max(1.0, duration_ms / 1000 / self._video_max_frames)
        keyframes_only = interval > 1.0
        frame_pattern = temp_dir / "sample-%03d.png"
        # ``prev_selected_t`` gives a sequential, actual-PTS cadence.  ``showinfo``
        # follows selection, so its PTS records pair one-for-one with frame files.
        select = f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.6f})"
        scale = (
            "scale=iw*sar:ih,setsar=1,"
            f"scale='min(iw,{self._video_max_edge})':'min(ih,{self._video_max_edge})'"
            ":force_original_aspect_ratio=decrease"
        )
        args = [
            _FFMPEG,
            "-y",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
        ]
        if keyframes_only:
            args.extend(["-skip_frame", "nokey"])
        args.extend(
            [
                "-i",
                str(path),
                "-map",
                f"0:{probe.video_stream_index}",
                "-vf",
                f"{select},{scale},showinfo",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(self._video_max_frames),
                str(frame_pattern),
            ]
        )
        result = await self._run_stage("sampling", args, deadline)
        files = await asyncio.to_thread(lambda: sorted(temp_dir.glob("sample-*.png")))
        timestamps = [
            float(match.group(1))
            for line in result.stderr.decode("utf-8", "replace").splitlines()
            if "Parsed_showinfo" in line
            if (match := _PTS_RE.search(line)) is not None
        ]
        if not files:
            raise InvalidMediaError("video decoded no frames")
        if len(files) != len(timestamps):
            raise MediaProcessingError("video frame/PTS records did not match")
        # One worker-thread call for every frame: native decode + hash stays off
        # the loop without a thread hop per sample. The video slot bounds it.
        samples = await asyncio.to_thread(_hash_frames_sync, files, timestamps)
        profile = "uniform-pts-keyframes-v1" if keyframes_only else "uniform-pts-v1"
        return samples, profile


def _read_header(path: Path, limit: int = 65_536) -> bytes:
    """Read the container sniff window; called via ``asyncio.to_thread``."""
    with path.open("rb") as source:
        return source.read(limit)


def _hash_frames_sync(files: list[Path], timestamps: list[float]) -> list[HashSample]:
    """Hash sampled frames with PTS offsets from the first frame; runs in a worker thread."""
    first = timestamps[0]
    return [
        HashSample(
            pdq=pdq_from_image_bytes(path.read_bytes()),
            sample_index=index,
            frame_timestamp_ms=max(0, round((timestamp - first) * 1000)),
        )
        for index, (path, timestamp) in enumerate(zip(files, timestamps, strict=True))
    ]
