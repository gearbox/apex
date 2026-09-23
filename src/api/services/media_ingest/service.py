"""Prepare newly stored originals without taking ownership of storage or DB work."""

from __future__ import annotations

import asyncio
import io
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
from src.api.services.media_ingest.errors import (
    InvalidMediaError,
    MediaProcessingError,
    UnsupportedMediaError,
)
from src.api.services.media_ingest.image_strip import strip_image_metadata
from src.api.services.media_ingest.pdq import pdq_from_image_bytes
from src.api.services.media_ingest.types import (
    ImageIngestPolicy,
    PreparedImage,
    PreparedVideo,
)
from src.api.services.media_tools import (
    MediaToolError,
    MediaToolExitError,
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


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    format: MediaFormat
    width: int
    height: int
    duration_ms: int
    video_stream_index: int
    audio_stream_index: int | None


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
    ) -> None:
        if max_image_megapixels <= 0 or max_input_bytes <= 0:
            raise ValueError("media byte and pixel limits must be positive")
        if not 1 <= video_max_frames <= 60 or not 1 <= video_max_edge <= 512:
            raise ValueError("video sample settings are outside their supported bounds")
        if video_concurrency < 1 or image_concurrency < 1 or max_animation_frames < 1:
            raise ValueError("media concurrency and animation limits must be positive")
        if min(admission_wait_seconds, video_deadline_seconds, stage_timeout_seconds) <= 0:
            raise ValueError("media timeouts must be positive")
        self._max_image_megapixels = max_image_megapixels
        self._max_input_bytes = max_input_bytes
        self._video_max_frames = video_max_frames
        self._video_max_edge = video_max_edge
        self._admission_wait_seconds = admission_wait_seconds
        self._video_deadline_seconds = video_deadline_seconds
        self._stage_timeout_seconds = stage_timeout_seconds
        self._max_animation_frames = max_animation_frames
        self._image_slots = asyncio.Semaphore(image_concurrency)
        self._video_slots = asyncio.Semaphore(video_concurrency)

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

    def _cancelled_video_done(self, task: asyncio.Task[PreparedVideo]) -> None:
        self._video_slots.release()
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
        try:
            await asyncio.wait_for(self._video_slots.acquire(), self._admission_wait_seconds)
        except TimeoutError as exc:
            raise MediaProcessingError("video preparation capacity is exhausted") from exc

        task = asyncio.create_task(self._prepare_video(data, max_duration_seconds))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            # The worker keeps the temporary directory and capacity slot until
            # ffmpeg has been reaped; releasing here would over-admit work.
            task.add_done_callback(self._cancelled_video_done)
            raise
        except Exception:
            self._video_slots.release()
            raise
        else:
            self._video_slots.release()
            return result

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
        temp_dir = Path(tempfile.mkdtemp(prefix="apex-media-ingest-"))
        input_path = temp_dir / "input.bin"
        try:
            await asyncio.to_thread(input_path.write_bytes, data)
            source_probe = await self._probe_video(input_path, deadline)
            self._validate_video_duration(source_probe.duration_ms, max_duration_seconds)
            output_path = temp_dir / f"prepared.{source_probe.format.value}"
            await self._remux(input_path, output_path, source_probe, deadline)
            prepared_probe = await self._probe_video(output_path, deadline)
            self._validate_remux_duration(source_probe.duration_ms, prepared_probe.duration_ms)
            samples, sampling_profile = await self._sample_video(
                output_path, prepared_probe, temp_dir, deadline
            )
            prepared_bytes = await asyncio.to_thread(output_path.read_bytes)
            self._validate_prepared_video_size(prepared_bytes)
            return PreparedVideo(
                data=prepared_bytes,
                format=prepared_probe.format,
                width=prepared_probe.width,
                height=prepared_probe.height,
                duration_ms=prepared_probe.duration_ms,
                hash_set=HashSet(
                    profile_id=f"pdq-video-rgb-white-v2-edge-{self._video_max_edge}",
                    sampling_profile=sampling_profile,
                    samples=tuple(samples),
                ),
            )
        except (InvalidMediaError, UnsupportedMediaError, MediaProcessingError):
            raise
        except (MediaToolError, TimeoutError, OSError) as exc:
            raise MediaProcessingError("video preparation failed operationally") from exc
        finally:
            await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise MediaProcessingError("video preparation exceeded its total deadline")
        return min(remaining, self._stage_timeout_seconds)

    @staticmethod
    def _validate_video_duration(duration_ms: int, maximum_seconds: float | None) -> None:
        if maximum_seconds is not None and duration_ms > round(maximum_seconds * 1000):
            raise InvalidMediaError("video duration exceeds the configured limit")

    @staticmethod
    def _validate_remux_duration(source_ms: int, prepared_ms: int) -> None:
        if source_ms - prepared_ms > 250:
            # Some demuxers only warn for a file cut mid-cluster and ffmpeg
            # still exits zero. The prepared timeline exposes that loss.
            logger.error(
                "media_ingest.video_duration_lost",
                stage="prepared_probe",
                source_duration_ms=source_ms,
                prepared_duration_ms=prepared_ms,
            )
            raise InvalidMediaError("video is not decodable")

    def _validate_prepared_video_size(self, data: bytes) -> None:
        if len(data) > self._max_input_bytes:
            raise InvalidMediaError("prepared video exceeds the configured byte limit")

    async def _run_stage(self, stage: str, args: list[str], deadline: float) -> bytes:
        try:
            result = await run_media_command(args, timeout_seconds=self._remaining(deadline))
        except MediaToolExitError as exc:
            if exc.returncode > 0:
                logger.exception(
                    "media_ingest.video_not_decodable",
                    stage=stage,
                    stderr_excerpt=exc.stderr_excerpt.strip(),
                )
                raise InvalidMediaError("video is not decodable") from exc
            raise
        return result.stdout

    async def _probe_video(self, path: Path, deadline: float) -> _VideoProbe:
        result = await self._run_stage(
            "probe",
            [
                _FFPROBE,
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,width,height,duration,duration_ts,time_base,disposition:stream_tags=DURATION:format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            deadline,
        )
        try:
            decoded = json.loads(result)
            with path.open("rb") as source:
                header = source.read(65_536)
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
            duration_seconds = self._stream_duration_seconds(visual, decoded["format"])
            width, height = int(visual["width"]), int(visual["height"])
            self._validate_probe_values(duration_seconds, width, height, visual.get("codec_name"))
            self._validate_video_pixel_limit(width, height)
            return _VideoProbe(
                format=container,
                width=width,
                height=height,
                duration_ms=round(duration_seconds * 1000),
                video_stream_index=int(visual["index"]),
                audio_stream_index=int(audio["index"]) if audio is not None else None,
            )
        except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
            raise InvalidMediaError("video has no usable visual stream") from exc

    @staticmethod
    def _stream_duration_seconds(stream: dict[str, object], container: dict[str, object]) -> float:
        """Use only the selected visual stream timeline, never audio duration."""

        def valid(value: object) -> float | None:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) and number > 0 else None

        format_duration = valid(container.get("duration"))
        candidates: list[float | None] = [valid(stream.get("duration"))]
        duration_ts = stream.get("duration_ts")
        time_base = stream.get("time_base")
        timeline: float | None = None
        if isinstance(time_base, str) and isinstance(duration_ts, (str, int)):
            try:
                numerator, denominator = time_base.split("/", 1)
                timeline = valid(int(duration_ts) * int(numerator) / int(denominator))
            except (ValueError, ZeroDivisionError):
                pass
        candidates.append(timeline)
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
        candidates.append(tagged)
        for candidate in candidates:
            if candidate is not None:
                if format_duration is not None and candidate > format_duration + 1:
                    raise InvalidMediaError("visual stream duration exceeds container duration")
                return candidate
        if format_duration is not None:
            return format_duration
        raise ValueError("visual stream duration is unavailable")

    @staticmethod
    def _validate_probe_values(
        duration_seconds: float, width: int, height: int, codec: object
    ) -> None:
        if (
            not math.isfinite(duration_seconds)
            or duration_seconds <= 0
            or width <= 0
            or height <= 0
        ):
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

        try:
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
        except ValueError:
            pass
        return None

    async def _remux(
        self, source: Path, destination: Path, probe: _VideoProbe, deadline: float
    ) -> None:
        args = [
            _FFMPEG,
            "-y",
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
        self, path: Path, probe: _VideoProbe, temp_dir: Path, deadline: float
    ) -> tuple[list[HashSample], str]:
        interval = max(1.0, probe.duration_ms / 1000 / self._video_max_frames)
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
        try:
            result = await run_media_command(args, timeout_seconds=self._remaining(deadline))
        except MediaToolExitError as exc:
            if exc.returncode > 0:
                logger.exception(
                    "media_ingest.video_not_decodable",
                    stage="sampling",
                    stderr_excerpt=exc.stderr_excerpt.strip(),
                )
                raise InvalidMediaError("video is not decodable") from exc
            raise
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
        first = timestamps[0]
        samples: list[HashSample] = []
        for index, (frame_path, timestamp) in enumerate(zip(files, timestamps, strict=True)):
            normalized_ms = max(0, round((timestamp - first) * 1000))
            samples.append(
                HashSample(
                    pdq=pdq_from_image_bytes(await asyncio.to_thread(frame_path.read_bytes)),
                    sample_index=index,
                    frame_timestamp_ms=normalized_ms,
                )
            )
        profile = "uniform-pts-keyframes-v1" if keyframes_only else "uniform-pts-v1"
        return samples, profile
