"""``MediaIngestService.prepare_audio`` against real ffmpeg / ffprobe (S2a W3 / W11).

Every file is generated inside the test run (``tests/audio_support.py``). The invariants the
review probes: the stored audio is **bit-identical** to the input's (a stream copy, never a
re-encode), carries **no tag and no attached picture**, and each refusal is its typed code.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest
import structlog

from src.api.services.media_ingest import (
    AudioCode,
    InvalidAudioError,
    InvalidMediaError,
    MediaIngestService,
    MediaProcessingError,
    UnsupportedAudioError,
)
from src.core.enums import MediaFormat
from tests.audio_support import (
    CONTAINERS,
    SECRET_TAGS,
    AudioFixtures,
    has_attached_picture,
    probe,
    require_ffmpeg,
    stream_md5,
    user_tags,
)

pytestmark = pytest.mark.unit

_FORMATS = {
    "mp3": MediaFormat.MP3,
    "m4a": MediaFormat.M4A,
    "wav": MediaFormat.WAV,
    "flac": MediaFormat.FLAC,
}


@pytest.fixture(autouse=True)
def _ffmpeg() -> None:
    require_ffmpeg()


def _service(**overrides: object) -> MediaIngestService:
    options: dict[str, object] = {
        "max_image_megapixels": 10,
        "max_input_bytes": 2 * 1024 * 1024,
        "max_audio_bytes": 20 * 1024 * 1024,
        "video_concurrency": 2,
    }
    options.update(overrides)
    return MediaIngestService(**options)  # type: ignore[arg-type]


class TestEachAcceptedContainer:
    @pytest.mark.parametrize("container", list(CONTAINERS))
    async def test_the_stored_audio_is_the_input_s_stream_without_tags_or_picture(
        self, audio_files: AudioFixtures, container: str
    ) -> None:
        source = audio_files.tone(container)
        # The fixture really carries what ingest has to remove (WAV cannot hold a picture).
        original = probe(source)
        assert user_tags(original).get("title") == SECRET_TAGS["title"]
        assert has_attached_picture(original) is (container != "wav")

        prepared = await _service().prepare_audio(source.read_bytes())

        stored = source.parent / f"stored-{container}.{CONTAINERS[container][0]}"
        stored.write_bytes(prepared.data)
        info = probe(stored)
        assert prepared.format is _FORMATS[container]
        assert user_tags(info) == {}, "a tag survived the remux"
        assert not has_attached_picture(info)
        assert [s["codec_type"] for s in info["streams"]] == ["audio"]  # type: ignore[index, union-attr]
        for value in SECRET_TAGS.values():
            assert value.encode() not in prepared.data
        # bit-identical audio: the packets hash equal, so nothing was re-encoded
        assert stream_md5(stored) == stream_md5(source)

    @pytest.mark.parametrize("container", list(CONTAINERS))
    async def test_the_facts_describe_the_stored_bytes(
        self, audio_files: AudioFixtures, container: str
    ) -> None:
        prepared = await _service().prepare_audio(audio_files.tone(container).read_bytes())

        assert prepared.sample_rate_hz == 44100
        assert prepared.channels == 1
        assert abs(prepared.duration_ms - 6000) <= 60
        assert len(prepared.peaks) == 2048
        assert prepared.peaks_version == 1
        assert prepared.content_type == _FORMATS[container].content_type
        expected_codec = {"mp3": "mp3", "m4a": "aac", "wav": "pcm_s16le", "flac": "flac"}
        assert prepared.codec == expected_codec[container]

    async def test_the_same_input_gives_the_same_stored_bytes_and_peaks(
        self, audio_files: AudioFixtures
    ) -> None:
        data = audio_files.tone("flac").read_bytes()

        first = await _service().prepare_audio(data)
        second = await _service().prepare_audio(data)

        assert first.data == second.data
        assert first.peaks == second.peaks

    async def test_a_loud_tone_has_peaks_and_the_peaks_are_signed_int8(
        self, audio_files: AudioFixtures
    ) -> None:
        prepared = await _service().prepare_audio(audio_files.tone("wav").read_bytes())

        peaks = np.frombuffer(prepared.peaks, dtype=np.int8)
        assert peaks.shape == (2048,)
        assert (peaks[0::2] <= peaks[1::2]).all()
        assert peaks[1::2].max() > 8  # a sine is not silence (ffmpeg's default amplitude is low)
        assert peaks[0::2].min() < -8


class TestPeaksOfKnownSignals:
    async def test_silence_gives_all_zeros(self, audio_files: AudioFixtures) -> None:
        prepared = await _service().prepare_audio(audio_files.silence.read_bytes())

        assert prepared.peaks == bytes(2048)

    async def test_a_full_scale_square_wave_gives_minus_128_and_127_in_every_bucket(
        self, audio_files: AudioFixtures
    ) -> None:
        prepared = await _service().prepare_audio(audio_files.square.read_bytes())

        peaks = np.frombuffer(prepared.peaks, dtype=np.int8)
        assert set(peaks[0::2].tolist()) == {-128}
        assert set(peaks[1::2].tolist()) == {127}


class TestRefusals:
    async def test_a_video_with_an_audio_track_is_not_audio(
        self, audio_files: AudioFixtures
    ) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            await _service().prepare_audio(audio_files.video_mp4.read_bytes())
        assert raised.value.code is AudioCode.HAS_VIDEO

    async def test_a_too_short_file_names_the_limits(self, audio_files: AudioFixtures) -> None:
        with pytest.raises(InvalidAudioError) as raised:
            await _service().prepare_audio(audio_files.short.read_bytes())
        assert raised.value.code is AudioCode.DURATION_OUT_OF_RANGE
        assert raised.value.detail == {"min_seconds": 5, "max_seconds": 900}

    async def test_ogg_opus_is_not_a_supported_format(self, audio_files: AudioFixtures) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            await _service().prepare_audio(audio_files.ogg_opus.read_bytes())
        assert raised.value.code is AudioCode.FORMAT_NOT_SUPPORTED

    async def test_float_pcm_in_a_wav_is_an_unsupported_codec(
        self, audio_files: AudioFixtures
    ) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            await _service().prepare_audio(audio_files.pcm_float.read_bytes())
        assert raised.value.code is AudioCode.CODEC_NOT_SUPPORTED

    async def test_a_picture_is_not_audio(self, audio_files: AudioFixtures) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            await _service().prepare_audio(audio_files.picture_only.read_bytes())
        assert raised.value.code is AudioCode.FORMAT_NOT_SUPPORTED

    @pytest.mark.parametrize("data", [b"not audio at all" * 40, bytes(range(256)) * 20])
    async def test_garbage_is_not_decodable(self, data: bytes) -> None:
        with pytest.raises(InvalidMediaError) as raised:
            await _service().prepare_audio(data)
        # ffprobe cannot read it at all (invalid_audio) or sniffs no accepted format.
        assert getattr(raised.value, "code", None) in {
            AudioCode.INVALID,
            AudioCode.FORMAT_NOT_SUPPORTED,
        }

    async def test_a_truncated_file_never_stores_a_partial_result(
        self, audio_files: AudioFixtures
    ) -> None:
        data = audio_files.tone("flac").read_bytes()[:200]
        with pytest.raises(InvalidMediaError):
            await _service().prepare_audio(data)

    async def test_the_sniffed_container_decides_not_the_extension_or_mime(
        self, audio_files: AudioFixtures
    ) -> None:
        """A FLAC sent as ``.mp3`` is stored as FLAC (ingest never sees the declared type)."""
        prepared = await _service().prepare_audio(audio_files.tone("flac").read_bytes())
        assert prepared.format is MediaFormat.FLAC

    async def test_a_body_over_the_audio_budget_is_refused_by_ingest_too(
        self, audio_files: AudioFixtures
    ) -> None:
        service = _service(max_audio_bytes=1000)
        with pytest.raises(InvalidMediaError, match="byte limit"):
            await service.prepare_audio(audio_files.tone("wav").read_bytes())

    async def test_the_audio_budget_defaults_to_the_general_one(self) -> None:
        service = MediaIngestService(max_image_megapixels=10, max_input_bytes=1000)
        with pytest.raises(InvalidMediaError, match="byte limit"):
            await service.prepare_audio(b"x" * 1001)

    def test_the_audio_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MediaIngestService(max_image_megapixels=10, max_input_bytes=10, max_audio_bytes=0)


class TestSharedVideoSlotsAndOperationalFailures:
    async def test_audio_takes_a_video_slot_and_gives_it_back(
        self, audio_files: AudioFixtures
    ) -> None:
        service = _service(video_concurrency=1)
        data = audio_files.tone("mp3").read_bytes()

        await service.prepare_audio(data)
        await service.prepare_audio(data)  # a leaked slot would time this one out

        assert service._video_slots._value == 1

    async def test_audio_waits_behind_video_work_and_is_refused_when_capacity_is_exhausted(
        self, audio_files: AudioFixtures
    ) -> None:
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()  # a video preparation holds the only slot
        service = _service(video_slots=semaphore, admission_wait_seconds=0.05)

        with pytest.raises(MediaProcessingError, match="capacity"):
            await service.prepare_audio(audio_files.tone("mp3").read_bytes())

    async def test_a_failure_gives_the_slot_back(self) -> None:
        service = _service(video_concurrency=1)

        with pytest.raises(InvalidMediaError):
            await service.prepare_audio(b"garbage" * 100)

        assert service._video_slots._value == 1

    async def test_a_full_temp_dir_is_an_operational_failure_not_the_client_s(self) -> None:
        with (
            patch(
                "src.api.services.media_ingest.service.tempfile.mkdtemp",
                side_effect=OSError(28, "No space left on device"),
            ),
            pytest.raises(MediaProcessingError, match="audio preparation failed operationally"),
        ):
            await _service().prepare_audio(b"x" * 100)


class TestNothingPrivateIsLogged:
    async def test_no_tag_value_or_file_name_reaches_the_logs(
        self, audio_files: AudioFixtures
    ) -> None:
        with structlog.testing.capture_logs() as logs:
            await _service().prepare_audio(audio_files.tone("mp3").read_bytes())
            with pytest.raises(InvalidMediaError):
                await _service().prepare_audio(b"garbage" * 100)

        rendered = repr(logs)
        for value in SECRET_TAGS.values():
            assert value not in rendered
        assert "input.bin" not in rendered


class TestServiceInternalsWithoutFfmpeg:
    """The paths that need a failing child process or a cancelled caller (no real ffmpeg)."""

    async def test_a_cancelled_caller_keeps_the_slot_until_the_work_has_finished(self) -> None:
        service = _service(video_concurrency=1)
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow(_data: bytes) -> object:
            started.set()
            await release.wait()
            raise InvalidMediaError("done")

        service._prepare_audio = slow  # type: ignore[method-assign]
        task = asyncio.create_task(service.prepare_audio(b"x" * 10))
        await started.wait()
        assert service._video_slots._value == 0  # the slot is taken

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # ffmpeg (here: the stand-in) is still running: the slot must still be held
        assert service._video_slots._value == 0
        release.set()
        for _ in range(50):
            if service._video_slots._value == 1:
                break
            await asyncio.sleep(0.01)
        assert service._video_slots._value == 1  # released once the work was reaped

    async def test_a_signal_exit_is_operational_not_the_files_fault(self) -> None:
        from src.api.services.media_tools import MediaToolExitError

        async def killed(*_args: object, **_kwargs: object) -> object:
            raise MediaToolExitError("ffprobe", -9, "killed")

        with (
            patch("src.api.services.media_ingest.service.run_media_command", killed),
            pytest.raises(MediaProcessingError, match="audio preparation failed operationally"),
        ):
            await _service().prepare_audio(b"x" * 100)

    async def test_a_positive_exit_is_invalid_audio_and_logs_the_stage_only(self) -> None:
        from src.api.services.media_tools import MediaToolExitError

        async def refused(*_args: object, **_kwargs: object) -> object:
            raise MediaToolExitError("ffprobe", 1, "title=SECRET_TITLE_VALUE: bad")

        with (
            structlog.testing.capture_logs() as logs,
            patch("src.api.services.media_ingest.service.run_media_command", refused),
            pytest.raises(InvalidAudioError) as raised,
        ):
            await _service().prepare_audio(b"x" * 100)

        assert raised.value.code is AudioCode.INVALID
        assert logs == [
            {"stage": "probe", "event": "media_ingest.audio_not_decodable", "log_level": "warning"}
        ]
        assert "SECRET_TITLE_VALUE" not in repr(logs)

    async def test_an_unreadable_probe_document_is_invalid_audio(self) -> None:
        from src.api.services.media_tools import ProcessResult

        async def not_json(*_args: object, **_kwargs: object) -> ProcessResult:
            return ProcessResult(stdout=b"<<not json>>", stderr=b"")

        with (
            patch("src.api.services.media_ingest.service.run_media_command", not_json),
            pytest.raises(InvalidAudioError) as raised,
        ):
            await _service().prepare_audio(b"x" * 100)

        assert raised.value.code is AudioCode.INVALID

    def test_a_remux_that_changed_the_container_is_our_fault(self) -> None:
        from src.api.services.media_ingest.audio import AudioProbe

        def probe_of(media_format: MediaFormat) -> AudioProbe:
            return AudioProbe(
                format=media_format,
                codec="flac",
                sample_rate_hz=44100,
                channels=2,
                bit_rate=None,
                duration_ms=6000,
            )

        MediaIngestService._validate_audio_remux(
            probe_of(MediaFormat.FLAC), probe_of(MediaFormat.FLAC)
        )
        with pytest.raises(MediaProcessingError, match="changed the container"):
            MediaIngestService._validate_audio_remux(
                probe_of(MediaFormat.FLAC), probe_of(MediaFormat.WAV)
            )

    def test_prepared_audio_over_the_budget_is_refused(self) -> None:
        service = _service(max_audio_bytes=10)
        service._validate_prepared_audio_size(b"x" * 10)
        with pytest.raises(InvalidMediaError, match="prepared audio exceeds"):
            service._validate_prepared_audio_size(b"x" * 11)
