"""Audio ingest policy that needs neither ffmpeg nor a database (S2a).

``parse_audio_probe`` applies the accepted-format / codec / duration table to a decoded
``ffprobe -of json`` document, ``remux_arguments`` builds the stream-copy options per container and
``compute_peaks`` reduces PCM to the stored peaks. ``PreparedAudio`` guards what is stored.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.api.services.media_ingest import (
    AudioCode,
    InvalidAudioError,
    PreparedAudio,
    UnsupportedAudioError,
)
from src.api.services.media_ingest.audio import (
    AUDIO_MAX_SECONDS,
    AUDIO_MIN_SECONDS,
    PEAKS_BUCKETS,
    PEAKS_BYTES,
    PEAKS_VERSION,
    compute_peaks,
    parse_audio_probe,
    remux_arguments,
)
from src.core.enums import MediaFormat

pytestmark = pytest.mark.unit

_MP4_FORMAT = "mov,mp4,m4a,3gp,3g2,mj2"


def _audio_stream(codec: str, **extra: object) -> dict[str, object]:
    return {
        "index": 0,
        "codec_type": "audio",
        "codec_name": codec,
        "sample_rate": "44100",
        "channels": 2,
        "duration": "6.000000",
        "bit_rate": "128000",
        "disposition": {"attached_pic": 0},
        **extra,
    }


def _probe(
    format_name: str = "mp3",
    streams: list[dict[str, object]] | None = None,
    **container: object,
) -> dict[str, object]:
    return {
        "streams": streams if streams is not None else [_audio_stream("mp3")],
        "format": {"format_name": format_name, "duration": "6.000000", **container},
    }


def _picture() -> dict[str, object]:
    return {
        "index": 1,
        "codec_type": "video",
        "codec_name": "png",
        "disposition": {"attached_pic": 1},
    }


def _real_video() -> dict[str, object]:
    return {
        "index": 1,
        "codec_type": "video",
        "codec_name": "h264",
        "disposition": {"attached_pic": 0},
    }


class TestAcceptedFiles:
    @pytest.mark.parametrize(
        ("format_name", "codec", "expected"),
        [
            ("mp3", "mp3", MediaFormat.MP3),
            (_MP4_FORMAT, "aac", MediaFormat.M4A),
            ("wav", "pcm_s16le", MediaFormat.WAV),
            ("wav", "pcm_s24le", MediaFormat.WAV),
            ("flac", "flac", MediaFormat.FLAC),
        ],
    )
    def test_each_accepted_container_and_codec(
        self, format_name: str, codec: str, expected: MediaFormat
    ) -> None:
        result = parse_audio_probe(_probe(format_name, [_audio_stream(codec)]))
        assert result.format is expected
        assert result.codec == codec
        assert result.sample_rate_hz == 44100
        assert result.channels == 2
        assert result.bit_rate == 128000
        assert result.duration_ms == 6000

    def test_an_attached_cover_picture_is_not_a_video_stream(self) -> None:
        result = parse_audio_probe(_probe("mp3", [_audio_stream("mp3"), _picture()]))
        assert result.format is MediaFormat.MP3

    def test_the_first_audio_stream_is_the_one_described(self) -> None:
        second = _audio_stream("aac", index=2, sample_rate="48000")
        result = parse_audio_probe(_probe(_MP4_FORMAT, [_audio_stream("aac"), second]))
        assert result.sample_rate_hz == 44100

    def test_the_bit_rate_may_be_absent(self) -> None:
        stream = _audio_stream("flac")
        del stream["bit_rate"]
        assert parse_audio_probe(_probe("flac", [stream])).bit_rate is None

    def test_the_container_bit_rate_is_the_fallback(self) -> None:
        stream = _audio_stream("flac")
        del stream["bit_rate"]
        result = parse_audio_probe(_probe("flac", [stream], bit_rate="900000"))
        assert result.bit_rate == 900000


class TestRefusals:
    """The table of the work order, in the order the checks run."""

    @pytest.mark.parametrize("format_name", ["ogg", "matroska,webm", "png_pipe", "aiff", "aac", ""])
    def test_a_format_outside_the_four_is_not_supported(self, format_name: str) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            parse_audio_probe(_probe(format_name))
        assert raised.value.code is AudioCode.FORMAT_NOT_SUPPORTED

    def test_a_real_video_stream_is_refused_even_in_an_accepted_container(self) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            parse_audio_probe(_probe(_MP4_FORMAT, [_audio_stream("aac"), _real_video()]))
        assert raised.value.code is AudioCode.HAS_VIDEO

    def test_no_audio_stream_is_invalid_audio(self) -> None:
        with pytest.raises(InvalidAudioError) as raised:
            parse_audio_probe(_probe("mp3", [_picture()]))
        assert raised.value.code is AudioCode.INVALID

    @pytest.mark.parametrize(
        ("format_name", "codec"),
        [
            ("wav", "pcm_f32le"),
            ("wav", "adpcm_ms"),
            ("mp3", "mp2"),
            (_MP4_FORMAT, "alac"),
            (_MP4_FORMAT, "mp3"),
            ("flac", "pcm_s16le"),
        ],
    )
    def test_a_codec_outside_the_container_s_accepted_one_is_not_supported(
        self, format_name: str, codec: str
    ) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            parse_audio_probe(_probe(format_name, [_audio_stream(codec)]))
        assert raised.value.code is AudioCode.CODEC_NOT_SUPPORTED

    def test_more_than_eight_channels_is_not_supported(self) -> None:
        with pytest.raises(UnsupportedAudioError) as raised:
            parse_audio_probe(_probe("flac", [_audio_stream("flac", channels=9)]))
        assert raised.value.code is AudioCode.CODEC_NOT_SUPPORTED

    @pytest.mark.parametrize("seconds", ["0.5", "4.99", "900.01", "7200"])
    def test_a_duration_outside_five_seconds_to_fifteen_minutes(self, seconds: str) -> None:
        with pytest.raises(InvalidAudioError) as raised:
            parse_audio_probe(_probe("mp3", [_audio_stream("mp3", duration=seconds)]))
        assert raised.value.code is AudioCode.DURATION_OUT_OF_RANGE
        assert raised.value.detail == {
            "min_seconds": AUDIO_MIN_SECONDS,
            "max_seconds": AUDIO_MAX_SECONDS,
        }

    @pytest.mark.parametrize("seconds", ["5", "5.0", "900", "900.0"])
    def test_the_limits_themselves_are_accepted(self, seconds: str) -> None:
        result = parse_audio_probe(_probe("mp3", [_audio_stream("mp3", duration=seconds)]))
        assert result.duration_ms == round(float(seconds) * 1000)

    def test_the_limit_constants_are_the_documented_ones(self) -> None:
        assert (AUDIO_MIN_SECONDS, AUDIO_MAX_SECONDS) == (5, 900)

    def test_the_container_duration_is_the_fallback(self) -> None:
        stream = _audio_stream("mp3")
        del stream["duration"]
        result = parse_audio_probe(_probe("mp3", [stream], duration="7.5"))
        assert result.duration_ms == 7500

    @pytest.mark.parametrize("bad", [None, "N/A", "0", "-3", "nan", "inf"])
    def test_a_file_with_no_positive_duration_anywhere_is_invalid(self, bad: str | None) -> None:
        stream = _audio_stream("mp3")
        container: dict[str, object] = {}
        if bad is None:
            del stream["duration"]
        else:
            stream["duration"] = bad
            container["duration"] = bad
        probe = _probe("mp3", [stream])
        if bad is None:
            del probe["format"]["duration"]  # type: ignore[call-overload]
        else:
            probe["format"]["duration"] = bad  # type: ignore[index]
        with pytest.raises(InvalidAudioError) as raised:
            parse_audio_probe(probe)
        assert raised.value.code is AudioCode.INVALID

    def test_the_audio_stream_duration_wins_over_the_container_s(self) -> None:
        result = parse_audio_probe(
            _probe("mp3", [_audio_stream("mp3", duration="10")], duration="900")
        )
        assert result.duration_ms == 10000

    @pytest.mark.parametrize("decoded", [None, [], "x", {}, {"streams": []}, {"format": {}}])
    def test_a_malformed_probe_document_is_invalid_audio(self, decoded: object) -> None:
        with pytest.raises(InvalidAudioError) as raised:
            parse_audio_probe(decoded)
        assert raised.value.code is AudioCode.INVALID

    def test_a_missing_sample_rate_is_invalid_audio(self) -> None:
        stream = _audio_stream("mp3")
        del stream["sample_rate"]
        with pytest.raises(InvalidAudioError):
            parse_audio_probe(_probe("mp3", [stream]))

    def test_the_messages_are_static_and_carry_no_probe_content(self) -> None:
        stream = _audio_stream("pcm_f32le", tags={"title": "SECRET"})
        with pytest.raises(UnsupportedAudioError) as raised:
            parse_audio_probe(_probe("wav", [stream]))
        assert "SECRET" not in str(raised.value)


_COMMON = [
    "-map",
    "0:a:0",
    "-c:a",
    "copy",
    "-map_metadata",
    "-1",
    "-map_metadata:s:a",
    "-1",
    "-map_chapters",
    "-1",
    "-fflags",
    "+bitexact",
    "-flags:a",
    "+bitexact",
]


class TestRemuxArguments:
    @pytest.mark.parametrize(
        "media_format", [MediaFormat.MP3, MediaFormat.M4A, MediaFormat.WAV, MediaFormat.FLAC]
    )
    def test_every_container_maps_only_the_first_audio_stream_and_copies_it(
        self, media_format: MediaFormat
    ) -> None:
        arguments = remux_arguments(media_format)
        assert arguments[: len(_COMMON)] == _COMMON
        assert "-c" not in arguments  # never a re-encode request
        assert "libmp3lame" not in arguments

    def test_mp3_writes_no_id3_and_a_xing_header(self) -> None:
        arguments = remux_arguments(MediaFormat.MP3)
        assert arguments[len(_COMMON) :] == [
            "-write_id3v1",
            "0",
            "-id3v2_version",
            "0",
            "-write_xing",
            "1",
            "-f",
            "mp3",
        ]

    def test_m4a_is_faststart_in_the_ipod_muxer(self) -> None:
        assert remux_arguments(MediaFormat.M4A)[len(_COMMON) :] == [
            "-movflags",
            "+faststart",
            "-f",
            "ipod",
        ]

    def test_wav_may_become_rf64_when_needed(self) -> None:
        assert remux_arguments(MediaFormat.WAV)[len(_COMMON) :] == [
            "-rf64",
            "auto",
            "-f",
            "wav",
        ]

    def test_flac_has_no_container_options(self) -> None:
        assert remux_arguments(MediaFormat.FLAC)[len(_COMMON) :] == ["-f", "flac"]

    def test_a_non_audio_format_has_no_remux(self) -> None:
        with pytest.raises(KeyError):
            remux_arguments(MediaFormat.PNG)


def _pcm(samples: list[int] | np.ndarray) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


class TestPeaks:
    def test_the_stored_layout_is_1024_interleaved_int8_min_max_pairs(self) -> None:
        peaks = compute_peaks(_pcm(np.arange(-30000, 30000)))
        assert (PEAKS_BUCKETS, PEAKS_BYTES, PEAKS_VERSION) == (1024, 2048, 1)
        assert len(peaks) == 2048

    def test_the_same_pcm_gives_the_same_peaks(self) -> None:
        pcm = _pcm(np.random.default_rng(7).integers(-32768, 32767, 50_000))
        first = compute_peaks(pcm)
        second = compute_peaks(pcm)
        assert first == second

    def test_silence_gives_all_zeros(self) -> None:
        assert compute_peaks(_pcm(np.zeros(60_000))) == bytes(2048)

    def test_a_full_scale_square_wave_gives_minus_128_and_127_in_every_bucket(self) -> None:
        square = np.tile(np.concatenate([np.full(25, 32767), np.full(25, -32768)]), 1200)
        peaks = np.frombuffer(compute_peaks(_pcm(square)), dtype=np.int8)
        assert set(peaks[0::2]) == {-128}
        assert set(peaks[1::2]) == {127}

    def test_min_comes_before_max_in_every_pair(self) -> None:
        ramp = np.linspace(-20000, 20000, 40_000).astype(np.int16)
        peaks = np.frombuffer(compute_peaks(_pcm(ramp)), dtype=np.int8)
        assert (peaks[0::2] <= peaks[1::2]).all()

    def test_the_shift_is_arithmetic_so_small_negatives_are_minus_one(self) -> None:
        peaks = np.frombuffer(compute_peaks(_pcm(np.full(5000, -1))), dtype=np.int8)
        assert set(peaks) == {-1}

    def test_buckets_cover_the_signal_in_order(self) -> None:
        # Loud first half, quiet second half: the first buckets peak high, the last ones low.
        signal = np.concatenate([np.full(30_000, 30000), np.full(30_000, 300)])
        peaks = np.frombuffer(compute_peaks(_pcm(signal)), dtype=np.int8)
        assert peaks[1] > 100
        assert peaks[-1] < 5

    def test_an_odd_trailing_byte_is_ignored(self) -> None:
        pcm = _pcm(np.zeros(5000)) + b"\x01"
        assert compute_peaks(pcm) == bytes(2048)

    @pytest.mark.parametrize("pcm", [b"", _pcm([1, 2, 3]), _pcm(np.zeros(1023))])
    def test_fewer_samples_than_buckets_is_invalid_audio(self, pcm: bytes) -> None:
        with pytest.raises(InvalidAudioError) as raised:
            compute_peaks(pcm)
        assert raised.value.code is AudioCode.INVALID

    def test_exactly_one_sample_per_bucket_is_enough(self) -> None:
        assert len(compute_peaks(_pcm(np.zeros(1024)))) == 2048


class TestPreparedAudio:
    @staticmethod
    def _build(**overrides: object) -> PreparedAudio:
        values: dict[str, object] = {
            "data": b"audio",
            "format": MediaFormat.FLAC,
            "codec": "flac",
            "sample_rate_hz": 44100,
            "channels": 2,
            "bit_rate": 900000,
            "duration_ms": 6000,
            "peaks": bytes(2048),
            "peaks_version": 1,
        }
        values.update(overrides)
        return PreparedAudio(**values)  # type: ignore[arg-type]

    def test_a_valid_value_reports_its_content_type(self) -> None:
        assert self._build().content_type == "audio/flac"
        assert self._build(format=MediaFormat.M4A).content_type == "audio/mp4"
        assert self._build(format=MediaFormat.MP3).content_type == "audio/mpeg"
        assert self._build(format=MediaFormat.WAV).content_type == "audio/wav"

    def test_the_bit_rate_may_be_none(self) -> None:
        assert self._build(bit_rate=None).bit_rate is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"format": MediaFormat.PNG},
            {"format": MediaFormat.MP4},
            {"peaks": bytes(2047)},
            {"peaks": bytes(2049)},
            {"peaks": b""},
            {"codec": ""},
            {"sample_rate_hz": 0},
            {"channels": 0},
            {"duration_ms": 0},
            {"duration_ms": -5},
            {"peaks_version": 0},
            {"data": b""},
            {"bit_rate": 0},
            {"bit_rate": -1},
        ],
        ids=lambda o: ",".join(sorted(o)) + ":" + repr(next(iter(o.values())))[:20],
    )
    def test_an_invalid_value_cannot_be_built(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError, match=r"prepared audio|PreparedAudio"):
            self._build(**overrides)

    def test_it_is_frozen(self) -> None:
        prepared = self._build()
        with pytest.raises(AttributeError):
            prepared.codec = "mp3"  # type: ignore[misc]
