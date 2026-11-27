"""The 20 ms detail peaks level (S3a W6): bucket arithmetic, values, ingest and the window rules.

The ``ffmpeg`` tests prove two things the arithmetic cannot: ingest decodes the stored file **once**
for both levels, and the backfill's decode of the stored bytes yields exactly what ingest did.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from sqlalchemy import CheckConstraint

from src.api.services.library import (
    MAX_PEAKS_WINDOW_MS,
    LibraryPeaksWindowError,
    detail_peaks_window,
)
from src.api.services.media_ingest import InvalidAudioError, MediaIngestService, PreparedAudio
from src.api.services.media_ingest.audio import (
    DETAIL_BUCKET_MS,
    DETAIL_PEAKS_VERSION,
    MAX_DETAIL_BUCKETS,
    MAX_DETAIL_PEAKS_BYTES,
    MAX_PEAKS_SAMPLES,
    PEAKS_SAMPLE_RATE_HZ,
    compute_detail_peaks,
)
from src.core.enums import MediaFormat
from src.core.media_limits import AUDIO_MAX_SECONDS
from src.db.models.library import LibraryAudioProfile
from tests.audio_support import CONTAINERS, require_ffmpeg, run

if TYPE_CHECKING:
    from sqlalchemy import Table

    from tests.audio_support import AudioFixtures

pytestmark = pytest.mark.unit


def _pcm(samples: np.ndarray[Any, Any]) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def _pairs(peaks: bytes) -> np.ndarray[Any, Any]:
    return np.frombuffer(peaks, dtype=np.int8).reshape(-1, 2)


def _reference(samples: np.ndarray[Any, Any]) -> bytes:
    """The definition, written out: bucket ``b`` is ``[b * 441 // 2, (b + 1) * 441 // 2)``."""
    out: list[int] = []
    bucket = 0
    while bucket * 441 // 2 < len(samples):
        chunk = samples[bucket * 441 // 2 : (bucket + 1) * 441 // 2]
        out += [int(chunk.min()) >> 8, int(chunk.max()) >> 8]
        bucket += 1
    return np.array(out, dtype=np.int8).tobytes()


class TestComputeDetailPeaks:
    @pytest.mark.parametrize(
        "n",
        [1, 219, 220, 221, 440, 441, 442, 661, 662, 11_025, 88_200, 88_201, 100_003],
    )
    def test_there_are_ceil_n_times_two_over_441_buckets(self, n: int) -> None:
        peaks = compute_detail_peaks(_pcm(np.zeros(n)))

        assert len(peaks) == 2 * -(-n * 2 // 441)

    def test_an_eight_second_file_has_exactly_400_buckets(self) -> None:
        assert len(compute_detail_peaks(_pcm(np.zeros(8 * PEAKS_SAMPLE_RATE_HZ)))) == 800

    @pytest.mark.parametrize("n", [1, 440, 441, 4_410, 30_001])
    def test_every_bucket_matches_the_definition(self, n: int) -> None:
        samples = np.random.default_rng(seed=n).integers(-32768, 32768, size=n, dtype=np.int16)

        assert compute_detail_peaks(_pcm(samples)) == _reference(samples)

    def test_buckets_alternate_220_and_221_samples(self) -> None:
        """A single spike lands in exactly one bucket, on the right side of every boundary."""
        for index, bucket in ((0, 0), (219, 0), (220, 1), (440, 1), (441, 2), (660, 2), (661, 3)):
            samples = np.zeros(1_000, dtype=np.int16)
            samples[index] = 32767

            maxima = _pairs(compute_detail_peaks(_pcm(samples)))[:, 1]

            assert list(np.flatnonzero(maxima)) == [bucket], index

    def test_silence_is_all_zeros(self) -> None:
        peaks = compute_detail_peaks(_pcm(np.zeros(60_000)))

        assert peaks == bytes(len(peaks))

    def test_a_full_scale_square_wave_reaches_both_rails_in_every_bucket(self) -> None:
        period = np.where(np.arange(88_200) % 50 < 25, 32767, -32768)

        pairs = _pairs(compute_detail_peaks(_pcm(period)))

        assert set(pairs[:, 0]) == {-128}
        assert set(pairs[:, 1]) == {127}

    def test_it_is_deterministic(self) -> None:
        pcm = _pcm(np.arange(-20_000, 20_000))

        first = compute_detail_peaks(pcm)
        second = compute_detail_peaks(pcm)

        assert first == second

    def test_a_trailing_half_sample_is_ignored(self) -> None:
        pcm = _pcm(np.full(441, 1000))

        assert compute_detail_peaks(pcm + b"\x7f") == compute_detail_peaks(pcm)

    @pytest.mark.parametrize("pcm", [b"", b"\x01"])
    def test_an_empty_decode_is_invalid_audio(self, pcm: bytes) -> None:
        with pytest.raises(InvalidAudioError):
            compute_detail_peaks(pcm)

    def test_the_longest_decode_fills_exactly_the_stored_bound(self) -> None:
        peaks = compute_detail_peaks(bytes(2 * MAX_PEAKS_SAMPLES))

        assert len(peaks) == MAX_DETAIL_PEAKS_BYTES == 90_100

    def test_a_decode_past_the_bound_is_refused(self) -> None:
        with pytest.raises(InvalidAudioError):
            compute_detail_peaks(bytes(2 * (MAX_PEAKS_SAMPLES + 1)))


class TestTheNumbers:
    def test_the_bound_derives_from_the_shared_cap_and_the_bucket_width(self) -> None:
        assert DETAIL_BUCKET_MS == 20
        assert DETAIL_PEAKS_VERSION == 1
        assert MAX_DETAIL_PEAKS_BYTES == (AUDIO_MAX_SECONDS + 1) * 50 * 2 == 90_100
        assert MAX_DETAIL_BUCKETS == 45_050

    def test_the_model_s_check_holds_the_same_bound(self) -> None:
        # ``__table__`` is typed as ``FromClause``; a declarative model's is always a ``Table``.
        table = cast("Table", LibraryAudioProfile.__table__)
        checks = {
            constraint.name: str(constraint.sqltext)
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }

        size = checks["ck_library_audio_profiles_detail_peaks_size"]
        assert re.findall(r"<= (\d+)", size) == [str(MAX_DETAIL_PEAKS_BYTES)]

    def test_the_migration_s_frozen_literals_compute_the_same_bound(self) -> None:
        path = (
            Path(__file__).resolve().parents[2] / "alembic/versions/063_studio_audio_timelines.py"
        )
        spec = importlib.util.spec_from_file_location("revision_063_literals", path)
        assert spec is not None
        assert spec.loader is not None
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)

        assert migration._MAX_DETAIL_PEAKS_BYTES == MAX_DETAIL_PEAKS_BYTES
        assert migration._AUDIO_MAX_SECONDS == AUDIO_MAX_SECONDS


class TestPreparedAudioDetail:
    def _prepared(self, **overrides: Any) -> PreparedAudio:
        values: dict[str, Any] = {
            "data": b"x",
            "format": MediaFormat.WAV,
            "codec": "pcm_s16le",
            "sample_rate_hz": 44_100,
            "channels": 1,
            "bit_rate": None,
            "duration_ms": 8_000,
            "peaks": bytes(2048),
            "peaks_version": 1,
            "detail_peaks": bytes(800),
            "detail_peaks_version": 1,
        } | overrides
        return PreparedAudio(**values)

    def test_a_valid_value(self) -> None:
        assert self._prepared(detail_peaks=bytes(90_100)).detail_peaks_version == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"detail_peaks": b""}, id="empty"),
            pytest.param({"detail_peaks": bytes(801)}, id="odd"),
            pytest.param({"detail_peaks": bytes(90_102)}, id="too-long"),
            pytest.param({"detail_peaks_version": 0}, id="version-0"),
        ],
    )
    def test_a_bad_detail_level_is_refused(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="prepared audio"):
            self._prepared(**overrides)


class TestWindow:
    def test_no_bounds_is_the_whole_file(self) -> None:
        assert detail_peaks_window(None, None) == (0, MAX_DETAIL_BUCKETS)

    @pytest.mark.parametrize(
        ("from_ms", "to_ms", "expected"),
        [
            (0, 20, (0, 1)),
            (0, 1, (0, 1)),  # [from // 20, ceil(to / 20))
            (19, 21, (0, 2)),
            (20, 40, (1, 1)),
            (1_000, 2_000, (50, 50)),
            (0, 600_000, (0, 30_000)),
            (899_990, 1_499_990, (44_999, 51)),  # kept within the largest bucket count
            (10**30, 10**30 + 20, (MAX_DETAIL_BUCKETS, 0)),  # can never overflow the query
        ],
    )
    def test_the_buckets_of_a_window(
        self, from_ms: int, to_ms: int, expected: tuple[int, int]
    ) -> None:
        assert detail_peaks_window(from_ms, to_ms) == expected

    @pytest.mark.parametrize(
        ("from_ms", "to_ms"),
        [
            (0, None),
            (None, 20),
            (-1, 20),
            (20, 20),
            (40, 20),
            (0, MAX_PEAKS_WINDOW_MS + 1),
            (-20, -10),
        ],
    )
    def test_what_is_not_a_window(self, from_ms: int | None, to_ms: int | None) -> None:
        with pytest.raises(LibraryPeaksWindowError):
            detail_peaks_window(from_ms, to_ms)

    def test_the_widest_window_is_ten_minutes(self) -> None:
        assert MAX_PEAKS_WINDOW_MS == 600_000


# -- through ffmpeg -------------------------------------------------------------------------------


@pytest.fixture
def sine_wav(tmp_path: Path) -> bytes:
    """An 8 s sine as 44.1 kHz WAV: it resamples to exactly 88,200 samples at 11,025 Hz."""
    path = tmp_path / "sine8.wav"
    run(
        [
            require_ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8:sample_rate=44100",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path.read_bytes()


def _service() -> MediaIngestService:
    return MediaIngestService(max_image_megapixels=10, max_input_bytes=20 * 1024 * 1024)


class TestThroughFfmpeg:
    async def test_ingest_computes_400_buckets_for_8_seconds(self, sine_wav: bytes) -> None:
        prepared = await _service().prepare_audio(sine_wav)

        assert len(prepared.detail_peaks) == 400 * 2
        assert prepared.detail_peaks_version == DETAIL_PEAKS_VERSION
        pairs = _pairs(prepared.detail_peaks)
        assert (pairs[:, 0] <= pairs[:, 1]).all()
        assert pairs[:, 1].max() > 0 > pairs[:, 0].min()

    async def test_ingest_decodes_once_for_both_levels(
        self, sine_wav: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.api.services.media_ingest import service as ingest_module

        real = ingest_module.run_media_command
        decodes: list[list[str]] = []

        async def counting(args: list[str], **kwargs: Any) -> Any:
            if "s16le" in args:  # the mono PCM decode the peaks are computed from
                decodes.append(args)
            return await real(args, **kwargs)

        monkeypatch.setattr(ingest_module, "run_media_command", counting)

        await _service().prepare_audio(sine_wav)

        assert len(decodes) == 1

    @pytest.mark.parametrize("container", list(CONTAINERS))
    async def test_the_backfill_decode_of_the_stored_bytes_equals_ingest_s(
        self, audio_files: AudioFixtures, container: str
    ) -> None:
        service = _service()
        prepared = await service.prepare_audio(audio_files.tone(container).read_bytes())

        recomputed = await service.compute_audio_detail_peaks(prepared.data)

        assert recomputed == prepared.detail_peaks

    async def test_undecodable_bytes_are_invalid_audio(self) -> None:
        require_ffmpeg()
        with pytest.raises(InvalidAudioError):
            await _service().compute_audio_detail_peaks(b"not audio at all" * 100)
