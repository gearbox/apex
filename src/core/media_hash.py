"""Immutable values shared by media preparation and the durable hash ledger.

The values in this module deliberately have no SQLAlchemy, storage, or API
dependencies.  A stored hash is meaningful only together with its profile and
sampling profile, so callers must carry all three values as one :class:`HashSet`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PdqHash:
    """One PDQ fingerprint in its canonical 32-byte, big-endian encoding."""

    bits: bytes
    quality: int

    def __post_init__(self) -> None:
        if len(self.bits) != 32:
            raise ValueError("PDQ hashes must contain exactly 32 bytes")
        if not 0 <= self.quality <= 100:
            raise ValueError("PDQ quality must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class HashSample:
    """A PDQ sample and its stable position in a source asset."""

    pdq: PdqHash
    sample_index: int
    frame_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.frame_timestamp_ms is not None and self.frame_timestamp_ms < 0:
            raise ValueError("frame_timestamp_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class HashSet:
    """Versioned, ordered PDQ samples for one stored original."""

    profile_id: str
    sampling_profile: str
    samples: tuple[HashSample, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or not self.sampling_profile:
            raise ValueError("hash profile identifiers are required")
        if not self.samples:
            raise ValueError("a hash set needs at least one sample")
        expected = tuple(range(len(self.samples)))
        actual = tuple(sample.sample_index for sample in self.samples)
        if actual != expected:
            raise ValueError("hash sample indexes must be unique and contiguous from zero")

        timestamps = tuple(sample.frame_timestamp_ms for sample in self.samples)
        is_video = any(timestamp is not None for timestamp in timestamps)
        if is_video:
            if any(timestamp is None for timestamp in timestamps):
                raise ValueError("video hash samples all require timestamps")
            concrete = tuple(timestamp for timestamp in timestamps if timestamp is not None)
            if concrete != tuple(sorted(concrete)):
                raise ValueError("video sample timestamps must be nondecreasing")
        elif len(self.samples) != 1 or self.samples[0].frame_timestamp_ms is not None:
            raise ValueError("still-image hash sets contain exactly one untimestamped sample")
