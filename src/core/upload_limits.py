"""The one place an upload's byte limit is decided: per media kind (S2a-4).

Images and videos share ``MAX_UPLOAD_SIZE_MB``; audio has its own ``AUDIO_UPLOAD_MAX_MB`` (a
lossless track is far larger than an image). The route reads the limit from the client-declared
content type (an early filter), and ingest re-checks it against the kind it sniffed, because the
client's MIME is never trusted. The storage layer reads it too, so no 20 MB constant is repeated.
"""

from __future__ import annotations

from typing import Protocol, assert_never

from src.core.enums import MediaKind

__all__ = ["MIB", "UploadLimits", "upload_size_limit_bytes"]

MIB = 1024 * 1024


class UploadLimits(Protocol):
    """What the helper reads: ``Settings`` and ``R2StorageSettings`` both provide it."""

    @property
    def max_upload_size_mb(self) -> int: ...

    @property
    def audio_upload_max_mb(self) -> int: ...


def upload_size_limit_bytes(kind: MediaKind, settings: UploadLimits) -> int:
    """The largest body, in bytes, an upload of ``kind`` may have."""
    match kind:
        case MediaKind.IMAGE | MediaKind.VIDEO:
            return settings.max_upload_size_mb * MIB
        case MediaKind.AUDIO:
            return settings.audio_upload_max_mb * MIB
        case _:
            assert_never(kind)
