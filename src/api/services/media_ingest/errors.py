"""Error categories for media preparation.

Only deterministic malformed/unsupported inputs use ``InvalidMediaError``.
Capacity, executable, timeout and native-library failures are operational and
must not be turned into a client-side "not decodable" response.
"""

from __future__ import annotations


class MediaIngestError(Exception):
    """Base class for preparation failures."""


class InvalidMediaError(MediaIngestError):
    """The source bytes are malformed or exceed a configured media limit."""


class UnsupportedMediaError(InvalidMediaError):
    """The source is valid media but outside the supported ingest policy."""


class MediaProcessingError(MediaIngestError):
    """A local operational dependency failed while preparing media."""
