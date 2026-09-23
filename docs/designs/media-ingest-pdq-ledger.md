# Media ingest and PDQ ledger

Every newly stored original passes through `MediaIngestService` before R2. The
service prepares immutable `PreparedImage` or `PreparedVideo` values; it does
not own an R2 write, database session, transaction, billing action, or job
transition. Composition creates one bounded service per process, so replica
counts multiply its image/video concurrency limits.

Image preparation accepts upload-compatible Pillow inputs or provider PNG,
JPEG, and WebP bytes. It checks pixel limits before full decode, bakes static
EXIF orientation, strips unapproved PNG/JPEG/WebP metadata carriers, then
computes PDQ from the exact stored bytes. ICC, colour, aspect and compressed
rendering payloads are retained functional exceptions: this policy removes
descriptive/application metadata and extensions; it does not promise removal
of every identifying byte inside retained rendering data. Animated provider
WebP/APNG keeps frames and timing, hashes the first displayed frame under a
separate profile, and rejects orientation-bearing animations.

Video preparation probes the real container before remuxing, retains one visual
stream and at most one audio stream, removes global/stream metadata, then hashes
sampled decoded frames from the remuxed output. `uniform-pts-v1` always includes
the first decoded frame, uses actual selected PTS values, caps frames, converts
sampling frames to square pixels, and stores normalized millisecond timestamps.

PDQ profiles are immutable identifiers. V1 uses Pillow's non-colour-managed RGB
conversion and composites transparency on white. Bits are packed big-endian:
bit zero is the most-significant bit of byte zero. A new rendering, colour, or
bit-serialization policy requires a new profile identifier.

`media_hashes` has no foreign key to expiring upload/output rows, so retention
and manual asset deletion keep ledger evidence. User deletion cascades ledger
rows; physical generation-job deletion sets `job_id` to null. There is no ledger
TTL in this feature.

For a persisted original, callers create and flush the parent row, stage its
ledger rows, flush that mandatory work, then attempt thumbnails. Each optional
thumbnail insert gets its own savepoint. Ingest failures split into deterministic
invalid media/limits and operational capacity, timeout, executable, disk, or
native-processing failures; only the former map to upload validation errors.
