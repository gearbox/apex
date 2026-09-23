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

Video preparation probes before classifying the container. ISO-BMFF requires an
MP4-compatible major or compatible brand, or a QuickTime major brand; WebM
requires an EBML `webm` DocType. It retains one visual stream and at most one
audio stream, removes global/stream metadata, then hashes sampled decoded
frames from the remuxed output. Visual duration uses, in order, stream duration,
duration ticks and time base, visual `DURATION` tag, then container duration.
The container fallback may include a longer audio track, making the upload cap
conservative. Stream durations over container duration plus one second are
rejected. No audio stream duration is used.
If the prepared timeline is more than 250 ms shorter than the source timeline,
the input is rejected. Matroska demuxing can warn about a truncated cluster
while returning success, and the lost timeline reveals that truncation.

`uniform-pts-v1` always includes the first decoded frame, uses actual selected
PTS values, caps frames, converts sampling frames to square pixels, and stores
normalized millisecond timestamps. ffmpeg showinfo stderr is captured up to
1 MiB; the 4096-byte limit applies only to error excerpts. Longer clips use
`uniform-pts-keyframes-v1`, which skips non-keyframes before decoding. Sparse
GOP sources can yield fewer samples; the minimum is one.

PDQ profiles are immutable identifiers. Video v2 corrects sample display width
for non-square pixels. Image v1 and video v2 use Pillow's non-colour-managed RGB
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
Positive ffprobe/ffmpeg exits during source probe, remux, prepared-output probe,
or sampling mean deterministic undecodable input. Signal exits, executable
absence, timeouts, and OS errors remain operational. A full disk can also make
ffmpeg exit positively during remux; input writes normally fail first with an
OS error, and stderr text is not used to classify it.
