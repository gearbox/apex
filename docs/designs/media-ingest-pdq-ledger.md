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
frames from the remuxed output. Visual duration uses, in order, stream duration
(`stream`), duration ticks and time base (`timeline`), visual `DURATION` tag
(`tag`), then container duration (`container`); each probe records which source
it used (`DurationSource`). The container fallback may include a longer audio
track, making the upload cap conservative. Stream durations over container
duration plus one second are rejected. No audio stream duration is used.

The source probe may resolve no duration at all (`unknown`): browser
`MediaRecorder` WebM carries neither stream, tag, nor container duration. When
the source duration is known, the duration cap is enforced before the remux as
an early exit. The prepared-output probe must resolve a duration — the remux
writes one — and the cap is enforced again on it; that check is authoritative,
and `PreparedVideo.duration_ms` and sampling both use the prepared probe.

If the prepared timeline is more than 250 ms shorter than the source timeline,
the input is rejected. Matroska demuxing can warn about a truncated cluster
while returning success, and the lost timeline reveals that truncation. The
comparison only runs when both sides came from stream-level sources (`stream`,
`timeline`, `tag`); a `container` or `unknown` side skips it, because a container
duration can span a longer audio track and would flag a healthy file.

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
OS error, and stderr text is not used to classify it. Creating the per-video
temporary directory (ENOSPC, inode quota, EACCES, missing `TMPDIR`) and reading
the container sniff header are inside the same classification: any `OSError`
there becomes `MediaProcessingError`, never `InvalidMediaError` or a raw 500.
Every blocking filesystem call (temp-dir create/remove, input write, header and
output reads) and the per-frame PDQ decode/hash run via `asyncio.to_thread`;
all video frames are hashed in one worker-thread call. Error excerpts keep the
tail of stderr (the fatal line comes last); ffmpeg runs with `-hide_banner`,
and child processes get a null stdin. A user-caused undecodable input is logged
at warning (`media_ingest.video_not_decodable`, no traceback); provider callers
log their own error-level events for pipeline anomalies.

`POST /v1/storage/upload` maps the two failure classes to distinct statuses:
an object-storage (R2) failure is `502 upstream_error`, while ingest capacity
exhaustion or an operational processing failure is `503 service_unavailable`
("Media processing is temporarily unavailable"). Invalid media is `400
validation_error`.

The ledger derives `source_media_type` from the row's persisted `format`; an
unknown or missing format raises instead of defaulting to image.
