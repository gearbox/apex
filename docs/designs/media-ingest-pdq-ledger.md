# Media ingest and PDQ ledger

Every newly stored original passes through `MediaIngestService` before R2. The
service prepares immutable `PreparedImage`, `PreparedVideo` or (uploads only)
`PreparedAudio` values. **Only images and videos get a perceptual hash and a
ledger row: audio has none** (see "Audio" below). The service does
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

`PreparedVideo` also carries a required `stream_profile: VideoStreamProfile` — the
probed facts of the *prepared* visual stream (container, codec, codec profile,
`pix_fmt`, colour transfer/primaries, display-matrix rotation in degrees
normalized into `[0, 360)`, sample aspect ratio, audio presence, and a derived `is_hdr`). It is
informational only and never a gate: every optional fact is read leniently and
degrades to `None`/`0`, so a missing or odd value can never become
`InvalidMediaError`, and acceptance behaviour is unchanged. The ingest boundary
owns no storage or database context, so it does not log the profile; the two
writers (`UserContentService._upload_video` and the Grok video materialization,
after the output row commits) emit one `media.video_profile` event each, tagged
with their `origin` and the original's `asset_ref`, through
`media_ingest.profile_log.log_video_profile`. Nothing is persisted: the logs answer
the one open question — how much stored video is HDR / rotated / non-H.264 — before
any browser-compatibility conversion is considered.

`POST /v1/storage/upload` maps the two failure classes to distinct statuses:
an object-storage (R2) failure is `502 upstream_error`, while ingest capacity
exhaustion or an operational processing failure is `503 service_unavailable`
("Media processing is temporarily unavailable"). Invalid media is `400
validation_error`.

The ledger derives `source_media_type` from the row's persisted `format`; an
unknown or missing format raises instead of defaulting to image — and an audio
format raises too (`MediaFormat.kind` is an exhaustive `match`): no writer may
register a hash for audio.

## Audio (S2a)

`prepare_audio(data)` is user uploads only (the provider materializers never
call it until generated audio exists). It shares the **video** admission slots
and deadline (`video_slots`, `video_deadline_seconds`, `stage_timeout_seconds`):
it is ffmpeg work of the same weight, and `MediaProcessingError` carries the
same operational meaning. Its own byte budget is `AUDIO_UPLOAD_MAX_MB`
(`max_audio_bytes`); the route enforces it from the *declared* type before
ingest runs and ingest re-checks it.

Pipeline: write a temp file → `ffprobe` (never the client MIME) → clean the
container → `ffprobe` of the **prepared** file (the stored bytes are what is
described and gated) → decoded-sample verification where required → bounded
peaks pass → read the prepared bytes.

* **Policy** (`media_ingest/audio.py::parse_audio_probe`, pure), in this order:
  format not mp3 / ISO-BMFF / wav / flac → `audio_format_not_supported`; any
  video stream other than an attached picture → `audio_has_video`; no audio
  stream → `invalid_audio`; a codec other than mp3 (mp3), aac (m4a),
  pcm_s16le / pcm_s24le (wav) or flac (flac), or more than 8 channels →
  `audio_codec_not_supported`; duration (the audio stream's, else the
  container's, positive) outside 5 s – 900 s → `audio_duration_out_of_range`
  (`detail: {min_seconds, max_seconds}`). The attached-picture flag is read from
  ffprobe's `stream_disposition` *section*: `disposition` listed inside
  `stream=` prints nothing, which would make cover art look like a video stream.
* **Container cleanup:** stream copy for MP3 (frames only), M4A and WAV;
  lossless re-encode for FLAC (bit-identical samples). The stream-copy options
  are `-map 0:a:0 -c:a copy -map_metadata -1
  -map_metadata:s:a -1 -map_chapters -1 -fflags +bitexact -flags:a +bitexact`,
  plus per container: mp3 `-write_id3v1 0 -id3v2_version 0 -write_xing 1 -f mp3`;
  m4a `-movflags +faststart -f ipod`; wav `-rf64 auto -f wav`. MP3 packets are
  scanned for complete Layer III frames; bytes outside confirmed frame chains
  are dropped, and when any are dropped the frames are remuxed again so the Xing
  header has the right count. A second scan must drop zero bytes. FLAC uses
  `-map 0:a:0 -c:a flac -sample_fmt <input-sample-fmt> -compression_level 5
  -map_metadata -1 -map_metadata:s:a -1 -map_chapters -1 -fflags +bitexact
  -flags:a +bitexact -f flac`; unsupported decoder sample formats are refused
  before encoding. The FLAC input and output decoded PCM `-f md5` values must
  match or ingest raises `MediaProcessingError`. Attached pictures are dropped
  because only the first audio stream is mapped. Every ffmpeg / ffprobe run adds
  `-protocol_whitelist file`.
* **Peaks:** a second ffmpeg pass on the prepared bytes (`-t 901 -ac 1 -ar 11025 -f s16le -`),
  read into a numpy `int16` array, split by `np.array_split` into exactly 1,024
  buckets; per bucket min and max `>> 8`, clipped to `[-128, 127]`, stored as
  interleaved `int8` (2,048 bytes, `PEAKS_VERSION = 1`).
* **Nothing private is logged:** the audio stages log the stage name only (never
  stderr: decoder warnings can quote tag values), and `media.audio_profile`
  (`profile_log.log_audio_profile`) carries format, codec, sample rate, channels,
  size and duration. Tag values are what ingest strips; they are never read,
  logged or returned.
* **Persistence** (`UserContentService._upload_audio`): the object is stored, then
  the `user_images` row (`width`/`height` `NULL`, `duration_ms`) and its
  `library_audio_profiles` row are written in one transaction; if the database
  step fails the object is deleted best-effort, so a failure leaves no orphan.
  No thumbnail, no poster and no ledger row.
