"""Application-wide constants."""

from uuid import UUID

# Sentinel user representing the system itself.
# Used as `created_by` for rows seeded during migrations (e.g. default pricing rules).
# Inserted into `users` by migration 002. Cannot authenticate — has no password and is inactive.
SYSTEM_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

# Video frame extraction request caps. Shared by src/api/schemas/frames.py (msgspec.Meta
# bounds) and src/core/config.py (frame_extract_stale_running_seconds validator) — the
# sweep threshold must exceed worst-case job runtime, which is derived from these caps.
MAX_PREVIEW_FRAME_COUNT = 60
MAX_EXTRACT_TIMESTAMPS = 50
