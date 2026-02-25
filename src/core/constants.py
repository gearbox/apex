"""Application-wide constants."""

from uuid import UUID

# Sentinel user representing the system itself.
# Used as `created_by` for rows seeded during migrations (e.g. default pricing rules).
# Inserted into `users` by migration 002. Cannot authenticate — has no password and is inactive.
SYSTEM_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
