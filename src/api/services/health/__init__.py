"""Health check system — extensible infrastructure and provider monitoring."""

# Shared Redis Pub/Sub channel for health snapshot streaming.
# Used by HealthSnapshotWorker (publisher) and SSE stream (subscriber).
HEALTH_STREAM_CHANNEL = "health:stream"
