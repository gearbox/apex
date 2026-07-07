"""Tests for health response schema serialization."""

import msgspec

from src.api.schemas.health import (
    CategoryHealthResponse,
    ComponentHealthResponse,
    DetailedHealthResponse,
    GpuSessionHealthResponse,
    HealthSnapshotResponse,
    LivenessResponse,
    ReadinessResponse,
)


class TestLivenessResponse:
    def test_serialize(self) -> None:
        r = LivenessResponse(status="alive")
        data = msgspec.json.decode(msgspec.json.encode(r))
        assert data == {"status": "alive"}


class TestReadinessResponse:
    def test_serialize(self) -> None:
        r = ReadinessResponse(
            status="ready",
            checks={"postgres": "healthy", "redis": "healthy"},
            build_sha="abc123",
        )
        data = msgspec.json.decode(msgspec.json.encode(r))
        assert data["status"] == "ready"
        assert data["checks"]["postgres"] == "healthy"
        assert data["build_sha"] == "abc123"


class TestDetailedHealthResponse:
    def test_full_roundtrip(self) -> None:
        r = DetailedHealthResponse(
            status="healthy",
            checked_at="2026-03-31T14:00:00+00:00",
            infrastructure=CategoryHealthResponse(
                status="healthy",
                components=[
                    ComponentHealthResponse(name="postgres", status="healthy", latency_ms=1.2),
                ],
            ),
            platform_apis=CategoryHealthResponse(status="inactive", components=[]),
            cloud_providers={},
            gpu_sessions=GpuSessionHealthResponse(
                status="inactive",
                total=0,
                healthy=0,
                stale=0,
            ),
        )
        encoded = msgspec.json.encode(r)
        decoded = msgspec.json.decode(encoded, type=DetailedHealthResponse)
        assert decoded.status == "healthy"
        assert decoded.infrastructure.components[0].name == "postgres"


class TestHealthSnapshotResponse:
    def test_serialize(self) -> None:
        r = HealthSnapshotResponse(
            checked_at="2026-03-31T14:00:00+00:00",
            overall_status="healthy",
            snapshot_data={"status": "healthy"},
        )
        data = msgspec.json.decode(msgspec.json.encode(r))
        assert data["overall_status"] == "healthy"
        assert data["snapshot_data"]["status"] == "healthy"
