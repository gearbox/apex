"""Tests for GpuSession model and GpuSessionStatus enum."""

from src.core.enums import GpuSessionStatus


class TestGpuSessionStatus:
    def test_all_values(self) -> None:
        assert GpuSessionStatus.pending.value == "pending"
        assert GpuSessionStatus.provisioning.value == "provisioning"
        assert GpuSessionStatus.active.value == "active"
        assert GpuSessionStatus.stale.value == "stale"
        assert GpuSessionStatus.stopping.value == "stopping"
        assert GpuSessionStatus.stopped.value == "stopped"
        assert GpuSessionStatus.failed.value == "failed"

    def test_from_string(self) -> None:
        assert GpuSessionStatus("active") == GpuSessionStatus.active
        assert GpuSessionStatus("stale") == GpuSessionStatus.stale
