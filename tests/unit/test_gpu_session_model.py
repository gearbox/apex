"""Tests for GpuSession model and GpuSessionStatus enum."""

from sqlalchemy import Table

from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession


def _table() -> Table:
    return GpuSession.__table__  # type: ignore[return-value]


class TestGpuSessionStatus:
    def test_all_values(self) -> None:
        assert GpuSessionStatus.pending.value == "pending"
        assert GpuSessionStatus.provisioning.value == "provisioning"
        assert GpuSessionStatus.active.value == "active"
        assert GpuSessionStatus.stale.value == "stale"
        assert GpuSessionStatus.paused.value == "paused"
        assert GpuSessionStatus.resuming.value == "resuming"
        assert GpuSessionStatus.stopping.value == "stopping"
        assert GpuSessionStatus.stopped.value == "stopped"
        assert GpuSessionStatus.failed.value == "failed"

    def test_member_count(self) -> None:
        assert len(GpuSessionStatus) == 9

    def test_from_string(self) -> None:
        assert GpuSessionStatus("active") == GpuSessionStatus.active
        assert GpuSessionStatus("stale") == GpuSessionStatus.stale
        assert GpuSessionStatus("paused") == GpuSessionStatus.paused
        assert GpuSessionStatus("resuming") == GpuSessionStatus.resuming

    def test_paused_resuming_ordering(self) -> None:
        members = list(GpuSessionStatus)
        stale_idx = members.index(GpuSessionStatus.stale)
        paused_idx = members.index(GpuSessionStatus.paused)
        resuming_idx = members.index(GpuSessionStatus.resuming)
        stopping_idx = members.index(GpuSessionStatus.stopping)
        assert stale_idx < paused_idx < resuming_idx < stopping_idx


class TestGpuSessionModelColumns:
    def test_new_columns_exist(self) -> None:
        cols = {c.name for c in _table().columns}
        expected = {
            "bundle_name",
            "bundle_version",
            "model_type",
            "cf_tunnel_id",
            "cf_dns_record_id",
            "tunnel_hostname",
            "vastai_offer_id",
            "vastai_cost_per_hour_micros",
            "vastai_gpu_name",
            "provision_attempt",
            "paused_at",
            "resumed_at",
            "callback_token",
        }
        assert expected.issubset(cols)

    def test_original_columns_still_exist(self) -> None:
        cols = {c.name for c in _table().columns}
        original = {
            "id",
            "user_id",
            "product_id",
            "status",
            "vastai_instance_id",
            "node_host",
            "node_port",
            "stale_detected_at",
            "stale_notified",
            "error_message",
            "created_at",
            "started_at",
            "stopped_at",
        }
        assert original.issubset(cols)

    def test_partial_unique_index_exists(self) -> None:
        index_names = {idx.name for idx in _table().indexes}
        assert "ix_gpu_sessions_active_user_model" in index_names

    def test_active_stale_index_exists(self) -> None:
        index_names = {idx.name for idx in _table().indexes}
        assert "ix_gpu_sessions_active_stale" in index_names
