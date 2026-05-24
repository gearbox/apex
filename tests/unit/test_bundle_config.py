"""Unit tests for bundle_config dataclasses."""

import dataclasses

import pytest

from src.core.bundle_config import BundleMapping, HardwareRequirements, ReadinessMarker


class TestHardwareRequirements:
    def _make(self, **overrides: object) -> HardwareRequirements:
        defaults: dict[str, object] = {
            "gpu_whitelist": ("RTX_4090", "A100_SXM4"),
            "min_disk_gb": 100,
            "min_network_upload_mbps": 100,
            "min_network_download_mbps": 500,
            "cuda_min_version": "12.1",
            "num_gpus": 1,
        } | overrides
        return HardwareRequirements(**defaults)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        hw = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            hw.min_disk_gb = 200  # type: ignore[misc]

    def test_slots(self) -> None:
        assert "__slots__" in HardwareRequirements.__dict__

    def test_default_comfyui_port(self) -> None:
        hw = self._make()
        assert hw.comfyui_port == 8188

    def test_custom_comfyui_port(self) -> None:
        hw = self._make(comfyui_port=8188)
        assert hw.comfyui_port == 8188

    def test_fields_stored_correctly(self) -> None:
        hw = self._make(
            gpu_whitelist=("RTX_4090",),
            min_disk_gb=200,
            min_network_upload_mbps=50,
            min_network_download_mbps=1000,
            cuda_min_version="12.4",
            num_gpus=2,
        )
        assert hw.gpu_whitelist == ("RTX_4090",)
        assert hw.min_disk_gb == 200
        assert hw.min_network_upload_mbps == 50
        assert hw.min_network_download_mbps == 1000
        assert hw.cuda_min_version == "12.4"
        assert hw.num_gpus == 2


class TestBundleMapping:
    def _hw(self) -> HardwareRequirements:
        return HardwareRequirements(
            gpu_whitelist=("RTX_4090",),
            min_disk_gb=100,
            min_network_upload_mbps=100,
            min_network_download_mbps=500,
            cuda_min_version="12.1",
            num_gpus=1,
        )

    def test_frozen(self) -> None:
        bm = BundleMapping(bundle_name="wan_2.2_i2v", bundle_version=None, hardware=self._hw())
        with pytest.raises(dataclasses.FrozenInstanceError):
            bm.bundle_name = "other"  # type: ignore[misc]

    def test_slots(self) -> None:
        assert "__slots__" in BundleMapping.__dict__

    def test_with_version(self) -> None:
        bm = BundleMapping(
            bundle_name="wan_2.2_i2v",
            bundle_version="260105-01",
            hardware=self._hw(),
        )
        assert bm.bundle_version == "260105-01"

    def test_without_version(self) -> None:
        bm = BundleMapping(bundle_name="wan_2.2_i2v", bundle_version=None, hardware=self._hw())
        assert bm.bundle_version is None

    def test_hardware_reference(self) -> None:
        hw = self._hw()
        bm = BundleMapping(bundle_name="wan_2.2_i2v", bundle_version=None, hardware=hw)
        assert bm.hardware is hw

    def test_default_readiness_marker_is_none(self) -> None:
        bm = BundleMapping(bundle_name="wan_2.2_i2v", bundle_version=None, hardware=self._hw())
        assert bm.readiness_marker is None

    def test_readiness_marker_stored_when_set(self) -> None:
        marker = ReadinessMarker(node_class="WanImageToVideo")
        bm = BundleMapping(
            bundle_name="wan_2.2_i2v",
            bundle_version=None,
            hardware=self._hw(),
            readiness_marker=marker,
        )
        assert bm.readiness_marker is marker


class TestHardwareRequirementsExtended:
    def _make(self, **overrides: object) -> HardwareRequirements:
        defaults: dict[str, object] = {
            "gpu_whitelist": ("RTX_4090",),
            "min_disk_gb": 100,
            "min_network_upload_mbps": 100,
            "min_network_download_mbps": 500,
            "cuda_min_version": "12.1",
            "num_gpus": 1,
        } | overrides
        return HardwareRequirements(**defaults)  # type: ignore[arg-type]

    def test_default_template_hash_id_is_none(self) -> None:
        hw = self._make()
        assert hw.template_hash_id is None

    def test_template_hash_id_stored_when_set(self) -> None:
        hw = self._make(template_hash_id="abc123")
        assert hw.template_hash_id == "abc123"


class TestReadinessMarker:
    def test_readiness_marker_frozen(self) -> None:
        marker = ReadinessMarker(node_class="WanImageToVideo")
        with pytest.raises(dataclasses.FrozenInstanceError):
            marker.node_class = "Other"  # type: ignore[misc]

    def test_readiness_marker_slots(self) -> None:
        assert "__slots__" in ReadinessMarker.__dict__

    def test_readiness_marker_stores_node_class(self) -> None:
        marker = ReadinessMarker(node_class="WanImageToVideo")
        assert marker.node_class == "WanImageToVideo"
