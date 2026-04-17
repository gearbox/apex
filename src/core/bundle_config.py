"""Bundle configuration dataclasses for GPU provisioning.

These represent hardware requirements and model-type mappings
parsed from the ai-bundles repository's bundle.yaml files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareRequirements:
    """GPU hardware requirements parsed from bundle.yaml."""

    gpu_whitelist: tuple[str, ...]
    """Vast.ai gpu_name values (e.g. RTX_4090, A100_SXM4)."""

    min_disk_gb: int
    """Minimum disk space in GB for models + ComfyUI + workspace."""

    min_network_upload_mbps: int
    """Minimum upload speed in Mbps."""

    min_network_download_mbps: int
    """Minimum download speed in Mbps — critical for model downloads."""

    cuda_min_version: str
    """Minimum CUDA version string (e.g. '12.1')."""

    num_gpus: int
    """Number of GPUs required."""

    comfyui_port: int = 18188
    """ComfyUI listen port inside the container."""


@dataclass(frozen=True, slots=True)
class BundleMapping:
    """Resolved bundle for a ModelType."""

    bundle_name: str
    """ai-bundles bundle name (e.g. wan_2.2_i2v)."""

    bundle_version: str | None
    """Specific version (e.g. 260105-01). None = use 'current' symlink."""

    hardware: HardwareRequirements
    """Hardware requirements for Vast.ai search."""
