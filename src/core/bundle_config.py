"""Bundle configuration dataclasses for GPU provisioning.

These represent hardware requirements and model-type mappings
parsed from the ai-bundles repository's bundle.yaml files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.services.workflow.contract import BundleCapabilities


class BundleDefinitionError(ValueError):
    """A bundle.yaml file is present but semantically invalid."""


@dataclass(frozen=True, slots=True)
class ReadinessMarker:
    """Bundle-declared evidence that bundle deployment succeeded.

    Apex's GpuProvisioningWorker probes ComfyUI's /object_info endpoint and
    requires this class name to be present in the registered class list before
    transitioning the session to 'active'. Without a marker, the probe falls
    back to 200-OK reachability and emits an ERROR-level log.
    """

    node_class: str
    """ComfyUI class name registered by one of the bundle's custom nodes.

    Pick a class central to the bundle's primary workflow — one that would
    cause catastrophic workflow failure if missing. Examples: 'WanImageToVideo'
    for the WAN i2v bundle; do NOT pick a class from ComfyUI-Manager or
    generic essentials, those can register on a half-broken node.
    """


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

    template_hash_id: str | None = None
    """Vast.ai template content hash. When set, apex calls create_instance with
    template_hash_id and omits the explicit image. None means use the legacy
    image+onstart path (for backward compatibility with any unmigrated bundles)."""


@dataclass(frozen=True, slots=True)
class BundleMapping:
    """Resolved bundle for a ModelType."""

    bundle_name: str
    """ai-bundles bundle name (e.g. wan_2.2_i2v)."""

    bundle_version: str | None
    """Specific version (e.g. 260105-01). None = use 'current' symlink."""

    hardware: HardwareRequirements
    """Hardware requirements for Vast.ai search."""

    readiness_marker: ReadinessMarker | None = None
    """Optional bundle-declared readiness marker. None means fall back to 200-OK probe."""

    capabilities: BundleCapabilities | None = None
    """Bound-workflow capabilities for this resolved bundle, when indexed."""

    declared_model_bytes: int | None = None
    """Sum of this bundle's models[].files[].size_bytes across the whole bundle (F8).

    None when any declared file omits size_bytes — a partial sum would let a
    batch-headroom guard downstream pass a batch that will not actually fit, so
    "unknown" must stay unknown rather than becoming a wrong number. Fed to
    GpuSessionCommandService.enqueue_batch, which puts it on batch index 0 only.
    """
