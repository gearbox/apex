from __future__ import annotations

from enum import StrEnum


class Product(StrEnum):
    """Product identifiers."""

    VEX = "vex"
    SYNTHARA = "synthara"


class Provider(StrEnum):
    """Generation provider."""

    AISHA = "aisha"
    GROK = "grok"

    @property
    def provisioning_mode(self) -> ProvisioningMode:
        """Provisioning mode for this provider. Raises if undeclared."""
        try:
            return _PROVISIONING_MODE_BY_PROVIDER[self]
        except KeyError as exc:  # pragma: no cover - guarded by completeness test
            raise RuntimeError(
                f"No provisioning mode declared for provider {self.value!r}; "
                f"add an entry to _PROVISIONING_MODE_BY_PROVIDER"
            ) from exc


class ProvisioningMode(StrEnum):
    """How a provider's compute is made available."""

    ALWAYS_ON = "always_on"  # cloud API; usable whenever the provider is configured
    ON_DEMAND = "on_demand"  # requires a per-user GPU session before generation


_PROVISIONING_MODE_BY_PROVIDER: dict[Provider, ProvisioningMode] = {
    Provider.AISHA: ProvisioningMode.ON_DEMAND,
    Provider.GROK: ProvisioningMode.ALWAYS_ON,
}


class ModelType(StrEnum):
    """Available model types."""

    AISHA_IMAGE = "aisha-image"
    AISHA_VIDEO = "aisha-video"

    # Grok models
    GROK_IMAGINE_IMAGE = "grok-imagine-image"  # T2I, I2I (editing)
    GROK_2_IMAGE = "grok-2-image-1212"  # T2I only (older, different pricing)
    GROK_IMAGINE_VIDEO = "grok-imagine-video"  # T2V, I2V, V2V, FLF2V

    @property
    def provider(self) -> Provider:
        """Get the provider for this model type."""
        from src.core.model_registry import get_model_meta

        return get_model_meta(self).provider

    @property
    def supports_image_input(self) -> bool:
        """Check if this model supports image input (I2I/I2V)."""
        return self.supports_generation_type(GenerationType.I2I) or self.supports_generation_type(
            GenerationType.I2V
        )

    @property
    def is_video_model(self) -> bool:
        """Check if this model generates video."""
        from src.core.model_registry import get_model_meta

        return get_model_meta(self).video is not None

    def supports_generation_type(self, gen_type: GenerationType) -> bool:
        """Check if this model supports the given generation type.

        Derives compatibility from registry metadata rather than
        hardcoded member lists.
        """
        from src.core.model_registry import get_model_meta

        meta = get_model_meta(self)

        if not gen_type.is_video:
            return meta.image is not None and gen_type in meta.image.supported_types
        return meta.video is not None and gen_type in meta.video.supported_types

    @property
    def max_concurrent_outputs(self) -> int:
        """Maximum number of outputs this model can produce per request."""
        from src.core.model_registry import get_model_meta

        return get_model_meta(self).max_concurrent_outputs

    @property
    def requires_age_verification(self) -> bool:
        """Check if this model requires age verification before generation."""
        from src.core.model_registry import get_model_meta

        return get_model_meta(self).requires_age_verification


class GenerationType(StrEnum):
    """Generation type - text-to-image or image-to-image."""

    T2I = "t2i"
    I2I = "i2i"
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"  # Video-to-video (editing)
    FLF2V = "flf2v"  # First-last-frame to video

    @property
    def is_video(self) -> bool:
        """Check if this is a video generation type."""
        return self in (
            GenerationType.T2V,
            GenerationType.I2V,
            GenerationType.V2V,
            GenerationType.FLF2V,
        )

    @property
    def requires_image_input(self) -> bool:
        """Check if this generation type requires an input image."""
        return self in (GenerationType.I2I, GenerationType.I2V, GenerationType.FLF2V)

    @property
    def requires_video_input(self) -> bool:
        """Check if this generation type requires an input video."""
        return self == GenerationType.V2V


class JobStatus(StrEnum):
    """Job execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MODERATED = "moderated"


class AspectRatio(StrEnum):
    """Supported aspect ratios."""

    RATIO_2_3 = "2:3"
    RATIO_3_2 = "3:2"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"
    RATIO_3_4 = "3:4"
    RATIO_4_3 = "4:3"

    def calculate_width(self, height: int) -> int:
        """Calculate width from height based on aspect ratio.

        Returns width rounded to nearest multiple of 8 for latent space compatibility.
        """
        rw, rh = self.as_fraction()
        width = int(height * rw / rh)
        return (width + 4) // 8 * 8

    def as_fraction(self) -> tuple[int, int]:
        """Return (w, h) integer ratio components, e.g. RATIO_3_4 -> (3, 4)."""
        w_str, h_str = self.value.split(":")
        return int(w_str), int(h_str)


class VideoResolution(StrEnum):
    """Supported video resolutions for Grok."""

    RES_480P = "480p"
    RES_720P = "720p"


class Resolution(StrEnum):
    """Image quality tier (maps to a target megapixel budget; see core.resolution)."""

    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"


class Sampler(StrEnum):
    """ComfyUI sampler names (common subset)."""

    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    EULER_CFG_PP = "euler_cfg_pp"
    HEUN = "heun"
    DPM_2 = "dpm_2"
    DPM_2_ANCESTRAL = "dpm_2_ancestral"
    LMS = "lms"
    DPMPP_2S_ANCESTRAL = "dpmpp_2s_ancestral"
    DPMPP_SDE = "dpmpp_sde"
    DPMPP_2M = "dpmpp_2m"
    DPMPP_2M_SDE = "dpmpp_2m_sde"
    DPMPP_3M_SDE = "dpmpp_3m_sde"
    DDIM = "ddim"
    UNI_PC = "uni_pc"
    UNI_PC_BH2 = "uni_pc_bh2"
    LCM = "lcm"
    RES_MULTISTEP = "res_multistep"


class Scheduler(StrEnum):
    """ComfyUI scheduler names (common subset)."""

    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"
    BETA = "beta"
    LINEAR_QUADRATIC = "linear_quadratic"
    KL_OPTIMAL = "kl_optimal"


class MediaFormat(StrEnum):
    """Supported media formats."""

    # Image formats
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    # Video formats
    MP4 = "mp4"

    @classmethod
    def from_content_type(cls, content_type: str) -> MediaFormat:
        """Get format from MIME type."""
        mapping = {
            "image/png": cls.PNG,
            "image/jpeg": cls.JPEG,
            "image/jpg": cls.JPEG,
            "image/webp": cls.WEBP,
        }
        fmt = mapping.get(content_type.lower())
        if fmt is None:
            raise ValueError(f"Unsupported content type: {content_type}")
        return fmt

    @classmethod
    def from_extension(cls, ext: str) -> MediaFormat:
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "png": cls.PNG,
            "jpeg": cls.JPEG,
            "jpg": cls.JPEG,
            "webp": cls.WEBP,
        }
        fmt = mapping.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported extension: {ext}")
        return fmt

    @property
    def content_type(self) -> str:
        """Get MIME type for format."""
        return {
            self.PNG: "image/png",
            self.JPEG: "image/jpeg",
            self.WEBP: "image/webp",
            self.MP4: "video/mp4",
        }[self]

    @property
    def extension(self) -> str:
        """Get file extension for format."""
        return self.value

    @property
    def is_video(self) -> bool:
        """Check if this is a video format."""
        return self == MediaFormat.MP4


class OutputMediaType(StrEnum):
    """Media type classification for gallery filtering."""

    IMAGE = "image"
    VIDEO = "video"


class GalleryBadge(StrEnum):
    """Badge type for gallery grid — describes the input source."""

    IMAGE = "image"
    PROMPT = "prompt"


class GallerySourceType(StrEnum):
    """Type of input source for lineage display."""

    UPLOAD = "upload"
    GENERATION = "generation"


class SubscriptionTier(StrEnum):
    """User subscription tiers."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class AccountType(StrEnum):
    """Token account types."""

    PERSONAL = "personal"
    ENTERPRISE = "enterprise"


class TransactionType(StrEnum):
    """Token transaction types."""

    CREDIT = "credit"
    DEBIT = "debit"
    REFUND = "refund"
    ADMIN_ADJUSTMENT = "admin_adjustment"


class PaymentStatus(StrEnum):
    """Payment processing status."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class OrgRole(StrEnum):
    """Organization member roles."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class UserRole(StrEnum):
    """User account roles.

    SYSTEM     — internal sentinel user (seeded by migration, cannot authenticate).
    SUPERADMIN — full administrative access including role management.
    ADMIN      — administrative access to the platform (no role escalation).
    USER       — standard authenticated user (default for all registrations).
    """

    SYSTEM = "system"
    SUPERADMIN = "superadmin"
    ADMIN = "admin"
    USER = "user"


class AdminPermission(StrEnum):
    """Granular permissions grantable to admin-role users by superadmins.

    BILLING_ADJUST — allows use of POST /v1/admin/accounts/{id}/adjust.
    """

    BILLING_ADJUST = "billing_adjust"


class NotificationLevel(StrEnum):
    """Severity level for system notifications and credit warnings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ComponentStatus(StrEnum):
    """Health status of a single component."""

    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"
    unknown = "unknown"
    inactive = "inactive"


class ComponentCategory(StrEnum):
    """Taxonomy bucket for a health component."""

    infrastructure = "infrastructure"
    cloud_provider = "cloud_provider"
    platform_api = "platform_api"
    gpu_session = "gpu_session"


class GpuSessionStatus(StrEnum):
    """GPU session lifecycle states."""

    pending = "pending"  # session requested, node not yet provisioned
    provisioning = "provisioning"  # Vast.ai node is starting up
    active = "active"  # node is up, ComfyUI is reachable
    stale = "stale"  # node was active but is now unreachable
    paused = "paused"  # user paused — Vast.ai instance stopped, disk retained
    resuming = "resuming"  # user resumed — Vast.ai instance restarting
    stopping = "stopping"  # user requested stop, teardown in progress
    stopped = "stopped"  # session ended normally
    failed = "failed"  # provisioning or runtime failure


# True lifecycle end. A session in one of these states no longer exists for
# routing purposes and FREES the (user, product, model_type) slot, so a new
# session may be started. `stopping` is deliberately EXCLUDED — teardown is in
# progress and the slot is still occupied.
TERMINAL_GPU_SESSION_STATUSES: frozenset[GpuSessionStatus] = frozenset(
    {GpuSessionStatus.stopped, GpuSessionStatus.failed}
)

# Teardown-in-progress OR ended. Used where `stop()` must be idempotent and where
# provisioning transitions must abort. Does NOT free the slot. This is the strict
# superset that ADDS `stopping`.
STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES: frozenset[GpuSessionStatus] = (
    TERMINAL_GPU_SESSION_STATUSES | {GpuSessionStatus.stopping}
)


class ModelSessionState(StrEnum):
    """Per-user readiness of an on-demand model, derived from GpuSessionStatus."""

    NONE = "none"  # no session occupying this model's slot
    PROVISIONING = "provisioning"  # pending / provisioning / resuming
    ACTIVE = "active"  # session active — ready to generate
    PAUSED = "paused"  # paused — needs resume
    STALE = "stale"  # was active, now unreachable
    STOPPING = "stopping"  # teardown in progress — not usable, cannot start a new one yet


def session_state_from_status(status: GpuSessionStatus) -> ModelSessionState:
    """Map a GpuSessionStatus to the UI-facing ModelSessionState."""
    match status:
        case GpuSessionStatus.active:
            return ModelSessionState.ACTIVE
        case GpuSessionStatus.pending | GpuSessionStatus.provisioning | GpuSessionStatus.resuming:
            return ModelSessionState.PROVISIONING
        case GpuSessionStatus.paused:
            return ModelSessionState.PAUSED
        case GpuSessionStatus.stale:
            return ModelSessionState.STALE
        case GpuSessionStatus.stopping:
            return ModelSessionState.STOPPING
        case GpuSessionStatus.stopped | GpuSessionStatus.failed:
            return ModelSessionState.NONE


class SupportedLocale(StrEnum):
    """Supported UI/email locales.

    Add new locales here AND create corresponding email template directories simultaneously.
    """

    EN = "en"  # English
    RU = "ru"  # Russian
    SR = "sr"  # Serbian (Latin)
