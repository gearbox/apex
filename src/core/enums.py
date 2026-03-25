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

        if gen_type == GenerationType.T2I:
            return meta.image is not None
        if gen_type == GenerationType.I2I:
            return meta.image is not None and meta.video is None and meta.image.supports_editing
        if gen_type in (GenerationType.T2V, GenerationType.V2V, GenerationType.FLF2V):
            return meta.video is not None
        if gen_type == GenerationType.I2V:
            return meta.video is not None
        return False

    @property
    def max_concurrent_outputs(self) -> int:
        """Maximum number of outputs this model can produce per request."""
        from src.core.model_registry import get_model_meta

        return get_model_meta(self).max_concurrent_outputs


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
    # RATIO_21_9 = "21:9"

    def calculate_width(self, height: int) -> int:
        """Calculate width from height based on aspect ratio.

        Returns width rounded to nearest multiple of 8 for latent space compatibility.
        """
        ratio_map = {
            "1:1": 1.0,
            "4:3": 4 / 3,
            "3:4": 3 / 4,
            "16:9": 16 / 9,
            "9:16": 9 / 16,
            "2:3": 2 / 3,
            "3:2": 3 / 2,
            # "21:9": 21 / 9,
        }
        ratio = ratio_map[self.value]
        width = int(height * ratio)
        # Round to nearest multiple of 8
        return (width + 4) // 8 * 8


class VideoResolution(StrEnum):
    """Supported video resolutions for Grok."""

    RES_480P = "480p"
    RES_720P = "720p"


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

    SYSTEM — internal sentinel user (seeded by migration, cannot authenticate).
    ADMIN  — full administrative access to the platform.
    USER   — standard authenticated user (default for all registrations).
    """

    SYSTEM = "system"
    ADMIN = "admin"
    USER = "user"


class SupportedLocale(StrEnum):
    """Supported UI/email locales.

    Add new locales here AND create corresponding email template directories simultaneously.
    """

    EN = "en"  # English
    RU = "ru"  # Russian
    SR = "sr"  # Serbian (Latin)
