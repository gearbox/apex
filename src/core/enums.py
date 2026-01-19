from enum import Enum


class ModelType(str, Enum):
    """Available model types."""

    AISHA = "aisha"
    # Future models:
    # SEEDREAM = "seedream"
    # Z_IMAGE = "z-image"


class GenerationType(str, Enum):
    """Generation type - text-to-image or image-to-image."""

    T2I = "t2i"
    I2I = "i2i"
    T2V = "t2v"
    I2V = "i2v"
    FLF2V = "flf2v"


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AspectRatio(str, Enum):
    """Supported aspect ratios."""

    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_2_3 = "2:3"
    RATIO_3_2 = "3:2"
    RATIO_21_9 = "21:9"

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
            "21:9": 21 / 9,
        }
        ratio = ratio_map[self.value]
        width = int(height * ratio)
        # Round to nearest multiple of 8
        return (width + 4) // 8 * 8
