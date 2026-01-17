"""Tests for storage schemas."""

import pytest

from src.api.services.storage.schemas import ImageFormat, StorageType


class TestImageFormat:
    """Tests for ImageFormat enum."""

    def test_from_content_type_png(self) -> None:
        """Test PNG content type parsing."""
        assert ImageFormat.from_content_type("image/png") == ImageFormat.PNG

    def test_from_content_type_jpeg(self) -> None:
        """Test JPEG content type parsing."""
        assert ImageFormat.from_content_type("image/jpeg") == ImageFormat.JPEG
        assert ImageFormat.from_content_type("image/jpg") == ImageFormat.JPEG

    def test_from_content_type_webp(self) -> None:
        """Test WebP content type parsing."""
        assert ImageFormat.from_content_type("image/webp") == ImageFormat.WEBP

    def test_from_content_type_case_insensitive(self) -> None:
        """Test content type parsing is case insensitive."""
        assert ImageFormat.from_content_type("IMAGE/PNG") == ImageFormat.PNG
        assert ImageFormat.from_content_type("Image/Jpeg") == ImageFormat.JPEG

    def test_from_content_type_invalid(self) -> None:
        """Test invalid content type raises error."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            ImageFormat.from_content_type("image/gif")

        with pytest.raises(ValueError):
            ImageFormat.from_content_type("text/plain")

    def test_from_extension_png(self) -> None:
        """Test PNG extension parsing."""
        assert ImageFormat.from_extension("png") == ImageFormat.PNG
        assert ImageFormat.from_extension(".png") == ImageFormat.PNG
        assert ImageFormat.from_extension("PNG") == ImageFormat.PNG

    def test_from_extension_jpeg(self) -> None:
        """Test JPEG extension parsing."""
        assert ImageFormat.from_extension("jpeg") == ImageFormat.JPEG
        assert ImageFormat.from_extension("jpg") == ImageFormat.JPEG
        assert ImageFormat.from_extension(".JPG") == ImageFormat.JPEG

    def test_from_extension_webp(self) -> None:
        """Test WebP extension parsing."""
        assert ImageFormat.from_extension("webp") == ImageFormat.WEBP

    def test_from_extension_invalid(self) -> None:
        """Test invalid extension raises error."""
        with pytest.raises(ValueError, match="Unsupported extension"):
            ImageFormat.from_extension("gif")

    def test_content_type_property(self) -> None:
        """Test content_type property returns correct MIME type."""
        assert ImageFormat.PNG.content_type == "image/png"
        assert ImageFormat.JPEG.content_type == "image/jpeg"
        assert ImageFormat.WEBP.content_type == "image/webp"

    def test_extension_property(self) -> None:
        """Test extension property returns correct extension."""
        assert ImageFormat.PNG.extension == "png"
        assert ImageFormat.JPEG.extension == "jpeg"
        assert ImageFormat.WEBP.extension == "webp"


class TestStorageType:
    """Tests for StorageType enum."""

    def test_upload_value(self) -> None:
        """Test upload storage type."""
        assert StorageType.UPLOAD.value == "upload"

    def test_output_value(self) -> None:
        """Test output storage type."""
        assert StorageType.OUTPUT.value == "output"
