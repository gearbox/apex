"""Tests for storage schemas."""

import pytest

from src.api.services.storage.schemas import MediaFormat, StorageType


class TestImageFormat:
    """Tests for ImageFormat enum."""

    def test_from_content_type_png(self) -> None:
        """Test PNG content type parsing."""
        assert MediaFormat.from_content_type("image/png") == MediaFormat.PNG

    def test_from_content_type_jpeg(self) -> None:
        """Test JPEG content type parsing."""
        assert MediaFormat.from_content_type("image/jpeg") == MediaFormat.JPEG
        assert MediaFormat.from_content_type("image/jpg") == MediaFormat.JPEG

    def test_from_content_type_webp(self) -> None:
        """Test WebP content type parsing."""
        assert MediaFormat.from_content_type("image/webp") == MediaFormat.WEBP

    def test_from_content_type_case_insensitive(self) -> None:
        """Test content type parsing is case insensitive."""
        assert MediaFormat.from_content_type("IMAGE/PNG") == MediaFormat.PNG
        assert MediaFormat.from_content_type("Image/Jpeg") == MediaFormat.JPEG

    def test_from_content_type_invalid(self) -> None:
        """Test invalid content type raises error."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            MediaFormat.from_content_type("image/gif")

        with pytest.raises(ValueError):
            MediaFormat.from_content_type("text/plain")

    def test_from_extension_png(self) -> None:
        """Test PNG extension parsing."""
        assert MediaFormat.from_extension("png") == MediaFormat.PNG
        assert MediaFormat.from_extension(".png") == MediaFormat.PNG
        assert MediaFormat.from_extension("PNG") == MediaFormat.PNG

    def test_from_extension_jpeg(self) -> None:
        """Test JPEG extension parsing."""
        assert MediaFormat.from_extension("jpeg") == MediaFormat.JPEG
        assert MediaFormat.from_extension("jpg") == MediaFormat.JPEG
        assert MediaFormat.from_extension(".JPG") == MediaFormat.JPEG

    def test_from_extension_webp(self) -> None:
        """Test WebP extension parsing."""
        assert MediaFormat.from_extension("webp") == MediaFormat.WEBP

    def test_from_extension_invalid(self) -> None:
        """Test invalid extension raises error."""
        with pytest.raises(ValueError, match="Unsupported extension"):
            MediaFormat.from_extension("gif")

    def test_content_type_property(self) -> None:
        """Test content_type property returns correct MIME type."""
        assert MediaFormat.PNG.content_type == "image/png"
        assert MediaFormat.JPEG.content_type == "image/jpeg"
        assert MediaFormat.WEBP.content_type == "image/webp"

    def test_extension_property(self) -> None:
        """Test extension property returns correct extension."""
        assert MediaFormat.PNG.extension == "png"
        assert MediaFormat.JPEG.extension == "jpeg"
        assert MediaFormat.WEBP.extension == "webp"


class TestStorageType:
    """Tests for StorageType enum."""

    def test_upload_value(self) -> None:
        """Test upload storage type."""
        assert StorageType.UPLOAD.value == "upload"

    def test_output_value(self) -> None:
        """Test output storage type."""
        assert StorageType.OUTPUT.value == "output"
