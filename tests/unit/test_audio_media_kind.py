"""``MediaKind.AUDIO`` plumbing (S2a): formats, content types, aliases and per-kind upload limits."""

from __future__ import annotations

import pytest

from src.api.services.library_capabilities import LibraryAction, resolve_library_actions
from src.api.services.storage.schemas import (
    ALLOWED_AUDIO_CONTENT_TYPES,
    ALLOWED_CLIENT_UPLOAD_CONTENT_TYPES,
    ALLOWED_IMAGE_CONTENT_TYPES,
    ALLOWED_UPLOAD_CONTENT_TYPES,
    ALLOWED_VIDEO_CONTENT_TYPES,
    AUDIO_CLIENT_CONTENT_TYPE_ALIASES,
)
from src.core.enums import (
    MediaFormat,
    MediaHashMediaType,
    MediaKind,
    UnknownMediaTypeError,
    media_kind_from_content_type,
)
from src.core.library_ref import LibraryAssetSource
from src.core.product_registry import VEX_CONFIG
from src.core.upload_limits import MIB, upload_size_limit_bytes
from tests.unit.helpers import hermetic_settings

pytestmark = pytest.mark.unit

_AUDIO_FORMATS = {
    MediaFormat.MP3: "audio/mpeg",
    MediaFormat.M4A: "audio/mp4",
    MediaFormat.WAV: "audio/wav",
    MediaFormat.FLAC: "audio/flac",
}


class TestMediaFormat:
    @pytest.mark.parametrize(("media_format", "content_type"), list(_AUDIO_FORMATS.items()))
    def test_each_audio_format_has_its_content_type_and_round_trips(
        self, media_format: MediaFormat, content_type: str
    ) -> None:
        assert media_format.content_type == content_type
        assert MediaFormat.from_content_type(content_type) is media_format
        assert MediaFormat.from_extension(media_format.value) is media_format
        assert media_format.extension == media_format.value

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("audio/mp3", MediaFormat.MP3),
            ("audio/x-m4a", MediaFormat.M4A),
            ("audio/aac", MediaFormat.M4A),
            ("audio/x-wav", MediaFormat.WAV),
            ("audio/wave", MediaFormat.WAV),
            ("audio/x-flac", MediaFormat.FLAC),
            ("AUDIO/MPEG", MediaFormat.MP3),
        ],
    )
    def test_client_aliases_are_accepted(self, alias: str, expected: MediaFormat) -> None:
        assert MediaFormat.from_content_type(alias) is expected

    @pytest.mark.parametrize("content_type", ["audio/ogg", "audio/opus", "audio/aiff", "audio/"])
    def test_other_audio_types_are_not_formats(self, content_type: str) -> None:
        with pytest.raises(ValueError, match="Unsupported content type"):
            MediaFormat.from_content_type(content_type)

    def test_kind_of_every_format(self) -> None:
        expected = {
            MediaFormat.PNG: MediaKind.IMAGE,
            MediaFormat.JPEG: MediaKind.IMAGE,
            MediaFormat.WEBP: MediaKind.IMAGE,
            MediaFormat.MP4: MediaKind.VIDEO,
            MediaFormat.WEBM: MediaKind.VIDEO,
            MediaFormat.MOV: MediaKind.VIDEO,
            **dict.fromkeys(_AUDIO_FORMATS, MediaKind.AUDIO),
        }
        assert {fmt: fmt.kind for fmt in MediaFormat} == expected

    def test_is_image_no_longer_means_not_video(self) -> None:
        for media_format in _AUDIO_FORMATS:
            assert media_format.is_image is False
            assert media_format.is_video is False
            assert media_format.is_audio is True
        assert MediaFormat.PNG.is_image and not MediaFormat.PNG.is_video
        assert MediaFormat.MP4.is_video and not MediaFormat.MP4.is_image
        assert not MediaFormat.MP4.is_audio

    def test_a_video_container_and_an_audio_container_are_not_confused(self) -> None:
        assert MediaFormat.from_content_type("video/mp4") is MediaFormat.MP4
        assert MediaFormat.from_content_type("audio/mp4") is MediaFormat.M4A


class TestMediaKindFromContentType:
    @pytest.mark.parametrize("content_type", sorted(ALLOWED_AUDIO_CONTENT_TYPES))
    def test_stored_audio_types_are_audio(self, content_type: str) -> None:
        assert media_kind_from_content_type(content_type) is MediaKind.AUDIO

    def test_the_other_kinds_are_unchanged(self) -> None:
        assert media_kind_from_content_type("image/png") is MediaKind.IMAGE
        assert media_kind_from_content_type("video/mp4") is MediaKind.VIDEO

    def test_an_unknown_family_is_still_an_error(self) -> None:
        with pytest.raises(UnknownMediaTypeError):
            media_kind_from_content_type("application/pdf")

    def test_the_perceptual_hash_ledger_has_no_audio_type(self) -> None:
        """Audio gets no PDQ ledger row: ``MediaHashMediaType`` is unchanged."""
        assert {member.value for member in MediaHashMediaType} == {"image", "video"}


class TestAllowedContentTypes:
    def test_the_four_stored_audio_types(self) -> None:
        assert {
            "audio/mpeg",
            "audio/mp4",
            "audio/wav",
            "audio/flac",
        } == ALLOWED_AUDIO_CONTENT_TYPES

    def test_the_storage_gate_includes_audio_and_nothing_else_new(self) -> None:
        assert ALLOWED_UPLOAD_CONTENT_TYPES == (
            ALLOWED_IMAGE_CONTENT_TYPES | ALLOWED_VIDEO_CONTENT_TYPES | ALLOWED_AUDIO_CONTENT_TYPES
        )

    def test_the_client_gate_also_takes_the_aliases(self) -> None:
        assert {
            "audio/mp3",
            "audio/x-m4a",
            "audio/aac",
            "audio/x-wav",
            "audio/wave",
            "audio/x-flac",
        } == AUDIO_CLIENT_CONTENT_TYPE_ALIASES
        assert AUDIO_CLIENT_CONTENT_TYPE_ALIASES <= ALLOWED_CLIENT_UPLOAD_CONTENT_TYPES
        assert not AUDIO_CLIENT_CONTENT_TYPE_ALIASES & ALLOWED_UPLOAD_CONTENT_TYPES  # not stored

    def test_every_client_alias_resolves_to_a_stored_type(self) -> None:
        for alias in AUDIO_CLIENT_CONTENT_TYPE_ALIASES:
            assert MediaFormat.from_content_type(alias).content_type in ALLOWED_AUDIO_CONTENT_TYPES


class TestUploadLimits:
    def test_defaults_are_20_mb_for_images_and_videos_and_80_mb_for_audio(self) -> None:
        settings = hermetic_settings()
        assert upload_size_limit_bytes(MediaKind.IMAGE, settings) == 20 * MIB
        assert upload_size_limit_bytes(MediaKind.VIDEO, settings) == 20 * MIB
        assert upload_size_limit_bytes(MediaKind.AUDIO, settings) == 80 * MIB

    def test_each_kind_reads_its_own_setting(self) -> None:
        settings = hermetic_settings(max_upload_size_mb=30, audio_upload_max_mb=60)
        assert upload_size_limit_bytes(MediaKind.IMAGE, settings) == 30 * MIB
        assert upload_size_limit_bytes(MediaKind.VIDEO, settings) == 30 * MIB
        assert upload_size_limit_bytes(MediaKind.AUDIO, settings) == 60 * MIB

    @pytest.mark.parametrize(("image_mb", "audio_mb"), [(20, 80), (100, 80), (50, 50), (1, 1)])
    def test_the_request_body_limit_is_the_largest_per_kind_limit(
        self, image_mb: int, audio_mb: int
    ) -> None:
        settings = hermetic_settings(max_upload_size_mb=image_mb, audio_upload_max_mb=audio_mb)
        largest = max(upload_size_limit_bytes(kind, settings) for kind in MediaKind)
        assert settings.max_request_body_bytes == largest == max(image_mb, audio_mb) * MIB

    @pytest.mark.parametrize("value", [0, -1, 96, 1000])
    def test_the_audio_limit_stays_under_the_proxy_cap(self, value: int) -> None:
        with pytest.raises(ValueError, match="audio_upload_max_mb"):
            hermetic_settings(audio_upload_max_mb=value)

    def test_the_audio_limit_may_be_95(self) -> None:
        assert hermetic_settings(audio_upload_max_mb=95).audio_upload_max_bytes == 95 * MIB

    def test_the_app_uses_the_largest_limit_as_its_body_size(self) -> None:
        import importlib
        import inspect

        module = importlib.import_module("src.api.app")  # the package re-exports ``app`` itself
        source = inspect.getsource(module.create_app)
        assert "request_max_body_size=settings.max_request_body_bytes" in source


class TestLibraryActions:
    def test_audio_has_only_the_always_on_actions(self) -> None:
        actions = resolve_library_actions(
            media_type=MediaKind.AUDIO,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=VEX_CONFIG,
        )
        assert actions == (
            LibraryAction.FAVORITE,
            LibraryAction.RENAME,
            LibraryAction.DOWNLOAD,
            LibraryAction.DELETE,
        )
