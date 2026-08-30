"""Tests for the providers discovery endpoint (v2)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest

from src.api.routes.providers import PROVIDER_DISPLAY_NAMES, _build_model_info
from src.api.schemas.providers import (
    ImageConstraints,
    ModelInfo,
    ProviderInfo,
    ProvidersResponse,
    UserContext,
    VideoConstraints,
)
from src.api.services.generation.service import GenerationService
from src.api.services.workflow.contract import (
    BoundWorkflow,
    BundleCapabilities,
    MediaSlot,
    WorkflowMap,
    WorkflowMediaInput,
    WorkflowRole,
)
from src.core.enums import (
    STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES,
    TERMINAL_GPU_SESSION_STATUSES,
    GenerationType,
    GpuSessionStatus,
    MediaKind,
    ModelSessionState,
    ModelType,
    Provider,
    ProvisioningMode,
    session_state_from_status,
)
from src.core.model_registry import get_model_meta


class TestProvidersResponseSchema:
    def test_roundtrip_empty(self) -> None:
        resp = ProvidersResponse(providers=[])
        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=ProvidersResponse)
        assert decoded.providers == []
        assert decoded.user_context is None

    def test_roundtrip_with_user_context(self) -> None:
        resp = ProvidersResponse(
            providers=[],
            user_context=UserContext(subscription_tier="pro"),
        )
        encoded = msgspec.json.encode(resp)
        decoded = msgspec.json.decode(encoded, type=ProvidersResponse)
        assert decoded.user_context is not None
        assert decoded.user_context.subscription_tier == "pro"

    def test_user_context_null_when_omitted(self) -> None:
        resp = ProvidersResponse(providers=[])
        data = msgspec.json.decode(msgspec.json.encode(resp))
        assert data["user_context"] is None


class TestModelInfoSchema:
    def test_image_model_fields(self) -> None:
        info = ModelInfo(
            model_key="grok-imagine-image",
            name="Grok Imagine Image",
            description="Flagship image gen",
            capabilities=["t2i", "i2i"],
            is_enabled=True,
            max_images=10,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1", "16:9"],
            image=ImageConstraints(output_resolutions=["1024x1024", "2048x2048"]),
            video=None,
        )
        assert info.video is None
        assert info.image is not None
        assert info.image.output_resolutions == ["1024x1024", "2048x2048"]
        assert info.image.min_height is None  # Grok: not user-controllable
        assert "t2i" in info.capabilities

    def test_aisha_model_with_height_control(self) -> None:
        info = ModelInfo(
            model_key="aisha-image",
            name="Aisha Image",
            description="ComfyUI-based generation",
            capabilities=["t2i", "i2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
            image=ImageConstraints(
                min_height=256,
                max_height=2048,
                default_height=1024,
            ),
        )
        assert info.image is not None
        assert info.image.min_height == 256
        assert info.image.max_height == 2048
        assert info.image.output_resolutions is None

    def test_video_model_fields(self) -> None:
        info = ModelInfo(
            model_key="grok-imagine-video",
            name="Grok Imagine Video",
            description="Video gen",
            capabilities=["t2v", "i2v", "v2v", "flf2v"],
            is_enabled=True,
            max_images=1,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1", "16:9"],
            video=VideoConstraints(max_duration=15, resolutions=["480p", "720p"]),
        )
        assert info.video is not None
        assert info.video.max_duration == 15

    def test_roundtrip_with_video(self) -> None:
        info = ModelInfo(
            model_key="test",
            name="Test",
            description="",
            capabilities=["t2v"],
            is_enabled=True,
            max_images=1,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1"],
            video=VideoConstraints(max_duration=15, resolutions=["480p", "720p"]),
        )
        encoded = msgspec.json.encode(info)
        decoded = msgspec.json.decode(encoded, type=ModelInfo)
        assert decoded.video is not None
        assert decoded.video.resolutions == ["480p", "720p"]

    def test_session_state_default_is_none(self) -> None:
        info = ModelInfo(
            model_key="aisha-image",
            name="Aisha",
            description="",
            capabilities=["t2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
        )
        assert info.session_state is None

    def test_session_state_explicit_value(self) -> None:
        info = ModelInfo(
            model_key="aisha-image",
            name="Aisha",
            description="",
            capabilities=["t2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
            session_state="active",
        )
        assert info.session_state == "active"

    def test_session_state_roundtrip(self) -> None:
        info = ModelInfo(
            model_key="aisha-image",
            name="Aisha",
            description="",
            capabilities=["t2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
            session_state="provisioning",
        )
        decoded = msgspec.json.decode(msgspec.json.encode(info), type=ModelInfo)
        assert decoded.session_state == "provisioning"

    @pytest.mark.parametrize("model_type", ModelType)
    def test_source_media_required_for_is_derived_from_capabilities(
        self, model_type: ModelType
    ) -> None:
        info = _build_model_info(
            model_type,
            SimpleNamespace(name="Test", description="", is_enabled=True),
            session_state=None,
        )
        source_media = info.inputs.source_media

        if source_media is None:
            assert get_model_meta(model_type).inputs.source_media is None
            return

        expected = [
            generation_type
            for generation_type in info.capabilities
            if GenerationType(generation_type).input_kinds
        ]
        assert source_media.required_for == expected
        assert set(source_media.required_for).issubset(info.capabilities)
        assert GenerationType.V2V.value not in source_media.required_for

    def test_source_media_required_for_tracks_narrowed_indexed_capabilities(self) -> None:
        media_input = WorkflowMediaInput(
            id="loader",
            class_name="LoadImage",
            input="image",
            kind=MediaKind.IMAGE,
            slot=MediaSlot.REFERENCE,
            target_role=WorkflowRole.POSITIVE_PROMPT,
            target_input="reference_image",
        )
        bound_workflow = BoundWorkflow(
            map=WorkflowMap(
                contract_version=2,
                media=MediaKind.IMAGE,
                nodes={},
                media_inputs=(media_input,),
                model_inputs=(),
            ),
            api_graph={},
        )
        capabilities = BundleCapabilities(
            media=MediaKind.IMAGE,
            generation_types=frozenset({GenerationType.I2I}),
            supports_negative_prompt=False,
            writable=frozenset(),
            max_batch_size=1,
            max_reference_images=1,
        )

        info = _build_model_info(
            ModelType.AISHA_IMAGE,
            SimpleNamespace(name="Test", description="", is_enabled=True),
            session_state=None,
            capabilities=capabilities,
            bound_workflow=bound_workflow,
        )

        assert info.capabilities == [GenerationType.I2I.value]
        assert info.inputs.source_media is not None
        assert info.inputs.source_media.required_for == [GenerationType.I2I.value]

    def test_aisha_image_lite_reports_t2i_only_no_negative_no_source_media(self) -> None:
        """Z-C1: aisha-image-lite (zit.cyberrealistic) is t2i-only, has no
        negative_prompt role, and declares no media_inputs at all."""
        bound_workflow = BoundWorkflow(
            map=WorkflowMap(
                contract_version=2,
                media=MediaKind.IMAGE,
                nodes={},
                media_inputs=(),
                model_inputs=(),
            ),
            api_graph={},
        )
        capabilities = BundleCapabilities(
            media=MediaKind.IMAGE,
            generation_types=frozenset({GenerationType.T2I}),
            supports_negative_prompt=False,
            writable=frozenset(
                {
                    "latent.width",
                    "latent.height",
                    "latent.batch_size",
                    "positive_prompt.text",
                    "sampler.seed",
                    "sampler.steps",
                    "sampler.cfg",
                    "sampler.sampler",
                    "sampler.scheduler",
                    "sampler.denoise",
                    "save.filename_prefix",
                }
            ),
            max_batch_size=4,
            max_reference_images=0,
        )

        info = _build_model_info(
            ModelType.AISHA_IMAGE_LITE,
            SimpleNamespace(name="Aisha Image Lite", description="", is_enabled=True),
            session_state=None,
            capabilities=capabilities,
            bound_workflow=bound_workflow,
        )

        assert info.capabilities == [GenerationType.T2I.value]
        assert info.supports_negative_prompt is False
        assert info.inputs.source_media is None
        assert "negative_prompt" in info.unsupported_parameters


class TestProviderInfoSchema:
    def test_provider_with_models(self) -> None:
        model = ModelInfo(
            model_key="aisha-image",
            name="Aisha Image",
            description="ComfyUI",
            capabilities=["t2i", "i2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
        )
        provider = ProviderInfo(
            provider="aisha",
            name="Aisha",
            available=True,
            provisioning_mode="on_demand",
            models=[model],
        )
        assert len(provider.models) == 1
        assert provider.models[0].supports_negative_prompt is True
        assert provider.provisioning_mode == "on_demand"

    def test_provider_with_empty_models(self) -> None:
        provider = ProviderInfo(
            provider="grok",
            name="xAI Grok",
            available=False,
            provisioning_mode="always_on",
            models=[],
        )
        assert provider.available is False
        assert provider.models == []
        assert provider.provisioning_mode == "always_on"

    def test_provisioning_mode_roundtrip(self) -> None:
        provider = ProviderInfo(
            provider="grok",
            name="xAI Grok",
            available=True,
            provisioning_mode="always_on",
            models=[],
        )
        decoded = msgspec.json.decode(msgspec.json.encode(provider), type=ProviderInfo)
        assert decoded.provisioning_mode == "always_on"


class TestRequiresAgeVerification:
    def test_default_is_false(self) -> None:
        info = ModelInfo(
            model_key="grok-imagine-image",
            name="Grok",
            description="",
            capabilities=["t2i"],
            is_enabled=True,
            max_images=10,
            max_prompt_length=4096,
            supports_negative_prompt=False,
            aspect_ratios=["1:1"],
        )
        assert info.requires_age_verification is False

    def test_aisha_models_have_flag_true_via_registry(self) -> None:
        for mt in [ModelType.AISHA_IMAGE, ModelType.AISHA_VIDEO]:
            meta = get_model_meta(mt)
            assert meta.requires_age_verification is True

    def test_grok_models_have_flag_false_via_registry(self) -> None:
        grok_models = [
            ModelType.GROK_IMAGINE_IMAGE,
            ModelType.GROK_2_IMAGE,
            ModelType.GROK_IMAGINE_VIDEO,
        ]
        for mt in grok_models:
            meta = get_model_meta(mt)
            assert meta.requires_age_verification is False

    def test_roundtrip_preserves_flag(self) -> None:
        info = ModelInfo(
            model_key="aisha-image",
            name="Aisha",
            description="",
            capabilities=["t2i"],
            is_enabled=True,
            max_images=4,
            max_prompt_length=4096,
            supports_negative_prompt=True,
            aspect_ratios=["1:1"],
            requires_age_verification=True,
        )
        decoded = msgspec.json.decode(msgspec.json.encode(info), type=ModelInfo)
        assert decoded.requires_age_verification is True


class TestCapabilitiesDerivation:
    """Verify capabilities list matches ModelType enum properties."""

    def test_grok_imagine_image_capabilities(self) -> None:
        mt = ModelType.GROK_IMAGINE_IMAGE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i", "i2i"]

    def test_grok_2_image_capabilities(self) -> None:
        mt = ModelType.GROK_2_IMAGE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i"]

    def test_grok_imagine_video_capabilities(self) -> None:
        mt = ModelType.GROK_IMAGINE_VIDEO
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2v", "i2v", "v2v"]

    def test_aisha_image_capabilities(self) -> None:
        mt = ModelType.AISHA_IMAGE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i", "i2i"]

    def test_aisha_video_capabilities(self) -> None:
        mt = ModelType.AISHA_VIDEO
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2v", "i2v", "flf2v"]

    def test_aisha_image_lite_capabilities(self) -> None:
        mt = ModelType.AISHA_IMAGE_LITE
        caps = [gt.value for gt in GenerationType if mt.supports_generation_type(gt)]
        assert caps == ["t2i"]

    def test_aspect_ratios_are_valid_enum_members(self) -> None:
        """Every model's aspect_ratios must be valid AspectRatio enum values."""
        from src.core.enums import AspectRatio

        all_values = {ar.value for ar in AspectRatio}
        for mt in ModelType:
            meta = get_model_meta(mt)
            model_values = {ar.value for ar in meta.aspect_ratios}
            assert model_values.issubset(all_values), (
                f"{mt.value} has invalid aspect ratios: {model_values - all_values}"
            )
            assert len(meta.aspect_ratios) > 0, f"{mt.value} has no aspect ratios"


class TestSessionStateFromStatus:
    """Exhaustive mapping of every GpuSessionStatus → ModelSessionState."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (GpuSessionStatus.active, ModelSessionState.ACTIVE),
            (GpuSessionStatus.pending, ModelSessionState.PROVISIONING),
            (GpuSessionStatus.provisioning, ModelSessionState.PROVISIONING),
            (GpuSessionStatus.resuming, ModelSessionState.PROVISIONING),
            (GpuSessionStatus.paused, ModelSessionState.PAUSED),
            (GpuSessionStatus.stale, ModelSessionState.STALE),
            (GpuSessionStatus.stopping, ModelSessionState.STOPPING),
            (GpuSessionStatus.stopped, ModelSessionState.NONE),
            (GpuSessionStatus.failed, ModelSessionState.NONE),
        ],
    )
    def test_mapping(self, status: GpuSessionStatus, expected: ModelSessionState) -> None:
        assert session_state_from_status(status) is expected

    def test_all_statuses_covered(self) -> None:
        """Fails if a new GpuSessionStatus member is added without a mapping."""
        for status in GpuSessionStatus:
            result = session_state_from_status(status)
            assert isinstance(result, ModelSessionState)


class TestGpuSessionStatusSets:
    """Invariants for TERMINAL_GPU_SESSION_STATUSES and STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES.

    These lock the membership that prevents double-start: `stopping` must never
    be terminal at the repo layer (or a second session can be created mid-teardown).
    """

    def test_stopping_not_in_terminal(self) -> None:
        assert GpuSessionStatus.stopping not in TERMINAL_GPU_SESSION_STATUSES

    def test_stopping_in_stopping_or_terminal(self) -> None:
        assert GpuSessionStatus.stopping in STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES

    def test_stopped_and_failed_in_terminal(self) -> None:
        assert GpuSessionStatus.stopped in TERMINAL_GPU_SESSION_STATUSES
        assert GpuSessionStatus.failed in TERMINAL_GPU_SESSION_STATUSES

    def test_stopping_or_terminal_is_superset(self) -> None:
        assert (
            TERMINAL_GPU_SESSION_STATUSES | {GpuSessionStatus.stopping}
            == STOPPING_OR_TERMINAL_GPU_SESSION_STATUSES
        )


class TestProviderTableCompleteness:
    """Guard: every Provider member must have an entry in both per-provider tables."""

    def test_every_provider_has_provisioning_mode(self) -> None:
        for p in Provider:
            # Raises RuntimeError if undeclared — that's a test failure
            mode = p.provisioning_mode
            assert isinstance(mode, ProvisioningMode), f"{p.value} returned non-ProvisioningMode"

    def test_every_provider_has_display_name(self) -> None:
        for p in Provider:
            assert p in PROVIDER_DISPLAY_NAMES, (
                f"Provider {p.value!r} has no entry in PROVIDER_DISPLAY_NAMES"
            )
            assert PROVIDER_DISPLAY_NAMES[p], f"Provider {p.value!r} has an empty display name"

    def test_known_provisioning_modes(self) -> None:
        assert Provider.GROK.provisioning_mode is ProvisioningMode.ALWAYS_ON
        assert Provider.AISHA.provisioning_mode is ProvisioningMode.ON_DEMAND


class TestConfiguredProvidersAccessor:
    """GenerationService.configured_providers returns the exact registered key set."""

    def test_empty_registry(self) -> None:
        svc = GenerationService(
            providers={},
            billing_service=MagicMock(),
            pricing_service=MagicMock(),
            rate_limiter=MagicMock(),
        )
        assert svc.configured_providers == frozenset()

    def test_single_provider(self) -> None:
        stub = MagicMock()
        svc = GenerationService(
            providers={Provider.GROK: stub},
            billing_service=MagicMock(),
            pricing_service=MagicMock(),
            rate_limiter=MagicMock(),
        )
        assert svc.configured_providers == frozenset({Provider.GROK})

    def test_multiple_providers(self) -> None:
        svc = GenerationService(
            providers={Provider.GROK: MagicMock(), Provider.AISHA: MagicMock()},
            billing_service=MagicMock(),
            pricing_service=MagicMock(),
            rate_limiter=MagicMock(),
        )
        assert svc.configured_providers == frozenset({Provider.GROK, Provider.AISHA})

    def test_returns_frozenset(self) -> None:
        svc = GenerationService(
            providers={Provider.GROK: MagicMock()},
            billing_service=MagicMock(),
            pricing_service=MagicMock(),
            rate_limiter=MagicMock(),
        )
        result = svc.configured_providers
        assert isinstance(result, frozenset)


class TestNoHardcodedProviderNames:
    """Regression: providers.py must not branch on provider names in logic."""

    def test_no_provider_literals_in_route_module(self) -> None:
        """Module-level explicit lookup tables (PROVIDER_DISPLAY_NAMES) are allowed;
        provider literals must not appear as discriminators inside function bodies."""
        import ast
        import pathlib

        source = pathlib.Path("src/api/routes/providers.py").read_text()
        tree = ast.parse(source)

        # Collect line ranges covered by function/async-function bodies so we can
        # restrict the check to function scope (dict keys in module-level constants are fine).
        function_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        function_lines.add(getattr(child, "lineno"))  # noqa: B009

        for node in ast.walk(tree):
            if not hasattr(node, "lineno") or getattr(node, "lineno") not in function_lines:  # noqa: B009
                continue
            if isinstance(node, ast.Constant) and node.value in {"grok", "aisha"}:
                raise AssertionError(
                    f"Literal {node.value!r} found inside a function in providers.py "
                    f"at line {node.lineno}; no provider name may be used as a discriminator"
                )
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "Provider"
                and node.attr in {"GROK", "AISHA"}
            ):
                raise AssertionError(
                    f"Provider.{node.attr} reference found inside a function in providers.py "
                    f"at line {node.lineno}; no provider name may be used as a discriminator"
                )
