"""Tests locking the ValueError → HTTP 400 contract on the unified generation endpoint.

F5 from review: confirm that schema-level and provider-level validation errors
always produce 400 and never leave a billing reservation on the ledger.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import msgspec
import pytest

from src.api.routes.unified_generation import UnifiedGenerationController
from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.service import GenerationError, GenerationService
from src.core.enums import (
    GenerationType,
    ModelType,
    Resolution,
    Sampler,
    Scheduler,
)
from src.core.generation_config import (
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gen_config(
    *,
    min_steps: int = 1,
    max_steps: int = 30,
    min_cfg: float = 0.0,
    max_cfg: float = 15.0,
    allowed_samplers: frozenset[Sampler] = frozenset(),
) -> BundleGenerationConfig:
    return BundleGenerationConfig(
        defaults=GenerationDefaults(
            resolution=Resolution.STANDARD,
            steps=12,
            cfg=1.1,
            sampler=Sampler.EULER,
            scheduler=Scheduler.BETA,
            denoise=1.0,
        ),
        constraints=GenerationConstraints(
            max_megapixels=1.05,
            latent_multiple=16,
            max_edge=1536,
            min_steps=min_steps,
            max_steps=max_steps,
            min_cfg=min_cfg,
            max_cfg=max_cfg,
            allowed_samplers=allowed_samplers,
            allowed_schedulers=frozenset(),
        ),
    )


def _make_route_handler_mocks(
    *,
    generate_side_effect: Exception | None = None,
) -> tuple[MagicMock, AsyncMock, AsyncMock]:
    """Return (generation_service_mock, idempotency_mock, session_mock)."""
    svc = MagicMock(spec=GenerationService)
    if generate_side_effect is not None:
        svc.generate = AsyncMock(side_effect=generate_side_effect)
    else:
        svc.generate = AsyncMock(return_value=MagicMock())

    idempotency = AsyncMock()
    idempotency.check = AsyncMock(return_value=uuid4())  # record_id (not a replay)
    idempotency.fail = AsyncMock()

    session = AsyncMock()
    return svc, idempotency, session


# ---------------------------------------------------------------------------
# §3.4-A  Schema-level validation → 400 (msgspec.ValidationError)
# ---------------------------------------------------------------------------


class TestSchemaValidation400:
    """Schema __post_init__ raises ValueError → msgspec converts to ValidationError."""

    def test_only_width_no_height_raises_validation_error(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","width":512}',
                type=UnifiedGenerationRequest,
            )

    def test_only_height_no_width_raises_validation_error(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","height":512}',
                type=UnifiedGenerationRequest,
            )

    def test_image_resolution_and_explicit_dims_raises_validation_error(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image",'
                b'"image_resolution":"standard","width":512,"height":512}',
                type=UnifiedGenerationRequest,
            )


# ---------------------------------------------------------------------------
# §3.4-B  Route handler maps ValueError / GenerationError → 400
# ---------------------------------------------------------------------------


class TestRouteHandlerReturns400:
    """Route handler must return 400 (not 500) for all validation errors."""

    async def _call_handler(
        self,
        request: UnifiedGenerationRequest,
        *,
        generation_service: MagicMock,
        idempotency_service: AsyncMock,
        session: AsyncMock,
    ) -> Any:
        """Invoke the route handler directly; return the response object."""
        with MagicMock() as controller_self:
            return await UnifiedGenerationController.generate.fn(
                controller_self,
                current_user_id=uuid4(),
                data=request,
                session=session,
                generation_service=generation_service,
                product_config=MagicMock(slug="vex"),
                product_id="vex",
                idempotency_service=idempotency_service,
                idempotency_key_header="test-key-123",
            )

    async def test_generation_error_from_provider_returns_400(self) -> None:
        svc, idempotency, session = _make_route_handler_mocks(
            generate_side_effect=GenerationError("steps 999 out of range")
        )
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            steps=999,
        )
        response = await self._call_handler(
            request,
            generation_service=svc,
            idempotency_service=idempotency,
            session=session,
        )
        assert response.status_code == 400
        idempotency.fail.assert_awaited_once()

    async def test_value_error_from_service_returns_400(self) -> None:
        svc, idempotency, session = _make_route_handler_mocks(
            generate_side_effect=ValueError("model does not support generation type")
        )
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )
        response = await self._call_handler(
            request,
            generation_service=svc,
            idempotency_service=idempotency,
            session=session,
        )
        assert response.status_code == 400
        idempotency.fail.assert_awaited_once()

    async def test_not_implemented_from_provider_returns_400(self) -> None:
        svc, idempotency, session = _make_route_handler_mocks(
            generate_side_effect=NotImplementedError(
                "Aisha does not support source_images inputs yet"
            )
        )
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )

        response = await self._call_handler(
            request,
            generation_service=svc,
            idempotency_service=idempotency,
            session=session,
        )

        assert response.status_code == 400
        assert response.content.error == "not_implemented"
        idempotency.fail.assert_awaited_once()

    async def test_unhandled_exception_is_not_swallowed_as_400(self) -> None:
        """Non-ValueError/GenerationError exceptions must not silently return 400."""
        svc, idempotency, session = _make_route_handler_mocks(
            generate_side_effect=RuntimeError("database exploded")
        )
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )
        with pytest.raises(RuntimeError, match="database exploded"):
            await self._call_handler(
                request,
                generation_service=svc,
                idempotency_service=idempotency,
                session=session,
            )
        idempotency.fail.assert_awaited_once()


# ---------------------------------------------------------------------------
# §3.4-C  Provider _resolve_* raises ValueError BEFORE check_and_reserve
# ---------------------------------------------------------------------------


class TestProviderValidationBeforeBilling:
    """_resolve_* errors must fire before billing.check_and_reserve is called."""

    def _make_provider(self, gen_config: BundleGenerationConfig) -> tuple[object, AsyncMock]:
        from src.api.services.generation.aisha_provider import AishaGenerationProvider

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock()

        bundle_index = MagicMock()
        bundle_index.get_generation_config.return_value = gen_config

        gpu_session_svc = AsyncMock()
        gpu_session = MagicMock()
        gpu_session.id = uuid4()
        gpu_session.bundle_name = "my_bundle"
        gpu_session.bundle_version = "current"
        gpu_session.tunnel_hostname = "abc.gpu.test"
        gpu_session_svc.get_active_session_for_model = AsyncMock(return_value=gpu_session)

        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=gpu_session_svc,
            bundle_index=bundle_index,
            r2_storage=None,
            tunnel_domain="gpu.test",
        )
        return provider, billing

    async def test_steps_out_of_range_raises_before_check_and_reserve(self) -> None:
        gen_cfg = _make_gen_config(min_steps=1, max_steps=30)
        provider, billing = self._make_provider(gen_cfg)

        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            steps=999,
        )
        with pytest.raises((ValueError, Exception), match="steps"):
            await provider.submit(  # type: ignore[attr-defined]
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=billing,
                account_id=uuid4(),
                token_cost=10,
                product_id="vex",
            )

        billing.check_and_reserve.assert_not_awaited()

    async def test_cfg_out_of_range_raises_before_check_and_reserve(self) -> None:
        gen_cfg = _make_gen_config(min_cfg=0.0, max_cfg=15.0)
        provider, billing = self._make_provider(gen_cfg)

        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            cfg=25.0,
        )
        with pytest.raises((ValueError, Exception), match="cfg"):
            await provider.submit(  # type: ignore[attr-defined]
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=billing,
                account_id=uuid4(),
                token_cost=10,
                product_id="vex",
            )

        billing.check_and_reserve.assert_not_awaited()

    async def test_sampler_not_in_allowed_list_raises_before_check_and_reserve(self) -> None:
        gen_cfg = _make_gen_config(allowed_samplers=frozenset({Sampler.EULER, Sampler.DPMPP_2M}))
        provider, billing = self._make_provider(gen_cfg)

        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            sampler=Sampler.HEUN,  # not in allowed list
        )
        with pytest.raises((ValueError, Exception), match="sampler"):
            await provider.submit(  # type: ignore[attr-defined]
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=billing,
                account_id=uuid4(),
                token_cost=10,
                product_id="vex",
            )

        billing.check_and_reserve.assert_not_awaited()
