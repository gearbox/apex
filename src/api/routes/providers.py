"""Provider discovery endpoint."""

from __future__ import annotations

from collections.abc import Sequence

import msgspec
import structlog
from litestar import Controller, get
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.grok.job_service import GrokJobService
from src.core.enums import GenerationType, ModelType, Provider
from src.db.repositories.generation_model import GenerationModelRepository

logger = structlog.get_logger(__name__)


class ProviderModelInfo(msgspec.Struct, kw_only=True):
    """A single model's capabilities — derived from ModelType enum properties."""

    model: str
    name: str
    description: str
    provider: str
    is_enabled: bool
    supports_t2i: bool
    supports_i2i: bool
    supports_t2v: bool
    supports_i2v: bool
    supports_v2v: bool
    supports_flf2v: bool
    max_images: int


class ProviderInfo(msgspec.Struct, kw_only=True):
    """Single provider status."""

    provider: str
    name: str
    available: bool


class ProvidersResponse(msgspec.Struct, kw_only=True):
    """Full providers discovery response."""

    providers: list[ProviderInfo]
    models: list[ProviderModelInfo]


def _build_model_info(mt: ModelType, record: object) -> ProviderModelInfo:
    """Build ProviderModelInfo entirely from ModelType enum properties.

    No hardcoded if/elif per model — adding a new ModelType member with the
    correct ``supports_image_input`` and ``is_video_model`` properties is enough.
    """
    return ProviderModelInfo(
        model=mt.value,
        name=record.name,  # type: ignore[attr-defined]
        description=record.description,  # type: ignore[attr-defined]
        provider=mt.provider.value,
        is_enabled=record.is_enabled,  # type: ignore[attr-defined]
        supports_t2i=mt.supports_generation_type(GenerationType.T2I),
        supports_i2i=mt.supports_generation_type(GenerationType.I2I),
        supports_t2v=mt.supports_generation_type(GenerationType.T2V),
        supports_i2v=mt.supports_generation_type(GenerationType.I2V),
        supports_v2v=mt.supports_generation_type(GenerationType.V2V),
        supports_flf2v=mt.supports_generation_type(GenerationType.FLF2V),
        max_images=mt.max_concurrent_outputs,
    )


class ProvidersController(Controller):
    """Provider and model discovery."""

    path = "/v1/providers"
    tags: Sequence[str] | None = ["Providers"]

    @get("/")
    async def list_providers(
        self,
        session: AsyncSession,
        grok_job_service: GrokJobService | None,
    ) -> ProvidersResponse:
        """List available providers and their models.

        Returns provider availability status and per-model capability flags.
        The frontend uses this to build the model selector and validate
        generation_type options before submission.
        """
        repo = GenerationModelRepository(session)
        all_models = await repo.list_enabled()

        providers = [
            ProviderInfo(
                provider=Provider.AISHA.value,
                name="Aisha",
                available=True,
            ),
            ProviderInfo(
                provider=Provider.GROK.value,
                name="xAI Grok",
                available=grok_job_service is not None,
            ),
        ]

        models = []
        for record in all_models:
            try:
                mt = ModelType(record.model_key)
            except ValueError:
                continue
            models.append(_build_model_info(mt, record))

        return ProvidersResponse(providers=providers, models=models)
