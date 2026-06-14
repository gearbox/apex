"""Per-model generation defaults and constraints parsed from bundle.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import Resolution, Sampler, Scheduler


class BundleConfigError(Exception):
    """Raised when a bundle.yaml generation section contains invalid values."""


@dataclass(frozen=True, slots=True)
class GenerationDefaults:
    resolution: Resolution
    steps: int
    cfg: float
    sampler: Sampler
    scheduler: Scheduler
    denoise: float


@dataclass(frozen=True, slots=True)
class GenerationConstraints:
    max_megapixels: float
    latent_multiple: int
    max_edge: int
    min_steps: int
    max_steps: int
    min_cfg: float
    max_cfg: float
    allowed_samplers: frozenset[Sampler]  # empty = any
    allowed_schedulers: frozenset[Scheduler]  # empty = any


@dataclass(frozen=True, slots=True)
class BundleGenerationConfig:
    defaults: GenerationDefaults
    constraints: GenerationConstraints
