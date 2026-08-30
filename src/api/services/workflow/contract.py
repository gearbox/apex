"""Types and tables for the bundle-declared workflow contract (v2).

The workflow map is deliberately a small, closed contract.  Bundles decide
*where* a value goes in an API graph; Apex decides which canonical values it
can provide.  Keeping those two decisions here prevents per-bundle conditionals
from reappearing in the submit path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from src.core.enums import GenerationType, MediaKind

SUPPORTED_CONTRACT_VERSION: Final[int] = 2


class WorkflowRole(StrEnum):
    """Addressable roles in a ComfyUI API workflow."""

    LATENT = "latent"
    POSITIVE_PROMPT = "positive_prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    SAMPLER = "sampler"
    MODEL_SAMPLING = "model_sampling"
    SAVE = "save"
    PREVIEW = "preview"


class MediaSlot(StrEnum):
    """Named media positions supplied by a generation request."""

    REFERENCE = "reference"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    SOURCE = "source"


ROLE_PARAMETERS: Final[Mapping[WorkflowRole, frozenset[str]]] = {
    WorkflowRole.LATENT: frozenset({"width", "height", "batch_size", "length"}),
    WorkflowRole.POSITIVE_PROMPT: frozenset({"text"}),
    WorkflowRole.NEGATIVE_PROMPT: frozenset({"text"}),
    WorkflowRole.SAMPLER: frozenset({"seed", "steps", "cfg", "sampler", "scheduler", "denoise"}),
    WorkflowRole.MODEL_SAMPLING: frozenset({"shift"}),
    WorkflowRole.SAVE: frozenset({"filename_prefix", "fps", "format"}),
    WorkflowRole.PREVIEW: frozenset(),
}

REQUIRED_ROLES: Final[frozenset[WorkflowRole]] = frozenset(
    {WorkflowRole.LATENT, WorkflowRole.POSITIVE_PROMPT, WorkflowRole.SAMPLER}
)
VIDEO_ONLY_PARAMETERS: Final[frozenset[str]] = frozenset({"length", "fps", "format"})
MEDIA_SLOT_KINDS: Final[Mapping[MediaSlot, MediaKind]] = {
    MediaSlot.REFERENCE: MediaKind.IMAGE,
    MediaSlot.FIRST_FRAME: MediaKind.IMAGE,
    MediaSlot.LAST_FRAME: MediaKind.IMAGE,
    MediaSlot.SOURCE: MediaKind.VIDEO,
}
MODEL_TYPE_MEDIA: Final[Mapping[str, MediaKind]] = {
    "aisha-image": MediaKind.IMAGE,
    "aisha-image-lite": MediaKind.IMAGE,
    "aisha-video": MediaKind.VIDEO,
}


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """One declared workflow node and its canonical-to-API input map."""

    id: str
    class_name: str
    inputs: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class WorkflowMediaInput:
    """One media loader and the role input it reconnects."""

    id: str
    class_name: str
    input: str
    kind: MediaKind
    slot: MediaSlot
    target_role: WorkflowRole
    target_input: str


@dataclass(frozen=True, slots=True)
class WorkflowModelInput:
    """One loader input whose model filename is owned by the bundle."""

    id: str
    class_name: str
    input: str
    model_type: str | None
    filename: str | None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowMap:
    """Parsed, but not yet graph-bound, workflow contract."""

    contract_version: int
    media: MediaKind
    nodes: Mapping[WorkflowRole, WorkflowNode]
    media_inputs: tuple[WorkflowMediaInput, ...]
    model_inputs: tuple[WorkflowModelInput, ...]


@dataclass(frozen=True, slots=True)
class BoundWorkflow:
    """A workflow map proven to address the accompanying API graph."""

    map: WorkflowMap
    api_graph: Mapping[str, Mapping[str, Any]]

    @property
    def media(self) -> MediaKind:
        return self.map.media


@dataclass(frozen=True, slots=True)
class BundleCapabilities:
    """Capabilities mechanically derived from a bound workflow."""

    media: MediaKind
    generation_types: frozenset[GenerationType]
    supports_negative_prompt: bool
    writable: frozenset[str]
    max_batch_size: int
    max_reference_images: int


@dataclass(frozen=True, slots=True)
class RequestParameterBinding:
    """A public request control and the workflow input that makes it writable.

    ``generation_default_attribute`` is ``None`` for controls with no bundle
    generation default.  ``request_default`` covers those request-shape
    defaults, such as one requested output, that are still a no-op.
    """

    parameter: str
    request_attribute: str
    writable: str
    generation_default_attribute: str | None = None
    request_default: object | None = None


# This is deliberately also the vocabulary used by request validation and
# provider discovery.  Do not repeat canonical parameter names elsewhere.
REQUEST_PARAMETER_BINDINGS: Final[tuple[RequestParameterBinding, ...]] = (
    RequestParameterBinding("width", "width", "latent.width"),
    RequestParameterBinding("height", "height", "latent.height"),
    RequestParameterBinding("batch_size", "n", "latent.batch_size", request_default=1),
    RequestParameterBinding("seed", "seed", "sampler.seed"),
    RequestParameterBinding("steps", "steps", "sampler.steps", "steps"),
    RequestParameterBinding("cfg", "cfg", "sampler.cfg", "cfg"),
    RequestParameterBinding("sampler", "sampler", "sampler.sampler", "sampler"),
    RequestParameterBinding("scheduler", "scheduler", "sampler.scheduler", "scheduler"),
    RequestParameterBinding("denoise", "denoise", "sampler.denoise", "denoise"),
)
NEGATIVE_PROMPT_PARAMETER: Final[str] = "negative_prompt"
DIMENSION_REQUEST_PARAMETERS: Final[tuple[str, ...]] = ("image_resolution", "aspect_ratio")
DIMENSION_WRITABLE_PARAMETERS: Final[frozenset[str]] = frozenset({"latent.width", "latent.height"})


RequestAccessor = Callable[[object, str], object]


def _request_value(name: str) -> RequestAccessor:
    return lambda request, _filename_prefix: getattr(request, name, None)


def _width(request: object, _filename_prefix: str) -> object:
    return getattr(request, "width", None)


def _filename_prefix(_request: object, filename_prefix: str) -> str:
    return filename_prefix


PARAMETER_ACCESSORS: Final[Mapping[tuple[WorkflowRole, str], RequestAccessor]] = {
    (WorkflowRole.LATENT, "width"): _width,
    (WorkflowRole.LATENT, "height"): _request_value("height"),
    (WorkflowRole.LATENT, "batch_size"): _request_value("max_images"),
    (WorkflowRole.LATENT, "length"): _request_value("length"),
    (WorkflowRole.POSITIVE_PROMPT, "text"): _request_value("prompt"),
    (WorkflowRole.NEGATIVE_PROMPT, "text"): _request_value("negative_prompt"),
    (WorkflowRole.SAMPLER, "seed"): _request_value("seed"),
    (WorkflowRole.SAMPLER, "steps"): _request_value("steps"),
    (WorkflowRole.SAMPLER, "cfg"): _request_value("cfg"),
    (WorkflowRole.SAMPLER, "sampler"): _request_value("sampler"),
    (WorkflowRole.SAMPLER, "scheduler"): _request_value("scheduler"),
    (WorkflowRole.SAMPLER, "denoise"): _request_value("denoise"),
    (WorkflowRole.MODEL_SAMPLING, "shift"): _request_value("shift"),
    (WorkflowRole.SAVE, "filename_prefix"): _filename_prefix,
    (WorkflowRole.SAVE, "fps"): _request_value("fps"),
    (WorkflowRole.SAVE, "format"): _request_value("format"),
}

# The workflow vocabulary is intentionally broader than the legacy request
# shape.  Keep this set as the checklist for request fields added by future
# generation APIs: until a request accessor can supply a value, a declared
# workflow input must not be advertised as writable.
PARAMETER_HAS_REQUEST_SOURCE: Final[frozenset[str]] = frozenset(
    {
        "latent.width",
        "latent.height",
        "latent.batch_size",
        "positive_prompt.text",
        "negative_prompt.text",
        "sampler.seed",
        "sampler.steps",
        "sampler.cfg",
        "sampler.sampler",
        "sampler.scheduler",
        "sampler.denoise",
        "save.filename_prefix",
    }
)
