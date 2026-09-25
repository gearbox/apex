"""Immutable, file-backed registry of versioned legal documents.

Layout (``Settings.legal_documents_dir``, default ``legal/``)::

    legal/manifest.toml
    legal/{product}/{doc_type}/{YYYY-MM-DD}.md

A version *is* its effective date (UTC). The **current** version of a type is
the latest one with ``version <= today``; future-dated files stay dormant until
their date. The **required** version is the latest effective version with
``requires_reacceptance = true`` — an unflagged later version (e.g. a typo fix)
becomes current without forcing anyone to re-accept.

The registry is loaded once at startup (``LegalDocumentRegistry.load``) and is
read-only afterwards. ``today`` is always passed in by the caller so the
midnight-UTC switchover is evaluated per request, never cached.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Self

import msgspec
import structlog

from src.api.services.legal.errors import LegalDocumentNotFoundError, LegalRegistryError
from src.core.enums import LegalDocumentType, Product

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from src.core.product import ProductConfig

logger = structlog.get_logger(__name__)

MANIFEST_FILENAME: Final = "manifest.toml"

_DRAFTING_NOTE_RE: Final = re.compile(r"DRAFTING NOTE")
_REFERENCE_DEFINITION_RE: Final = re.compile(r"^ {0,3}\[([^\]\n]+)\]:[ \t]*\S")
_BRACKET_SPAN_RE: Final = re.compile(r"(?<!\\)\[([^\]\n]+)\]")


class ManifestEntry(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """One version of one document, as declared in ``manifest.toml``."""

    version: date
    requires_reacceptance: bool
    sha256: Annotated[str, msgspec.Meta(pattern=r"^[0-9a-f]{64}$")]


LegalManifest = dict[Product, dict[LegalDocumentType, list[ManifestEntry]]]


@dataclass(frozen=True, slots=True)
class LegalDocument:
    """One immutable version of a legal document."""

    product: Product
    doc_type: LegalDocumentType
    version: date
    requires_reacceptance: bool
    content_md: str
    sha256: str  # hex digest of content_md encoded as UTF-8 (LF line endings)


def normalize_content(raw: str) -> str:
    """Normalise line endings to ``\\n`` so the hash is platform-independent."""
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def content_sha256(content_md: str) -> str:
    """Hex sha256 of already-normalised markdown content."""
    return hashlib.sha256(content_md.encode("utf-8")).hexdigest()


def _normalise_reference_label(label: str) -> str:
    """Normalise a CommonMark reference label for case-insensitive lookup."""
    return " ".join(label.split()).casefold()


def _hygiene_findings(document: LegalDocument) -> list[tuple[int, str]]:
    """Return ``(line_number, matched_text)`` for placeholders and drafting notes.

    Markdown reference links use bracket notation too. Their labels can only be
    identified after collecting definitions from the whole document, so this is
    intentionally a small document-level scanner rather than one regex.
    """
    lines = document.content_md.split("\n")
    definition_lines: set[int] = set()
    definitions: set[str] = set()
    for lineno, line in enumerate(lines, start=1):
        if match := _REFERENCE_DEFINITION_RE.match(line):
            definition_lines.add(lineno)
            definitions.add(_normalise_reference_label(match.group(1)))

    findings: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, start=1):
        findings.extend((lineno, m.group(0)) for m in _DRAFTING_NOTE_RE.finditer(line))
        if lineno in definition_lines:
            continue

        position = 0
        while match := _BRACKET_SPAN_RE.search(line, position):
            text = match.group(0)
            label = match.group(1)
            end = match.end()

            # [text](url)
            if line.startswith("(", end):
                position = end
                continue

            # [text][reference] and [text][] (only if the reference resolves).
            if line.startswith("[", end):
                if line.startswith("[]", end):
                    if _normalise_reference_label(label) in definitions:
                        position = end + 2
                        continue
                elif (trailing := _BRACKET_SPAN_RE.match(line, end)) and _normalise_reference_label(
                    trailing.group(1)
                ) in definitions:
                    position = trailing.end()
                    continue

            # [reference] shortcut link.
            if _normalise_reference_label(label) in definitions:
                position = end
                continue

            # GFM task-list marker at the start of a list item.
            if text in {"[ ]", "[x]", "[X]"} and re.fullmatch(
                r"\s*[-*+]\s+", line[: match.start()]
            ):
                position = end
                continue

            findings.append((lineno, text))
            position = end
    return findings


class LegalDocumentRegistry:
    """Read-only index of every legal document version, per product and type."""

    def __init__(self, documents: Iterable[LegalDocument]) -> None:
        """Index documents. Ordering/consistency validation lives in :meth:`load`.

        Args:
            documents: Every document version to serve.
        """
        grouped: dict[tuple[Product, LegalDocumentType], list[LegalDocument]] = defaultdict(list)
        for doc in documents:
            grouped[(doc.product, doc.doc_type)].append(doc)
        self._docs: Mapping[tuple[Product, LegalDocumentType], tuple[LegalDocument, ...]] = (
            MappingProxyType(
                {key: tuple(sorted(docs, key=lambda d: d.version)) for key, docs in grouped.items()}
            )
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        products: Iterable[ProductConfig],
        environment: str,
        today: date,
    ) -> Self:
        """Load and validate ``root/manifest.toml`` and every document it names.

        Args:
            root: The legal documents directory.
            products: Product configs whose ``required_legal_documents`` must be
                satisfiable on ``today``.
            environment: ``Settings.environment``; ``"production"`` turns
                placeholder/drafting-note findings into a hard failure.
            today: The UTC date to validate effectiveness against.

        Returns:
            A validated registry.

        Raises:
            LegalRegistryError: On any manifest/file inconsistency, ordering
                violation, unsatisfiable requirement, or (production only)
                unfilled placeholder.
        """
        manifest_path = root / MANIFEST_FILENAME
        try:
            manifest = msgspec.toml.decode(manifest_path.read_bytes(), type=LegalManifest)
        except FileNotFoundError as exc:
            raise LegalRegistryError(f"Legal manifest not found: {manifest_path}") from exc
        except msgspec.ValidationError as exc:
            raise LegalRegistryError(f"Invalid legal manifest {manifest_path}: {exc}") from exc

        documents: list[LegalDocument] = []
        expected_files: set[Path] = set()
        for product, types in manifest.items():
            for doc_type, entries in types.items():
                cls._validate_entries(product, doc_type, entries)
                for entry in entries:
                    path = root / product.value / doc_type.value / f"{entry.version.isoformat()}.md"
                    expected_files.add(path)
                    try:
                        raw = path.read_text(encoding="utf-8")
                    except FileNotFoundError as exc:
                        raise LegalRegistryError(
                            f"Manifest entry {product}/{doc_type}/{entry.version} has no file "
                            f"at {path}"
                        ) from exc
                    content = normalize_content(raw)
                    computed_sha256 = content_sha256(content)
                    if computed_sha256 != entry.sha256:
                        raise LegalRegistryError(
                            f"{product}/{doc_type}/{entry.version}: manifest sha256 "
                            f"{entry.sha256} does not match computed content sha256 "
                            f"{computed_sha256}. Published legal documents are immutable — "
                            "publish a new version file instead. If this version has never been "
                            "accepted anywhere, update the manifest hash."
                        )
                    documents.append(
                        LegalDocument(
                            product=product,
                            doc_type=doc_type,
                            version=entry.version,
                            requires_reacceptance=entry.requires_reacceptance,
                            content_md=content,
                            sha256=computed_sha256,
                        )
                    )

        if stray := sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
            and p != manifest_path
            and not p.name.startswith(".")
            and p not in expected_files
        ):
            raise LegalRegistryError(f"Files under {root} without a manifest entry: {stray}")

        registry = cls(documents)
        registry._validate_requirements(products, today=today)
        registry._check_hygiene(environment=environment)
        return registry

    @staticmethod
    def _validate_entries(
        product: Product, doc_type: LegalDocumentType, entries: list[ManifestEntry]
    ) -> None:
        if not entries:
            raise LegalRegistryError(f"{product}/{doc_type}: manifest lists no versions")
        if not entries[0].requires_reacceptance:
            raise LegalRegistryError(
                f"{product}/{doc_type}: the first version ({entries[0].version}) must set "
                "requires_reacceptance = true"
            )
        for prev, nxt in pairwise(entries):
            if nxt.version <= prev.version:
                raise LegalRegistryError(
                    f"{product}/{doc_type}: versions must be strictly increasing "
                    f"({prev.version} then {nxt.version})"
                )

    def _validate_requirements(self, products: Iterable[ProductConfig], *, today: date) -> None:
        for config in products:
            for doc_type in sorted(config.required_legal_documents):
                try:
                    self.current(config.product, doc_type, today=today)
                except LegalDocumentNotFoundError as exc:
                    raise LegalRegistryError(
                        f"{config.product}/{doc_type} is required but has no version effective "
                        f"on {today.isoformat()}"
                    ) from exc

    def _check_hygiene(self, *, environment: str) -> None:
        failures: list[str] = []
        for docs in self._docs.values():
            for doc in docs:
                for lineno, text in _hygiene_findings(doc):
                    if environment == "production":
                        failures.append(
                            f"{doc.product}/{doc.doc_type}/{doc.version}:{lineno}: {text}"
                        )
                    else:
                        logger.warning(
                            "legal.placeholder_detected",
                            product=doc.product.value,
                            doc_type=doc.doc_type.value,
                            version=doc.version.isoformat(),
                            line=lineno,
                        )
        if failures:
            raise LegalRegistryError(
                "Legal documents contain unfilled placeholders or drafting notes; refusing to "
                "publish in production:\n" + "\n".join(failures)
            )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _versions(self, product: Product, doc_type: LegalDocumentType) -> tuple[LegalDocument, ...]:
        if docs := self._docs.get((product, doc_type)):
            return docs
        raise LegalDocumentNotFoundError(doc_type.value)

    def has_type(self, product: Product, doc_type: LegalDocumentType) -> bool:
        """Whether any version of ``doc_type`` exists for ``product``."""
        return bool(self._docs.get((product, doc_type)))

    def current(
        self, product: Product, doc_type: LegalDocumentType, *, today: date
    ) -> LegalDocument:
        """Latest version with ``version <= today``.

        Raises:
            LegalDocumentNotFoundError: If the type is unknown or nothing is effective yet.
        """
        if effective := [d for d in self._versions(product, doc_type) if d.version <= today]:
            return effective[-1]
        raise LegalDocumentNotFoundError(doc_type.value)

    def get(self, product: Product, doc_type: LegalDocumentType, version: date) -> LegalDocument:
        """Exact version lookup (effective or not).

        Raises:
            LegalDocumentNotFoundError: If that version doesn't exist.
        """
        for doc in self._versions(product, doc_type):
            if doc.version == version:
                return doc
        raise LegalDocumentNotFoundError(doc_type.value, version)

    def list_current(self, product: Product, *, today: date) -> tuple[LegalDocument, ...]:
        """Current version of every document type that has one, ordered by type."""
        result: list[LegalDocument] = []
        for (doc_product, doc_type), _docs in sorted(self._docs.items()):
            if doc_product != product:
                continue
            try:
                result.append(self.current(product, doc_type, today=today))
            except LegalDocumentNotFoundError:
                continue
        return tuple(result)

    def required_versions(
        self, product: ProductConfig, *, today: date
    ) -> Mapping[LegalDocumentType, date]:
        """For each required type: the latest effective version flagged for re-acceptance.

        Raises:
            LegalDocumentNotFoundError: If a required type has no flagged effective
                version (prevented at startup by :meth:`load`).
        """
        required: dict[LegalDocumentType, date] = {}
        for doc_type in sorted(product.required_legal_documents):
            if flagged := [
                d
                for d in self._versions(product.product, doc_type)
                if d.requires_reacceptance and d.version <= today
            ]:
                required[doc_type] = flagged[-1].version
            else:
                raise LegalDocumentNotFoundError(doc_type.value)
        return MappingProxyType(required)

    def required_digest(self, product: ProductConfig, *, today: date) -> str | None:
        """Short digest of the required (type, version) set — the ``lgl`` JWT claim.

        Returns:
            ``None`` when the product requires nothing, else 16 hex chars.
        """
        required = self.required_versions(product, today=today)
        if not required:
            return None
        canonical = "|".join(f"{t}:{v.isoformat()}" for t, v in sorted(required.items()))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
