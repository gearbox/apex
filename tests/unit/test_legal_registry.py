"""LegalDocumentRegistry: loading, validation, current/required resolution.

Contracts covered: C1 (bijection), C2 (ordering), C3 (current resolution),
C4 (hash stability — registry half), C5 (production hygiene), and the
"unflagged v2 does not change the digest" half of C11.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import pytest
import structlog

from src.api.services.legal.errors import LegalDocumentNotFoundError, LegalRegistryError
from src.api.services.legal.registry import LegalDocumentRegistry, content_sha256
from src.core.enums import LegalDocumentType, Product
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from tests.legal_support import make_legal_document, make_legal_registry

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.product import ProductConfig

pytestmark = pytest.mark.unit

V1 = date(2026, 10, 1)
V2 = date(2026, 11, 1)
TERMS_ONLY: ProductConfig = replace(
    VEX_CONFIG, required_legal_documents=frozenset({LegalDocumentType.TERMS})
)


def _write_tree(
    root: Path,
    manifest: str,
    files: dict[str, str | bytes],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.toml").write_text(manifest, encoding="utf-8")
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return root


def _terms_manifest(*entries: tuple[str, bool]) -> str:
    rows = ",\n".join(
        f"  {{ version = {v}, requires_reacceptance = {str(flag).lower()} }}" for v, flag in entries
    )
    return f"[vex]\nterms = [\n{rows},\n]\n"


def _load(
    root: Path,
    *,
    today: date = V2,
    environment: str = "development",
    products: tuple[ProductConfig, ...] = (TERMS_ONLY,),
) -> LegalDocumentRegistry:
    return LegalDocumentRegistry.load(root, products=products, environment=environment, today=today)


class TestBijection:
    """C1 — manifest entries and files must match one-to-one."""

    def test_valid_tree_loads(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "# Terms\n"},
        )
        registry = _load(root)
        assert registry.current(Product.VEX, LegalDocumentType.TERMS, today=V2).version == V1

    def test_manifest_entry_without_file_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True), ("2026-11-01", False)),
            {"vex/terms/2026-10-01.md": "# Terms\n"},
        )
        with pytest.raises(LegalRegistryError, match="has no file"):
            _load(root)

    def test_file_without_manifest_entry_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "# Terms\n", "vex/terms/2026-12-01.md": "# Stray\n"},
        )
        with pytest.raises(LegalRegistryError, match="without a manifest entry"):
            _load(root)

    def test_stray_non_markdown_file_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "# Terms\n", "vex/notes.txt": "draft"},
        )
        with pytest.raises(LegalRegistryError, match="without a manifest entry"):
            _load(root)

    def test_hidden_files_are_ignored(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "# Terms\n", "vex/.DS_Store": b"\x00"},
        )
        _load(root)

    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        with pytest.raises(LegalRegistryError, match="manifest not found"):
            _load(tmp_path)

    def test_unknown_document_type_in_manifest_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            "[vex]\ncookies = [ { version = 2026-10-01, requires_reacceptance = true } ]\n",
            {"vex/cookies/2026-10-01.md": "# Cookies\n"},
        )
        with pytest.raises(LegalRegistryError, match="Invalid legal manifest"):
            _load(root)

    def test_unknown_manifest_field_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            "[vex]\nterms = [ { version = 2026-10-01, requires_reacceptance = true, x = 1 } ]\n",
            {"vex/terms/2026-10-01.md": "# Terms\n"},
        )
        with pytest.raises(LegalRegistryError, match="Invalid legal manifest"):
            _load(root)


class TestOrdering:
    """C2 — strictly increasing versions; first version must force acceptance."""

    def test_non_increasing_versions_fail(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-11-01", True), ("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "a\n", "vex/terms/2026-11-01.md": "b\n"},
        )
        with pytest.raises(LegalRegistryError, match="strictly increasing"):
            _load(root)

    def test_duplicate_versions_fail(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True), ("2026-10-01", False)),
            {"vex/terms/2026-10-01.md": "a\n"},
        )
        with pytest.raises(LegalRegistryError, match="strictly increasing"):
            _load(root)

    def test_first_version_without_reacceptance_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", False)),
            {"vex/terms/2026-10-01.md": "a\n"},
        )
        with pytest.raises(LegalRegistryError, match="first version"):
            _load(root)

    def test_empty_version_list_fails(self, tmp_path: Path) -> None:
        root = _write_tree(tmp_path / "legal", "[vex]\nterms = []\n", {})
        with pytest.raises(LegalRegistryError, match="no versions"):
            _load(root)


class TestRequiredAvailability:
    def test_required_type_not_yet_effective_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "a\n"},
        )
        with pytest.raises(LegalRegistryError, match="no version effective on 2026-09-30"):
            _load(root, today=date(2026, 9, 30))

    def test_required_type_missing_entirely_fails(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "a\n"},
        )
        with pytest.raises(LegalRegistryError, match="privacy is required"):
            _load(root, products=(VEX_CONFIG,))

    def test_product_with_empty_required_set_needs_nothing(self, tmp_path: Path) -> None:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": "a\n"},
        )
        _load(root, products=(TERMS_ONLY, SYNTHARA_CONFIG))


class TestCurrentResolution:
    """C3 — a future-dated version stays dormant until its date."""

    @pytest.fixture
    def registry(self, tmp_path: Path) -> LegalDocumentRegistry:
        root = _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True), ("2026-11-01", True)),
            {"vex/terms/2026-10-01.md": "v1\n", "vex/terms/2026-11-01.md": "v2\n"},
        )
        return _load(root, today=V1)

    def test_day_before_boundary_resolves_v1(self, registry: LegalDocumentRegistry) -> None:
        doc = registry.current(Product.VEX, LegalDocumentType.TERMS, today=date(2026, 10, 31))
        assert doc.version == V1
        assert registry.required_versions(TERMS_ONLY, today=date(2026, 10, 31)) == {
            LegalDocumentType.TERMS: V1
        }

    def test_on_boundary_resolves_v2(self, registry: LegalDocumentRegistry) -> None:
        doc = registry.current(Product.VEX, LegalDocumentType.TERMS, today=V2)
        assert doc.version == V2
        assert doc.content_md == "v2\n"
        assert registry.required_versions(TERMS_ONLY, today=V2) == {LegalDocumentType.TERMS: V2}

    def test_digest_changes_at_boundary(self, registry: LegalDocumentRegistry) -> None:
        before = registry.required_digest(TERMS_ONLY, today=date(2026, 10, 31))
        after = registry.required_digest(TERMS_ONLY, today=V2)
        assert before is not None
        assert after is not None
        assert before != after

    def test_future_version_is_fetchable_by_exact_get(
        self, registry: LegalDocumentRegistry
    ) -> None:
        assert registry.get(Product.VEX, LegalDocumentType.TERMS, V2).content_md == "v2\n"

    def test_unknown_version_raises_not_found(self, registry: LegalDocumentRegistry) -> None:
        with pytest.raises(LegalDocumentNotFoundError):
            registry.get(Product.VEX, LegalDocumentType.TERMS, date(2026, 10, 2))

    def test_unknown_type_raises_not_found(self, registry: LegalDocumentRegistry) -> None:
        with pytest.raises(LegalDocumentNotFoundError):
            registry.current(Product.VEX, LegalDocumentType.PRIVACY, today=V2)

    def test_nothing_effective_raises_not_found(self, registry: LegalDocumentRegistry) -> None:
        with pytest.raises(LegalDocumentNotFoundError):
            registry.current(Product.VEX, LegalDocumentType.TERMS, today=date(2026, 1, 1))


class TestHashStability:
    """C4 — CRLF and LF sources hash identically."""

    def test_crlf_and_lf_produce_same_sha(self, tmp_path: Path) -> None:
        text = "# Terms\n\nLine one.\nLine two.\n"
        lf_root = _write_tree(
            tmp_path / "lf",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": text.encode()},
        )
        crlf_root = _write_tree(
            tmp_path / "crlf",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": text.replace("\n", "\r\n").encode()},
        )
        lf = _load(lf_root).current(Product.VEX, LegalDocumentType.TERMS, today=V2)
        crlf = _load(crlf_root).current(Product.VEX, LegalDocumentType.TERMS, today=V2)
        assert lf.sha256 == crlf.sha256
        assert crlf.content_md == text
        assert lf.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestProductionHygiene:
    """C5 — unfilled placeholders fail production, only warn elsewhere."""

    def _root(self, tmp_path: Path, body: str) -> Path:
        return _write_tree(
            tmp_path / "legal",
            _terms_manifest(("2026-10-01", True)),
            {"vex/terms/2026-10-01.md": body},
        )

    def test_placeholder_fails_in_production(self, tmp_path: Path) -> None:
        root = self._root(tmp_path, "Operated by [PLACEHOLDER].\n")
        with pytest.raises(LegalRegistryError, match=r"vex/terms/2026-10-01:1: \[PLACEHOLDER\]"):
            _load(root, environment="production")

    def test_drafting_note_fails_in_production(self, tmp_path: Path) -> None:
        root = self._root(tmp_path, "Intro\n> DRAFTING NOTE: remove me\n")
        with pytest.raises(LegalRegistryError, match="DRAFTING NOTE"):
            _load(root, environment="production")

    def test_markdown_link_passes_in_production(self, tmp_path: Path) -> None:
        root = self._root(tmp_path, "See our [Privacy Policy](https://vex.pics/privacy).\n")
        _load(root, environment="production")

    def test_staging_only_logs_placeholder(self, tmp_path: Path) -> None:
        root = self._root(tmp_path, "line one\nOperated by [PLACEHOLDER].\n")
        with structlog.testing.capture_logs() as logs:
            _load(root, environment="staging")
        events = [e for e in logs if e["event"] == "legal.placeholder_detected"]
        assert events == [
            {
                "event": "legal.placeholder_detected",
                "log_level": "warning",
                "product": "vex",
                "doc_type": "terms",
                "version": "2026-10-01",
                "line": 2,
            }
        ]


class TestRequiredDigest:
    def test_empty_required_set_has_no_digest(self) -> None:
        assert make_legal_registry().required_digest(SYNTHARA_CONFIG, today=V2) is None

    def test_digest_format_matches_spec(self) -> None:
        registry = make_legal_registry()
        expected_canonical = "|".join(
            f"{t}:2020-01-01" for t in sorted(VEX_CONFIG.required_legal_documents)
        )
        assert (
            registry.required_digest(VEX_CONFIG, today=V2)
            == (hashlib.sha256(expected_canonical.encode()).hexdigest()[:16])
        )

    def test_unflagged_v2_does_not_change_digest(self) -> None:
        """C11 (second half) — a non-material v2 becomes current without re-acceptance."""
        base = make_legal_registry()
        v1_terms = base.current(Product.VEX, LegalDocumentType.TERMS, today=V2)
        v2_terms = make_legal_document(LegalDocumentType.TERMS, V1, requires_reacceptance=False)
        others = [
            base.current(Product.VEX, t, today=V2)
            for t in (LegalDocumentType.PRIVACY, LegalDocumentType.SENSITIVE_DATA_CONSENT)
        ]
        updated = make_legal_registry(v1_terms, v2_terms, *others)

        assert updated.current(Product.VEX, LegalDocumentType.TERMS, today=V2).version == V1
        assert updated.required_digest(VEX_CONFIG, today=V2) == base.required_digest(
            VEX_CONFIG, today=V2
        )

    def test_list_current_is_ordered_and_product_scoped(self) -> None:
        registry = make_legal_registry(
            *(make_legal_document(t) for t in LegalDocumentType),
            make_legal_document(LegalDocumentType.TERMS, product=Product.SYNTHARA),
        )
        vex = registry.list_current(Product.VEX, today=V2)
        assert [d.doc_type for d in vex] == sorted(LegalDocumentType)
        assert [d.product for d in registry.list_current(Product.SYNTHARA, today=V2)] == [
            Product.SYNTHARA
        ]

    def test_document_sha_is_content_hash(self) -> None:
        doc = make_legal_document(LegalDocumentType.TERMS, content="hello\n")
        assert doc.sha256 == content_sha256("hello\n")
