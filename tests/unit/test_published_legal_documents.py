"""The repository's own ``legal/`` directory loads (the test suite otherwise uses fixtures).

``LegalDocumentRegistry.load`` runs at startup: a manifest hash that does not match its
file, or versions out of order, would stop the API from booting. Nothing else loads the
real directory in CI, so a new document version is checked here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.api.services.legal.registry import LegalDocumentRegistry
from src.core.enums import LegalDocumentType, Product
from src.core.product_registry import PRODUCT_REGISTRY

_LEGAL_DIR = Path(__file__).parent.parent.parent / "legal"


def _load(today: date) -> LegalDocumentRegistry:
    return LegalDocumentRegistry.load(
        _LEGAL_DIR, products=PRODUCT_REGISTRY.values(), environment="test", today=today
    )


def test_published_documents_load_on_every_version_date() -> None:
    for today in (date(2026, 9, 25), date(2026, 11, 30), date(2026, 12, 1), date(2026, 12, 2)):
        assert _load(today) is not None


def test_synthara_privacy_names_alibaba_cloud_from_2026_12_01() -> None:
    before = _load(date(2026, 11, 30)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 11, 30)
    )
    after = _load(date(2026, 12, 1)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 1)
    )

    assert before.version == date(2026, 9, 25)
    assert "Alibaba Cloud" not in before.content_md
    assert after.version == date(2026, 12, 1)
    assert "Alibaba Cloud (Singapore) Private Limited" in after.content_md
    assert "Studio content" in after.content_md


_REFERENCES_ROW = (
    "| Uploaded Inputs and Outputs used as Studio references | For as long as a Studio item "
    "(such as a cast member) uses them; the period above starts again when the last such use "
    "is removed |"
)


def test_synthara_privacy_keeps_assets_that_studio_items_use_from_2026_12_02() -> None:
    before = _load(date(2026, 12, 1)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 1)
    )
    after = _load(date(2026, 12, 2)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 2)
    )

    assert before.version == date(2026, 12, 1)
    assert _REFERENCES_ROW not in before.content_md
    assert after.version == date(2026, 12, 2)
    assert _REFERENCES_ROW in after.content_md
    assert after.requires_reacceptance is False  # synthara requires no acceptance


def test_the_new_version_differs_from_the_previous_one_by_that_row_only() -> None:
    previous = _load(date(2026, 12, 1)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 1)
    )
    current = _load(date(2026, 12, 2)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 2)
    )

    previous_lines = previous.content_md.splitlines()
    current_lines = current.content_md.splitlines()

    assert [line for line in current_lines if line not in previous_lines] == [_REFERENCES_ROW]
    assert [line for line in previous_lines if line not in current_lines] == []
    outputs = next(i for i, line in enumerate(current_lines) if line.startswith("| Outputs |"))
    assert current_lines[outputs + 1] == _REFERENCES_ROW  # directly after the Outputs row


def test_the_published_synthara_versions_are_untouched_and_vex_is_not_changed() -> None:
    # Published versions are immutable: the earlier files still carry their pinned hashes (the
    # registry would refuse to load otherwise) and the vex documents were not touched.
    registry = _load(date(2026, 12, 2))
    assert registry.current(
        Product.VEX, LegalDocumentType.PRIVACY, today=date(2026, 12, 2)
    ).version == date(2026, 9, 25)
    old = registry.current(Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 11, 30))
    assert old.version == date(2026, 9, 25)
