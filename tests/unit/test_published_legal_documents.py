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


_INPUTS_BEFORE = "- **Inputs:** text prompts, and images or videos you upload."
_INPUTS_AFTER = "- **Inputs:** text prompts, and images, videos or audio files you upload."
_NO_AUDIO_FINGERPRINTS = (
    "We do not compute perceptual hashes or other fingerprints from audio files."
)


def test_synthara_privacy_names_audio_uploads_from_2026_12_02() -> None:
    """S2a-12: ``2026-12-02`` is amended in place (it is dormant until its date, so no one can
    have accepted it); the earlier version keeps the old text."""
    before = _load(date(2026, 12, 1)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 1)
    )
    after = _load(date(2026, 12, 2)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 2)
    )

    assert before.version == date(2026, 12, 1)
    assert _INPUTS_BEFORE in before.content_md
    assert _INPUTS_AFTER not in before.content_md
    assert _NO_AUDIO_FINGERPRINTS not in before.content_md
    assert after.version == date(2026, 12, 2)
    assert _INPUTS_AFTER in after.content_md
    assert _INPUTS_BEFORE not in after.content_md
    assert _NO_AUDIO_FINGERPRINTS in after.content_md


def test_the_audio_sentence_is_its_own_paragraph_after_the_fingerprints_paragraph() -> None:
    after = _load(date(2026, 12, 2)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 2)
    )
    paragraphs = after.content_md.split("\n\n")
    fingerprints = next(
        i for i, text in enumerate(paragraphs) if text.startswith("### 2.3. Content fingerprints")
    )
    assert "Perceptual hashes of uploaded files and Outputs." in paragraphs[fingerprints]
    assert paragraphs[fingerprints + 1] == _NO_AUDIO_FINGERPRINTS


def test_the_new_version_differs_from_the_previous_one_by_the_documented_changes_only() -> None:
    previous = _load(date(2026, 12, 1)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 1)
    )
    current = _load(date(2026, 12, 2)).current(
        Product.SYNTHARA, LegalDocumentType.PRIVACY, today=date(2026, 12, 2)
    )

    previous_lines = previous.content_md.splitlines()
    current_lines = current.content_md.splitlines()

    # The retention row (S1-P3) and the audio wording (S2a): nothing else changed.
    assert [line for line in current_lines if line not in previous_lines] == [
        _INPUTS_AFTER,
        _NO_AUDIO_FINGERPRINTS,
        _REFERENCES_ROW,
    ]
    assert [line for line in previous_lines if line not in current_lines] == [_INPUTS_BEFORE]
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
