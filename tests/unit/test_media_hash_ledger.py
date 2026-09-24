"""Media-type resolution for staged ledger rows."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.api.services.media_hash_ledger import MediaHashLedger
from src.core.enums import MediaHashMediaType
from src.core.media_hash import HashSample, HashSet, PdqHash

pytestmark = pytest.mark.unit


def _hash_set() -> HashSet:
    return HashSet(
        profile_id="pdq-image-rgb-white-v1",
        sampling_profile="still-v1",
        samples=(HashSample(pdq=PdqHash(bits=b"\x00" * 32, quality=100), sample_index=0),),
    )


def _row(media_format: object) -> MagicMock:
    return MagicMock(
        id=uuid4(),
        user_id=uuid4(),
        job_id=uuid4(),
        product_id="vex",
        is_thumbnail=False,
        format=media_format,
    )


@pytest.mark.parametrize(
    "media_format,expected",
    [("png", MediaHashMediaType.IMAGE), ("webm", MediaHashMediaType.VIDEO)],
)
async def test_media_type_follows_persisted_format(
    media_format: str, expected: MediaHashMediaType
) -> None:
    session = MagicMock()
    await MediaHashLedger(session).register_output(_row(media_format), _hash_set())
    (rows,) = session.add_all.call_args.args
    assert [row.source_media_type for row in rows] == [expected]


@pytest.mark.parametrize("media_format", [None, "", "bmp", MagicMock()])
async def test_unknown_format_is_refused_rather_than_defaulted(media_format: object) -> None:
    session = MagicMock()
    with pytest.raises(ValueError, match="unknown media format for ledger row"):
        await MediaHashLedger(session).register_upload(_row(media_format), _hash_set())
    session.add_all.assert_not_called()
