"""Re-check of declared length limits after normalization (RD6).

msgspec enforces ``max_length`` on the **raw** request. Normalization (NFC) can then *expand* a
value: composition exclusions such as U+0958 decompose to two code points and U+FB2C to three, and
never recompose. A value that was within its limit on the wire can therefore leave normalization
past it — and be stored (unreadable versions, silently truncated array columns, 500s).

:func:`collect_length_issues` closes that gap with **one generic walker** over
``msgspec.inspect.type_info``: the limits already declared on the wire Structs are the *only*
source, so they cannot drift from a second copy. Every violating value yields one
:class:`DocumentIssue` (``too_long_after_normalization``) at its wire path, collected together
with the other validation issues (P1-6). Lengths are counted in code points, like msgspec does.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Final, Literal

import msgspec
import msgspec.inspect as mi

from src.studio.errors import DocumentIssue

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "AUDIO_LABEL_MAX",
    "CAST_APPEARANCE_MAX",
    "CAST_DESCRIPTION_MAX",
    "CAST_LABEL_MAX",
    "CAST_LOOK_NAME_MAX",
    "CAST_NAME_MAX",
    "CHAT_MESSAGE_MAX",
    "CHAT_REFUSAL_SENTINEL",
    "CHAT_SENTINEL_BUFFER",
    "CHAT_TITLE_MAX",
    "CHAT_WINDOW_BUDGET",
    "CHAT_WINDOW_MESSAGES",
    "DISCOGRAPHY_BUDGET",
    "DISCOGRAPHY_TITLES_MAX",
    "MAX_ACTIVE_MEMORIES_PER_ARTIST",
    "MAX_AUDIO_VERSIONS_PER_TRACK",
    "MAX_CAST_LOOKS",
    "MAX_CAST_MEMBERS_PER_OWNER",
    "MAX_CAST_REFERENCES",
    "MAX_CHATS_PER_ARTIST",
    "MAX_MESSAGES_PER_CHAT",
    "MAX_TIMELINES_PER_AUDIO",
    "MEMORY_BODY_MAX",
    "MEMORY_CONTEXT_BUDGET",
    "MEMORY_IMAGERY_MAX",
    "MEMORY_IMAGE_MAX",
    "MEMORY_THEMES_MAX",
    "MEMORY_THEME_MAX",
    "MEMORY_TITLE_MAX",
    "OUT_OF_RANGE",
    "TOO_LONG",
    "TOO_MANY",
    "collect_count_issues",
    "collect_length_issues",
    "collect_range_issues",
]

TOO_LONG: Final = "too_long_after_normalization"
"""Issue code: a value is within its wire limit as sent, but past it once normalized."""

TOO_MANY: Final = "too_many_items"
"""Issue code: a list or tuple holds more items than its declared ``max_length``."""

OUT_OF_RANGE: Final = "out_of_range"
"""Issue code: an integer is outside its declared ``ge`` / ``le`` / ``gt`` / ``lt`` bounds."""

MEMORY_TITLE_MAX: Final = 120
"""Characters in an artist memory's title (a single line)."""
MEMORY_BODY_MAX: Final = 2_000
"""Characters in an artist memory's body."""
MEMORY_THEMES_MAX: Final = 12
"""Themes per memory."""
MEMORY_THEME_MAX: Final = 60
"""Characters in one theme."""
MEMORY_IMAGERY_MAX: Final = 20
"""Imagery items per memory."""
MEMORY_IMAGE_MAX: Final = 120
"""Characters in one imagery item."""
MAX_ACTIVE_MEMORIES_PER_ARTIST: Final = 200
"""Active memories one artist can hold (``409 memory_limit_reached``)."""
MEMORY_CONTEXT_BUDGET: Final = 12_000
"""Characters of artist memories one run may send to the model, as rendered into the prompt."""

# -- artist chat (S1-P2c) ---------------------------------------------------------------------

CHAT_TITLE_MAX: Final = 120
"""Characters in a chat's title (a single line)."""
CHAT_MESSAGE_MAX: Final = 4_000
"""Characters in one user message."""
MAX_MESSAGES_PER_CHAT: Final = 200
"""Messages (both roles) one chat holds; then a new chat is started (``409 chat_full``)."""
MAX_CHATS_PER_ARTIST: Final = 50
"""Chats one artist holds (``409 chat_limit_reached``)."""
CHAT_WINDOW_MESSAGES: Final = 20
"""Most messages of a chat's history (the new user message included) one reply is given."""
CHAT_WINDOW_BUDGET: Final = 16_000
"""Characters of history (the new message included) one reply is given."""
DISCOGRAPHY_BUDGET: Final = 6_000
"""Characters of the artist's own tracks a chat reply is given, as rendered into the prompt."""
DISCOGRAPHY_TITLES_MAX: Final = 30
"""Track titles listed in the discography block."""
CHAT_REFUSAL_SENTINEL: Final = "⟦declined⟧"
"""What the model answers, and nothing else, when answering would break the content rules."""
CHAT_SENTINEL_BUFFER: Final = 32
"""Characters of a reply held back before the first delta is published, so a refusal never
reaches the client and is then retracted."""

# -- cast (S1-P3) -----------------------------------------------------------------------------

MAX_CAST_MEMBERS_PER_OWNER: Final = 50
"""Cast members one user holds per product (``409 cast_limit_reached``, a soft cap)."""
MAX_AUDIO_VERSIONS_PER_TRACK: Final = 20
"""Audio versions one track holds (a hard cap, counted under the track lock; S2a-8)."""
AUDIO_LABEL_MAX: Final = 60
"""Characters in an audio version's label."""
MAX_TIMELINES_PER_AUDIO: Final = 50
"""Realized timeline versions one audio version holds (a hard cap, counted under the track lock;
S3-2)."""
MAX_CAST_REFERENCES: Final = 12
"""Reference images one cast member holds (a hard cap, counted under the member lock)."""
MAX_CAST_LOOKS: Final = 12
"""Named looks one cast member holds (a hard cap, counted under the member lock)."""
CAST_NAME_MAX: Final = 100
"""Characters in a cast member's name."""
CAST_LABEL_MAX: Final = 40
"""Characters in a reference's label."""
CAST_LOOK_NAME_MAX: Final = 60
"""Characters in a look's name."""
CAST_APPEARANCE_MAX: Final = 320
"""Characters in a cast member's appearance descriptor (a prompt fragment from S4 on)."""
CAST_DESCRIPTION_MAX: Final = 2_000
"""Characters in a cast member's free-form description."""

type _Check = Literal["length", "count", "range"]


@functools.cache
def _struct_type(cls: type[msgspec.Struct]) -> mi.StructType:
    """The (immutable) ``msgspec.inspect`` description of ``cls``; built once per class."""
    info = mi.type_info(cls)
    assert isinstance(info, mi.StructType)  # noqa: S101 — type_info of a Struct is a StructType
    return info


def collect_length_issues(
    value: msgspec.Struct, issues: list[DocumentIssue], path: str = ""
) -> None:
    """Re-check every ``max_length`` declared on ``type(value)`` against the (normalized) values.

    Args:
        value: A Struct holding *normalized* values. It is usually built with
            ``msgspec.structs.replace`` or a plain constructor, neither of which validates —
            which is the point: this is the validation they skipped.
        issues: Receives one ``DocumentIssue(path, TOO_LONG)`` per violating value, in field
            order. Nothing is raised, so the caller can report them with all other issues.
        path: Prefix of every reported path (the wire path of ``value``; ``""`` for the root).
    """
    _walk(value, _struct_type(type(value)), path, issues, check="length")


def collect_count_issues(
    value: msgspec.Struct, issues: list[DocumentIssue], path: str = ""
) -> None:
    """Re-check every array ``max_length`` declared on ``type(value)`` (the item *counts*).

    Normalization never changes how many items a request holds, so a document that came off the
    wire needs no such check. A document **built in code** (the Suno importer) skips msgspec's
    decoding altogether and must not be able to store more tags or sections than the schema
    allows. The same walker, the same single source of limits; one ``DocumentIssue(path,
    TOO_MANY)`` per offending array, at the array's own path.
    """
    _walk(value, _struct_type(type(value)), path, issues, check="count")


def collect_range_issues(
    value: msgspec.Struct, issues: list[DocumentIssue], path: str = ""
) -> None:
    """Re-check every integer bound declared on ``type(value)`` (``ge`` / ``le`` / ``gt`` / ``lt``).

    Like :func:`collect_count_issues` this is for a document **built in code** (the planner's
    output), which never went through msgspec's decoding: a ``tempo_bpm`` of 9000 or a section
    ``duration_seconds`` of 0 must not reach storage because nothing enforced the wire limit.
    One ``DocumentIssue(path, OUT_OF_RANGE)`` per offending integer.
    """
    _walk(value, _struct_type(type(value)), path, issues, check="range")


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _walk(
    value: object, info: mi.Type, path: str, issues: list[DocumentIssue], *, check: _Check
) -> None:
    # PATCH fields: ``type_info`` drops ``UnsetType`` from the declared type, so an unset field
    # reports a bare ``StrType`` / ``ListType``. Skip it by identity *before* any type dispatch.
    if value is msgspec.UNSET or value is None:
        return
    match info:
        case mi.UnionType(types=branches):
            # Pick the branch the value actually is; walking every branch would check an
            # optional struct both as ``None`` and as itself.
            branch = _branch_for(value, branches)
            if branch is not None:
                _walk(value, branch, path, issues, check=check)
        case mi.StructType(fields=fields):
            for field in fields:
                _walk(
                    getattr(value, field.name),
                    field.type,
                    _join(path, field.encode_name),
                    issues,
                    check=check,
                )
        case mi.StrType(max_length=limit) if limit is not None:
            if check == "length" and isinstance(value, str) and len(value) > limit:
                issues.append(DocumentIssue(path, TOO_LONG))
        case mi.IntType(ge=ge, le=le, gt=gt, lt=lt):
            if check == "range" and isinstance(value, int) and not isinstance(value, bool):
                below = (ge is not None and value < ge) or (gt is not None and value <= gt)
                above = (le is not None and value > le) or (lt is not None and value >= lt)
                if below or above:
                    issues.append(DocumentIssue(path, OUT_OF_RANGE))
        case (
            mi.ListType(item_type=item, max_length=limit)
            | mi.VarTupleType(item_type=item, max_length=limit)
        ):
            if isinstance(value, list | tuple):
                if check == "count" and limit is not None and len(value) > limit:
                    issues.append(DocumentIssue(path, TOO_MANY))
                for index, element in enumerate(value):
                    _walk(element, item, f"{path}[{index}]", issues, check=check)
        case _:
            # int / UUID / enum / bool / ...: untouched by normalization.
            return


def _branch_for(value: object, branches: Iterable[mi.Type]) -> mi.Type | None:
    """The union member that describes ``value`` at runtime (``None`` when none does)."""
    for branch in branches:
        match branch:
            case mi.StrType():
                if isinstance(value, str):
                    return branch
            case mi.IntType():
                if isinstance(value, int) and not isinstance(value, bool):
                    return branch
            case mi.StructType(cls=cls):
                if isinstance(value, cls):
                    return branch
            case mi.ListType() | mi.VarTupleType():
                if isinstance(value, list | tuple):
                    return branch
            case _:
                continue
    return None
