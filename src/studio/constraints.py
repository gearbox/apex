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
from typing import TYPE_CHECKING, Final

import msgspec
import msgspec.inspect as mi

from src.studio.errors import DocumentIssue

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["TOO_LONG", "TOO_MANY", "collect_count_issues", "collect_length_issues"]

TOO_LONG: Final = "too_long_after_normalization"
"""Issue code: a value is within its wire limit as sent, but past it once normalized."""

TOO_MANY: Final = "too_many_items"
"""Issue code: a list or tuple holds more items than its declared ``max_length``."""


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
    _walk(value, _struct_type(type(value)), path, issues, counts=False)


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
    _walk(value, _struct_type(type(value)), path, issues, counts=True)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _walk(
    value: object, info: mi.Type, path: str, issues: list[DocumentIssue], *, counts: bool
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
                _walk(value, branch, path, issues, counts=counts)
        case mi.StructType(fields=fields):
            for field in fields:
                _walk(
                    getattr(value, field.name),
                    field.type,
                    _join(path, field.encode_name),
                    issues,
                    counts=counts,
                )
        case mi.StrType(max_length=limit) if limit is not None:
            if not counts and isinstance(value, str) and len(value) > limit:
                issues.append(DocumentIssue(path, TOO_LONG))
        case (
            mi.ListType(item_type=item, max_length=limit)
            | mi.VarTupleType(item_type=item, max_length=limit)
        ):
            if isinstance(value, list | tuple):
                if counts and limit is not None and len(value) > limit:
                    issues.append(DocumentIssue(path, TOO_MANY))
                for index, element in enumerate(value):
                    _walk(element, item, f"{path}[{index}]", issues, counts=counts)
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
            case mi.StructType(cls=cls):
                if isinstance(value, cls):
                    return branch
            case mi.ListType() | mi.VarTupleType():
                if isinstance(value, list | tuple):
                    return branch
            case _:
                continue
    return None
