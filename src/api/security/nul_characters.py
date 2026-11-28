"""The API's one rule about U+0000: a request that carries it is refused (``invalid_character``).

PostgreSQL cannot store U+0000: ``text`` / ``varchar`` and ``jsonb`` both refuse it. msgspec
decodes the JSON escape for it into a real NUL, so without this rule one escaped character in any
string field turns the request into a 500 at the INSERT. One check at the boundary covers every
route, the ones not written yet included, which a pattern on each string field would not.

Text that did not come with a request (model output, an aligner's) never passes here; the studio's
walker checks what the server builds (``src/studio/constraints.py``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final, Literal

from litestar.enums import RequestEncodingType
from litestar.params import BodyKwarg

if TYPE_CHECKING:
    from litestar import Request

__all__ = ["NulCharacterError", "NulLocation", "has_json_nul_escape", "refuse_nul_characters"]

type NulLocation = Literal["body", "query", "path"]

_NUL: Final = "\x00"

_ESCAPE_BYTES: Final = rb"\u0000"
"""The six bytes of the escape. Looked for first: a substring search costs a few milliseconds on
the largest body, the pattern below costs most of a second."""

# The escape after an even number of backslashes. After an odd number it is not an escape at all:
# a literal backslash followed by the text "u0000". The ``*+`` is possessive on purpose: with a
# plain ``*``, a body made of backslashes builds about 33 bytes of backtracking state per byte
# (673 MB for 20 MiB). Backtracking can never produce a match here, so the verdicts are the same.
_JSON_NUL_ESCAPE: Final = re.compile(rb"(?<!\\)(?:\\\\)*+\\u0000")

_FORM_ENCODINGS: Final = frozenset(
    {RequestEncodingType.MULTI_PART, RequestEncodingType.URL_ENCODED}
)


class NulCharacterError(Exception):
    """A request carried U+0000. It says where, never what."""

    def __init__(self, location: NulLocation) -> None:
        super().__init__("Text must not contain the NUL character (U+0000).")
        self.location: NulLocation = location


def has_json_nul_escape(body: bytes) -> bool:
    """Whether raw JSON bytes hold an escape that decodes to U+0000.

    The body is not decoded. A raw U+0000 byte needs no check: JSON forbids one inside a string,
    and msgspec refuses it.
    """
    return _ESCAPE_BYTES in body and _JSON_NUL_ESCAPE.search(body) is not None


def _is_json_media_type(media_type: str) -> bool:
    """``application/json`` or ``application/*+json`` (Litestar lower-cases the media type)."""
    return media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )


def _is_json_body(request: Request[Any, Any, Any]) -> bool:
    """Whether this request's body is, or will be read as, JSON.

    A handler with a ``data`` argument decides how its body is decoded by its own declaration:
    Litestar JSON-decodes it **whatever Content-Type the request names**, or when it names none.
    Going by the header alone would let ``text/plain`` carry a NUL past this check. A handler
    without one (a webhook that reads the bytes itself) goes by the header.
    """
    data = request.route_handler.parsed_data_field
    if data is None:
        return _is_json_media_type(request.content_type[0])
    kwarg = data.kwarg_definition
    return not (isinstance(kwarg, BodyKwarg) and kwarg.media_type in _FORM_ENCODINGS)


async def refuse_nul_characters(request: Request[Any, Any, Any]) -> None:
    """The app's ``before_request`` hook: no U+0000 in path parameters, query or JSON body.

    It runs after the route's guards and before any dependency is resolved, so a refused request
    has opened no database session: nothing is read or written, no idempotency key is consumed
    and no tokens are reserved.

    Litestar uses only the innermost ``before_request``, so one declared on a router, controller
    or handler would **replace** this one for those routes. None may be declared; a test holds
    every route of the app to this hook.

    The body is read with ``request.body()``, which enforces the route's body size limit while it
    streams and caches the bytes, so the handler does not read the request a second time.

    **Multipart and other bodies are not scanned**, and a form handler's body is not even read
    here. Uploads can be 80 MiB, and the upload form's text fields (``source_asset_ref``,
    ``source_timestamp_ms``) are parsed into typed values; file names are already cleaned
    (``user_content._CONTROL_CHAR_RUNS``). A new form handler with a free-text field must refuse
    U+0000 itself.

    Raises:
        NulCharacterError: With the location of the first U+0000 found: path, query, then body.
    """
    if any(_NUL in str(value) for value in request.path_params.values()):
        raise NulCharacterError("path")
    if any(_NUL in key or _NUL in value for key, value in request.query_params.multi_items()):
        raise NulCharacterError("query")
    if _is_json_body(request) and has_json_nul_escape(await request.body()):
        raise NulCharacterError("body")
