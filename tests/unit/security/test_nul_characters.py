"""N2: the API never accepts U+0000 - the escape pattern, the hook, and that no route escapes it."""

from __future__ import annotations

import itertools
import json
import os
import tracemalloc
from typing import TYPE_CHECKING, Annotated, Any, Final

os.environ.setdefault("JWT_SECRET_KEY", "test-nul-characters-key-32-bytes-long!!")

import msgspec
import pytest
from litestar import Litestar, Request, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body, BodyKwarg
from litestar.routes import HTTPRoute
from litestar.testing import TestClient
from structlog.testing import capture_logs

from src.api.app import create_app, nul_character_handler
from src.api.security.nul_characters import (
    NulCharacterError,
    has_json_nul_escape,
    refuse_nul_characters,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx

pytestmark = pytest.mark.unit

BS: Final = "\\"
NUL: Final = chr(0)
ESCAPE: Final = BS + "u0000"
"""The six characters a JSON document writes for U+0000."""
MARKER: Final = "SECRET-MARKER"
MESSAGE: Final = "Text must not contain the NUL character (U+0000)."


def _json_string(inner: str) -> bytes:
    return f'"{inner}"'.encode()


# ---------------------------------------------------------------------------
# The pattern
# ---------------------------------------------------------------------------


class TestTheEscapePattern:
    @pytest.mark.parametrize(
        ("body", "found"),
        [
            pytest.param(_json_string("a" + ESCAPE + "b"), True, id="an-escape"),
            pytest.param(_json_string("a" + BS + ESCAPE + "b"), False, id="escaped-backslash"),
            pytest.param(_json_string(BS + BS + ESCAPE), True, id="backslash-then-a-nul-escape"),
            pytest.param(_json_string(ESCAPE + "1"), True, id="followed-by-a-digit"),
            pytest.param(_json_string(BS + "u0001"), False, id="another-control-character"),
            pytest.param(b'{"n": 1, "ok": true}', False, id="no-string-at-all"),
            pytest.param(b"", False, id="empty"),
        ],
    )
    def test_table(self, body: bytes, found: bool) -> None:
        assert has_json_nul_escape(body) is found

    def test_it_matches_exactly_when_the_decoded_string_holds_a_nul(self) -> None:
        """Every sequence of up to five of these pieces, judged by a real JSON decoder."""
        pieces = (
            BS + BS,  # an escaped backslash
            ESCAPE,
            BS + "u0001",
            BS + "n",
            BS + '"',
            "u0000",
            "0",
            "a",
        )
        verdicts: set[bool] = set()
        for length in range(6):
            for chosen in itertools.product(pieces, repeat=length):
                body = _json_string("".join(chosen))
                expected = NUL in json.loads(body)
                assert has_json_nul_escape(body) is expected, body
                verdicts.add(expected)

        assert verdicts == {True, False}

    def test_a_body_of_backslashes_costs_no_backtracking_memory(self) -> None:
        """A plain ``*`` in the pattern allocates about 40 MB here (673 MB at 20 MiB)."""
        # An even run, then "u0000": the six bytes are there, but they are not an escape.
        body = BS.encode() * (1 << 20) + b"u0000"
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            found = has_json_nul_escape(body)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert found is False
        assert peak - before < 1_000_000


# ---------------------------------------------------------------------------
# The hook, on a small app
# ---------------------------------------------------------------------------


class _Note(msgspec.Struct):
    text: str


@post("/json")
async def _json_route(data: _Note) -> dict[str, str]:
    return {"text": data.text}


@post("/raw")
async def _raw_route(request: Request[Any, Any, Any]) -> dict[str, int]:
    """Like a webhook: no ``data`` argument, the bytes are read by hand."""
    return {"size": len(await request.body())}


@post("/upload")
async def _upload_route(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> dict[str, int]:
    return {"size": len(await data.read())}


@get("/items/{name:str}")
async def _item_route(name: str) -> dict[str, str]:
    return {"name": name}


@get("/search")
async def _search_route(term: str = "") -> dict[str, str]:
    return {"term": term}


def _small_app(**overrides: Any) -> Litestar:
    return Litestar(
        route_handlers=[_json_route, _raw_route, _upload_route, _item_route, _search_route],
        before_request=refuse_nul_characters,
        exception_handlers={NulCharacterError: nul_character_handler},
        signature_types=[UploadFile],
        **overrides,
    )


@pytest.fixture
def client() -> Iterator[TestClient[Litestar]]:
    with TestClient(app=_small_app()) as test_client:
        yield test_client


def _assert_refused(response: httpx.Response, location: str) -> None:
    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": "invalid_character",
        "message": MESSAGE,
        "status_code": 400,
        "detail": {"location": location},
    }


def _note(inner: str) -> bytes:
    return f'{{"text":"{inner}"}}'.encode()


class TestJsonBodies:
    @pytest.mark.parametrize(
        "content_type",
        [
            "application/json",
            "Application/JSON; charset=utf-8",
            "application/problem+json",
            # Litestar JSON-decodes a ``data`` argument whatever the header says, or without one.
            "text/plain",
            "application/octet-stream",
            "multipart/form-data; boundary=x",
            "application/x-www-form-urlencoded",
            None,
        ],
    )
    def test_a_json_handler_is_checked_whatever_the_content_type(
        self, client: TestClient[Litestar], content_type: str | None
    ) -> None:
        headers = {"content-type": content_type} if content_type else {}

        response = client.post("/json", content=_note(MARKER + ESCAPE), headers=headers)

        _assert_refused(response, "body")
        assert MARKER not in response.text

    def test_a_nul_in_an_object_key_is_refused(self, client: TestClient[Litestar]) -> None:
        response = client.post(
            "/json",
            content=f'{{"te{ESCAPE}xt":"a"}}'.encode(),
            headers={"content-type": "application/json"},
        )

        _assert_refused(response, "body")

    def test_a_literal_backslash_u0000_is_ordinary_text(self, client: TestClient[Litestar]) -> None:
        response = client.post("/json", json={"text": ESCAPE})

        assert response.status_code == 201, response.text
        assert response.json() == {"text": ESCAPE}

    @pytest.mark.parametrize(
        "content_type", ["application/json", "application/vnd.apex.event+json"]
    )
    def test_a_handler_that_reads_the_bytes_itself_is_checked_when_they_are_json(
        self, client: TestClient[Litestar], content_type: str
    ) -> None:
        response = client.post(
            "/raw", content=_note(ESCAPE), headers={"content-type": content_type}
        )

        _assert_refused(response, "body")

    def test_a_handler_that_reads_the_bytes_itself_gets_other_bodies_unscanned(
        self, client: TestClient[Litestar]
    ) -> None:
        body = _note(ESCAPE)

        response = client.post("/raw", content=body, headers={"content-type": "text/plain"})

        assert response.status_code == 201, response.text
        assert response.json() == {"size": len(body)}

    def test_the_handler_still_reads_a_body_the_hook_has_read(
        self, client: TestClient[Litestar]
    ) -> None:
        body = _note("plain")

        response = client.post("/raw", content=body, headers={"content-type": "application/json"})

        assert response.json() == {"size": len(body)}

    def test_the_body_limit_applies_before_anything_is_scanned(self) -> None:
        with TestClient(app=_small_app(request_max_body_size=16)) as limited:
            response = limited.post(
                "/json",
                content=_note(MARKER + ESCAPE),
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 413, response.text

    def test_a_multipart_handler_is_never_read_by_the_hook(
        self, client: TestClient[Litestar], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _must_not_be_called(_self: Request[Any, Any, Any]) -> bytes:
            raise AssertionError("the hook read a multipart body")

        monkeypatch.setattr(Request, "body", _must_not_be_called)
        payload = _note(ESCAPE) + b"\x00"

        response = client.post("/upload", files={"data": ("a.bin", payload)})

        assert response.status_code == 201, response.text
        assert response.json() == {"size": len(payload)}


class TestQueryAndPath:
    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("/search?term=a%00b", id="value"),
            pytest.param("/search?term=ok&term=%00", id="second-value-of-a-key"),
            pytest.param("/search?%00=1", id="key"),
            pytest.param("/search?undeclared=%00", id="a-parameter-no-handler-reads"),
        ],
    )
    def test_a_nul_in_the_query_is_refused(self, client: TestClient[Litestar], url: str) -> None:
        _assert_refused(client.get(url), "query")

    def test_a_nul_in_a_path_parameter_is_refused(self, client: TestClient[Litestar]) -> None:
        _assert_refused(client.get("/items/a%00b"), "path")

    @pytest.mark.parametrize(
        "url",
        [
            "/search?term=a%2500b",
            "/search?term=0",
            # The test client percent-decodes a path twice (a server does it once), so this is
            # what a server delivers for "a%2500": the three characters "%00".
            "/items/a%252500",
        ],
    )
    def test_an_encoded_percent_is_ordinary_text(
        self, client: TestClient[Litestar], url: str
    ) -> None:
        assert client.get(url).status_code == 200


class TestTheLogEvent:
    def test_it_says_where_and_nothing_else(self, client: TestClient[Litestar]) -> None:
        with capture_logs() as logs:
            client.post("/json", json={"text": f"{MARKER}{NUL}tail"})

        (event,) = [log for log in logs if log["event"] == "http.invalid_character"]
        assert event == {
            "event": "http.invalid_character",
            "log_level": "warning",
            "path": "/json",
            "method": "POST",
            "location": "body",
        }
        assert MARKER not in repr(logs)


# ---------------------------------------------------------------------------
# Every route of the real app
# ---------------------------------------------------------------------------


def _app_routes() -> list[Any]:
    return list(create_app().routes)


class TestNoRouteEscapes:
    def test_every_route_of_the_app_runs_the_hook(self) -> None:
        """Litestar runs only the innermost ``before_request``: declaring one on a router,
        controller or handler would silently switch this check off for its routes. A route that
        is not HTTP has no ``before_request`` at all and would need its own check."""
        routes = _app_routes()
        assert all(isinstance(route, HTTPRoute) for route in routes)

        handlers = [(route.path, handler) for route in routes for handler in route.route_handlers]
        replaced = [
            (path, sorted(handler.http_methods))
            for path, handler in handlers
            if handler.resolve_before_request() is not refuse_nul_characters
        ]

        assert len(handlers) > 100
        assert not replaced

    def test_the_only_form_handler_is_the_multipart_upload(self) -> None:
        """Form bodies are not scanned. A new form handler has to decide what refuses U+0000 in
        its text fields, then join this set."""
        form_encodings = {RequestEncodingType.MULTI_PART, RequestEncodingType.URL_ENCODED}
        found = {
            (method, route.path)
            for route in _app_routes()
            for handler in route.route_handlers
            if handler.parsed_data_field is not None
            and isinstance(handler.parsed_data_field.kwarg_definition, BodyKwarg)
            and handler.parsed_data_field.kwarg_definition.media_type in form_encodings
            for method in handler.http_methods
        }

        assert found == {("POST", "/v1/storage/upload")}
