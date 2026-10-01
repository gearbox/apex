"""Keep all four account recovery route response metadata aligned with runtime HTTP 200 responses."""

from __future__ import annotations

from litestar import Litestar
from litestar.openapi.config import OpenAPIConfig

from src.api.routes.auth import AuthController


def test_account_recovery_routes_document_http_200() -> None:
    app = Litestar(
        [AuthController],
        openapi_config=OpenAPIConfig(title="test", version="1"),
    )
    schema = app.openapi_schema.to_schema()

    for path in (
        "/v1/auth/forgot-password",
        "/v1/auth/verify-email",
        "/v1/auth/resend-verification",
        "/v1/auth/reset-password",
    ):
        responses = schema["paths"][path]["post"]["responses"]
        assert "200" in responses, path
        assert "201" not in responses, path
