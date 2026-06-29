"""Unit tests for JWTService content token methods."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from src.api.security.jwt import JWTConfig, JWTService

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"


@pytest.fixture
def jwt_config() -> JWTConfig:
    return JWTConfig(secret_key=TEST_SECRET)


@pytest.fixture
def jwt_service(jwt_config: JWTConfig) -> JWTService:
    return JWTService(jwt_config)


class TestCreateContentToken:
    def test_returns_token_and_expiry(self, jwt_service: JWTService) -> None:
        token, expires_at = jwt_service.create_content_token(
            uuid4(), product_id="vex", ttl=timedelta(hours=24)
        )
        assert isinstance(token, str)
        assert len(token) > 0
        assert expires_at is not None

    def test_round_trip_valid(self, jwt_service: JWTService) -> None:
        uid = uuid4()
        token, _ = jwt_service.create_content_token(uid, product_id="vex", ttl=timedelta(hours=1))
        payload = jwt_service.decode_content_token(token)
        assert payload is not None
        assert payload.sub == str(uid)
        assert payload.type == "content"
        assert payload.product_id == "vex"

    def test_carries_product_id(self, jwt_service: JWTService) -> None:
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id="synthara", ttl=timedelta(hours=1)
        )
        payload = jwt_service.decode_content_token(token)
        assert payload is not None
        assert payload.product_id == "synthara"


class TestDecodeContentToken:
    def test_expired_returns_none(self) -> None:
        expired_service = JWTService(JWTConfig(secret_key=TEST_SECRET))
        token, _ = expired_service.create_content_token(
            uuid4(), product_id="vex", ttl=timedelta(seconds=-1)
        )
        assert expired_service.decode_content_token(token) is None

    def test_tampered_returns_none(self, jwt_service: JWTService) -> None:
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id="vex", ttl=timedelta(hours=1)
        )
        tampered = f"{token[:-5]}XXXXX"
        assert jwt_service.decode_content_token(tampered) is None

    def test_wrong_secret_returns_none(self, jwt_service: JWTService) -> None:
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id="vex", ttl=timedelta(hours=1)
        )
        other = JWTService(JWTConfig(secret_key="completely_different_secret_key_32bytes_"))
        assert other.decode_content_token(token) is None

    def test_access_token_rejected_as_content(self, jwt_service: JWTService) -> None:
        """An access token must not authenticate via the content cookie slot."""
        uid = uuid4()
        access_token, _ = jwt_service.create_access_token(uid, product_id="vex")
        assert jwt_service.decode_content_token(access_token) is None

    def test_content_token_rejected_as_access(self, jwt_service: JWTService) -> None:
        """A content token must not authenticate via the Bearer access slot."""
        token, _ = jwt_service.create_content_token(
            uuid4(), product_id="vex", ttl=timedelta(hours=1)
        )
        assert jwt_service.decode_access_token(token) is None

    def test_garbage_input_returns_none(self, jwt_service: JWTService) -> None:
        assert jwt_service.decode_content_token("not.a.jwt") is None

    def test_empty_string_returns_none(self, jwt_service: JWTService) -> None:
        assert jwt_service.decode_content_token("") is None
