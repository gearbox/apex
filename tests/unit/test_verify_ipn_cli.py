"""Tests for the offline NowPayments IPN verification CLI (D4)."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from src.api.services.payments.ipn_canonical import canonical_bytes, parse_ipn_body
from src.cli.verify_ipn import app

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()

_SECRET = "cli-test-ipn-secret"


def _sign(raw: bytes) -> str:
    canonical = canonical_bytes(parse_ipn_body(raw))
    return hmac.new(_SECRET.encode(), canonical, hashlib.sha512).hexdigest()


def _write_body(tmp_path: Path, raw: bytes) -> Path:
    path = tmp_path / "capture.json"
    path.write_bytes(raw)
    return path


def _valid_raw_body() -> bytes:
    order_id = json.dumps({"payment_id": "9d2b8f2e-7f2b-4c9a-8f2e-7f2b4c9a8f2e"})
    return json.dumps(
        {"payment_status": "waiting", "payment_id": "np-1", "order_id": order_id}
    ).encode()


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32bytes")
    monkeypatch.setenv("NOWPAYMENTS_IPN_SECRET_VEX", _SECRET)
    monkeypatch.delenv("NOWPAYMENTS_IPN_SIGNATURE", raising=False)


def test_valid_body_and_signature_succeeds(tmp_path: Path) -> None:
    raw = _valid_raw_body()
    body_path = _write_body(tmp_path, raw)

    result = runner.invoke(
        app, ["--body", str(body_path), "--signature", _sign(raw), "--product", "vex"]
    )

    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_tampered_signature_reports_signature_mismatch(tmp_path: Path) -> None:
    body_path = _write_body(tmp_path, _valid_raw_body())

    result = runner.invoke(
        app,
        ["--body", str(body_path), "--signature", "deadbeef", "--product", "vex"],
    )

    assert result.exit_code == 1
    assert "signature_mismatch" in result.output
    assert _SECRET not in result.output


def test_broken_json_reports_malformed_json(tmp_path: Path) -> None:
    body_path = _write_body(tmp_path, b"not json")

    result = runner.invoke(
        app,
        ["--body", str(body_path), "--signature", "deadbeef", "--product", "vex"],
    )

    assert result.exit_code == 1
    assert "malformed_json" in result.output


def test_missing_signature_fails_fast_without_running_verification(tmp_path: Path) -> None:
    body_path = _write_body(tmp_path, b"{}")

    result = runner.invoke(app, ["--body", str(body_path), "--product", "vex"])

    assert result.exit_code == 1
    assert "No signature provided" in result.output


def test_signature_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _valid_raw_body()
    monkeypatch.setenv("NOWPAYMENTS_IPN_SIGNATURE", _sign(raw))
    body_path = _write_body(tmp_path, raw)

    result = runner.invoke(app, ["--body", str(body_path), "--product", "vex"])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_missing_body_file_fails() -> None:
    result = runner.invoke(
        app,
        ["--body", "/nonexistent/capture.json", "--signature", "x", "--product", "vex"],
    )

    assert result.exit_code != 0


def test_matrix_mode_marks_the_matching_recipe(tmp_path: Path) -> None:
    raw = _valid_raw_body()
    body_path = _write_body(tmp_path, raw)

    result = runner.invoke(
        app,
        ["--body", str(body_path), "--signature", _sign(raw), "--product", "vex", "--matrix"],
    )

    assert result.exit_code == 0, result.output
    assert "<<< MATCH" in result.output
    assert "canonical_rawnumber(shipped)  <<< MATCH" in result.output
    assert _SECRET not in result.output


def test_matrix_mode_marks_nothing_for_bad_signature(tmp_path: Path) -> None:
    body_path = _write_body(tmp_path, _valid_raw_body())

    result = runner.invoke(
        app,
        ["--body", str(body_path), "--signature", "deadbeef", "--product", "vex", "--matrix"],
    )

    assert result.exit_code == 0, result.output
    assert "<<< MATCH" not in result.output
