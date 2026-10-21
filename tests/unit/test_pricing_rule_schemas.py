"""Admin pricing-rule request validation for per-second and per-resolution rules."""

from __future__ import annotations

import msgspec
import pytest

from src.api.schemas.billing import CreatePricingRuleRequest, PatchPricingRuleRequest
from src.core.enums import VideoResolution

pytestmark = pytest.mark.unit


def _decode(payload: str) -> CreatePricingRuleRequest:
    return msgspec.json.decode(payload.encode(), type=CreatePricingRuleRequest)


def test_per_second_rule_may_have_zero_flat_cost() -> None:
    rule = _decode(
        '{"provider":"alibaba","generation_type":"t2v","token_cost":0,'
        '"per_second_token_cost":4,"video_resolution":"1080p"}'
    )

    assert rule.per_second_token_cost == 4
    assert rule.video_resolution is VideoResolution.RES_1080P


def test_rule_costing_nothing_is_rejected() -> None:
    with pytest.raises(msgspec.ValidationError, match="token_cost > 0 or per_second_token_cost"):
        _decode('{"provider":"p","generation_type":"t2v","token_cost":0}')


def test_defaults_keep_existing_flat_rules_valid() -> None:
    rule = _decode('{"provider":"grok","generation_type":"t2i","token_cost":5}')

    assert rule.per_second_token_cost == 0
    assert rule.video_resolution is None


def test_unknown_resolution_is_rejected() -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode('{"provider":"p","generation_type":"t2v","token_cost":1,"video_resolution":"4k"}')


def test_patch_can_change_per_second_cost_but_not_resolution() -> None:
    patch_request = msgspec.json.decode(
        b'{"per_second_token_cost":9}', type=PatchPricingRuleRequest
    )
    assert patch_request.per_second_token_cost == 9

    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"video_resolution":"720p"}', type=PatchPricingRuleRequest)
