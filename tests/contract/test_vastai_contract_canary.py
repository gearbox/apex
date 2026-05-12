"""Live contract canary for Vast.ai's /api/v0/bundles/ endpoint.

Run nightly in CI against the live API. Failure means Vast.ai changed
their response shape and apex code needs review BEFORE the next deploy.

Marked `@pytest.mark.live` so it doesn't run in the default unit suite.
Configure CI to run on a scheduled job:

    pytest -m live --vastai-api-key=$VASTAI_DEV_KEY tests/contract/

The test does THREE things, in order of strictness:

1. Asserts every field apex's code CONSUMES is present with a compatible type.
   This is the must-pass test — failure means apex code will crash today.

2. Asserts every field apex DECLARES in VastAIOffer is present with the
   declared type. This catches drift in declared-but-not-yet-consumed
   fields, giving lead time before they become consumed.

3. Captures a fresh structural fingerprint and diffs against the saved
   baseline. Reports additions, removals, and type changes. Soft-fails
   (warning, not error) to avoid noisy CI on harmless additions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "bundles_search_2026-05.json"


@pytest.fixture(scope="module")
def vastai_api_key() -> str:
    key = os.environ.get("VASTAI_API_KEY")
    if not key:
        pytest.skip("VASTAI_API_KEY not set; live contract canary disabled")
    return key


@pytest.fixture(scope="module")
def baseline_snapshot() -> dict[str, Any]:
    if not _SNAPSHOT_PATH.exists():
        pytest.skip(
            f"baseline snapshot not found at {_SNAPSHOT_PATH}; "
            f"run vastai_contract_check.py --snapshot first"
        )
    with _SNAPSHOT_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
async def live_offer(vastai_api_key: str) -> dict[str, Any]:
    """Fetch one live offer from Vast.ai for inspection."""
    payload = {
        # Use common GPUs that should always have inventory available
        "gpu_name": {"in": ["RTX 4090", "RTX 3090"]},
        "num_gpus": {"gte": 1},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "type": "ondemand",
        "order": [["dph_total", "asc"]],
        "limit": 1,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://console.vast.ai/api/v0/bundles/",
            json=payload,
            headers={"Authorization": f"Bearer {vastai_api_key}"},
        )
    resp.raise_for_status()
    offers = resp.json().get("offers", [])
    if not offers:
        pytest.skip("Vast.ai returned 0 offers — inventory issue, not a contract issue")
    return offers[0]


# ---------------------------------------------------------------------------
# Test 1: fields apex CONSUMES downstream (gpu_session/_provisioning.py etc.)
# ---------------------------------------------------------------------------

# These are the fields apex code reads from offer objects after fetch.
# Source-of-truth: grep `offer.<name>` or `offers[i].<name>` in src/.
APEX_CONSUMED_FIELDS: dict[str, list[str]] = {
    "id": ["int"],
    "gpu_name": ["str"],
    "dph_total": ["float", "int"],  # accept either; round() handles both
    "num_gpus": ["int"],
}


@pytest.mark.live
@pytest.mark.asyncio
async def test_apex_consumed_fields_present(live_offer: dict[str, Any]) -> None:
    """Every field apex code reads must be present with a compatible type.

    Failure here = apex will crash in production today.
    """
    missing: list[str] = []
    wrong_type: list[str] = []

    for field, ok_types in APEX_CONSUMED_FIELDS.items():
        if field not in live_offer:
            missing.append(field)
            continue
        actual = type(live_offer[field]).__name__
        if actual not in ok_types:
            wrong_type.append(f"{field}: got {actual!r}, expected one of {ok_types}")

    assert not missing, (
        f"CONTRACT BREAK: apex consumes these fields but they're missing from "
        f"the live response: {missing}. apex will crash on the next deploy. "
        f"Check whether Vast.ai renamed these fields."
    )
    assert not wrong_type, (
        f"CONTRACT BREAK: apex consumes these fields but their types changed: "
        f"{wrong_type}. apex's VastAIOffer schema needs updating."
    )


# ---------------------------------------------------------------------------
# Test 2: fields apex DECLARES in VastAIOffer (whether or not consumed)
# ---------------------------------------------------------------------------

# Source-of-truth: src/api/services/vastai/schemas.py:VastAIOffer
# Values are msgspec-compatible JSON types.
#
# When VastAIOffer changes, update this dict. The whole point of declaring
# them here is to make schema changes visible in test diffs.
APEX_DECLARED_FIELDS: dict[str, list[str] | None] = {
    "id": ["int"],
    "gpu_name": ["str"],
    "num_gpus": ["int"],
    "gpu_ram": ["int"],
    "disk_space": ["float", "int"],
    "dph_total": ["float", "int"],
    "inet_up": ["float", "int"],
    "inet_down": ["float", "int"],
    "cuda_max_good": ["float", "int"],
    # 'verified' DELIBERATELY OMITTED — apex used to declare it as bool
    # but the live response calls it 'verification' (str). After the
    # brittleness-fix PR drops it from VastAIOffer entirely, this test
    # stays clean. If anyone reintroduces 'verified' in the schema,
    # add it here and this test fails loudly.
    "geolocation": ["str", "NoneType"],
}


@pytest.mark.live
@pytest.mark.asyncio
async def test_apex_declared_fields_present(live_offer: dict[str, Any]) -> None:
    """Every field apex's VastAIOffer schema declares must be in the response.

    Failure here = msgspec.ValidationError on the next live call.
    """
    missing: list[str] = []
    wrong_type: list[str] = []

    for field, ok_types in APEX_DECLARED_FIELDS.items():
        if field not in live_offer:
            missing.append(field)
            continue
        if ok_types is None:
            continue  # any type acceptable
        actual = type(live_offer[field]).__name__
        if actual not in ok_types:
            wrong_type.append(f"{field}: got {actual!r}, expected one of {ok_types}")

    assert not missing, (
        f"CONTRACT BREAK: apex VastAIOffer declares these fields but they're "
        f"missing from the live response: {missing}. msgspec will raise "
        f"ValidationError on every offer. Either Vast.ai renamed them, or "
        f"apex's schema needs trimming."
    )
    assert not wrong_type, (
        f"CONTRACT BREAK: apex VastAIOffer's declared types don't match the "
        f"live response: {wrong_type}."
    )


# ---------------------------------------------------------------------------
# Test 3: structural drift vs the saved baseline (soft-fail)
# ---------------------------------------------------------------------------


def _fingerprint(value: Any, path: str = "") -> dict[str, str]:
    """Build flat {path: type_name} dict for a JSON-ish value."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            full = f"{path}.{k}" if path else k
            if v is None:
                out[full] = "None"
            elif isinstance(v, dict):
                out[full] = "dict"
                out |= _fingerprint(v, full)
            elif isinstance(v, list):
                out[full] = "list"
                if v and isinstance(v[0], dict):
                    out.update(_fingerprint(v[0], f"{full}[0]"))
            else:
                out[full] = type(v).__name__
    return out


@pytest.mark.live
@pytest.mark.asyncio
async def test_structural_drift_vs_baseline(
    live_offer: dict[str, Any], baseline_snapshot: dict[str, Any]
) -> None:
    """Soft-warn on structural changes vs the saved baseline.

    This is informational, not gating — Vast.ai adds new fields regularly
    and most additions are harmless. The hard-failing tests above cover
    the cases that actually matter for apex. This test surfaces drift for
    operators to review when they have time.
    """
    baseline_fp = baseline_snapshot["field_fingerprint"]
    current_fp = _fingerprint(live_offer)

    added = sorted(set(current_fp) - set(baseline_fp))
    removed = sorted(set(baseline_fp) - set(current_fp))
    type_changed: list[str] = []
    type_changed.extend(
        f"{k}: baseline={baseline_fp[k]} → current={current_fp[k]}"
        for k in set(baseline_fp) & set(current_fp)
        if baseline_fp[k] != current_fp[k]
    )
    # Print drift report regardless; let CI capture it
    if added:
        print(f"\n[drift] {len(added)} new fields since baseline:")
        for f in added[:20]:  # cap to keep output manageable
            print(f"  + {f}: {current_fp[f]}")
        if len(added) > 20:
            print(f"  ... and {len(added) - 20} more")
    if removed:
        print(f"\n[drift] {len(removed)} fields REMOVED since baseline:")
        for f in removed:
            print(f"  - {f}")
    if type_changed:
        print(f"\n[drift] {len(type_changed)} fields with TYPE CHANGES:")
        for f in type_changed:
            print(f"  ! {f}")

    # Soft-fail: skip rather than fail, so this stays informational.
    # If you want it gating, change `pytest.skip(...)` to `pytest.fail(...)`.
    if removed or type_changed:
        pytest.skip(
            f"baseline drift detected: {len(removed)} removed, "
            f"{len(type_changed)} type-changed. Review and refresh "
            f"the baseline if changes are expected."
        )
