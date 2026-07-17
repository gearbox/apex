"""Pure health-subsystem transition detection.

No I/O — feed it the previous cycle's persisted ``snapshot_data`` and the
current cycle's ``detailed()`` dict (same shape — see
``HealthService._build_detailed_response``) and get back the list of
subsystem transitions worth an ops notification. Unit-testable in complete
isolation from ``HealthSnapshotWorker``/the DB/Redis.
"""

from __future__ import annotations

from typing import Any, Final

from src.api.schemas.ops_events import HealthTransitionOpsPayload

# A subsystem in one of these states is considered unhealthy for transition
# purposes. `inactive` is deliberately excluded — an inactive checker (e.g. a
# provider that isn't configured) is not a failure.
BAD_STATUSES: Final[frozenset[str]] = frozenset({"degraded", "unhealthy", "unknown"})


def _flatten_snapshot(detailed: dict[str, Any]) -> dict[str, str]:
    """Flatten a DetailedHealthResponse-shaped dict to {subsystem_name: status}.

    Must be used identically against both the previous snapshot's
    ``snapshot_data`` and the current cycle's ``detailed()`` output — they
    are written from the exact same shape (``HealthSnapshotRepository.insert``
    persists ``detailed`` verbatim).
    """
    flat: dict[str, str] = {}

    for category_key in ("infrastructure", "platform_apis"):
        category = detailed.get(category_key)
        if not isinstance(category, dict):
            continue
        for component in category.get("components", []):
            name = component.get("name")
            status = component.get("status")
            if name and status:
                flat[str(name)] = str(status)

    cloud_providers = detailed.get("cloud_providers")
    if isinstance(cloud_providers, dict):
        for product_id, category in cloud_providers.items():
            if not isinstance(category, dict):
                continue
            for component in category.get("components", []):
                name = component.get("name")
                status = component.get("status")
                if name and status:
                    # Namespaced by product — the same checker name (e.g. "grok")
                    # is registered once per product and must not collide.
                    flat[f"{product_id}.{name}"] = str(status)

    gpu_sessions = detailed.get("gpu_sessions")
    if isinstance(gpu_sessions, dict) and (status := gpu_sessions.get("status")):
        flat["gpu_sessions"] = str(status)

    return flat


def detect_transitions(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[HealthTransitionOpsPayload]:
    """Diff two DetailedHealthResponse-shaped dicts into subsystem transitions.

    Semantics (locked):
      - ``previous is None`` -> ``[]`` (first-ever run — baseline, not a
        transition).
      - A subsystem with no previous entry (newly appeared) is also treated
        as baseline for that subsystem — skipped, not fired.
      - Status changed AND new status is bad -> "degraded" payload. Notably,
        degraded -> unhealthy re-fires with the new status — intentional,
        it's a worsening, not a duplicate.
      - Status changed AND old status was bad AND new status is not bad ->
        "restored" payload.
      - Subsystems that disappear from the current snapshot are ignored.
      - Every payload carries ``overall_status`` from the *current* cycle.
    """
    if previous is None:
        return []

    previous_flat = _flatten_snapshot(previous)
    current_flat = _flatten_snapshot(current)
    overall_status = str(current.get("status", ""))

    transitions: list[HealthTransitionOpsPayload] = []
    for subsystem, current_status in current_flat.items():
        previous_status = previous_flat.get(subsystem)
        if previous_status is None or previous_status == current_status:
            continue

        is_now_bad = current_status in BAD_STATUSES
        was_bad = previous_status in BAD_STATUSES
        if is_now_bad or was_bad:
            transitions.append(
                HealthTransitionOpsPayload(
                    subsystem=subsystem,
                    previous_status=previous_status,
                    current_status=current_status,
                    overall_status=overall_status,
                )
            )

    return transitions
