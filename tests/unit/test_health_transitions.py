"""Unit tests for the pure health-subsystem transition detector."""

from __future__ import annotations

from src.api.services.health.transitions import BAD_STATUSES, detect_transitions


def _detailed(
    *,
    overall: str = "healthy",
    infra_status: str = "healthy",
    grok_vex_status: str | None = None,
    gpu_sessions_status: str | None = None,
) -> dict:  # type: ignore[type-arg]
    detailed: dict = {  # type: ignore[type-arg]
        "status": overall,
        "checked_at": "2026-01-01T00:00:00Z",
        "infrastructure": {
            "status": infra_status,
            "components": [{"name": "redis", "status": infra_status, "latency_ms": 1}],
        },
        "platform_apis": {"status": "inactive", "components": []},
        "cloud_providers": {},
    }
    if grok_vex_status is not None:
        detailed["cloud_providers"]["vex"] = {
            "status": grok_vex_status,
            "components": [{"name": "grok", "status": grok_vex_status, "latency_ms": 5}],
        }
    if gpu_sessions_status is not None:
        detailed["gpu_sessions"] = {
            "status": gpu_sessions_status,
            "total": 1,
            "healthy": 1,
            "stale": 0,
        }
    return detailed


class TestBaseline:
    def test_previous_none_returns_empty(self) -> None:
        current = _detailed(infra_status="degraded")
        assert detect_transitions(None, current) == []

    def test_no_change_returns_empty(self) -> None:
        previous = _detailed(infra_status="healthy")
        current = _detailed(infra_status="healthy")
        assert detect_transitions(previous, current) == []


class TestDegrading:
    def test_healthy_to_degraded_fires(self) -> None:
        previous = _detailed(infra_status="healthy")
        current = _detailed(infra_status="degraded", overall="degraded")

        transitions = detect_transitions(previous, current)

        assert len(transitions) == 1
        t = transitions[0]
        assert t.subsystem == "redis"
        assert t.previous_status == "healthy"
        assert t.current_status == "degraded"
        assert t.overall_status == "degraded"

    def test_degraded_to_unhealthy_refires_with_new_status(self) -> None:
        """A worsening within the bad range re-fires — not deduplicated."""
        previous = _detailed(infra_status="degraded")
        current = _detailed(infra_status="unhealthy", overall="unhealthy")

        transitions = detect_transitions(previous, current)

        assert len(transitions) == 1
        assert transitions[0].previous_status == "degraded"
        assert transitions[0].current_status == "unhealthy"
        assert transitions[0].current_status in BAD_STATUSES


class TestRestoring:
    def test_unhealthy_to_healthy_fires_restored(self) -> None:
        previous = _detailed(infra_status="unhealthy")
        current = _detailed(infra_status="healthy")

        transitions = detect_transitions(previous, current)

        assert len(transitions) == 1
        assert transitions[0].previous_status == "unhealthy"
        assert transitions[0].current_status == "healthy"

    def test_degraded_to_inactive_fires_restored(self) -> None:
        """inactive is not a bad status — degraded -> inactive counts as restored."""
        previous = _detailed(grok_vex_status="degraded")
        current = _detailed(grok_vex_status="inactive")

        transitions = detect_transitions(previous, current)

        assert len(transitions) == 1
        assert transitions[0].subsystem == "vex.grok"
        assert transitions[0].previous_status == "degraded"
        assert transitions[0].current_status == "inactive"


class TestIgnoredCases:
    def test_disappearing_subsystem_is_ignored(self) -> None:
        previous = _detailed(grok_vex_status="unhealthy")
        current = _detailed()  # no cloud_providers entry at all

        assert detect_transitions(previous, current) == []

    def test_new_subsystem_with_no_baseline_is_ignored(self) -> None:
        previous = _detailed()  # no gpu_sessions entry
        current = _detailed(gpu_sessions_status="unhealthy")

        assert detect_transitions(previous, current) == []


class TestOverallStatusCarriesFromCurrent:
    def test_overall_status_is_current_not_previous(self) -> None:
        previous = _detailed(infra_status="healthy", overall="healthy")
        current = _detailed(infra_status="degraded", overall="degraded")

        transitions = detect_transitions(previous, current)

        assert transitions[0].overall_status == "degraded"
