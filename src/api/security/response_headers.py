"""Shared response headers for endpoints that end the caller's own session."""

from __future__ import annotations

CLEAR_SITE_DATA_HEADER: dict[str, str] = {"Clear-Site-Data": '"cache", "storage"'}
"""Purges this origin's HTTP cache and storage. Attach to every response that
ends the calling device's session (single-device logout, logout-all, password
change, password reset, deactivation) so cached content-proxy responses don't
outlive the session that authorized them.

Deliberately omits "executionContexts": that directive forces a
browsing-context reload, which would fight the SPA's own navigation and PWA
lifecycle. "cache" + "storage" is the full set actually needed here.

"storage" clears storage for the origin that *sent* the header — today that's
the API origin only (API and frontend are separate origins), which holds
nothing but cached content responses. See docs/CONFIGURATION.md's API Server
section before ever reverse-proxying this API same-origin with the frontend.
"""
