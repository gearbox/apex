"""Versioned legal documents and append-only acceptance records.

- ``registry`` — immutable, file-backed ``LegalDocumentRegistry`` (loaded once at startup).
- ``acceptance`` — ``LegalAcceptanceService``: validates submissions, records events,
  computes the ``lgl`` JWT digest a user has satisfied.
- ``errors`` — domain exceptions mapped to HTTP responses in ``src/api/app.py``.

Deliberately empty of re-exports: ``src.api.security.guards`` imports ``errors``
and must not drag the repository/ORM layer in with it.
"""
