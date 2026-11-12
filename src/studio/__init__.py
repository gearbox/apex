"""Media Project Studio: documents, validation, services and renderers.

Bounded context (D2). Layout::

    documents.py     versioned Plan / Lyrics document types
    validation.py    pure normalization, issue collection, content hashing
    errors.py        StudioError hierarchy (one handler in src/api/app.py)
    services/        per-request services (artists, projects, tracks, versions, export)
    renderers/       deterministic target renderers (Suno) over projections

Persistence lives in ``src/db/{models,repositories}/studio.py``; HTTP in
``src/api/routes/studio_*.py``. The library package never imports this one (D12).
"""
