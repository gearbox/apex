"""The planner: turns a brief and the artist's context into a validated plan.

It depends on **ports only** (``src/llm/port.py``, the artist-context provider): no HTTP
client, no database, no API layer - a graph node is a pure function of its state plus the port.
"""
