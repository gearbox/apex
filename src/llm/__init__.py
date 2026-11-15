"""Shared LLM access: the port, a strict-schema converter, the model registry and one adapter.

Nothing here knows who the caller is. Consumers (a feature's planner, a prompt-extension step, ...)
depend on :mod:`src.llm.port` and are handed a ready port by wiring; only wiring sees the registry
and the factory.
"""
