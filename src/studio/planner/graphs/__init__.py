"""LangGraph graphs of the planner - the only place ``langgraph`` may be imported.

Importing this package first pins LangSmith / LangChain tracing to ``false`` (and refuses to go on
if the environment asks for tracing), *before* any ``langgraph`` module can read those variables:
the brief and the artist's persona must reach the LLM provider and nowhere else (P2-M9).
"""

from src.core.llm_tracing import force_tracing_off

force_tracing_off()
