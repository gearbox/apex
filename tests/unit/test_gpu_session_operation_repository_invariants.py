"""Static invariants for durable GPU operation mutations."""

from __future__ import annotations

import ast
from pathlib import Path


def _is_operation_update(node: ast.expr) -> bool:
    """Whether a chained expression originates at update(GpuSessionOperation)."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "update":
            return (
                len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == ("GpuSessionOperation")
            )
        if isinstance(node.func, ast.Attribute):
            return _is_operation_update(node.func.value)
    return False


def test_every_operation_update_increments_the_public_revision() -> None:
    """A new read-model mutator must not bypass the SSE synchronization token."""
    source_dir = Path(__file__).resolve().parent.parent.parent / "src"
    values_calls = []
    for source_path in source_dir.rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        values_calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "values"
            and _is_operation_update(node.func.value)
        )

    assert values_calls
    for values_call in values_calls:
        revisions = [keyword for keyword in values_call.keywords if keyword.arg == "revision"]
        assert revisions, "Every update(GpuSessionOperation) must increment revision"
        assert all(
            isinstance(keyword.value, ast.BinOp) and isinstance(keyword.value.op, ast.Add)
            for keyword in revisions
        ), "Every update(GpuSessionOperation) must increment revision with addition"
