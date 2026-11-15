"""The shared LLM layer's structural rules, as durable tests (invariants 21 and 23).

``src/llm`` is a package any feature may use: it must not know who calls it. These tests read the
source with ``ast`` (never a text grep, so a comment or a string cannot trip them).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
LLM_DIR = SRC / "llm"
LLM_FILES = sorted(LLM_DIR.glob("*.py"))
SHARED_FILES = [*LLM_FILES, SRC / "core/llm_models.py"]

ALLOWED_THIRD_PARTY = frozenset({"httpx", "msgspec", "structlog", "pydantic"})
"""Everything ``src/llm`` may import besides the standard library and ``src.core``."""
FORBIDDEN_FIRST_PARTY = ("src.studio", "src.api", "src.db", "src.workers")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _code_string_literals(path: Path) -> list[str]:
    """Every string constant of a module that is not a docstring (a bare string statement)."""
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC))


def test_the_file_set_is_what_the_layout_says() -> None:
    assert {p.name for p in LLM_FILES} == {
        "__init__.py",
        "port.py",
        "strict_schema.py",
        "openai_compatible.py",
        "registry.py",
        "factory.py",
    }
    # No Python *source* lives under the old studio location. The directory itself may still exist
    # as a leftover ``__pycache__`` from another checkout, which says nothing about this branch.
    assert not list((SRC / "studio" / "llm").rglob("*.py"))


# -- invariant 21: the layer imports nothing from its consumers ------------------------------------


@pytest.mark.parametrize("path", SHARED_FILES, ids=_rel)
def test_the_shared_layer_never_imports_a_consumer(path: Path) -> None:
    offenders = {
        m
        for m in _imported_modules(path)
        if any(m == name or m.startswith(f"{name}.") for name in FORBIDDEN_FIRST_PARTY)
    }
    assert not offenders, f"{path.relative_to(ROOT)} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", LLM_FILES, ids=_rel)
def test_the_llm_package_imports_only_the_allowed_libraries(path: Path) -> None:
    for module in _imported_modules(path):
        root = module.split(".")[0]
        if root == "src":
            assert module.startswith(("src.core", "src.llm")), (path.name, module)
        else:
            assert root in sys.stdlib_module_names or root in ALLOWED_THIRD_PARTY, (
                path.name,
                module,
            )


def test_core_llm_models_imports_no_first_party_application_code() -> None:
    for module in _imported_modules(SRC / "core/llm_models.py"):
        assert not module.startswith("src"), module


@pytest.mark.parametrize("path", SHARED_FILES, ids=_rel)
def test_the_shared_layer_does_not_name_its_consumers(path: Path) -> None:
    """No ``StudioRunKind``, no "studio" anywhere - not even in a docstring or a message."""
    assert not re.search(r"studio", path.read_text(), flags=re.IGNORECASE), path


def test_only_wiring_and_the_executor_see_the_registry_and_the_factory() -> None:
    """Consumers import ``src.llm.port``; the registry, factory and adapter are for wiring."""
    wiring_only = {"src.llm.registry", "src.llm.factory", "src.llm.openai_compatible"}
    allowed = {
        "api/dependencies/common.py",
        "studio/services/run_executor.py",  # the registry type, for the model id of a claimed run
    }
    importers = {
        _rel(path)
        for path in SRC.rglob("*.py")
        if LLM_DIR not in path.parents and wiring_only & _imported_modules(path)
    }
    assert importers <= allowed, sorted(importers - allowed)
    assert "api/dependencies/common.py" in importers


# -- the adapter names no vendor and no model id ---------------------------------------------------


def test_the_adapter_names_no_vendor() -> None:
    text = (LLM_DIR / "openai_compatible.py").read_text().replace("OpenAICompatible", "")
    assert not re.search(r"xai|grok|qwen|dashscope|aliyun|openai", text, flags=re.IGNORECASE)


@pytest.mark.parametrize("path", SHARED_FILES, ids=_rel)
def test_no_model_id_or_vendor_literal_in_the_shared_layer(path: Path) -> None:
    literals = _code_string_literals(path)
    assert not [
        lit for lit in literals if re.search(r"grok|qwen|xai|dashscope|gpt-|claude-", lit, re.I)
    ], path


# -- privacy: nothing here logs or raises with request or response text ----------------------------

_LOGGER_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical"})
_LOG_FIELDS = frozenset(
    {"model_key", "status_code", "duration_ms", "prompt_tokens", "completion_tokens", "attempt"}
)
_PORT_ERRORS = frozenset(
    {"LlmUnavailable", "LlmRejected", "LlmOutputInvalid", "LlmOutputTruncated"}
)
"""The errors that stand for a request or a reply; a registry error names configuration only."""
_ERROR_PARTS = frozenset({"status", "status_code", "failure", "usage"})
"""Names an ``Llm*`` error may be built from: numbers, a fixed failure label, token counts."""


def _is_logger_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _LOGGER_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id in {"logger", "log", "_logger"}
    )


def _only_static_or_numeric(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        return expr.id in _ERROR_PARTS
    if isinstance(expr, ast.Attribute):  # ``exc.status_code`` / ``usage.prompt_tokens``
        return expr.attr in _ERROR_PARTS | _LOG_FIELDS
    if isinstance(expr, ast.JoinedStr):
        return all(
            isinstance(part, ast.Constant)
            or (isinstance(part, ast.FormattedValue) and _only_static_or_numeric(part.value))
            for part in expr.values
        )
    return False


def _privacy_violations(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_logger_call(node):
            event = node.args[0] if node.args else None
            if not (isinstance(event, ast.Constant) and isinstance(event.value, str)):
                found.append(f"line {node.lineno}: a log event must be a constant string")
            if len(node.args) > 1:
                found.append(f"line {node.lineno}: log fields must be keywords")
            found.extend(
                f"line {node.lineno}: log field {kw.arg!r} is not an allowed metric"
                for kw in node.keywords
                if kw.arg not in _LOG_FIELDS
            )
            if any(kw.arg is None for kw in node.keywords):
                found.append(f"line {node.lineno}: no **kwargs in a log call")
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name) and func.id in _PORT_ERRORS:
                args = [*node.exc.args, *(kw.value for kw in node.exc.keywords)]
                found.extend(
                    f"line {node.lineno}: {func.id} is built from something that may be text"
                    for arg in args
                    if not _only_static_or_numeric(arg)
                )
    return found


@pytest.mark.parametrize("path", LLM_FILES, ids=_rel)
def test_nothing_in_the_llm_package_logs_or_raises_with_request_or_response_text(
    path: Path,
) -> None:
    assert _privacy_violations(ast.parse(path.read_text())) == []


class TestThePrivacyCheckSeesWhatItShould:
    """The detector itself: it must flag the patterns it exists to forbid."""

    @pytest.mark.parametrize(
        "source",
        [
            'logger.info("x", body=body)',
            'logger.info("x", model_key=k, text=response.text)',
            "logger.info(message)",
            'logger.info("x", **fields)',
            'logger.info("x", "positional")',
            'raise LlmOutputInvalid(f"bad reply: {content}")',
            "raise LlmUnavailable(response.text)",
            'raise LlmRejected(f"{response.text}")',
        ],
    )
    def test_it_flags(self, source: str) -> None:
        assert _privacy_violations(ast.parse(source))

    @pytest.mark.parametrize(
        "source",
        [
            'logger.info("llm.call", model_key=k, status_code=s, attempt=1)',
            'raise LlmOutputInvalid("LLM reply is empty", usage=usage)',
            'raise LlmUnavailable(f"LLM provider unavailable ({failure})")',
            "raise LlmRejected(status)",
        ],
    )
    def test_it_allows(self, source: str) -> None:
        assert _privacy_violations(ast.parse(source)) == []


# -- the old names are gone -------------------------------------------------------------------------


def test_the_studio_prefixed_registry_names_are_gone_everywhere() -> None:
    needles = [
        "STUDIO_LLM_" + "MODELS",
        "STUDIO_LLM_" + "DEFAULT_MODEL",
        "studio_llm_" + "models",
        "studio_llm_" + "default_model",
        "src.studio." + "llm",
        "studio/" + "llm",
        '"studio.llm' + '.call"',
    ]
    pattern = re.compile("|".join(re.escape(n) for n in needles))
    roots = [SRC, ROOT / "tests", ROOT / "docs", ROOT / "CLAUDE.md", ROOT / ".env.example"]
    offenders: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths:
            if path.suffix not in {".py", ".md", ".example", ""} or "__pycache__" in path.parts:
                continue
            if path == Path(__file__).resolve():
                continue
            if pattern.search(path.read_text(errors="ignore")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
