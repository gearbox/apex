"""Enforce coverage floors for the bundle-declared workflow remediation."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class CoverageFloor:
    """A minimum line-coverage percentage for one or more source files."""

    label: str
    paths: tuple[str, ...]
    minimum: float


FLOORS: tuple[CoverageFloor, ...] = (
    CoverageFloor("workflow package", ("api/services/workflow/",), 90.0),
    CoverageFloor("workflow capabilities", ("api/services/workflow/capabilities.py",), 100.0),
    CoverageFloor("workflow applier", ("api/services/workflow/applier.py",), 95.0),
    CoverageFloor(
        "Aisha workflow handler",
        ("api/services/generation/aisha/handlers.py",),
        92.0,
    ),
)


def _matching_classes(root: ElementTree.Element, paths: Iterable[str]) -> list[ElementTree.Element]:
    """Return Cobertura classes whose filenames exactly or prefix-match ``paths``."""
    classes = root.findall(".//class")
    result: list[ElementTree.Element] = []
    for class_ in classes:
        filename = class_.get("filename", "")
        if any(filename == path or filename.startswith(path) for path in paths):
            result.append(class_)
    return result


def _line_coverage(classes: Iterable[ElementTree.Element]) -> float:
    lines = [line for class_ in classes for line in class_.findall("./lines/line")]
    if not lines:
        return 0.0
    hit_lines = sum(int(line.get("hits", "0")) > 0 for line in lines)
    return 100 * hit_lines / len(lines)


def check_coverage(report: Path) -> list[str]:
    """Return human-readable floor violations for a Cobertura XML report."""
    # The report is produced by pytest-cov in the preceding CI step; it is
    # not user-provided XML handled by the application.
    root = ElementTree.parse(report).getroot()  # noqa: S314
    failures: list[str] = []
    for floor in FLOORS:
        classes = _matching_classes(root, floor.paths)
        percentage = _line_coverage(classes)
        if not classes:
            failures.append(f"{floor.label}: no matching files found in {report}")
        elif percentage < floor.minimum:
            failures.append(f"{floor.label}: {percentage:.1f}% < required {floor.minimum:.1f}%")
        else:
            print(f"{floor.label}: {percentage:.1f}% (minimum {floor.minimum:.1f}%)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Cobertura XML coverage report")
    args = parser.parse_args()
    failures = check_coverage(args.report)
    if failures:
        print("B1 coverage floors failed:", *failures, sep="\n- ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
