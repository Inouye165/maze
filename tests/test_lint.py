"""Lint regression tests that run as part of pytest."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


def test_pylint_line_length_and_final_newline() -> None:
    """Pylint should not introduce new long-line or final-newline regressions."""

    repo_root = Path(__file__).resolve().parents[1]
    python_files = sorted((repo_root / "src" / "maze_rl").rglob("*.py"))
    python_files.extend(sorted((repo_root / "tests").glob("test_*.py")))
    baseline_path = repo_root / "tests" / "pylint_c0301_c0304_baseline.txt"

    command = [
        sys.executable,
        "-m",
        "pylint",
        "--disable=all",
        "--enable=C0301,C0304",
        "--max-line-length=100",
        *[str(path) for path in python_files],
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    actual = sorted(_normalize_pylint_output(result.stdout, repo_root))
    expected = sorted(
        _normalize_baseline_entry(line.strip())
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    details: list[str] = []
    if added:
        details.append("New lint violations:")
        details.extend(added)
    if removed:
        details.append("Baseline entries no longer present:")
        details.extend(removed)

    assert actual == expected, "\n".join(details)


def _normalize_pylint_output(output: str, repo_root: Path) -> list[str]:
    """Normalize Pylint messages into stable path:code:message records."""

    pattern = re.compile(
        r"^(?P<path>.+?):(?P<line>\d+):\d+: (?P<code>C030[14]): (?P<message>.+?) "
        r"\((?:line-too-long|missing-final-newline)\)$"
    )
    normalized: list[str] = []
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        relative_path = Path(match.group("path")).as_posix()
        if ":/" in relative_path or relative_path.startswith("C:/"):
            relative_path = Path(relative_path).resolve().relative_to(repo_root).as_posix()
        normalized.append(
            f"{relative_path}:{match.group('code')}:{match.group('message')}"
        )
    return normalized


def _normalize_baseline_entry(entry: str) -> str:
    """Drop baseline line numbers so refactors do not cause false failures."""

    parts = entry.split(":", 3)
    if len(parts) == 4 and parts[1].isdigit():
        return f"{parts[0]}:{parts[2]}:{parts[3]}"
    return entry
