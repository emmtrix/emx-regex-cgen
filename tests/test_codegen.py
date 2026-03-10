"""Parameterised tests for emx-regex-cgen.

Test strategy:
1. Generate C code from a regex pattern.
2. Compile the generated C code with ``gcc``.
3. Execute the compiled binary with each test subject.
4. Compare the exit code (0 = match, 1 = no match) against the expected result.

Test cases are loaded from ``tests/data/re2_compat_results.json`` which was
produced by ``tests/data/parse_pcre2_tests.py`` from the PCRE2 test suite,
filtered for re2 compatibility.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._support import build_matcher, run_matcher


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("engine", ["dfa", "bitnfa"])
def test_fullmatch(
    case_idx: int,
    codegen_cases: list[dict],
    codegen_errors: dict[tuple[str, int], str],
    engine: str,
    tmp_path: Path,
) -> None:
    """Generate -> compile -> execute for every subject of *pattern*."""
    case = codegen_cases[case_idx]
    pattern: str = case["pattern"]
    flags = "".join(c for c in case.get("flags", "") if c in "imsx")
    subjects: list[dict] = case["subjects"]
    expected_error = codegen_errors.get((engine, case_idx))

    try:
        exe = build_matcher(pattern, tmp_path, flags=flags, engine=engine)
    except Exception as exc:
        actual_error = _format_exception(exc)
        if expected_error is None:
            pytest.fail(
                f"Unexpected compile failure for pattern {pattern!r} "
                f"(flags={flags!r}, engine={engine}): {actual_error}"
            )
        assert actual_error == expected_error, (
            f"Compile failure mismatch for pattern {pattern!r} "
            f"(flags={flags!r}, engine={engine}): expected {expected_error!r}, "
            f"got {actual_error!r}"
        )
        return

    if expected_error is not None:
        pytest.fail(
            f"Expected compile failure for pattern {pattern!r} "
            f"(flags={flags!r}, engine={engine}), but build succeeded: {expected_error}"
        )

    for subj in subjects:
        inp: str = subj["input"]
        expected: bool = subj["match"]

        try:
            actual = run_matcher(exe, inp, tmp_path)
        except subprocess.TimeoutExpired:
            pytest.fail(f"Execution failed for input {inp!r}")
        assert actual == expected, (
            f"Pattern {pattern!r} (flags={flags!r}, engine={engine}) with input {inp!r}: "
            f"expected {'match' if expected else 'no match'}, "
            f"got {'match' if actual else 'no match'}"
        )
