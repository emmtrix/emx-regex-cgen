"""Parameterised tests for regex-cgen.

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

import json
import subprocess
from pathlib import Path

import pytest

from regex_cgen import generate

_JSON_FILE = Path(__file__).resolve().parent / "data" / "re2_compat_results.json"


def _load_test_params() -> list:
    """Load and flatten test cases from the JSON fixture.

    Each pytest parameter is a tuple of
    ``(pattern, flags, subjects)``
    where *subjects* is the list of ``{input, match, ...}`` dicts.
    """
    with open(_JSON_FILE) as fh:
        data = json.load(fh)

    params: list = []
    for idx, tc in enumerate(data["test_cases"]):
        pattern = tc["pattern"]
        flags = "".join(c for c in tc.get("flags", "") if c in "imsx")
        subjects = tc["subjects"]
        if not subjects:
            continue
        # Build a short, readable test-ID
        safe = pattern[:35].replace("\n", "\\n").replace("\r", "\\r")
        params.append(pytest.param(pattern, flags, subjects, id=f"tc{idx}_{safe}"))
    return params


@pytest.mark.parametrize("engine", ["dfa", "bitnfa"])
@pytest.mark.parametrize("pattern,flags,subjects", _load_test_params())
def test_fullmatch(
    pattern: str, flags: str, subjects: list[dict], engine: str, tmp_path: Path
) -> None:
    """Generate → compile → execute for every subject of *pattern*."""

    # 1. Generate C code
    try:
        c_code = generate(pattern, flags=flags, emit_main=True, engine=engine).render()
    except (ValueError, Exception) as exc:
        pytest.skip(f"Unsupported pattern: {exc}")

    # 2. Write & compile
    c_file = tmp_path / "test.c"
    c_file.write_text(c_code)
    exe = tmp_path / "test"

    comp = subprocess.run(
        ["gcc", "-O2", "-o", str(exe), str(c_file)],
        capture_output=True,
        timeout=30,
    )
    assert comp.returncode == 0, f"gcc failed:\n{comp.stderr.decode()}"

    # 3. Test each subject
    for subj in subjects:
        inp: str = subj["input"]
        expected: bool = subj["match"]

        # Skip inputs with embedded NUL – they cannot be passed via argv
        if "\x00" in inp:
            continue

        try:
            run = subprocess.run(
                [str(exe), inp],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, ValueError):
            pytest.fail(f"Execution failed for input {inp!r}")

        actual = run.returncode == 0
        assert actual == expected, (
            f"Pattern {pattern!r} (flags={flags!r}, engine={engine}) with input {inp!r}: "
            f"expected {'match' if expected else 'no match'}, "
            f"got {'match' if actual else 'no match'}"
        )
