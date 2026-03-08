"""Tests for transition-table row deduplication in codegen.

Row deduplication emits only unique rows in ``dfa_transitions`` and adds a
``dfa_row_map`` indirection array so that duplicate DFA states (which have the
same transition vector) share a single row rather than each occupying 256
entries.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from regex_cgen import generate
from regex_cgen.codegen import generate_c_code
from regex_cgen.compiler import compile_regex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(pattern: str, tmp_path: Path, flags: str = "") -> Path:
    """Generate, write and compile a C matcher; return the executable path."""
    c_code = generate(pattern, flags=flags, emit_main=True)
    c_file = tmp_path / "test.c"
    c_file.write_text(c_code)
    exe = tmp_path / "test"
    comp = subprocess.run(
        ["gcc", "-O2", "-o", str(exe), str(c_file)],
        capture_output=True,
        timeout=30,
    )
    assert comp.returncode == 0, f"gcc failed:\n{comp.stderr.decode()}"
    return exe


def _run(exe: Path, inp: str) -> bool:
    """Run *exe* with *inp* as a command-line argument; return True on match."""
    result = subprocess.run([str(exe), inp], capture_output=True, timeout=10)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Structural tests: verify deduplicated table is emitted
# ---------------------------------------------------------------------------


def test_dedup_emits_row_map_when_duplicates_exist() -> None:
    """When duplicate rows exist, regex_row_map must be properly declared and used."""
    # 'a[bc]+d' has states 0 (dead) and 2 (accept after match) with the same
    # all-zero transition row; deduplication should kick in.
    dfa = compile_regex("a[bc]+d")
    n = dfa["num_states"]
    code = generate_c_code(dfa)

    # Declaration must be present, sized correctly
    assert re.search(rf"regex_row_map\[{n}\]", code), (
        "regex_row_map not declared with the correct state count"
    )
    # The hot loop must reference the row map for the table lookup
    assert "regex_transitions[regex_row_map[state]]" in code, (
        "hot loop does not use regex_row_map for table indexing"
    )


def test_dedup_reduces_table_rows() -> None:
    """The emitted regex_transitions must have fewer rows than num_states."""
    dfa = compile_regex("a[bc]+d")
    n = dfa["num_states"]
    code = generate_c_code(dfa)
    # Extract the dimension from the table declaration, e.g. "regex_transitions[4][256]"
    m = re.search(r"regex_transitions\[(\d+)\]\[256\]", code)
    assert m is not None
    num_rows = int(m.group(1))
    assert num_rows < n, f"Expected fewer rows than states ({n}), got {num_rows}"


def test_dedup_no_row_map_when_all_rows_unique() -> None:
    """When every DFA state has a unique row, no regex_row_map should be emitted."""
    # Construct a DFA that is known to have no duplicate rows by providing a
    # synthetic dfa dict where all rows are distinct.
    dfa = {
        "num_states": 2,
        "initial": 1,
        "accept": {1},
        "first_accept": 1,
        "transitions": {
            (0, ord("a")): 1,
            (1, ord("b")): 0,
        },
    }
    code = generate_c_code(dfa)
    assert "regex_row_map" not in code


# ---------------------------------------------------------------------------
# Functional tests: deduplicated code must produce correct match results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # 'a[bc]+d' – duplicate dead/accept rows
        ("a[bc]+d", "abcd", True),
        ("a[bc]+d", "abd", True),
        ("a[bc]+d", "abbd", True),
        ("a[bc]+d", "ad", False),
        ("a[bc]+d", "acd", True),
        # 'abc' – accept state shares row with dead state
        ("abc", "abc", True),
        ("abc", "ab", False),
        ("abc", "abcd", False),
        # alternation: many structurally similar states
        ("(foo|bar|baz)+", "foo", True),
        ("(foo|bar|baz)+", "foobar", True),
        ("(foo|bar|baz)+", "foobaz", True),
        ("(foo|bar|baz)+", "fox", False),
    ],
)
def test_dedup_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """Deduplicated generated code must match/reject exactly as expected."""
    exe = _build(pattern, tmp_path)
    assert _run(exe, inp) == expected, (
        f"Pattern {pattern!r} with input {inp!r}: "
        f"expected {'match' if expected else 'no match'}"
    )
