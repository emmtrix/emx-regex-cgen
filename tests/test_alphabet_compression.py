"""Tests for alphabet compression in codegen.

Alphabet compression maps byte values (0-255) to equivalence classes.
Two bytes belong to the same class when they produce identical transitions
in every DFA state.  This replaces the second dimension of the transition
table (256) with the number of equivalence classes, and adds a
``{prefix}_alphabet`` look-up array.
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


def _build(pattern: str, tmp_path: Path, flags: str = "", **kwargs) -> Path:
    """Generate, write and compile a C matcher; return the executable path."""
    c_code = generate(pattern, flags=flags, emit_main=True, **kwargs).render()
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
# Structural tests: verify compressed table is emitted
# ---------------------------------------------------------------------------


def test_alphabet_emits_alphabet_map() -> None:
    """When alphabet compression is enabled, the alphabet map must be present."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, alphabet_compression="yes").render()
    assert re.search(r"regex_alphabet\[256\]", code), (
        "regex_alphabet[256] not found in generated code"
    )


def test_alphabet_reduces_table_columns() -> None:
    """The emitted transition table must have fewer than 256 columns."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, alphabet_compression="yes").render()
    m = re.search(r"regex_transitions\[\d+\]\[(\d+)\]", code)
    assert m is not None, "Could not find transitions declaration"
    num_cols = int(m.group(1))
    assert num_cols < 256, f"Expected fewer than 256 columns, got {num_cols}"


def test_alphabet_no_map_when_disabled() -> None:
    """When alphabet compression is disabled, no alphabet map should be emitted."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, alphabet_compression="no").render()
    assert "regex_alphabet" not in code


def test_alphabet_auto_below_threshold() -> None:
    """Auto mode must not compress when the table is below the threshold."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, alphabet_compression="auto", size_threshold=8192).render()
    assert "regex_alphabet" not in code


def test_alphabet_auto_above_threshold() -> None:
    """Auto mode must compress when the table exceeds the threshold."""
    dfa = compile_regex("hello")
    n = dfa["num_states"]
    # Set threshold below the actual table size
    code = generate_c_code(dfa, alphabet_compression="auto", size_threshold=n * 256 - 1).render()
    assert "regex_alphabet" in code


def test_alphabet_hot_loop_uses_map() -> None:
    """When compression is active, the hot loop must reference the alphabet map."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, alphabet_compression="yes").render()
    assert "regex_alphabet[(unsigned char)input[i]]" in code


def test_alphabet_combined_with_row_dedup() -> None:
    """Alphabet compression and row dedup can be enabled together."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, alphabet_compression="yes", row_dedup="yes").render()
    assert "regex_alphabet" in code
    assert "regex_row_map" in code
    # Hot loop must reference both
    assert "regex_row_map[state]" in code
    assert "regex_alphabet[(unsigned char)input[i]]" in code


# ---------------------------------------------------------------------------
# Functional tests: compressed code must produce correct match results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # Literal – only a few unique byte classes
        ("hello", "hello", True),
        ("hello", "world", False),
        ("hello", "hell", False),
        ("hello", "helloo", False),
        # Character class
        ("[a-z]+", "abc", True),
        ("[a-z]+", "ABC", False),
        ("[a-z]+", "abc123", False),
        # Digit escapes
        (r"\d{4}-\d{2}-\d{2}", "2024-01-15", True),
        (r"\d{4}-\d{2}-\d{2}", "abcd-ef-gh", False),
        # Alternation
        ("cat|dog|fish", "cat", True),
        ("cat|dog|fish", "dog", True),
        ("cat|dog|fish", "fish", True),
        ("cat|dog|fish", "bird", False),
        # UTF-8 pattern
        (r"\x{00e9}+", "\u00e9\u00e9", True),
        (r"\x{00e9}+", "ee", False),
    ],
)
def test_alphabet_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """Code generated with alphabet compression must match/reject correctly."""
    exe = _build(pattern, tmp_path, alphabet_compression="yes")
    assert _run(exe, inp) == expected, (
        f"Pattern {pattern!r} with input {inp!r}: "
        f"expected {'match' if expected else 'no match'}"
    )


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        ("a[bc]+d", "abcd", True),
        ("a[bc]+d", "abd", True),
        ("a[bc]+d", "ad", False),
        ("(foo|bar|baz)+", "foobar", True),
        ("(foo|bar|baz)+", "fox", False),
    ],
)
def test_alphabet_and_dedup_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """Code with both alphabet compression and row dedup must match correctly."""
    exe = _build(pattern, tmp_path, alphabet_compression="yes", row_dedup="yes")
    assert _run(exe, inp) == expected


# ---------------------------------------------------------------------------
# Auto-mode threshold tests
# ---------------------------------------------------------------------------


def test_row_dedup_auto_below_threshold() -> None:
    """Auto mode must not deduplicate when the table is below the threshold."""
    dfa = compile_regex("hello")
    code = generate_c_code(dfa, row_dedup="auto", size_threshold=8192).render()
    assert "regex_row_map" not in code


def test_row_dedup_auto_above_threshold() -> None:
    """Auto mode must deduplicate when the table exceeds the threshold."""
    dfa = compile_regex("hello")
    n = dfa["num_states"]
    code = generate_c_code(dfa, row_dedup="auto", size_threshold=n * 256 - 1).render()
    # The hello pattern has duplicate rows (dead + accept), so row_map should appear
    assert "regex_row_map" in code
