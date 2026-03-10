"""Targeted tests for the bit-parallel NFA backend.

Tests cover all four codegen variants (uint8, uint16, uint32, uint32_array)
and verify both structural properties of the generated code and functional
correctness via compile-and-execute.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from emx_regex_cgen import generate
from emx_regex_cgen.codegen_bitnfa import generate_bitnfa_c_code
from emx_regex_cgen.compiler import compile_nfa
from tests._support import build_matcher, run_matcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(pattern: str, tmp_path: Path, flags: str = "", **kwargs) -> Path:
    """Generate, write and compile a bitnfa C matcher; return the executable."""
    return build_matcher(pattern, tmp_path, flags=flags, engine="bitnfa", **kwargs)


def _run(exe: Path, inp: str) -> bool:
    """Run *exe* with *inp* and return True on match."""
    return run_matcher(exe, inp, exe.parent)


# ---------------------------------------------------------------------------
# Variant selection tests
# ---------------------------------------------------------------------------


def test_variant_uint8() -> None:
    """Pattern with ≤8 NFA positions must use uint8_t."""
    # Use a multi-entry pattern (\d{3}) to exercise the 256-byte table path.
    nfa = compile_nfa(r"\d{3}")
    assert nfa["num_positions"] <= 8
    code = generate_bitnfa_c_code(nfa).render()
    assert "uint8_t" in code
    assert re.search(r"regex_trans_\d+\[256\]", code)


def test_variant_uint16() -> None:
    """Pattern with 9-16 NFA positions must use uint16_t."""
    # Use a multi-entry pattern (\d{9}) to exercise the 256-byte table path.
    nfa = compile_nfa(r"\d{9}")
    assert 8 < nfa["num_positions"] <= 16
    code = generate_bitnfa_c_code(nfa).render()
    assert "uint16_t" in code
    assert re.search(r"regex_trans_\d+\[256\]", code)


def test_variant_uint32() -> None:
    """Pattern with 17-32 NFA positions must use uint32_t (single word)."""
    nfa = compile_nfa("abcdefghijklmnopq")
    assert 16 < nfa["num_positions"] <= 32
    code = generate_bitnfa_c_code(nfa).render()
    assert "uint32_t" in code
    # Must not be the array variant
    assert "uint32_t[" not in code


def test_variant_uint32_array() -> None:
    """Pattern with >32 NFA positions must use uint32_t array."""
    # Use a multi-entry pattern (\d{33}) to exercise the 2-D table path.
    nfa = compile_nfa(r"\d{33}")
    assert nfa["num_positions"] > 32
    code = generate_bitnfa_c_code(nfa).render()
    m = re.search(r"regex_trans_\d+\[256\]\[(\d+)\]", code)
    assert m is not None, "Expected per-position 2-D transition arrays for uint32_array"
    num_words = int(m.group(1))
    assert num_words == (nfa["num_positions"] + 31) // 32


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_no_dynamic_alloc() -> None:
    """Generated code must not contain malloc, calloc, free, or realloc."""
    for pat in ["ab", "helloworld", "abcdefghijklmnopq", "a{33}"]:
        code = generate(pat, engine="bitnfa").render()
        for fn in ("malloc", "calloc", "free", "realloc"):
            assert fn not in code, f"{fn} found in generated code for {pat!r}"


def test_metadata_comment() -> None:
    """The generated code must include a metadata comment with the engine name."""
    code = generate("hello", engine="bitnfa").render()
    assert "bitnfa" in code


def test_hot_loop_unrolled_uint32_array() -> None:
    """The uint32_array hot loop must contain no loops over words."""
    nfa = compile_nfa("a{33}")
    code = generate_bitnfa_c_code(nfa).render()
    # The main for-loop is the only loop; word operations are unrolled
    loop_count = code.count("for (")
    assert loop_count == 1, "Expected exactly one for-loop (the input loop)"


def test_prefix() -> None:
    """Custom prefix must be used for table and function names."""
    # Use \d which has a multi-entry table to verify the prefix appears in the
    # table name; "ab" would have no table at all due to the inline optimisation.
    code = generate(r"\d", engine="bitnfa", prefix="my_re").render()
    assert "my_re_trans" in code
    assert "my_re_match" in code
    assert "regex_" not in code


# ---------------------------------------------------------------------------
# Single-bit inline optimisation
# ---------------------------------------------------------------------------


def test_inline_opt_no_table_for_single_bit_positions() -> None:
    """Positions with one non-zero power-of-two entry must not emit a table."""
    # "ab" has two positions each mapping one byte to a single-bit mask.
    nfa = compile_nfa("ab")
    code = generate_bitnfa_c_code(nfa).render()
    assert not re.search(r"regex_trans_\d+\[256\]", code), (
        "Expected no 256-byte tables for 'ab' (all positions are single-bit)"
    )
    assert "(b == 'a')" in code
    assert "(b == 'b')" in code


def test_inline_opt_tables_preserved_for_multi_entry_positions() -> None:
    """Positions with multiple non-zero entries must still use a table."""
    # "\d{3}" has three digit positions each with 10 byte entries; none qualify.
    nfa = compile_nfa(r"\d{3}")
    code = generate_bitnfa_c_code(nfa).render()
    assert re.search(r"regex_trans_\d+\[256\]", code), (
        "Expected 256-byte tables for '\\d{3}' (multi-entry digit positions)"
    )
    assert "(b == " not in code, "Unexpected inline expression for multi-entry position"


def test_inline_opt_mixed_pattern() -> None:
    r"""Mixed patterns emit both tables (multi-entry) and inline exprs (single-bit)."""
    # "\d{4}-\d{2}-\d{2}": digit positions have 10 entries (tables), while the
    # two '-' positions each have a single byte mapping to a power-of-two mask.
    nfa = compile_nfa(r"\d{4}-\d{2}-\d{2}")
    code = generate_bitnfa_c_code(nfa).render()
    assert re.search(r"regex_trans_\d+\[256\]", code), "Digit positions must keep tables"
    assert "(b == '-')" in code, "'-' positions must use inline comparison"


def test_inline_opt_array_variant() -> None:
    """Single-bit positions in the uint32_array variant also use inline expressions."""
    # "abcdefghijklmnopqrstuvwxyz012345" has 36 unique chars → >32 positions,
    # each position mapping exactly one byte to a single-bit mask.
    nfa = compile_nfa("abcdefghijklmnopqrstuvwxyz012345")
    assert nfa["num_positions"] > 32
    code = generate_bitnfa_c_code(nfa).render()
    assert not re.search(r"regex_trans_\d+\[256\]\[\d+\]", code), (
        "Expected no 2-D tables for single-unique-char pattern in array variant"
    )
    assert "(b == 'a')" in code


def test_inline_opt_functional_correctness(tmp_path: Path) -> None:
    """Patterns using the inline optimisation must still match correctly."""
    # "hello" and "ab" have all-inline transitions; verify compile+run.
    cases = [
        ("hello", ["hello"], ["hell", "helloo", ""]),
        ("ab", ["ab"], ["a", "b", "abc", ""]),
    ]
    for i, (pattern, matches, rejects) in enumerate(cases):
        sub = tmp_path / str(i)
        sub.mkdir()
        exe = _build(pattern, sub)
        for inp in matches:
            assert _run(exe, inp), f"{pattern!r} should match {inp!r}"
        for inp in rejects:
            assert not _run(exe, inp), f"{pattern!r} should not match {inp!r}"


# ---------------------------------------------------------------------------
# Position-NFA reduction tests
# ---------------------------------------------------------------------------


def test_position_reduction_fewer_positions() -> None:
    """Position NFA must have fewer positions than the raw Thompson NFA."""
    from emx_regex_cgen.compiler import _build_nfa

    for pat in ["hello", "abc", "(a|b)*c", "a{2,4}", "cat|dog"]:
        builder, _start, _accept = _build_nfa(pat)
        thompson_count = builder._next
        nfa = compile_nfa(pat)
        assert nfa["num_positions"] < thompson_count, (
            f"Pattern {pat!r}: position NFA ({nfa['num_positions']}) "
            f"not smaller than Thompson NFA ({thompson_count})"
        )


def test_position_reduction_no_empty_rows() -> None:
    """Every position in the reduced NFA (except accept) should have transitions."""
    for pat in ["ab", "hello", "(a|b)*c", "[a-z]+", r"\d{3}"]:
        nfa = compile_nfa(pat)
        empty_count = sum(
            1 for masks in nfa["trans_masks"] if not masks
        )
        # At most one empty row (the accept position)
        assert empty_count <= 1, (
            f"Pattern {pat!r}: {empty_count} empty rows in position NFA"
        )


def test_position_reduction_contiguous_bits() -> None:
    """Bit positions must be contiguous (0..N-1) with no gaps."""
    for pat in ["abc", "a|b|c", "(xy)+", r"\w+"]:
        nfa = compile_nfa(pat)
        n = nfa["num_positions"]
        all_bits = nfa["initial"] | nfa["accept"]
        for masks in nfa["trans_masks"]:
            for bits in masks.values():
                all_bits |= bits
        # All referenced bits must be within [0, n)
        assert all_bits < (1 << n), (
            f"Pattern {pat!r}: bits exceed num_positions={n}"
        )


# ---------------------------------------------------------------------------
# Functional correctness – all four variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # uint8 patterns (≤8 positions)
        ("ab", "ab", True),
        ("ab", "a", False),
        ("ab", "abc", False),
        ("ab", "", False),
        (r"\d+", "123", True),
        (r"\d+", "abc", False),
        (r"\d+", "", False),
        ("[a-z]+", "hello", True),
        ("[a-z]+", "HELLO", False),
    ],
)
def test_uint8_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """uint8 variant must match/reject correctly."""
    nfa = compile_nfa(pattern)
    assert nfa["num_positions"] <= 8, f"Expected ≤8 positions, got {nfa['num_positions']}"
    exe = _build(pattern, tmp_path)
    assert _run(exe, inp) == expected


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # uint16 patterns (9-16 positions after position-NFA reduction)
        ("helloworld", "helloworld", True),
        ("helloworld", "helloworl", False),
        ("helloworld", "helloworldx", False),
        ("helloworld", "", False),
        ("a{2,8}", "aa", True),
        ("a{2,8}", "aaaaaaaa", True),
        ("a{2,8}", "a", False),
        ("a{2,8}", "aaaaaaaaa", False),
        ("cat|dog|fish", "cat", True),
    ],
)
def test_uint16_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """uint16 variant must match/reject correctly."""
    nfa = compile_nfa(pattern)
    assert 8 < nfa["num_positions"] <= 16
    exe = _build(pattern, tmp_path)
    assert _run(exe, inp) == expected


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # uint32 patterns (17-32 positions after position-NFA reduction)
        ("abcdefghijklmnopq", "abcdefghijklmnopq", True),
        ("abcdefghijklmnopq", "abcdefghijklmnop", False),
        ("abcdefghijklmnopq", "abcdefghijklmnopqr", False),
        ("abcdefghijklmnopq", "", False),
        ("[a-z]{20}", "abcdefghijklmnopqrst", True),
        ("cat|dog|fish|bird|snake|horse", "cat", True),
        ("cat|dog|fish|bird|snake|horse", "horse", True),
        ("cat|dog|fish|bird|snake|horse", "lion", False),
    ],
)
def test_uint32_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """uint32 variant must match/reject correctly."""
    nfa = compile_nfa(pattern)
    assert 16 < nfa["num_positions"] <= 32
    exe = _build(pattern, tmp_path)
    assert _run(exe, inp) == expected


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # uint32_array patterns (>32 positions after position-NFA reduction)
        ("a{33}", "a" * 33, True),
        ("a{33}", "a" * 32, False),
        ("a{33}", "a" * 34, False),
        ("a{33}", "", False),
    ],
)
def test_uint32_array_correctness(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """uint32_array variant must match/reject correctly."""
    nfa = compile_nfa(pattern)
    assert nfa["num_positions"] > 32
    exe = _build(pattern, tmp_path)
    assert _run(exe, inp) == expected


# ---------------------------------------------------------------------------
# Cross-variant correctness (same patterns, both engines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,inp,expected",
    [
        # Simple patterns
        ("hello", "hello", True),
        ("hello", "world", False),
        (r"\d{4}-\d{2}-\d{2}", "2024-01-15", True),
        (r"\d{4}-\d{2}-\d{2}", "abcd-ef-gh", False),
        # Quantifiers
        ("a*", "", True),
        ("a*", "aaa", True),
        ("a*", "b", False),
        ("a+", "a", True),
        ("a+", "", False),
        ("colou?r", "color", True),
        ("colou?r", "colour", True),
        ("colou?r", "colouur", False),
        # Character classes
        ("[a-z0-9_]+", "hello_123", True),
        ("[a-z0-9_]+", "HELLO", False),
        ("[^aeiou]+", "xyz", True),
        ("[^aeiou]+", "aei", False),
        # Alternation
        ("cat|dog|fish", "dog", True),
        ("cat|dog|fish", "rat", False),
    ],
)
def test_bitnfa_matches_dfa(
    pattern: str, inp: str, expected: bool, tmp_path: Path
) -> None:
    """bitnfa and dfa engines must agree on match results."""
    bitnfa_dir = tmp_path / "bitnfa"
    dfa_dir = tmp_path / "dfa"
    bitnfa_dir.mkdir()
    dfa_dir.mkdir()
    exe_bitnfa = _build(pattern, bitnfa_dir)
    # Build DFA version
    exe_dfa = build_matcher(pattern, dfa_dir, engine="dfa")
    assert _run(exe_bitnfa, inp) == expected
    assert _run(exe_dfa, inp) == expected
