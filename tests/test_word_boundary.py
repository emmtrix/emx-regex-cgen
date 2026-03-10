r"""Tests for \b and \B word-boundary assertions.

Verifies that the DFA and bitnfa engines correctly handle word-boundary (\b)
and non-word-boundary (\B) zero-width assertions via generate → compile →
execute round-trips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._support import build_matcher, run_matcher


@pytest.mark.parametrize(
    "pattern,flags,input_str,expected",
    [
        # --- \b at start and end ---
        (r"\bword\b", "", "word", True),
        (r"\bword\b", "", "words", False),
        (r"\bword\b", "", "sword", False),

        # --- \b at end only ---
        (r"a\b", "", "a", True),
        (r"a\b", "", "ab", False),

        # --- \b at start only ---
        (r"\ba", "", "a", True),

        # --- \B (non-boundary) ---
        (r"a\Bb", "", "ab", True),   # word-word ⇒ no boundary ⇒ \B succeeds
        (r"a\Bb", "", "a b", False),  # word-nonword ⇒ boundary ⇒ \B fails
        (r"a\B.", "", "ab", True),   # word-word ⇒ no boundary ⇒ \B succeeds
        (r"a\B.", "", "a ", False),  # word-nonword ⇒ boundary ⇒ \B fails

        # --- \b with character classes ---
        (r"\b\w+\b", "", "hello", True),
        (r"\b\d+\b", "", "42", True),

        # --- \B at start and end ---
        (r"\Biss\B", "", "iss", False),

        # --- \b with wildcard ---
        (r"\b.*", "", "abc", True),
        (r"\b.*", "", " abc", False),
        (r"\b.*\b", "", "hello", True),

        # --- \b...\B ---
        # \b at start checks start-of-string vs first byte;
        # \B at end checks last byte vs end-of-string (non-word).
        (r"\b...\B", "", "ab ", True),   # \b: start→word=bnd, \B: nw→end(nw)=non-bnd
        (r"\b...\B", "", "abc", False),  # \B: 'c'(word)→end(nw)=boundary ⇒ fails
        (r"\b...\B", "", "   ", False),  # \b: start(nw)→' '(nw)=non-boundary ⇒ fails
        (r"\b...\B", "", "ab!", True),   # \B: '!'(nw)→end(nw)=non-boundary ⇒ succeeds

        # --- Mixed boundary / non-boundary ---
        (r"\bthe cat\b", "", "the cat", True),
        (r"\bthe cat\b", "", "the cats", False),

        # --- \b with case insensitive flag ---
        (r"\bWord\b", "i", "word", True),
        (r"\bWord\b", "i", "WORD", True),

        # --- bytes encoding ---
        (r"\bfoo\b", "", "foo", True),
    ],
    ids=lambda x: repr(x) if isinstance(x, str) else str(x),
)
@pytest.mark.parametrize("engine", ["dfa", "bitnfa"])
def test_word_boundary(
    pattern: str, flags: str, input_str: str, expected: bool,
    engine: str, tmp_path: Path,
) -> None:
    r"""Generate → compile → execute for \b / \B patterns."""
    exe = build_matcher(pattern, tmp_path, flags=flags, engine=engine)
    actual = run_matcher(exe, input_str, tmp_path)
    assert actual == expected, (
        f"Pattern {pattern!r} (flags={flags!r}, engine={engine}) "
        f"with input {input_str!r}: "
        f"expected {'match' if expected else 'no match'}, "
        f"got {'match' if actual else 'no match'}"
    )
