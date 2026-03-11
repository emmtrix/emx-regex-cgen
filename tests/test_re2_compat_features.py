from __future__ import annotations

from pathlib import Path

import pytest

from emx_regex_cgen import generate
from tests._support import build_matcher, run_matcher


@pytest.mark.parametrize("engine", ["dfa", "bitnfa"])
@pytest.mark.parametrize(
    "pattern,flags,input_data,expected",
    [
        ("a(?<ONE>b)c(?<TWO>d)e", "", "abcde", True),
        ("a(?<ONE>b)c(?<TWO>d)e", "", "abxde", False),
        ("a(?i)b", "", "ab", True),
        ("a(?i)b", "", "aB", True),
        ("a(?i)b", "", "AB", False),
        ("((?-i)[[:lower:]])[[:lower:]]", "i", "aB", True),
        ("((?-i)[[:lower:]])[[:lower:]]", "i", "Ab", False),
        ("(a.b(?s)c.d|x.y)p.q", "", "a+bc\ndp+q", True),
        ("(a.b(?s)c.d|x.y)p.q", "", "a\nbc\ndp+q", False),
        (r"a\Q\Eb", "", "ab", True),
        (r"\Q\E", "", "", True),
        (r"ab\Cde", "", b"abXde", True),
        (r"ab\Cde", "", b"ab\x80de", True),
        (r"a\x{d800}b", "", b"a\xed\xa0\x80b", False),
        (r"\777", "", "\u01ff", True),
        (r"(?i:A{1,}\6666666666)", "", "A\u01b66666666", True),
    ],
)
def test_re2_compat_features(
    pattern: str,
    flags: str,
    input_data: str | bytes,
    expected: bool,
    engine: str,
    tmp_path: Path,
) -> None:
    exe = build_matcher(pattern, tmp_path, flags=flags, engine=engine)
    assert run_matcher(exe, input_data, tmp_path) is expected


def test_pcaret_unicode_property_compiles() -> None:
    generate(r"\p{^Lu}", flags="i")
