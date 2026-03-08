"""Golden tests: verify that generate() output matches the committed C reference files.

Each golden file in tests/golden/ captures the exact C output for a specific
regex feature or CLI option.  If the generated output ever changes, update the
golden files by running::

    python tests/update_golden.py
"""

from pathlib import Path

import pytest

from regex_cgen import generate

GOLDEN_DIR = Path(__file__).parent / "golden"

# (golden_filename, pattern, generate() keyword arguments)
CASES: list[tuple[str, str, dict]] = [
    # --- regex features ---
    ("literal.c",              r"hello",                 {}),
    ("char_class.c",           r"[a-z0-9_]+",            {}),
    ("negated_class.c",        r"[^aeiou]+",              {}),
    ("dot.c",                  r".+",                     {}),
    ("alternation.c",          r"cat|dog|fish",           {}),
    ("quantifier_star.c",      r"ab*c",                   {}),
    ("quantifier_plus.c",      r"ab+c",                   {}),
    ("quantifier_optional.c",  r"colou?r",                {}),
    ("quantifier_repeat.c",    r"a{2,4}",                 {}),
    ("escape_digit.c",         r"\d{4}-\d{2}-\d{2}",     {}),
    ("escape_word.c",          r"\w+",                    {}),
    ("escape_space.c",         r"\s+",                    {}),
    ("unicode.c",              r"\x{00e9}+",              {}),
    ("anchors.c",              r"^start.*end$",           {}),
    # --- CLI options / flags ---
    ("flag_ignorecase.c",      r"[a-z]+",                 {"flags": "i"}),
    ("flag_dotall.c",          r".+",                     {"flags": "s"}),
    ("flag_multiline.c",       r"[a-z]+",                 {"flags": "m"}),
    ("flag_verbose.c",         r"(?x) [a-z]+ # letters", {"flags": "x"}),
    ("encoding_bytes.c",       r"[\x80-\xff]+",           {"encoding": "bytes"}),
    ("prefix.c",               r"[a-z]+",                 {"prefix": "my_matcher"}),
    ("emit_main.c",            r"\d+",                    {"emit_main": True}),
    ("alphabet_compression.c", r"hello",                  {"alphabet_compression": "yes"}),
    ("row_dedup.c",            r"hello",                  {"row_dedup": "yes"}),
    ("early_exit.c",           r"hello",                  {"early_exit": True}),
]


@pytest.mark.parametrize("filename,pattern,kwargs", CASES, ids=[c[0] for c in CASES])
def test_golden(filename: str, pattern: str, kwargs: dict) -> None:
    """Generated output must match the committed golden file."""
    golden_path = GOLDEN_DIR / filename
    actual = generate(pattern, **kwargs)
    expected = golden_path.read_text()
    assert actual == expected, (
        f"Golden file '{filename}' does not match the current output of generate().\n"
        "If the change is intentional, regenerate golden files with:\n"
        "    python tests/update_golden.py"
    )
