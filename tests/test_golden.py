"""Golden tests: verify that generate() output matches the committed C reference files.

Each golden file in tests/golden/ captures the exact C output for a specific
regex feature or CLI option.  If the generated output ever changes, update the
golden files by running::

    python tests/update_golden.py
"""

from pathlib import Path

import pytest

from emx_regex_cgen import generate

GOLDEN_DIR = Path(__file__).parent / "golden"

# (golden_filename, pattern, generate() keyword arguments)
CASES: list[tuple[str, str, dict]] = [
    # --- regex features (DFA) ---
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
    # --- word boundary ---
    ("word_boundary.c",        r"\bword\b",               {}),
    ("non_word_boundary.c",    r"a\Bb",                   {}),
    # --- unicode property ---
    ("unicode_property.c",     r"\p{Nd}+",                {}),
    ("unicode_property_bitnfa.c", r"\p{Nd}+",             {"engine": "bitnfa"}),
    # --- word boundary (bitnfa) ---
    ("word_boundary_bitnfa.c",     r"\bword\b",           {"engine": "bitnfa"}),
    ("non_word_boundary_bitnfa.c", r"a\Bb",               {"engine": "bitnfa"}),
    # --- CLI options / flags (DFA) ---
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
    # --- regex features (bitnfa) ---
    ("literal_bitnfa.c",              r"hello",                 {"engine": "bitnfa"}),
    ("char_class_bitnfa.c",           r"[a-z0-9_]+",            {"engine": "bitnfa"}),
    ("negated_class_bitnfa.c",        r"[^aeiou]+",              {"engine": "bitnfa"}),
    ("dot_bitnfa.c",                  r".+",                     {"engine": "bitnfa"}),
    ("alternation_bitnfa.c",          r"cat|dog|fish",           {"engine": "bitnfa"}),
    ("quantifier_star_bitnfa.c",      r"ab*c",                   {"engine": "bitnfa"}),
    ("quantifier_plus_bitnfa.c",      r"ab+c",                   {"engine": "bitnfa"}),
    ("quantifier_optional_bitnfa.c",  r"colou?r",                {"engine": "bitnfa"}),
    ("quantifier_repeat_bitnfa.c",    r"a{2,4}",                 {"engine": "bitnfa"}),
    ("escape_digit_bitnfa.c",         r"\d{4}-\d{2}-\d{2}",     {"engine": "bitnfa"}),
    ("escape_word_bitnfa.c",          r"\w+",                    {"engine": "bitnfa"}),
    ("escape_space_bitnfa.c",         r"\s+",                    {"engine": "bitnfa"}),
    ("unicode_bitnfa.c",              r"\x{00e9}+",              {"engine": "bitnfa"}),
    ("anchors_bitnfa.c",              r"^start.*end$",           {"engine": "bitnfa"}),
    # --- CLI options / flags (bitnfa) ---
    ("flag_ignorecase_bitnfa.c",  r"[a-z]+",     {"engine": "bitnfa", "flags": "i"}),
    ("flag_dotall_bitnfa.c",      r".+",          {"engine": "bitnfa", "flags": "s"}),
    ("flag_multiline_bitnfa.c",   r"[a-z]+",      {"engine": "bitnfa", "flags": "m"}),
    ("flag_verbose_bitnfa.c",     r"(?x) [a-z]+ # letters",
     {"engine": "bitnfa", "flags": "x"}),
    ("encoding_bytes_bitnfa.c",   r"[\x80-\xff]+",
     {"engine": "bitnfa", "encoding": "bytes"}),
    ("prefix_bitnfa.c",           r"[a-z]+",
     {"engine": "bitnfa", "prefix": "my_matcher"}),
    ("emit_main_bitnfa.c",        r"\d+",
     {"engine": "bitnfa", "emit_main": True}),
    # --- bitnfa variant-specific ---
    ("bitnfa_uint8.c",        r"ab",                              {"engine": "bitnfa"}),
    ("bitnfa_uint16.c",       r"cat|dog|fish",                    {"engine": "bitnfa"}),
    ("bitnfa_uint32.c",       r"abcdefghijklmnopq",               {"engine": "bitnfa"}),
    ("bitnfa_uint32_array.c", r"abcdefghijklmnopqrstuvwxyz012345", {"engine": "bitnfa"}),
]


@pytest.mark.parametrize("filename,pattern,kwargs", CASES, ids=[c[0] for c in CASES])
def test_golden(filename: str, pattern: str, kwargs: dict) -> None:
    """Generated output must match the committed golden file."""
    golden_path = GOLDEN_DIR / filename
    actual = generate(pattern, **kwargs).render()
    expected = golden_path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"Golden file '{filename}' does not match the current output of generate().\n"
        "If the change is intentional, regenerate golden files with:\n"
        "    python tests/update_golden.py"
    )
