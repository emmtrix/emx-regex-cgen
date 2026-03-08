#!/usr/bin/env python
"""Regenerate all golden C reference files in tests/golden/.

Run this script whenever the code generator output intentionally changes::

    python tests/update_golden.py
"""

from pathlib import Path

from regex_cgen import generate

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

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
]

for filename, pattern, kwargs in CASES:
    content = generate(pattern, **kwargs)
    path = GOLDEN_DIR / filename
    path.write_text(content)
    print(f"  wrote {path}")

print(f"\nUpdated {len(CASES)} golden files.")
