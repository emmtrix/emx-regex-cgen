"""Tests for the GeneratedCode structured result type returned by generate().

Verifies that:
- generate() returns a GeneratedCode instance with the correct fields.
- render() produces output identical to the original string-based generate().
- Individual parts (includes, globals, match_function, main_function) have
  the expected content and structure.
"""

from __future__ import annotations

import pytest

from emx_regex_cgen import GeneratedCode, generate
from emx_regex_cgen.codegen import generate_c_code
from emx_regex_cgen.codegen_bitnfa import generate_bitnfa_c_code
from emx_regex_cgen.compiler import compile_nfa, compile_regex

# ---------------------------------------------------------------------------
# Type and structure tests
# ---------------------------------------------------------------------------


def test_generate_returns_generated_code() -> None:
    """generate() must return a GeneratedCode instance."""
    result = generate("hello")
    assert isinstance(result, GeneratedCode)


def test_generated_code_has_expected_fields() -> None:
    """GeneratedCode must expose includes, globals, match_function, main_function."""
    result = generate("hello")
    assert hasattr(result, "includes")
    assert hasattr(result, "globals")
    assert hasattr(result, "match_function")
    assert hasattr(result, "main_function")


# ---------------------------------------------------------------------------
# includes field
# ---------------------------------------------------------------------------


def test_includes_is_list_of_strings() -> None:
    """includes must be a list of header name strings (no angle brackets)."""
    result = generate("hello")
    assert isinstance(result.includes, list)
    assert all(isinstance(inc, str) for inc in result.includes)
    # No angle brackets – just bare names
    for inc in result.includes:
        assert "<" not in inc and ">" not in inc


def test_includes_contains_required_headers_dfa() -> None:
    """DFA result must include stddef.h, stdbool.h, stdint.h."""
    result = generate("hello")
    assert "stddef.h" in result.includes
    assert "stdbool.h" in result.includes
    assert "stdint.h" in result.includes


def test_includes_no_main_headers_without_emit_main() -> None:
    """string.h and stdio.h must not be in includes when emit_main=False."""
    result = generate("hello", emit_main=False)
    assert "string.h" not in result.includes
    assert "stdio.h" not in result.includes


def test_includes_has_main_headers_with_emit_main() -> None:
    """string.h and stdio.h must be in includes when emit_main=True."""
    result = generate("hello", emit_main=True)
    assert "string.h" in result.includes
    assert "stdio.h" in result.includes


def test_includes_bitnfa() -> None:
    """bitnfa result must also include stddef.h, stdbool.h, stdint.h."""
    result = generate("hello", engine="bitnfa")
    assert "stddef.h" in result.includes
    assert "stdbool.h" in result.includes
    assert "stdint.h" in result.includes


# ---------------------------------------------------------------------------
# globals field
# ---------------------------------------------------------------------------


def test_globals_contains_transition_table_dfa() -> None:
    """globals must contain the DFA transition table declaration."""
    result = generate("hello")
    assert "regex_transitions" in result.globals
    assert "static const" in result.globals


def test_globals_contains_transition_table_bitnfa() -> None:
    """globals must contain the bitnfa transition table declaration."""
    # Use \d which has multi-entry digit tables that are not inlined.
    result = generate(r"\d", engine="bitnfa")
    assert "regex_trans" in result.globals
    assert "static const" in result.globals


def test_globals_contains_alphabet_when_requested() -> None:
    """globals must contain the alphabet map when alphabet compression is on."""
    result = generate("hello", alphabet_compression="yes")
    assert "regex_alphabet" in result.globals


def test_globals_no_alphabet_when_disabled() -> None:
    """globals must not contain the alphabet map when compression is off."""
    result = generate("hello", alphabet_compression="no")
    assert "regex_alphabet" not in result.globals


def test_globals_contains_row_map_when_dedup_applies() -> None:
    """globals must contain regex_row_map when row deduplication is active."""
    result = generate("hello", row_dedup="yes")
    assert "regex_row_map" in result.globals


def test_globals_does_not_contain_match_function() -> None:
    """globals must not contain the match function itself."""
    result = generate("hello")
    assert "bool regex_match" not in result.globals


# ---------------------------------------------------------------------------
# match_function field
# ---------------------------------------------------------------------------


def test_match_function_contains_function_signature() -> None:
    """match_function must contain the bool {prefix}_match(...) signature."""
    result = generate("hello")
    assert "bool regex_match(const char *input, size_t len)" in result.match_function


def test_match_function_contains_metadata_comment() -> None:
    """match_function must contain the metadata comment with the pattern."""
    result = generate(r"\d+")
    assert r'\d+' in result.match_function


def test_match_function_prefix_respected() -> None:
    """Custom prefix must be reflected in the match function name."""
    result = generate("hello", prefix="my_re")
    assert "bool my_re_match(" in result.match_function


def test_match_function_does_not_contain_includes() -> None:
    """match_function must not contain #include directives."""
    result = generate("hello")
    assert "#include" not in result.match_function


def test_match_function_does_not_contain_transition_table() -> None:
    """match_function must not contain the transition table declaration."""
    result = generate("hello")
    assert "regex_transitions[" not in result.match_function.split("bool")[0]


# ---------------------------------------------------------------------------
# main_function field
# ---------------------------------------------------------------------------


def test_main_function_none_when_not_requested() -> None:
    """main_function must be None when emit_main=False."""
    result = generate("hello", emit_main=False)
    assert result.main_function is None


def test_main_function_present_when_requested() -> None:
    """main_function must be a non-empty string when emit_main=True."""
    result = generate("hello", emit_main=True)
    assert isinstance(result.main_function, str)
    assert len(result.main_function) > 0


def test_main_function_contains_int_main() -> None:
    """main_function must start with 'int main'."""
    result = generate("hello", emit_main=True)
    assert result.main_function is not None
    assert result.main_function.startswith("int main(")


def test_main_function_calls_match() -> None:
    """main_function must call the match function."""
    result = generate("hello", emit_main=True)
    assert result.main_function is not None
    assert "regex_match(" in result.main_function


def test_main_function_bitnfa() -> None:
    """main_function for bitnfa must also be present when emit_main=True."""
    result = generate("hello", engine="bitnfa", emit_main=True)
    assert result.main_function is not None
    assert "int main(" in result.main_function


# ---------------------------------------------------------------------------
# render() method – output equivalence
# ---------------------------------------------------------------------------


def test_render_contains_header_comment() -> None:
    """render() output must start with the standard generated-file header."""
    result = generate("hello")
    rendered = result.render()
    assert rendered.startswith("/* Generated by emx-regex-cgen – do not edit. */")


def test_render_contains_includes() -> None:
    """render() output must contain #include lines for every entry in includes."""
    result = generate("hello")
    rendered = result.render()
    for inc in result.includes:
        assert f"#include <{inc}>" in rendered


def test_render_contains_globals() -> None:
    """render() output must contain the globals string verbatim."""
    result = generate("hello")
    assert result.globals in result.render()


def test_render_contains_match_function() -> None:
    """render() output must contain the match_function string verbatim."""
    result = generate("hello")
    assert result.match_function in result.render()


def test_render_contains_main_when_present() -> None:
    """render() output must contain the main_function string when not None."""
    result = generate("hello", emit_main=True)
    assert result.main_function is not None
    assert result.main_function in result.render()


def test_render_ends_with_newline() -> None:
    """render() output must end with a newline character."""
    for kwargs in [{}, {"emit_main": True}, {"engine": "bitnfa"}]:
        result = generate("hello", **kwargs)
        assert result.render().endswith("\n"), f"render() does not end with newline for {kwargs}"


@pytest.mark.parametrize(
    "pattern,kwargs",
    [
        ("hello", {}),
        (r"\d+", {"emit_main": True}),
        ("[a-z]+", {"flags": "i"}),
        ("hello", {"alphabet_compression": "yes"}),
        ("hello", {"row_dedup": "yes"}),
        ("hello", {"early_exit": True}),
        (r"\d", {"engine": "bitnfa"}),
        (r"\d+", {"engine": "bitnfa", "emit_main": True}),
        (r"\d{33}", {"engine": "bitnfa"}),
    ],
)
def test_render_sections_order(pattern: str, kwargs: dict) -> None:
    """render() must emit sections in order: header, includes, globals, match, [main]."""
    result = generate(pattern, **kwargs)
    rendered = result.render()

    header_pos = rendered.index("/* Generated by emx-regex-cgen")
    # Find first include
    include_pos = rendered.index("#include <")
    # Find globals (first static const declaration)
    globals_pos = rendered.index("static const")
    # Find match function
    match_pos = rendered.index(f"bool {kwargs.get('prefix', 'regex')}_match(")

    assert header_pos < include_pos < globals_pos < match_pos

    if kwargs.get("emit_main"):
        main_pos = rendered.index("int main(")
        assert match_pos < main_pos


# ---------------------------------------------------------------------------
# generate_c_code / generate_bitnfa_c_code return type
# ---------------------------------------------------------------------------


def test_generate_c_code_returns_generated_code() -> None:
    """generate_c_code() must return a GeneratedCode instance."""
    dfa = compile_regex("hello")
    result = generate_c_code(dfa)
    assert isinstance(result, GeneratedCode)


def test_generate_bitnfa_c_code_returns_generated_code() -> None:
    """generate_bitnfa_c_code() must return a GeneratedCode instance."""
    nfa = compile_nfa("hello")
    result = generate_bitnfa_c_code(nfa)
    assert isinstance(result, GeneratedCode)
