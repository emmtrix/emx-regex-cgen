#!/usr/bin/env python3
"""
Parse PCRE2 test cases from pcre2-org/testdata and check compatibility
with Python's re2 (google-re2) library.

Usage:
    python parse_pcre2_tests.py > re2_compat_results.json
"""

import json
import re
import sys
from pathlib import Path

import re2

TESTDATA_DIR = Path(__file__).parent / "pcre2-org" / "testdata"


def find_closing_delimiter(content, start):
    """Find the closing '/' delimiter of a PCRE2 pattern.

    Scans from 'start' (position right after the opening '/'), handling
    backslash escapes and character classes where '/' does not terminate
    the pattern.  Multi-line patterns are supported – the scan continues
    across newlines.

    Returns ``(pattern_text, flags_str, end_pos)`` where *end_pos* points
    to the character immediately after the flags, or ``None`` when no
    closing delimiter is found.
    """
    i = start
    in_char_class = False

    while i < len(content):
        c = content[i]
        if c == "\\" and i + 1 < len(content):
            next_c = content[i + 1]
            # \cX is a 3-character control escape (e.g. \c[ = ESC); skip all 3
            if next_c == "c" and i + 2 < len(content):
                i += 3
            else:
                i += 2
            continue
        if c == "[" and not in_char_class:
            in_char_class = True
        elif c == "]" and in_char_class:
            in_char_class = False
        elif c == "/" and not in_char_class:
            pattern = content[start:i]
            flags_start = i + 1
            flags_end = flags_start
            # Collect letter flags (e.g. 'imsx')
            while flags_end < len(content) and (
                content[flags_end].isalpha() or content[flags_end] == "_"
            ):
                flags_end += 1
            flags = content[flags_start:flags_end]
            return (pattern, flags, flags_end)
        i += 1

    return None


def decode_pcre2_subject(subject):
    """Decode PCRE2 test-subject escape sequences to a Python string.

    Handles ``\\n``, ``\\r``, ``\\t``, ``\\f``, ``\\a``, ``\\e``, hex
    escapes ``\\xNN`` / ``\\x{NNNN}``, and octal escapes ``\\NNN``.
    """
    result = []
    i = 0
    while i < len(subject):
        if subject[i] != "\\" or i + 1 >= len(subject):
            result.append(subject[i])
            i += 1
            continue
        esc = subject[i + 1]
        if esc == "n":
            result.append("\n")
            i += 2
        elif esc == "r":
            result.append("\r")
            i += 2
        elif esc == "t":
            result.append("\t")
            i += 2
        elif esc == "f":
            result.append("\f")
            i += 2
        elif esc == "a":
            result.append("\a")
            i += 2
        elif esc == "e":
            result.append("\x1b")
            i += 2
        elif esc == "b":
            result.append("\b")
            i += 2
        elif esc == "v":
            result.append("\v")
            i += 2
        elif esc == "\\":
            result.append("\\")
            i += 2
        elif esc == '"':
            result.append('"')
            i += 2
        elif esc == "'":
            result.append("'")
            i += 2
        elif esc == "x":
            # \x{NNNN} or \xNN
            if i + 2 < len(subject) and subject[i + 2] == "{":
                end = subject.find("}", i + 3)
                if end != -1:
                    hex_str = subject[i + 3 : end]
                    try:
                        result.append(chr(int(hex_str, 16)))
                    except (ValueError, OverflowError):
                        result.append(subject[i])
                        i += 1
                        continue
                    i = end + 1
                else:
                    result.append(subject[i])
                    i += 1
            elif i + 3 < len(subject) and all(
                c in "0123456789abcdefABCDEF" for c in subject[i + 2 : i + 4]
            ):
                try:
                    result.append(chr(int(subject[i + 2 : i + 4], 16)))
                except (ValueError, OverflowError):
                    result.append(subject[i])
                    i += 1
                    continue
                i += 4
            else:
                result.append(subject[i])
                i += 1
        elif esc in "01234567":
            # Octal: up to 3 digits
            j = i + 1
            while j < min(i + 4, len(subject)) and subject[j] in "01234567":
                j += 1
            try:
                result.append(chr(int(subject[i + 1 : j], 8)))
            except (ValueError, OverflowError):
                result.append(subject[i])
                i += 1
                continue
            i = j
        else:
            # Unknown escape – keep both characters
            result.append(subject[i])
            result.append(subject[i + 1])
            i += 2
    return "".join(result)


def strip_subject_modifiers(subject):
    r"""Strip pcre2test subject modifiers (``\=keyword``) from *subject*."""
    idx = subject.find("\\=")
    if idx != -1:
        return subject[:idx]
    return subject


def build_re2_options(flags):
    """Build an re2.Options instance from PCRE2 flag characters."""
    opts = re2.Options()
    opts.log_errors = False
    for ch in flags:
        if ch == "i":
            opts.case_sensitive = False
        elif ch == "m":
            opts.one_line = False
        elif ch == "s":
            opts.dot_nl = True
        # 'x', 'u', and others have no direct re2.Options equivalent
    return opts


def parse_test_file(filepath):
    """Parse a PCRE2 test file and return a list of test-case dicts.

    Each dict has keys ``pattern``, ``flags``, ``flags_raw``, and
    ``subjects`` (list of ``{subject_raw, expect_match}`` dicts).
    """
    try:
        content = filepath.read_bytes().decode("latin-1")
    except Exception:
        return []

    test_cases = []
    i = 0
    lines = content.splitlines(keepends=True)

    # Rebuild content as a single string for positional scanning
    # We also keep a line-offset map for human-readable error messages
    flat = content

    pos = 0  # current position in flat
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx]

        # Skip blank lines and comment/directive lines
        stripped = line.rstrip("\r\n")
        if not stripped or stripped.startswith("#"):
            line_idx += 1
            pos += len(line)
            continue

        # Pattern line starts with '/'
        if stripped.startswith("/"):
            pat_start = pos + 1  # skip the opening '/'
            result = find_closing_delimiter(flat, pat_start)
            if result is None:
                line_idx += 1
                pos += len(line)
                continue

            pattern, flags, end_pos = result

            # Collect any extra options that come after the letter flags
            # (e.g. ',hex', ',B') up to end of line
            extra_start = end_pos
            while extra_start < len(flat) and flat[extra_start] not in ("\r", "\n"):
                extra_start += 1
            flags_raw = flags + flat[end_pos:extra_start].strip()

            # Skip binary/hex patterns – they are not regular patterns
            if "hex" in flags_raw or ",B" in flags_raw:
                pos = extra_start
                while pos < len(flat) and flat[pos] not in ("\r", "\n"):
                    pos += 1
                if pos < len(flat):
                    pos += 1  # skip the newline
                line_idx = flat[:pos].count("\n")
                continue

            # Advance past the closing delimiter line
            pos = extra_start
            while pos < len(flat) and flat[pos] not in ("\r", "\n"):
                pos += 1
            if pos < len(flat):
                if flat[pos] == "\r" and pos + 1 < len(flat) and flat[pos + 1] == "\n":
                    pos += 2
                else:
                    pos += 1
            line_idx = flat[:pos].count("\n")

            # Now read subject lines
            subjects = []
            expect_match = True

            while line_idx < len(lines):
                sline = lines[line_idx]
                sstripped = sline.rstrip("\r\n")

                # Next pattern or directive starts a new block
                if sstripped.startswith("/") or sstripped.startswith("#"):
                    break

                # "Expect no match" marker
                if sstripped.startswith("\\= Expect no match"):
                    expect_match = False
                    line_idx += 1
                    pos += len(sline)
                    continue

                # Other \= markers (reset, anchored, etc.) – skip
                if sstripped.startswith("\\="):
                    line_idx += 1
                    pos += len(sline)
                    continue

                # Blank line between patterns
                if not sstripped:
                    line_idx += 1
                    pos += len(sline)
                    continue

                # Subject lines are indented (start with whitespace) or
                # sometimes not indented in older test files
                if sstripped:
                    subjects.append(
                        {
                            "subject_raw": sstripped.lstrip(),
                            "expect_match": expect_match,
                        }
                    )

                line_idx += 1
                pos += len(sline)

            test_cases.append(
                {
                    "pattern": pattern,
                    "flags": flags,
                    "flags_raw": flags_raw,
                    "subjects": subjects,
                }
            )
            continue

        line_idx += 1
        pos += len(line)

    return test_cases


def check_re2_support(test_case):
    """Try to compile a pattern with re2 and run all subject tests.

    Uses ``fullmatch`` – the primary operation expected by a regex-to-C-code
    generator, which checks whether the *entire* input string is matched by
    the pattern (equivalent to anchoring the pattern with ``^`` and ``$``).

    Returns a dict with:
    - ``re2_supported`` (bool)
    - ``compile_error`` (str or None)
    - ``num_groups``    total number of capture groups in the pattern
    - ``named_groups``  mapping of group name → group index (1-based)
    - ``subjects`` – list of per-subject result dicts, each containing:
      - ``input``         decoded subject string ready for C-level testing
      - ``match``         whether re2 fullmatch succeeded (bool)
      - ``span``          [start, end] of the full match, or null
      - ``match_string``  the matched substring (= input on fullmatch), or null
      - ``groups``        list of per-group dicts, or null when no match:
                          each entry has ``index`` (1-based), ``value``
                          (string or null if group did not participate), and
                          ``span`` ([start, end] or null)
    """
    pattern = test_case["pattern"]
    flags = test_case["flags"]
    opts = build_re2_options(flags)

    try:
        compiled = re2.compile(pattern, opts)
    except Exception as exc:
        return {
            "re2_supported": False,
            "compile_error": str(exc),
            "num_groups": 0,
            "named_groups": {},
            "subjects": [],
        }

    num_groups = compiled.groups
    # groupindex maps name → 1-based index
    named_groups = dict(compiled.groupindex)

    subject_results = []
    for subj in test_case["subjects"]:
        subject_clean = strip_subject_modifiers(subj["subject_raw"])
        try:
            subject_decoded = decode_pcre2_subject(subject_clean)
        except Exception:
            subject_decoded = subject_clean

        try:
            m = compiled.fullmatch(subject_decoded)
        except Exception:
            # Subject caused a runtime error – skip
            continue

        if m is not None:
            span = list(m.span(0))
            match_string = m.group(0)

            groups = []
            for idx in range(1, num_groups + 1):
                value = m.group(idx)
                try:
                    grp_span = list(m.span(idx))
                    # span (-1, -1) means the group did not participate
                    if grp_span[0] == -1:
                        grp_span = None
                except Exception:
                    grp_span = None
                groups.append(
                    {
                        "index": idx,
                        "value": value,
                        "span": grp_span,
                    }
                )
        else:
            span = None
            match_string = None
            groups = None

        subject_results.append(
            {
                "input": subject_decoded,
                "match": m is not None,
                "span": span,
                "match_string": match_string,
                "groups": groups,
            }
        )

    return {
        "re2_supported": True,
        "compile_error": None,
        "num_groups": num_groups,
        "named_groups": named_groups,
        "subjects": subject_results,
    }


def main():
    testdata_dir = TESTDATA_DIR
    if not testdata_dir.exists():
        print(
            f"Error: testdata directory not found at {testdata_dir}", file=sys.stderr
        )
        sys.exit(1)

    test_files = sorted(
        f for f in testdata_dir.iterdir()
        if f.name.startswith("testinput") and f.suffix not in (".bz2", ".gz")
    )

    test_cases_out = []
    total_patterns = 0
    total_supported = 0
    total_unsupported = 0

    for test_file in test_files:
        parsed = parse_test_file(test_file)

        for tc in parsed:
            total_patterns += 1
            result = check_re2_support(tc)

            if not result["re2_supported"]:
                total_unsupported += 1
                continue

            total_supported += 1

            # Only emit test cases that have at least one usable subject
            if not result["subjects"]:
                continue

            test_cases_out.append(
                {
                    "pattern": tc["pattern"],
                    "flags": tc["flags"],
                    "num_groups": result["num_groups"],
                    "named_groups": result["named_groups"],
                    "source_file": test_file.name,
                    "subjects": result["subjects"],
                }
            )

    output = {
        "summary": {
            "total_patterns": total_patterns,
            "re2_supported": total_supported,
            "re2_unsupported": total_unsupported,
            "test_cases_with_subjects": len(test_cases_out),
        },
        "test_cases": test_cases_out,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
