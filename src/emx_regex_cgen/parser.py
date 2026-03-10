"""Custom regex parser for emx-regex-cgen.

Replaces Python's ``sre_parse`` with a self-contained recursive-descent
parser that produces an equivalent AST consumed by ``NFABuilder``.

The grammar handled is a subset of PCRE / Python ``re`` syntax — enough
for all features the code-generator already supports (literals, character
classes, alternation, groups, repetition, anchors, boundaries, and the
common escape sequences ``\\d``, ``\\w``, ``\\s``, etc.).
"""

from __future__ import annotations

import re as _re
import sys as _sys

from .unicode_data import resolve_property as _resolve_property

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAXREPEAT: int = 4294967295  # 2**32 - 1, same as _sre.MAXREPEAT

# Upper bound accepted for a repetition count in ``{m,n}`` syntax.
_MAXREPEAT_PARSE = MAXREPEAT


class _Const:
    """Lightweight sentinel for AST-node opcodes with a human-readable repr."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Const):
            return self._name == other._name
        return NotImplemented


# -- Node opcodes -----------------------------------------------------------

LITERAL = _Const("LITERAL")
NOT_LITERAL = _Const("NOT_LITERAL")
ANY = _Const("ANY")
IN = _Const("IN")
BRANCH = _Const("BRANCH")
SUBPATTERN = _Const("SUBPATTERN")
MAX_REPEAT = _Const("MAX_REPEAT")
MIN_REPEAT = _Const("MIN_REPEAT")
AT = _Const("AT")
NEGATE = _Const("NEGATE")
RANGE = _Const("RANGE")
CATEGORY = _Const("CATEGORY")
UNICODE_PROPERTY = _Const("UNICODE_PROPERTY")

# -- AT sub-types -----------------------------------------------------------

AT_BEGINNING = _Const("AT_BEGINNING")
AT_BEGINNING_STRING = _Const("AT_BEGINNING_STRING")
AT_END = _Const("AT_END")
AT_END_STRING = _Const("AT_END_STRING")
AT_BOUNDARY = _Const("AT_BOUNDARY")
AT_NON_BOUNDARY = _Const("AT_NON_BOUNDARY")

# -- CATEGORY sub-types -----------------------------------------------------

CATEGORY_DIGIT = _Const("CATEGORY_DIGIT")
CATEGORY_NOT_DIGIT = _Const("CATEGORY_NOT_DIGIT")
CATEGORY_WORD = _Const("CATEGORY_WORD")
CATEGORY_NOT_WORD = _Const("CATEGORY_NOT_WORD")
CATEGORY_SPACE = _Const("CATEGORY_SPACE")
CATEGORY_NOT_SPACE = _Const("CATEGORY_NOT_SPACE")


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    """Parse a regex string into an AST list compatible with *NFABuilder*."""

    def __init__(self, source: str, flags: int = 0) -> None:
        self.source = source
        self.pos = 0
        self.flags = flags
        self._group_count = 0

    # -- helpers -------------------------------------------------------------

    @property
    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _peek(self) -> str | None:
        if self._at_end:
            return None
        return self.source[self.pos]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        return ch

    def _error(self, msg: str, pos: int | None = None) -> _re.error:
        if pos is None:
            pos = self.pos
        return _re.error(msg, self.source, pos)

    # -- top-level -----------------------------------------------------------

    def parse(self) -> list:
        self._skip_leading_flags()
        result = self._regex()
        if not self._at_end:
            raise self._error("unbalanced parenthesis")
        return result

    def _skip_leading_flags(self) -> None:
        """Consume zero or more leading ``(?flags)`` directives."""
        while self.pos + 2 <= len(self.source):
            if self.source[self.pos] != "(" or self.source[self.pos + 1] != "?":
                break
            saved = self.pos
            self.pos += 2
            if self._at_end or self._peek() not in "imsx":
                self.pos = saved
                break
            while not self._at_end and self._peek() in "imsx":
                ch = self._advance()
                if ch == "x":
                    self.flags |= _re.VERBOSE
            if self._peek() == ")":
                self._advance()
                continue
            # Not a bare flag directive (could be ``(?i:...)``) — backtrack.
            self.pos = saved
            break

    # -- verbose-mode helpers ------------------------------------------------

    def _skip_verbose(self) -> None:
        """When VERBOSE is active, skip whitespace and ``#`` comments."""
        if not (self.flags & _re.VERBOSE):
            return
        while not self._at_end:
            ch = self._peek()
            if ch is not None and ch in " \t\n\r\v\f":
                self._advance()
            elif ch == "#":
                while not self._at_end and self._peek() != "\n":
                    self._advance()
                if not self._at_end:
                    self._advance()  # consume '\n'
            else:
                break

    # -- grammar -------------------------------------------------------------

    def _regex(self) -> list:
        """``regex → branch ('|' branch)*``"""
        branches = [self._branch()]
        while self._peek() == "|":
            self._advance()
            branches.append(self._branch())
        if len(branches) == 1:
            return branches[0]
        return [(BRANCH, (None, branches))]

    def _branch(self) -> list:
        """``branch → piece*``"""
        items: list = []
        self._skip_verbose()
        while not self._at_end and self._peek() not in ("|", ")"):
            result = self._piece()
            if isinstance(result, list):
                items.extend(result)
            else:
                items.append(result)
            self._skip_verbose()
        return items

    def _piece(self):
        """``piece → atom quantifier?``

        Returns a single ``tuple`` or a ``list`` of tuples when a
        non-capturing group is inlined.
        """
        atom_result = self._atom()
        quant = self._quantifier()
        if quant is not None:
            op, lo, hi = quant
            if isinstance(atom_result, list):
                return (op, (lo, hi, atom_result))
            return (op, (lo, hi, [atom_result]))
        return atom_result

    def _atom(self):
        """Parse a single atom.

        Returns a ``tuple`` (normal node) or a ``list`` of tuples (inlined
        non-capturing group content).
        """
        ch = self._peek()
        if ch is None:
            raise self._error("unexpected end of pattern")
        if ch == "(":
            return self._group()
        if ch == "[":
            return self._char_class()
        if ch == ".":
            self._advance()
            return (ANY, None)
        if ch == "^":
            self._advance()
            return (AT, AT_BEGINNING)
        if ch == "$":
            self._advance()
            return (AT, AT_END)
        if ch == "\\":
            return self._escape()
        if ch in "*+?":
            raise self._error("nothing to repeat")
        # Literal character (includes ``{``, ``}``, and anything else).
        self._advance()
        return (LITERAL, ord(ch))

    # -- groups --------------------------------------------------------------

    def _group(self):
        start_pos = self.pos
        self._advance()  # consume '('

        if self._peek() == "?":
            self._advance()  # consume '?'
            ch = self._peek()

            if ch == ":":
                self._advance()
                items = self._regex()
                if self._peek() != ")":
                    raise self._error(
                        "missing ), unterminated subpattern", start_pos
                    )
                self._advance()
                return items  # inline the content

            if ch == "P":
                self._advance()
                return self._named_group_P(start_pos)

            if ch is not None and ch in "imsx-":
                return self._flag_group(start_pos)

            # Unknown extension — e.g. ``(?<...)``, ``(?U)``
            raise self._error(
                f"unknown extension ?{ch if ch is not None else ''}", start_pos + 1
            )

        # Capturing group.
        self._group_count += 1
        gid = self._group_count
        items = self._regex()
        if self._peek() != ")":
            raise self._error("missing ), unterminated subpattern", start_pos)
        self._advance()
        return (SUBPATTERN, (gid, 0, 0, items))

    def _named_group_P(self, start_pos: int):
        """Parse ``(?P<name>...)``."""
        if self._peek() != "<":
            c = self._peek() or ""
            raise self._error(f"unknown extension ?P{c}", start_pos + 1)
        self._advance()  # consume '<'

        name_start = self.pos
        while not self._at_end and self._peek() != ">":
            self._advance()
        if self._at_end:
            raise self._error("missing >, unterminated name", name_start)
        self._advance()  # consume '>'

        self._group_count += 1
        gid = self._group_count
        items = self._regex()
        if self._peek() != ")":
            raise self._error("missing ), unterminated subpattern", start_pos)
        self._advance()
        return (SUBPATTERN, (gid, 0, 0, items))

    def _flag_group(self, start_pos: int):
        """Parse ``(?flags)``, ``(?flags:...)`` and ``(?-flags:...)``."""
        while not self._at_end and self._peek() in "imsx-":
            self._advance()

        if self._peek() == ":":
            self._advance()
            items = self._regex()
            if self._peek() != ")":
                raise self._error(
                    "missing ), unterminated subpattern", start_pos
                )
            self._advance()
            return items  # inline

        if self._peek() == ")":
            self._advance()
            if start_pos > 0:
                raise self._error(
                    "global flags not at the start of the expression",
                    start_pos + 1,
                )
            # Leading directive — already consumed by _skip_leading_flags in
            # the normal path.  Return empty list so it splices as a no-op.
            return []

        raise self._error("missing :", self.pos)

    # -- quantifiers ---------------------------------------------------------

    def _quantifier(self):
        ch = self._peek()
        if ch == "*":
            self._advance()
            lazy = self._peek() == "?"
            if lazy:
                self._advance()
            return (MIN_REPEAT if lazy else MAX_REPEAT, 0, MAXREPEAT)
        if ch == "+":
            self._advance()
            lazy = self._peek() == "?"
            if lazy:
                self._advance()
            return (MIN_REPEAT if lazy else MAX_REPEAT, 1, MAXREPEAT)
        if ch == "?":
            self._advance()
            lazy = self._peek() == "?"
            if lazy:
                self._advance()
            return (MIN_REPEAT if lazy else MAX_REPEAT, 0, 1)
        if ch == "{":
            return self._bounded_quantifier()
        return None

    def _bounded_quantifier(self):
        """Try to parse ``{m}``, ``{m,}``, ``{m,n}``.

        Returns ``None`` (and backtracks) when the braces do not form a
        valid quantifier.
        """
        saved = self.pos
        self._advance()  # consume '{'

        m_str = ""
        while not self._at_end and self._peek().isdigit():
            m_str += self._advance()

        if not m_str:
            self.pos = saved
            return None

        m = int(m_str)
        if m >= _MAXREPEAT_PARSE:
            raise OverflowError("the repetition number is too large")

        if self._peek() == "}":
            self._advance()
            lazy = self._peek() == "?"
            if lazy:
                self._advance()
            return (MIN_REPEAT if lazy else MAX_REPEAT, m, m)

        if self._peek() == ",":
            self._advance()
            n_str = ""
            while not self._at_end and self._peek().isdigit():
                n_str += self._advance()

            if self._peek() == "}":
                self._advance()
                n = MAXREPEAT
                if n_str:
                    n = int(n_str)
                    if n >= _MAXREPEAT_PARSE:
                        raise OverflowError("the repetition number is too large")
                lazy = self._peek() == "?"
                if lazy:
                    self._advance()
                return (MIN_REPEAT if lazy else MAX_REPEAT, m, n)

        # Not a valid quantifier — backtrack.
        self.pos = saved
        return None

    # -- character class -----------------------------------------------------

    def _char_class(self):
        """Parse ``[...]`` character class."""
        self._advance()  # consume '['
        items: list = []
        negate = False

        if self._peek() == "^":
            self._advance()
            negate = True
            items.append((NEGATE, None))

        # ``]`` immediately after ``[`` or ``[^`` is a literal.
        if self._peek() == "]":
            items.append((LITERAL, ord("]")))
            self._advance()

        while not self._at_end and self._peek() != "]":
            atom = self._cc_atom()

            # Look-ahead for a range ``a-z``.
            if (
                self._peek() == "-"
                and self.pos + 1 < len(self.source)
                and self.source[self.pos + 1] != "]"
            ):
                self._advance()  # consume '-'
                hi_atom = self._cc_atom()
                if atom[0] is LITERAL and hi_atom[0] is LITERAL:
                    items.append((RANGE, (atom[1], hi_atom[1])))
                else:
                    items.append(atom)
                    items.append((LITERAL, ord("-")))
                    items.append(hi_atom)
            else:
                items.append(atom)

        if self._at_end:
            raise self._error("unterminated character set")
        self._advance()  # consume ']'

        # --- optimisations matching sre_parse behaviour --------------------
        non_negate_items = items[1:] if negate else items
        if len(non_negate_items) == 1 and non_negate_items[0][0] is LITERAL:
            if negate:
                return (NOT_LITERAL, non_negate_items[0][1])
            return non_negate_items[0]

        return (IN, items)

    def _cc_atom(self):
        """Parse one atom inside a character class."""
        if self._peek() == "\\":
            return self._cc_escape()
        return (LITERAL, ord(self._advance()))

    def _cc_escape(self):
        r"""Handle ``\`` escape inside a character class."""
        esc_pos = self.pos
        self._advance()  # consume '\'
        if self._at_end:
            raise self._error("bad escape (end of pattern)", esc_pos)
        ch = self._advance()

        if ch == "d":
            return (CATEGORY, CATEGORY_DIGIT)
        if ch == "D":
            return (CATEGORY, CATEGORY_NOT_DIGIT)
        if ch == "w":
            return (CATEGORY, CATEGORY_WORD)
        if ch == "W":
            return (CATEGORY, CATEGORY_NOT_WORD)
        if ch == "s":
            return (CATEGORY, CATEGORY_SPACE)
        if ch == "S":
            return (CATEGORY, CATEGORY_NOT_SPACE)

        if ch == "n":
            return (LITERAL, 10)
        if ch == "t":
            return (LITERAL, 9)
        if ch == "r":
            return (LITERAL, 13)
        if ch == "f":
            return (LITERAL, 12)
        if ch == "v":
            return (LITERAL, 11)
        if ch == "a":
            return (LITERAL, 7)

        if ch == "x":
            return (LITERAL, self._hex_escape(esc_pos))

        # Octal: \0–\7 start an octal escape inside character classes.
        if "0" <= ch <= "7":
            return (LITERAL, self._octal_escape(ch, esc_pos))

        if ch in "pP":
            name = self._unicode_property(ch, esc_pos)
            return (UNICODE_PROPERTY, (name, ch == "P"))

        if ch in "CQEz":
            raise self._error(f"bad escape \\{ch}", esc_pos)
        return (LITERAL, ord(ch))

    # -- escape outside character class --------------------------------------

    def _escape(self):
        r"""Handle ``\`` escape outside a character class."""
        esc_pos = self.pos
        self._advance()  # consume '\'
        if self._at_end:
            raise self._error("bad escape (end of pattern)", esc_pos)
        ch = self._advance()

        # Character-class shorthands (wrapped in IN).
        if ch == "d":
            return (IN, [(CATEGORY, CATEGORY_DIGIT)])
        if ch == "D":
            return (IN, [(CATEGORY, CATEGORY_NOT_DIGIT)])
        if ch == "w":
            return (IN, [(CATEGORY, CATEGORY_WORD)])
        if ch == "W":
            return (IN, [(CATEGORY, CATEGORY_NOT_WORD)])
        if ch == "s":
            return (IN, [(CATEGORY, CATEGORY_SPACE)])
        if ch == "S":
            return (IN, [(CATEGORY, CATEGORY_NOT_SPACE)])

        # Anchors / boundaries.
        if ch == "b":
            return (AT, AT_BOUNDARY)
        if ch == "B":
            return (AT, AT_NON_BOUNDARY)
        if ch == "A":
            return (AT, AT_BEGINNING_STRING)
        if ch == "Z":
            return (AT, AT_END_STRING)

        # Common character escapes.
        if ch == "n":
            return (LITERAL, 10)
        if ch == "t":
            return (LITERAL, 9)
        if ch == "r":
            return (LITERAL, 13)
        if ch == "f":
            return (LITERAL, 12)
        if ch == "v":
            return (LITERAL, 11)
        if ch == "a":
            return (LITERAL, 7)

        # Hex escape.
        if ch == "x":
            return (LITERAL, self._hex_escape(esc_pos))

        # Octal escape — ``\0`` always starts one outside char classes.
        if ch == "0":
            return (LITERAL, self._octal_escape(ch, esc_pos))

        # Digit escapes 1–7 can be octal inside char classes in sre_parse
        # but outside they are group references.  We don't support
        # back-references, so treat multi-digit sequences starting with
        # 1–7 as octal (matching sre_parse behaviour when no groups
        # exist) and single digits as errors.
        if "1" <= ch <= "7":
            # Peek ahead: if at least one more octal digit follows, treat
            # the whole sequence as an octal escape for compatibility with
            # patterns that use e.g. ``\177``.
            if not self._at_end and "0" <= self._peek() <= "7":
                return (LITERAL, self._octal_escape(ch, esc_pos))
            raise self._error(f"bad escape \\{ch}", esc_pos)
        if ch in "89":
            raise self._error(f"bad escape \\{ch}", esc_pos)

        # Unicode property escapes.
        if ch in "pP":
            name = self._unicode_property(ch, esc_pos)
            negate = ch == "P"
            items: list = []
            if negate:
                items.append((NEGATE, None))
            items.append((UNICODE_PROPERTY, (name, False)))
            return (IN, items)

        # Unsupported PCRE2 / Perl escapes.
        if ch in "CQEz":
            raise self._error(f"bad escape \\{ch}", esc_pos)

        # Literal escape — ``\.``, ``\\``, ``\*``, etc.
        return (LITERAL, ord(ch))

    # -- shared escape helpers -----------------------------------------------

    def _hex_escape(self, esc_pos: int) -> int:
        r"""Parse ``\xNN`` — two hex digits."""
        if self.pos + 2 > len(self.source):
            raise self._error("incomplete escape \\x", esc_pos)
        hex_str = self.source[self.pos : self.pos + 2]
        try:
            val = int(hex_str, 16)
        except ValueError:
            raise self._error(f"bad escape \\x{hex_str}", esc_pos) from None
        self.pos += 2
        return val

    def _octal_escape(self, first_digit: str, esc_pos: int) -> int:
        r"""Parse an octal escape.

        *first_digit* has already been consumed.  Up to two more octal
        digits (``0``–``7``) are read.
        """
        digits = first_digit
        for _ in range(2):
            if not self._at_end and "0" <= self._peek() <= "7":
                digits += self._advance()
            else:
                break
        val = int(digits, 8)
        if val > 0o377:
            raise self._error(
                f"octal escape value \\{digits} outside of range 0-0o377",
                esc_pos,
            )
        return val

    def _unicode_property(self, esc_ch: str, esc_pos: int) -> str:
        r"""Parse Unicode property name after ``\p`` or ``\P``.

        Accepts both ``\p{Name}`` (braced) and ``\pX`` (single-letter) forms.
        Returns the resolved property name.
        """
        if self._at_end:
            raise self._error(f"bad escape \\{esc_ch}", esc_pos)
        if self._peek() == "{":
            self._advance()  # consume '{'
            start = self.pos
            while not self._at_end and self._peek() != "}":
                self._advance()
            if self._at_end:
                raise self._error(
                    f"incomplete Unicode property \\{esc_ch}{{...", esc_pos,
                )
            name = self.source[start : self.pos]
            self._advance()  # consume '}'
        else:
            # Single-letter form: \pL, \pN, etc.
            name = self._advance()
        if not name:
            raise self._error("empty Unicode property name", esc_pos)
        try:
            _resolve_property(name)
        except _re.error:
            raise self._error(
                f"unknown Unicode property name: {name!r}", esc_pos,
            ) from None
        return name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(pattern: str, flags: int = 0) -> list:
    """Parse *pattern* into an AST list compatible with ``NFABuilder``.

    The *flags* parameter accepts ``re`` module flag values but is currently
    only used for basic validation; actual flag handling is done by
    *NFABuilder*.
    """
    # Temporarily raise the recursion limit so that deeply nested groups
    # (e.g. 200+ levels) don't hit Python's default ceiling.
    old_limit = _sys.getrecursionlimit()
    try:
        _sys.setrecursionlimit(max(old_limit, 10_000))
        return _Parser(pattern, flags).parse()
    finally:
        _sys.setrecursionlimit(old_limit)
