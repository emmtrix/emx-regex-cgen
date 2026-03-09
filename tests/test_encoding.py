"""Tests for --encoding {utf8,bytes} mode.

Strategy: generate C code, compile with gcc, execute with raw byte inputs,
compare exit codes.

UTF-8 mode is the default and preserves existing Unicode-aware behaviour.
Bytes mode uses pure byte semantics: '.' = one arbitrary byte, literals and
character classes operate on raw byte values 0-255.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from regex_cgen import generate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(pattern: str, tmp_path: Path, encoding: str = "utf8", flags: str = "") -> Path:
    """Generate, write and compile a C matcher; return the executable path."""
    c_code = generate(pattern, flags=flags, emit_main=True, encoding=encoding).render()
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


def _run(exe: Path, inp: bytes) -> bool:
    """Run *exe* with *inp* passed as a single argument; return True on match."""
    run = subprocess.run([str(exe), inp], capture_output=True, timeout=10)
    return run.returncode == 0


# ---------------------------------------------------------------------------
# Bytes-mode tests
# ---------------------------------------------------------------------------

class TestBytesMode:
    def test_dot_matches_any_byte(self, tmp_path: Path) -> None:
        """'.' in bytes mode must match every byte 0x00-0xFF (except \\n)."""
        exe = _build(".", tmp_path, encoding="bytes")
        # ASCII bytes should match
        assert _run(exe, b"a")
        assert _run(exe, b"Z")
        # High bytes (non-ASCII) should also match
        assert _run(exe, bytes([0x80]))
        assert _run(exe, bytes([0xFF]))
        assert _run(exe, bytes([0xC0]))
        # Newline must NOT match (dotall off)
        assert not _run(exe, b"\n")
        # Two bytes must not match (fullmatch)
        assert not _run(exe, bytes([0x80, 0x81]))

    def test_dot_dotall_matches_newline(self, tmp_path: Path) -> None:
        """With flag 's', '.' in bytes mode matches \\n too."""
        exe = _build(".", tmp_path, encoding="bytes", flags="s")
        assert _run(exe, b"\n")
        assert _run(exe, bytes([0xFF]))

    def test_literal_high_byte(self, tmp_path: Path) -> None:
        r"""Pattern with a literal non-ASCII char matches its byte value."""
        # '\x80' as a single-byte pattern in bytes mode
        exe = _build("\x80", tmp_path, encoding="bytes")
        assert _run(exe, bytes([0x80]))
        assert not _run(exe, bytes([0x81]))
        assert not _run(exe, b"a")

    def test_char_class_high_bytes(self, tmp_path: Path) -> None:
        """[\\x80-\\x82] in bytes mode matches the three byte values."""
        exe = _build("[\x80-\x82]", tmp_path, encoding="bytes")
        assert _run(exe, bytes([0x80]))
        assert _run(exe, bytes([0x81]))
        assert _run(exe, bytes([0x82]))
        assert not _run(exe, bytes([0x83]))
        assert not _run(exe, b"a")

    def test_negated_class_bytes(self, tmp_path: Path) -> None:
        """[^a] in bytes mode matches all bytes except 0x61."""
        exe = _build("[^a]", tmp_path, encoding="bytes")
        assert not _run(exe, b"a")
        assert _run(exe, b"b")
        assert _run(exe, bytes([0x80]))
        assert _run(exe, bytes([0xFF]))

    def test_dot_star_bytes(self, tmp_path: Path) -> None:
        """'.*' in bytes mode matches any byte sequence (no multi-byte grouping)."""
        exe = _build(".*", tmp_path, encoding="bytes")
        assert _run(exe, b"")
        assert _run(exe, b"hello")
        assert _run(exe, bytes([0x80, 0x90, 0xC0, 0xFF]))
        # With dotall off, a newline in the middle breaks the match
        assert not _run(exe, b"a\nb")

    def test_not_literal_high_byte(self, tmp_path: Path) -> None:
        r"""[^\x80] in bytes mode accepts every byte except 0x80."""
        exe = _build("[^\x80]", tmp_path, encoding="bytes")
        assert not _run(exe, bytes([0x80]))
        assert _run(exe, bytes([0x81]))
        assert _run(exe, bytes([0x7F]))
        assert _run(exe, b"a")

    def test_digit_class_bytes(self, tmp_path: Path) -> None:
        r"""\d in bytes mode matches ASCII digits only."""
        exe = _build(r"\d+", tmp_path, encoding="bytes")
        assert _run(exe, b"0")
        assert _run(exe, b"9")
        assert _run(exe, b"123")
        assert not _run(exe, b"a")
        assert not _run(exe, bytes([0x80]))

    def test_invalid_encoding_raises(self) -> None:
        """generate() with an unknown encoding must raise ValueError."""
        with pytest.raises(ValueError, match="encoding"):
            generate("a", encoding="latin1")


# ---------------------------------------------------------------------------
# UTF-8 mode (default) – regression tests
# ---------------------------------------------------------------------------

class TestUtf8Mode:
    def test_utf8_default(self, tmp_path: Path) -> None:
        """Default encoding is utf8; basic ASCII pattern still works."""
        exe = _build("[a-z]+", tmp_path)
        assert _run(exe, b"hello")
        assert not _run(exe, b"HELLO")

    def test_utf8_dot_does_not_match_high_byte(self, tmp_path: Path) -> None:
        """In UTF-8 mode, '.' does NOT match a lone high byte (invalid UTF-8)."""
        exe = _build(".", tmp_path, encoding="utf8")
        # A lone 0x80 byte is not valid UTF-8; should not match
        assert not _run(exe, bytes([0x80]))

    def test_utf8_dot_matches_multibyte(self, tmp_path: Path) -> None:
        """In UTF-8 mode, '.' matches a valid multi-byte UTF-8 character."""
        exe = _build(".", tmp_path, encoding="utf8")
        # U+00E9 encoded as UTF-8: 0xC3 0xA9
        assert _run(exe, "\u00e9".encode("utf-8"))

    def test_utf8_explicit_encoding(self, tmp_path: Path) -> None:
        """Explicitly passing encoding='utf8' is equivalent to the default."""
        exe = _build(".", tmp_path, encoding="utf8")
        assert _run(exe, b"x")
        assert not _run(exe, bytes([0x80]))

    def test_bytes_and_utf8_differ_on_high_byte(self, tmp_path: Path) -> None:
        """The two modes produce different results for high-byte input."""
        bytes_dir = tmp_path / "bytes"
        utf8_dir = tmp_path / "utf8"
        bytes_dir.mkdir()
        utf8_dir.mkdir()
        exe_bytes = _build(".", bytes_dir, encoding="bytes")
        exe_utf8 = _build(".", utf8_dir, encoding="utf8")
        hi = bytes([0x80])
        assert _run(exe_bytes, hi)       # bytes mode: matches
        assert not _run(exe_utf8, hi)    # utf8 mode: invalid UTF-8, no match
