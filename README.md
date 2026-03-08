# regex-cgen

**Regex to C Code Generator** — compile regular expressions into portable,
static C code for embedded and performance-critical applications.

[![CI](https://github.com/emmtrix/regex-cgen/actions/workflows/ci.yml/badge.svg)](https://github.com/emmtrix/regex-cgen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Key Characteristics

- **Table-driven DFA** — every regex is compiled to a minimised
  deterministic finite automaton; the generated C code performs a single
  linear scan over the input.
- **No dynamic memory allocation** — all data is `static const`; no
  `malloc`, no `free`.
- **Branch-free inner loop** — the matching loop contains a single table
  lookup per byte, making it highly suitable for auto-vectorisation.
- **re2 feature set** — supports the same subset of regular expressions
  as Google's [re2](https://github.com/google/re2) library (no
  back-references, no look-around).
- **Fullmatch semantics** — the generated function checks whether the
  *entire* input matches the pattern.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Python Library

```python
from regex_cgen import generate

# Generate a match function (UTF-8 mode, default)
c_code = generate(r"\d{4}-\d{2}-\d{2}")
print(c_code)

# Include a main() for standalone testing
c_code = generate(r"[a-z]+", emit_main=True)

# Bytes mode: '.' matches any single byte, classes work on raw byte values
c_code = generate(r"[\x80-\xff]+", encoding="bytes")
```

### CLI

```bash
# Write generated C code to stdout
regex-cgen '[a-z]+\d+'

# Write to a file with a main() function
regex-cgen '[a-z]+\d+' --emit-main -o matcher.c

# Compile and test
gcc -O2 -o matcher matcher.c
./matcher "hello42"   # exit 0 (match)
./matcher "HELLO"     # exit 1 (no match)

# Bytes mode: match any sequence of high bytes
regex-cgen --encoding bytes '[\x80-\xff]+' --emit-main -o byte_matcher.c
```

## CLI Reference

```
usage: regex-cgen [-h] [-o OUTPUT] [--emit-main] [--func-name NAME]
                  [--flags FLAGS] [--encoding {utf8,bytes}] pattern

Generate C code that performs a fullmatch for a regular expression.

positional arguments:
  pattern               Regular expression pattern

options:
  -o, --output          Output file (default: stdout)
  --emit-main           Also emit a main() function (exit 0=match, 1=no match, 2=error)
  --func-name NAME      Name of the generated match function (default: regex_match)
  --flags FLAGS         Regex flags: i (case-insensitive), s (dot-all), m (multiline)
  --encoding {utf8,bytes}
                        Input encoding: utf8 (default, Unicode-aware) or bytes
                        (raw byte semantics: '.' = one arbitrary byte, no UTF-8
                        sequence handling, literals/classes work byte-wise 0-255)
```

## Supported Features

### Regex Features

| Feature | Example pattern | Golden reference |
|---------|-----------------|------------------|
| Literal string | `hello` | [literal.c](tests/golden/literal.c) |
| Character class | `[a-z0-9_]+` | [char\_class.c](tests/golden/char_class.c) |
| Negated class | `[^aeiou]+` | [negated\_class.c](tests/golden/negated_class.c) |
| Dot (any char except `\n`) | `.+` | [dot.c](tests/golden/dot.c) |
| Alternation | `cat|dog|fish` | [alternation.c](tests/golden/alternation.c) |
| Star quantifier `*` | `ab*c` | [quantifier\_star.c](tests/golden/quantifier_star.c) |
| Plus quantifier `+` | `ab+c` | [quantifier\_plus.c](tests/golden/quantifier_plus.c) |
| Optional quantifier `?` | `colou?r` | [quantifier\_optional.c](tests/golden/quantifier_optional.c) |
| Bounded repeat `{m,n}` | `a{2,4}` | [quantifier\_repeat.c](tests/golden/quantifier_repeat.c) |
| Digit escape `\d` | `\d{4}-\d{2}-\d{2}` | [escape\_digit.c](tests/golden/escape_digit.c) |
| Word escape `\w` | `\w+` | [escape\_word.c](tests/golden/escape_word.c) |
| Space escape `\s` | `\s+` | [escape\_space.c](tests/golden/escape_space.c) |
| Unicode / UTF-8 | `\x{00e9}+` | [unicode.c](tests/golden/unicode.c) |
| Anchors `^` / `$` | `^start.*end$` | [anchors.c](tests/golden/anchors.c) |

### CLI Options

| Option | Effect | Golden reference |
|--------|--------|------------------|
| `--flags i` | Case-insensitive matching | [flag\_ignorecase.c](tests/golden/flag_ignorecase.c) |
| `--flags s` | Dot matches `\n` (dot-all) | [flag\_dotall.c](tests/golden/flag_dotall.c) |
| `--flags m` | Multiline anchors | [flag\_multiline.c](tests/golden/flag_multiline.c) |
| `--flags x` | Verbose / free-spacing mode | [flag\_verbose.c](tests/golden/flag_verbose.c) |
| `--encoding bytes` | Raw byte semantics (no UTF-8) | [encoding\_bytes.c](tests/golden/encoding_bytes.c) |
| `--func-name NAME` | Custom match-function name | [func\_name.c](tests/golden/func_name.c) |
| `--emit-main` | Include standalone `main()` | [emit\_main.c](tests/golden/emit_main.c) |

## Generated Code Structure

The generator produces:

1. A **transition table** (`static const uint16_t dfa_transitions[N][256]`)
   mapping `(state, byte) → next_state`.
2. An **accept table** (`static const bool dfa_accept[N]`) marking
   accepting states.
3. A **match function** with the signature:
   ```c
   bool regex_match(const char *input, size_t len);
   ```
4. Optionally a **`main()` function** for standalone executables.

### Example Output (pattern `a[bc]+d`)

```c
bool regex_match(const char *input, size_t len) {
    uint16_t state = 1;
    for (size_t i = 0; i < len; i++) {
        state = dfa_transitions[state][(unsigned char)input[i]];
    }
    return dfa_accept[state];
}
```

## Compilation Pipeline

```
Regex pattern (string)
        │
        ▼
   sre_parse AST
        │
        ▼
   Thompson NFA
        │
        ▼
   Subset-construction DFA
        │
        ▼
   Hopcroft-minimised DFA
        │
        ▼
   Table-driven C code
```

## Testing

Tests are parameterised from `re2_compat_results.json`, which contains
2 500+ patterns extracted from the PCRE2 test suite and validated against
Google re2.

```bash
# Run all tests (parallel)
pytest -n auto -q

# Run linter
ruff check src/ tests/
```

### Test Strategy

1. **Generate** C code from the regex pattern.
2. **Compile** with `gcc -O2`.
3. **Execute** the binary with each test subject as `argv[1]`.
4. **Compare** the exit code against the expected match/no-match result.

## Development

```bash
# Clone
git clone --recurse-submodules https://github.com/emmtrix/regex-cgen.git
cd regex-cgen

# Install
pip install -e ".[dev]"

# Test
pytest -n auto -q

# Lint
ruff check src/ tests/
```

## License

[MIT](LICENSE) — Copyright (c) 2026 emmtrix Technologies GmbH

## Maintained by

[emmtrix Technologies GmbH](https://emmtrix.com)