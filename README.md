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

# Generate a match function
c_code = generate(r"\d{4}-\d{2}-\d{2}")
print(c_code)

# Include a main() for standalone testing
c_code = generate(r"[a-z]+", emit_main=True)
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
```

## CLI Reference

```
usage: regex-cgen [-h] [-o OUTPUT] [--emit-main] [--func-name NAME]
                  [--flags FLAGS] pattern

Generate C code that performs a fullmatch for a regular expression.

positional arguments:
  pattern           Regular expression pattern

options:
  -o, --output      Output file (default: stdout)
  --emit-main       Also emit a main() function (exit 0=match, 1=no match, 2=error)
  --func-name NAME  Name of the generated match function (default: regex_match)
  --flags FLAGS     Regex flags: i (case-insensitive), s (dot-all), m (multiline)
```

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