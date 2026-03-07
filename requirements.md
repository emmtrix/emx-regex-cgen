# Requirements

## Functional Requirements

| ID | Description | Status |
|----|-------------|--------|
| FR-01 | Generate C code that performs a **fullmatch** of a given regular expression against an input string. | ✅ |
| FR-02 | The generated match function has the signature `bool regex_match(const char *input, size_t len)`. | ✅ |
| FR-03 | Optionally emit a `main()` function that reads `argv[1]` and returns exit-code 0 (match), 1 (no match), or 2 (usage error). | ✅ |
| FR-04 | Support the **re2** feature set (no backreferences, no look-around). | ✅ |
| FR-05 | Support regex flags: case-insensitive (`i`), dot-all (`s`), multiline (`m`). | ✅ |
| FR-06 | Usable as a **Python library** (`from regex_cgen import generate`). | ✅ |
| FR-07 | Usable as a **CLI tool** (`regex-cgen <pattern> [options]`). | ✅ |

## Non-Functional Requirements

| ID | Description | Status |
|----|-------------|--------|
| NF-01 | Generated C code is **static** – no dynamic memory allocation, no external dependencies beyond standard C headers. | ✅ |
| NF-02 | The inner matching loop is **branch-free** and suitable for auto-vectorisation when embedded in a `for` loop. | ✅ |
| NF-03 | The DFA transition table uses `static const` arrays for read-only, position-independent data. | ✅ |
| NF-04 | Python ≥ 3.10 support. | ✅ |
| NF-05 | Generated C code compiles with GCC and Clang without warnings under `-Wall -Wextra`. | ✅ |

## Testing Requirements

| ID | Description | Status |
|----|-------------|--------|
| TR-01 | Parameterised pytest suite driven by `re2_compat_results.json`. | ✅ |
| TR-02 | Test strategy: generate C → compile → execute → compare exit code. | ✅ |
| TR-03 | CI pipeline with pytest and ruff linting via GitHub Actions. | ✅ |

## Code Generation Pipeline

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
