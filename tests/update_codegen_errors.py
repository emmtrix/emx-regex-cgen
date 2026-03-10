#!/usr/bin/env python3
"""Regenerate expected compile-time errors for re2 compatibility tests."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS_JSON = ROOT / "tests" / "data" / "re2_compat_results.json"
ERRORS_JSON = ROOT / "tests" / "data" / "re2_compat_errors.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emx_regex_cgen import generate  # noqa: E402


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> None:
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        cases = json.load(fh)["test_cases"]

    warnings.simplefilter("ignore", FutureWarning)

    errors: list[dict[str, object]] = []
    for engine in ("dfa", "bitnfa"):
        for case_idx, case in enumerate(cases):
            if not case.get("subjects"):
                continue
            pattern = case["pattern"]
            flags = "".join(ch for ch in case.get("flags", "") if ch in "imsx")
            try:
                generate(pattern, flags=flags, engine=engine)
            except Exception as exc:
                errors.append(
                    {
                        "case_idx": case_idx,
                        "engine": engine,
                        "pattern": pattern,
                        "flags": flags,
                        "error": _format_exception(exc),
                    }
                )

    payload = {
        "generated_from": "tests/data/re2_compat_results.json",
        "total_expected_errors": len(errors),
        "errors": errors,
    }
    with open(ERRORS_JSON, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
