"""Command-line interface for regex-cgen."""

from __future__ import annotations

import argparse
import sys

from .codegen import generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="regex-cgen",
        description="Generate C code that performs a fullmatch for a regular expression.",
    )
    parser.add_argument("pattern", help="Regular expression pattern")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--emit-main",
        action="store_true",
        help="Also emit a main() function (exit 0=match, 1=no match, 2=error)",
    )
    parser.add_argument(
        "--func-name",
        default="regex_match",
        help="Name of the generated match function (default: regex_match)",
    )
    parser.add_argument(
        "--flags",
        default="",
        help="Regex flags: i (case-insensitive), s (dot-all), m (multiline), x (verbose)",
    )

    args = parser.parse_args(argv)

    try:
        code = generate(
            args.pattern,
            flags=args.flags,
            emit_main=args.emit_main,
            func_name=args.func_name,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output == "-":
        sys.stdout.write(code)
    else:
        with open(args.output, "w") as fh:
            fh.write(code)


if __name__ == "__main__":
    main()
