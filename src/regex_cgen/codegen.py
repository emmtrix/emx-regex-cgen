"""Generate C source code from a compiled DFA or bit-parallel NFA."""

from __future__ import annotations

from .compiler import compile_nfa, compile_regex
from .result import GeneratedCode


def _should_apply(mode: str, table_size: int, threshold: int) -> bool:
    """Resolve an ``"auto"``/``"yes"``/``"no"`` mode flag."""
    if mode == "yes":
        return True
    if mode == "no":
        return False
    # auto – apply only when the uncompressed table exceeds the threshold
    return table_size > threshold


def _compute_alphabet_classes(
    n: int, trans: dict
) -> tuple[list[int], int, list[int]]:
    """Compute byte equivalence classes for alphabet compression.

    Two byte values belong to the same class when they produce identical
    transitions in *every* DFA state.

    Returns ``(class_map, num_classes, class_representatives)`` where

    * ``class_map[byte]`` is the class-id for each byte 0-255,
    * ``num_classes`` is the total number of distinct classes, and
    * ``class_representatives[class_id]`` is one byte from each class
      (used to read the transition value for that class).
    """
    sig_to_class: dict[tuple[int, ...], int] = {}
    class_map: list[int] = [0] * 256
    class_reps: list[int] = []

    for b in range(256):
        sig = tuple(trans.get((s, b), 0) for s in range(n))
        if sig not in sig_to_class:
            sig_to_class[sig] = len(class_reps)
            class_reps.append(b)
        class_map[b] = sig_to_class[sig]

    return class_map, len(class_reps), class_reps


def generate_c_code(
    dfa: dict,
    *,
    prefix: str = "regex",
    emit_main: bool = False,
    row_dedup: str = "auto",
    alphabet_compression: str = "auto",
    size_threshold: int = 8192,
    pattern: str | None = None,
    flags: str = "",
    encoding: str = "utf8",
    early_exit: bool = False,
) -> GeneratedCode:
    """Emit C code for a table-driven DFA matcher.

    The generated code contains:

    * An optional ``{prefix}_alphabet`` byte-to-class mapping array
      (emitted when *alphabet_compression* is active and yields fewer
      than 256 equivalence classes).
    * A ``static const`` transition table (``{prefix}_transitions``).
    * An optional ``{prefix}_row_map`` indirection array (emitted when
      *row_dedup* is active and duplicate rows exist).
    * A match function named ``{prefix}_match`` with the signature::

          bool {prefix}_match(const char *input, size_t len);

    States are ordered so that all accepting states have indices >=
    ``first_accept``; the match function uses ``return state >= first_accept;``
    as the acceptance test (no boolean lookup table required).

    When *emit_main* is ``True`` an additional ``main`` function is emitted
    that reads ``argv[1]`` and returns exit-code **0** on match, **1** on
    mismatch, and **2** on usage error.

    Parameters
    ----------
    row_dedup:
        ``"yes"`` always deduplicate identical transition rows,
        ``"no"`` never, ``"auto"`` only when the uncompressed table
        exceeds *size_threshold* cells.
    alphabet_compression:
        ``"yes"`` always compress the alphabet,
        ``"no"`` never, ``"auto"`` only when the uncompressed table
        exceeds *size_threshold* cells.
    size_threshold:
        Number of transition-table cells (states × 256) above which
        ``"auto"`` mode enables the corresponding optimisation.
    early_exit:
        When ``True``, emit ``if (state == 0) break;`` inside the DFA
        loop so matching terminates as soon as the dead state is reached.
    """
    n = dfa["num_states"]
    initial = dfa["initial"]
    first_accept = dfa["first_accept"]
    trans = dfa["transitions"]

    # Choose the narrowest unsigned type that fits
    if n <= 256:
        state_t = "uint8_t"
    elif n <= 65536:
        state_t = "uint16_t"
    else:
        state_t = "uint32_t"

    # --- resolve auto modes ------------------------------------------------
    table_size = n * 256
    do_dedup = _should_apply(row_dedup, table_size, size_threshold)
    do_alphabet = _should_apply(alphabet_compression, table_size, size_threshold)

    # --- alphabet compression -----------------------------------------------
    if do_alphabet:
        class_map, num_classes, class_reps = _compute_alphabet_classes(n, trans)
        if num_classes >= 256:
            do_alphabet = False  # no benefit

    if do_alphabet:
        num_cols = num_classes
    else:
        num_cols = 256
        class_reps = list(range(256))

    # --- build per-state rows (possibly compressed columns) -----------------
    all_rows: list[tuple[int, ...]] = []
    for s in range(n):
        row = tuple(trans.get((s, class_reps[c]), 0) for c in range(num_cols))
        all_rows.append(row)

    # --- row deduplication --------------------------------------------------
    if do_dedup:
        unique_rows: list[tuple[int, ...]] = []
        row_index: dict[tuple[int, ...], int] = {}
        state_to_row: list[int] = []
        for s in range(n):
            row = all_rows[s]
            if row not in row_index:
                row_index[row] = len(unique_rows)
                unique_rows.append(row)
            state_to_row.append(row_index[row])
        num_unique = len(unique_rows)
        has_row_map = num_unique < n
    else:
        unique_rows = all_rows
        state_to_row = list(range(n))
        num_unique = n
        has_row_map = False

    # Build reverse mapping: row index → list of states that use it
    row_to_states: list[list[int]] = [[] for _ in range(num_unique)]
    for s, r in enumerate(state_to_row):
        row_to_states[r].append(s)

    # --- build includes -----------------------------------------------------
    includes = ["stddef.h", "stdbool.h", "stdint.h"]
    if emit_main:
        includes.extend(["string.h", "stdio.h"])

    # --- build globals (tables) --------------------------------------------
    global_lines: list[str] = []

    # Alphabet map (byte → equivalence-class id)
    if do_alphabet:
        alpha_t = "uint8_t" if num_classes <= 256 else "uint16_t"
        global_lines.append(f"static const {alpha_t} {prefix}_alphabet[256] = {{")
        for row_start in range(0, 256, 16):
            row_end = min(row_start + 16, 256)
            vals = ", ".join(str(class_map[b]) for b in range(row_start, row_end))
            if row_end < 256:
                global_lines.append(f"    {vals},")
            else:
                global_lines.append(f"    {vals}")
        global_lines.append("};")
        global_lines.append("")  # blank line between tables

    # Transition table
    global_lines.append(
        f"static const {state_t} {prefix}_transitions[{num_unique}][{num_cols}] = {{"
    )
    for row_idx, row in enumerate(unique_rows):
        states = row_to_states[row_idx]
        if len(states) == 1:
            comment = f"/* state {states[0]} */"
        else:
            comment = f"/* states {', '.join(str(s) for s in states)} */"
        non_zero = [(i, v) for i, v in enumerate(row) if v != 0]
        if not non_zero:
            row_str = "{ 0 }"
        else:
            entries = []
            for i, v in non_zero:
                if do_alphabet:
                    idx = f"[{i}]"
                else:
                    if 32 <= i <= 126:
                        c = chr(i)
                        if c == "'":
                            idx = "['\\'']"
                        elif c == "\\":
                            idx = "['\\\\']"
                        else:
                            idx = f"['{c}']"
                    else:
                        idx = f"[{i}]"
                entries.append(f"{idx} = {v}")
            row_str = "{ " + ", ".join(entries) + " }"
        global_lines.append(f"    {comment} {row_str},")
    global_lines.append("};")

    # Row-index map: state → index into {prefix}_transitions (only when needed)
    if has_row_map:
        if num_unique <= 256:
            row_t = "uint8_t"
        elif num_unique <= 65536:
            row_t = "uint16_t"
        else:
            row_t = "uint32_t"
        global_lines.append("")  # blank line between tables
        global_lines.append(f"static const {row_t} {prefix}_row_map[{n}] = {{")
        global_lines.append(f"    {', '.join(str(i) for i in state_to_row)}")
        global_lines.append("};")

    globals_str = "\n".join(global_lines)

    # --- build match function -----------------------------------------------
    match_lines: list[str] = []

    # Metadata comment
    if pattern is not None:
        match_lines.append(f'/* regex:                "{pattern}"')
        match_lines.append(f' * flags:                "{flags}"')
        match_lines.append(f' * encoding:             {encoding}')
        match_lines.append(f' * alphabet-compression: {"yes" if do_alphabet else "no"}')
        match_lines.append(f' * row-deduplication:    {"yes" if do_dedup else "no"}')
        match_lines.append(f' * early-exit:           {"yes" if early_exit else "no"}')
        match_lines.append(' */')
    else:
        match_lines.append(f'/* alphabet-compression: {"yes" if do_alphabet else "no"}')
        match_lines.append(f' * row-deduplication:    {"yes" if do_dedup else "no"}')
        match_lines.append(f' * early-exit:           {"yes" if early_exit else "no"}')
        match_lines.append(' */')

    col_expr = (
        f"{prefix}_alphabet[(unsigned char)input[i]]"
        if do_alphabet
        else "(unsigned char)input[i]"
    )
    row_expr = f"{prefix}_row_map[state]" if has_row_map else "state"
    func_name = f"{prefix}_match"

    match_lines.append(f"bool {func_name}(const char *input, size_t len) {{")
    match_lines.append(f"    {state_t} state = {initial};")
    match_lines.append("    for (size_t i = 0; i < len; i++) {")
    match_lines.append(
        f"        state = {prefix}_transitions[{row_expr}][{col_expr}];"
    )
    if early_exit:
        match_lines.append("        if (state == 0) break;")
    match_lines.append("    }")
    match_lines.append(f"    return state >= {first_accept};")
    match_lines.append("}")
    match_str = "\n".join(match_lines)

    # --- build main function ------------------------------------------------
    if emit_main:
        main_lines: list[str] = []
        main_lines.append("int main(int argc, char *argv[]) {")
        main_lines.append("    if (argc != 2) {")
        main_lines.append('        fprintf(stderr, "Usage: %s <input>\\n", argv[0]);')
        main_lines.append("        return 2;")
        main_lines.append("    }")
        main_lines.append(f"    return {func_name}(argv[1], strlen(argv[1])) ? 0 : 1;")
        main_lines.append("}")
        main_str: str | None = "\n".join(main_lines)
    else:
        main_str = None

    return GeneratedCode(
        includes=includes,
        globals=globals_str,
        match_function=match_str,
        main_function=main_str,
    )


def generate(
    pattern: str,
    flags: str = "",
    *,
    emit_main: bool = False,
    prefix: str = "regex",
    encoding: str = "utf8",
    engine: str = "dfa",
    row_dedup: str = "auto",
    alphabet_compression: str = "auto",
    size_threshold: int = 8192,
    early_exit: bool = False,
) -> GeneratedCode:
    """High-level API: compile *pattern* and return generated C code.

    Parameters
    ----------
    pattern:
        Regular expression (re2-compatible subset).
    flags:
        Flag characters: ``i`` (case-insensitive), ``s`` (dot-all),
        ``m`` (multiline), ``x`` (verbose).
    emit_main:
        When ``True``, emit a ``main()`` function that reads ``argv[1]``
        and returns exit-code 0/1/2.
    prefix:
        Prefix for all generated C identifiers: arrays are named
        ``{prefix}_transitions``, ``{prefix}_alphabet``,
        ``{prefix}_row_map`` and the match function is named
        ``{prefix}_match``.  Defaults to ``"regex"``.
    encoding:
        ``"utf8"`` (default) for Unicode/UTF-8 semantics; ``"bytes"`` for
        raw byte semantics where ``.`` matches any single byte and
        literals/classes operate on byte values 0-255.
    engine:
        ``"dfa"`` (default) uses the table-driven minimised-DFA backend.
        ``"bitnfa"`` uses a bit-parallel NFA backend.
    row_dedup:
        ``"yes"`` always deduplicate identical transition rows,
        ``"no"`` never, ``"auto"`` (default) only when the uncompressed
        table exceeds *size_threshold* cells.  (DFA only.)
    alphabet_compression:
        ``"yes"`` always compress the byte alphabet into equivalence
        classes, ``"no"`` never, ``"auto"`` (default) only when the
        uncompressed table exceeds *size_threshold* cells.  (DFA only.)
    size_threshold:
        Number of transition-table cells (states × 256) above which
        ``"auto"`` mode enables the corresponding optimisation.
        Defaults to ``8192``.  (DFA only.)
    early_exit:
        When ``True``, emit ``if (state == 0) break;`` inside the DFA
        loop so matching terminates as soon as the dead state is reached.
        Defaults to ``False``.  (DFA only.)

    Returns
    -------
    GeneratedCode
        Structured result containing the separate parts of the generated C
        code.  Call :meth:`~GeneratedCode.render` to obtain the combined
        source string.
    """
    if engine == "bitnfa":
        from .codegen_bitnfa import generate_bitnfa_c_code

        nfa = compile_nfa(pattern, flags, encoding=encoding)
        return generate_bitnfa_c_code(
            nfa,
            prefix=prefix,
            emit_main=emit_main,
            pattern=pattern,
            flags=flags,
            encoding=encoding,
        )

    dfa = compile_regex(pattern, flags, encoding=encoding)
    return generate_c_code(
        dfa,
        prefix=prefix,
        emit_main=emit_main,
        row_dedup=row_dedup,
        alphabet_compression=alphabet_compression,
        size_threshold=size_threshold,
        pattern=pattern,
        flags=flags,
        encoding=encoding,
        early_exit=early_exit,
    )
