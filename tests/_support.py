from __future__ import annotations

import subprocess
from pathlib import Path

from emx_regex_cgen import generate

HARNESS_TEMPLATE = """#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

bool __FUNC_NAME__(const char *input, size_t len);

int main(int argc, char *argv[]) {
    FILE *fp;
    long size;
    size_t len;
    char *buf = NULL;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input-file>\\n", argv[0]);
        return 2;
    }

    fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        perror("fopen");
        return 2;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(fp);
        return 2;
    }
    size = ftell(fp);
    if (size < 0) {
        perror("ftell");
        fclose(fp);
        return 2;
    }
    rewind(fp);
    len = (size_t)size;
    if (len > 0) {
        buf = (char *)malloc(len);
        if (buf == NULL) {
            fprintf(stderr, "malloc failed\\n");
            fclose(fp);
            return 2;
        }
        if (fread(buf, 1, len, fp) != len) {
            fprintf(stderr, "fread failed\\n");
            free(buf);
            fclose(fp);
            return 2;
        }
    }
    fclose(fp);

    if (__FUNC_NAME__(buf != NULL ? buf : "", len)) {
        free(buf);
        return 0;
    }

    free(buf);
    return 1;
}
"""


def build_matcher(
    pattern: str,
    tmp_path: Path,
    *,
    engine: str = "dfa",
    flags: str = "",
    encoding: str = "utf8",
    **kwargs,
) -> Path:
    """Generate, write and compile a matcher executable."""
    prefix = kwargs.get("prefix", "regex")
    func_name = f"{prefix}_match"
    c_code = generate(
        pattern,
        flags=flags,
        emit_main=False,
        engine=engine,
        encoding=encoding,
        **kwargs,
    ).render()
    generated_c = tmp_path / "generated.c"
    generated_c.write_text(c_code, encoding="utf-8")

    harness_c = tmp_path / "harness.c"
    harness = HARNESS_TEMPLATE.replace("__FUNC_NAME__", func_name)
    harness_c.write_text(harness, encoding="utf-8")

    exe = tmp_path / "test.exe"
    comp = subprocess.run(
        ["gcc", "-O2", "-o", str(exe), str(generated_c), str(harness_c)],
        capture_output=True,
        timeout=30,
    )
    assert comp.returncode == 0, f"gcc failed:\n{comp.stderr.decode(errors='replace')}"
    return exe


def run_matcher(
    exe: Path,
    inp: str | bytes,
    tmp_path: Path,
    *,
    text_encoding: str = "utf-8",
) -> bool:
    """Run *exe* against *inp* via a temporary input file."""
    data = inp.encode(text_encoding) if isinstance(inp, str) else inp
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(data)
    result = subprocess.run([str(exe), str(input_file)], capture_output=True, timeout=10)
    return result.returncode == 0
