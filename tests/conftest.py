from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:
    import xdist  # noqa: F401
except ImportError:
    HAS_XDIST = False
else:
    HAS_XDIST = True

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEGEN_JSON = ROOT / "tests" / "data" / "re2_compat_results.json"
CODEGEN_ERRORS_JSON = ROOT / "tests" / "data" / "re2_compat_errors.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_codegen_cases() -> list[dict]:
    with open(CODEGEN_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["test_cases"]


def _load_codegen_errors() -> dict[tuple[str, int], str]:
    with open(CODEGEN_ERRORS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        (entry["engine"], entry["case_idx"]): entry["error"]
        for entry in data["errors"]
    }


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, "workerinput"):
        config._codegen_cases = config.workerinput["codegen_cases"]
        config._codegen_errors = config.workerinput["codegen_errors"]
    else:
        config._codegen_cases = _load_codegen_cases()
        config._codegen_errors = _load_codegen_errors()


if HAS_XDIST:
    def pytest_configure_node(node) -> None:
        node.workerinput["codegen_cases"] = node.config._codegen_cases
        node.workerinput["codegen_errors"] = node.config._codegen_errors


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.module.__name__ != "tests.test_codegen" or "case_idx" not in metafunc.fixturenames:
        return

    params: list[pytest.ParameterSet] = []
    for idx, tc in enumerate(metafunc.config._codegen_cases):
        if not tc["subjects"]:
            continue
        safe = tc["pattern"][:35].replace("\n", "\\n").replace("\r", "\\r")
        params.append(pytest.param(idx, id=f"tc{idx}_{safe}"))
    metafunc.parametrize("case_idx", params)


@pytest.fixture(scope="session")
def codegen_cases(pytestconfig: pytest.Config) -> list[dict]:
    return pytestconfig._codegen_cases


@pytest.fixture(scope="session")
def codegen_errors(pytestconfig: pytest.Config) -> dict[tuple[str, int], str]:
    return pytestconfig._codegen_errors
