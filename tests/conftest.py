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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_codegen_cases() -> list[dict]:
    with open(CODEGEN_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["test_cases"]


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, "workerinput"):
        config._codegen_cases = config.workerinput["codegen_cases"]
    else:
        config._codegen_cases = _load_codegen_cases()


if HAS_XDIST:
    def pytest_configure_node(node) -> None:
        node.workerinput["codegen_cases"] = node.config._codegen_cases


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
