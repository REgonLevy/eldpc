# SPDX-License-Identifier: Apache-2.0
"""The README's deterministic quick starts print what the README says they print.

Each Python block is taken from README.md itself and run in a fresh interpreter in a temporary
directory, so the test follows the README text; the C quick start is docs/tools/example.c, which
docs/tools/test_results_tools.py checks is the README's program. The qLDPC quick start is not
here: its estimator samples unseeded errors, so its value changes from run to run (the README
gives a range). The code-to-DEM numbers depend on the qLDPC and stim versions the README names,
and the test skips with other versions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import eldpc
from eldpc_testlib import cpu_core, requires_gpu

PACKAGE = Path(__file__).resolve().parent.parent
README = PACKAGE / "README.md"
EXAMPLE_C = PACKAGE / "docs" / "tools" / "example.c"
#: The versions the README's code-to-DEM numbers were produced with.
QLDPC_VERSION = "0.4.2.dev20261004"
STIM_VERSION = "1.16.0"


def readme_block(heading: str, language: str = "python", index: int = 0) -> str:
    """The index-th {language} block of the README section under '## {heading}'."""
    if not README.exists():
        pytest.skip("README.md is not next to the tests")
    text = README.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"README.md has no section '## {heading}'"
    blocks = re.findall(rf"^```{language}\n(.*?)^```", m.group(1), re.S | re.M)
    assert len(blocks) > index, f"README.md has no {language} block {index} under '## {heading}'"
    return blocks[index]


def run_block(code: str, cwd: Path) -> list[str]:
    env = os.environ.copy()
    env.pop("ELDPC_ISA", None)
    package_parent = str(Path(eldpc.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          cwd=str(cwd), timeout=600)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - not installed
        return None


@requires_gpu
def test_quick_start_from_a_stim_dem(tmp_path):
    lines = run_block(readme_block("Quick start: Python from a stim detector error model"), tmp_path)
    assert lines == ["[1 0 0]", "[0]", "False 3"]


@requires_gpu
def test_quick_start_from_h_o_and_priors(tmp_path):
    lines = run_block(readme_block("Quick start: Python from H, O and priors"), tmp_path)
    assert lines == ["[1 0 0] [1] 131073"]


@requires_gpu
def test_quick_start_c_program_output(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    package = Path(eldpc.__file__).resolve().parent
    if not (package / "include" / "eldpc.h").is_file() or not EXAMPLE_C.is_file():
        pytest.skip("the installed C header or docs/tools/example.c is not available")
    libs = package / "_libs"
    exe = tmp_path / "example"
    build = subprocess.run([cc, f"-I{package / 'include'}", str(EXAMPLE_C), f"-L{libs}", "-leldpc",
                            f"-Wl,-rpath,{libs}", "-o", str(exe)], capture_output=True, text=True,
                           timeout=300)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == ["correction 10000, observable 0, 3 iterations"]


@requires_gpu
def test_quick_start_from_a_code_to_a_dem_and_a_decode(tmp_path):
    pytest.importorskip("qldpc")
    if _version("qldpc") != QLDPC_VERSION or _version("stim") != STIM_VERSION:
        pytest.skip(f"the README's numbers are for qLDPC {QLDPC_VERSION} and stim {STIM_VERSION} "
                    f"(installed: {_version('qldpc')}, {_version('stim')})")
    lines = run_block(readme_block("Quick start: from a code to a circuit, a DEM and a decode"), tmp_path)
    assert "m=504 n=5184 k=12" in lines[0], lines[0]
    assert lines[1] == "logical error rate 0.0043"
    assert (tmp_path / "model.dem").is_file()


@cpu_core
def test_quick_start_cpu_core(tmp_path):
    if _version("stim") != STIM_VERSION:
        pytest.skip(f"the README's value is for stim {STIM_VERSION} (installed: {_version('stim')})")
    lines = run_block(readme_block("CPU core"), tmp_path)
    assert lines == ["cpu cpu", "logical error rate 0.0"]


@cpu_core
def test_cpu_core_cpus_example(tmp_path):
    """The README's cpus example: the preset string with ;cpus= and the dict built from the
    preset's keys both create a CPU decoder, and neither raises the placement warning."""
    if not {1, 2, 3, 4} <= os.sched_getaffinity(0):
        pytest.skip("the README's example pins to CPUs 1-4, which this process may not use")
    code = ("import warnings; warnings.simplefilter('error', RuntimeWarning)\n"
            + readme_block("CPU core", index=1))
    assert "cpus=1,2,3,4" in code and 'eldpc.presets()["a03125c3"]' in code
    assert run_block(code, tmp_path) == ["cpu", "cpu"]
