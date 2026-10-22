# SPDX-License-Identifier: Apache-2.0
"""What the installed native libraries need from the system (docs/SUPPORT.md, docs/BUILD.md).

- The glibc symbol versions: no library may need a newer ``GLIBC_x.y`` than the one SUPPORT.md's
  glibc row states for all of them (a front built against glibc 2.38 once needed
  ``__isoc23_strtol@GLIBC_2.38`` through ``std::atoi``, which glibc 2.34-2.37 systems lack).
- No ``RUNPATH`` or ``RPATH``: the front finds its cores in its own directory, and nothing else
  is searched for.
- The CUDA runtime is linked statically into every GPU core: no ``libcudart`` in ``NEEDED``, and
  the runtime's own strings (its error names, and the driver library it opens) are inside the
  core. tests/test_devices.py checks that a decode maps no libcudart.

The checks read the installed libraries next to the package and skip without them (a source-tree
run) or without ``objdump`` / ``readelf``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import eldpc

PACKAGE = Path(__file__).resolve().parent.parent
SUPPORT = PACKAGE / "docs" / "SUPPORT.md"
#: The glibc version SUPPORT.md states, used when the tests run without the source tree's docs.
DEFAULT_GLIBC = "2.34"
GLIBC_ROW = re.compile(r"^\| glibc \|[^|]*\| `GLIBC_([0-9.]+)` for all seven libraries", re.M)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("."))


def _libs() -> list[Path]:
    libs = sorted((Path(eldpc.__file__).resolve().parent / "_libs").glob("*.so"))
    if not libs:
        pytest.skip("no installed native libraries next to the package (source-tree run)")
    return libs


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not available")
    return path


def supported_glibc() -> str:
    """The number of SUPPORT.md's glibc row, or DEFAULT_GLIBC without the docs."""
    try:
        text = SUPPORT.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_GLIBC
    m = GLIBC_ROW.search(text)
    assert m, "docs/SUPPORT.md: no glibc row of the form '| glibc | ... | `GLIBC_x.y` for all seven libraries'"
    return m.group(1)


def test_support_states_the_glibc_version_these_tests_default_to():
    if not SUPPORT.exists():
        pytest.skip("docs/SUPPORT.md is not next to the tests")
    assert supported_glibc() == DEFAULT_GLIBC


def test_glibc_symbol_versions_are_at_most_the_supported_one():
    objdump = _tool("objdump")
    stated = supported_glibc()
    libs = _libs()
    assert len(libs) in (3, 7), [p.name for p in libs]   # CPU-only build: front + two CPU cores
    for lib in libs:
        out = subprocess.run([objdump, "-T", str(lib)], capture_output=True, text=True, check=True).stdout
        found = {}
        for line in out.splitlines():
            for v in re.findall(r"\bGLIBC_([0-9]+(?:\.[0-9]+)+)\b", line):
                found.setdefault(v, []).append(line.split()[-1])
        assert found, f"{lib.name}: no GLIBC_ symbol version found"
        newest = max(found, key=_version)
        assert _version(newest) <= _version(stated), (
            f"{lib.name} needs GLIBC_{newest} ({', '.join(sorted(set(found[newest]))[:5])}), "
            f"but docs/SUPPORT.md states GLIBC_{stated} for all seven libraries")


def test_libraries_have_no_runpath_or_rpath():
    readelf = _tool("readelf")
    for lib in _libs():
        out = subprocess.run([readelf, "-d", str(lib)], capture_output=True, text=True, check=True).stdout
        assert "(RUNPATH)" not in out and "(RPATH)" not in out, f"{lib.name} has a RUNPATH or RPATH"


def test_gpu_cores_carry_the_cuda_runtime_statically():
    readelf = _tool("readelf")
    cores = [p for p in _libs() if p.name.startswith("libeldpc_core_")]
    if not cores:
        pytest.skip("this build has no GPU cores")
    for lib in cores:
        out = subprocess.run([readelf, "-d", str(lib)], capture_output=True, text=True, check=True).stdout
        needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)
        assert not [n for n in needed if "cudart" in n], f"{lib.name} needs {needed}"
        data = lib.read_bytes()
        assert b"cudaErrorMemoryAllocation" in data, f"{lib.name}: no CUDA runtime error names inside"
        assert b"libcuda.so.1" in data, f"{lib.name}: the runtime's driver library name is not inside"
