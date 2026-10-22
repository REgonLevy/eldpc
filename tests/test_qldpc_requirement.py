# SPDX-License-Identifier: Apache-2.0
"""eldpc.qldpc names the qLDPC it needs and the install command when qLDPC is missing or too old.

These tests run in a subprocess with a hidden or a stand-in qLDPC, so they need neither qLDPC nor
a GPU and run in every environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

INSTALL = 'pip install ".[qldpc]"'
REQUIREMENT = "qLDPC 0.4.2.dev20261004 (qLDPC git main, commit 4c4f801) or newer"


def _import_layer(prelude: str, extra_path: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if extra_path is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(extra_path)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    code = f"{prelude}\nimport eldpc.qldpc\n"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120, check=False)


def _last_error_line(proc: subprocess.CompletedProcess) -> str:
    lines = [line for line in proc.stderr.splitlines() if "Error: " in line]
    assert lines, proc.stderr
    return lines[-1]


def test_missing_qldpc_names_the_requirement_and_the_install_command():
    proc = _import_layer("import sys; sys.modules['qldpc'] = None")
    assert proc.returncode != 0
    line = _last_error_line(proc)
    assert line.startswith("ModuleNotFoundError: eldpc.qldpc needs " + REQUIREMENT), line
    assert INSTALL in line


def test_a_qldpc_directory_without_the_package_counts_as_missing(tmp_path):
    # tests/qldpc, or any directory named qldpc on sys.path, imports as a namespace package when no
    # qLDPC is installed; with an installed qLDPC the package wins, so hide it as well.
    (tmp_path / "qldpc").mkdir()
    proc = _import_layer(
        "import sys, importlib.machinery\n"
        "class _Hide:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'qldpc':\n"
        f"            return importlib.machinery.PathFinder.find_spec(name, [{str(tmp_path)!r}])\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Hide())", extra_path=tmp_path)
    assert proc.returncode != 0
    line = _last_error_line(proc)
    assert line.startswith("ModuleNotFoundError: eldpc.qldpc needs " + REQUIREMENT), line
    assert "directory without the package" in line and INSTALL in line


def test_too_old_qldpc_names_its_version_and_the_install_command(tmp_path):
    # A stand-in for the released qLDPC 0.4.1: the modules exist, the decoder-specification API
    # that eldpc.qldpc imports does not (0.4.1 lacks ErrorDecodeResult).
    root = tmp_path / "qldpc"
    (root / "decoders" / "construction").mkdir(parents=True)
    (root / "__init__.py").write_text('__version__ = "0.4.1"\n')
    (root / "decoders" / "__init__.py").write_text("")
    (root / "decoders" / "common.py").write_text("PLACEHOLDER_ERROR_RATE = 1e-3\n")
    (root / "decoders" / "construction" / "__init__.py").write_text("")
    (root / "decoders" / "construction" / "specs.py").write_text(
        "PcmOrDem = object\n\ndef decoder_spec(*args, **kwargs):\n    raise NotImplementedError\n")
    (root / "decoders" / "protocols.py").write_text(
        "class BatchErrorDecoder:\n    pass\n\nclass BatchObservableDecoder:\n    pass\n")
    proc = _import_layer("", extra_path=tmp_path)
    assert proc.returncode != 0
    line = _last_error_line(proc)
    assert line.startswith("ImportError: eldpc.qldpc needs " + REQUIREMENT), line
    assert "installed qLDPC 0.4.1" in line and "ErrorDecodeResult" in line
    assert "0.4.0 and 0.4.1 are too old" in line and INSTALL in line


def test_the_layer_imports_with_a_supported_qldpc():
    proc = _import_layer("")
    if proc.returncode != 0 and "eldpc.qldpc needs" in proc.stderr:
        pytest.skip("no supported qLDPC is installed here (the tests above cover that case)")
    assert proc.returncode == 0, proc.stderr
