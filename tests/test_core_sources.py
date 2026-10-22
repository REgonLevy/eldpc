# SPDX-License-Identifier: Apache-2.0
"""The decoder sources: pinned by src/core/SOURCES.sha256 (raw bytes) and src/core/TOKENS.sha256
(text with the comments removed: identical to the benched sources, which differ only in comments)."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent
CORE = PACKAGE / "src" / "core"
# Optional external references (the tests that need them skip when they are not set):
#   ELDPC_CORE_SNAPSHOT  a directory holding the six decoder sources of the reference tree
#   ELDPC_CORE_TARBALL   the decoder source tarball the sources came from
SNAPSHOT = Path(os.environ.get("ELDPC_CORE_SNAPSHOT", "/nonexistent"))
TARBALL = Path(os.environ.get("ELDPC_CORE_TARBALL", "/nonexistent"))

SOURCES = ("fastbpgpu6g.cpp", "bp_gpu.cu", "bp_engine.hpp", "osd0_engine.hpp", "osd_caps.hpp",
           "fastbp_ext.h")


def read_manifest() -> dict[str, str]:
    entries = {}
    for line in (CORE / "SOURCES.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        assert len(digest) == 64
        assert name not in entries
        entries[name] = digest
    return entries


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_lists_exactly_the_six_sources():
    assert set(read_manifest()) == set(SOURCES)


@pytest.mark.parametrize("name", SOURCES)
def test_core_file_matches_manifest(name):
    manifest = read_manifest()
    path = CORE / name
    assert path.is_file(), f"{path} is missing"
    assert sha256(path) == manifest[name], f"{name} does not match SOURCES.sha256"


def test_core_dir_has_no_extra_files():
    present = {p.name for p in CORE.iterdir() if p.is_file()}
    assert present == set(SOURCES) | {"SOURCES.sha256", "TOKENS.sha256"}


def _comment_tool():
    spec = importlib.util.spec_from_file_location("check_comment_only",
                                                  PACKAGE / "tools" / "check_comment_only.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="ELDPC_CORE_SNAPSHOT is not set to a directory")
@pytest.mark.parametrize("name", SOURCES)
def test_core_file_matches_the_snapshot_except_comments(name):
    tool = _comment_tool()
    ours = (CORE / name).read_bytes().decode("utf-8")
    theirs = (SNAPSHOT / name).read_bytes().decode("utf-8")
    assert tool.strip_comments(ours) == tool.strip_comments(theirs), \
        f"{name}: code tokens or line structure differ from {SNAPSHOT / name}"


@pytest.mark.skipif(not TARBALL.is_file(), reason="ELDPC_CORE_TARBALL is not set to a file")
def test_tarball_matches_manifest():
    proc = subprocess.run([sys.executable, str(PACKAGE / "tools" / "check_core_tarball.py"), str(TARBALL)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
