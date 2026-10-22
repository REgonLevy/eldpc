# SPDX-License-Identifier: Apache-2.0
"""The CPU core sources: pinned by src/cpu/SOURCES.sha256 (raw bytes) and src/cpu/TOKENS.sha256 (text
without comments)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eldpc_testlib import cpu_core

TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent
CPU = PACKAGE / "src" / "cpu"
CORE = PACKAGE / "src" / "core"

# The four CPU core sources: the only new source is the shim; the two OSD-0 files and the BP
# engine are copies of their originals with the same code (the OSD-0 files are byte-identical to
# the ones of src/core).
SOURCES = ("fastbpcpu6g.cpp", "bp_engine.hpp", "osd0_engine.hpp", "osd_caps.hpp")
OSD_FILES = ("osd0_engine.hpp", "osd_caps.hpp")

pytestmark = cpu_core


def read_manifest() -> dict[str, str]:
    entries = {}
    for line in (CPU / "SOURCES.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        assert len(digest) == 64
        assert name not in entries
        entries[name] = digest
    return entries


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cpu_manifest_lists_exactly_the_four_sources():
    assert set(read_manifest()) == set(SOURCES)


@pytest.mark.parametrize("name", SOURCES)
def test_cpu_file_matches_manifest(name):
    manifest = read_manifest()
    path = CPU / name
    assert path.is_file(), f"{path} is missing"
    assert sha256(path) == manifest[name], f"{name} does not match SOURCES.sha256"


def test_cpu_dir_has_no_extra_files():
    present = {p.name for p in CPU.iterdir() if p.is_file()}
    assert present == set(SOURCES) | {"SOURCES.sha256", "TOKENS.sha256"}


@pytest.mark.parametrize("name", OSD_FILES)
def test_cpu_osd_files_are_the_core_osd_files(name):
    """The CPU core carries its own copies of the two OSD-0 files; they equal src/core's."""
    assert (CPU / name).read_bytes() == (CORE / name).read_bytes(), \
        f"src/cpu/{name} differs from src/core/{name}"
